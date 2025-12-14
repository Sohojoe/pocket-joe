# Refactor Messages

## Overview

I need messages to support images / media, so now is a good time to refactor the message layer.

This refactor is **not just about media** — it’s about cleaning up conceptual leaks that are already showing up as the framework grows (options, durability, multiple policies, etc).

### Current pain points

- `actor` is overloaded and confusing:
  - sometimes means *which policy created the message*
  - sometimes is abused as *LLM role* (`user`, `assistant`, etc.)
- `action_call` / `action_result` names leak provider terminology
- payloads are untyped `dict[str, Any]`, but the code already assumes hidden schemas
- no clean way to interleave text + media
- upcoming durability / replay work will make implicit assumptions explode

---

## Design principles

1. **Framework semantics first**

   - Policies choose options
   - Options produce results
   - Messages are observations/actions/events

2. **Provider semantics are adapter concerns**

   - OpenAI / Gemini roles, tool-call pairing rules, etc. do *not* leak into core types

3. **Strong typing over conventions**

   - Use Pydantic discriminated unions
   - Eliminate "stringly-typed" payload assumptions

4. **Media is URL-based and modality-aware**

   - Do not require exact MIME
   - Distinguish image vs audio vs video vs document

---

## Proposed Message shape (conceptual)

A `Message` is still the atomic, immutable unit flowing through the system.

### Replace `actor`

Replace `actor` with two explicit concepts:

- `policy: str`
  - Which policy created the message
  - For `option_call`: the *caller* policy (the decider)
  - For `option_result`: the *invoked* policy (the option that ran)
  - Reserve a policy name like `"runtime"` / `"executor"` for executor-generated errors/results when appropriate

- `role_hint_for_llm: Optional[Literal["user","assistant","tool","system"]]`
  - **Hint only** for provider adapters
  - Never used by the core framework for semantics

---

## Rename message kinds

Rename provider-leaky terminology:

- `action_call`   → `option_call`
- `action_result` → `option_result`

Rationale:

- "option" matches the options framework
- "tool" is provider jargon
- keeps POMDP framing clean

---

## Message kinds (top-level)

Keep **exactly three** top-level kinds:

1. `parts`

   - natural language + media
   - observations or prompts

2. `option_call`

   - a policy-selected action
   - contains option name + arguments

3. `option_result`

   - result of executing an option
   - success or error

Adapters may project these into provider-specific structures. For OpenAI specifically, adapters should serialize only complete `option_call`+`option_result` pairs (same invocation_id) and skip incomplete calls.

---

## Parts model (strongly typed)

### TextPart

```python
class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
```

### MediaPart

Media is **URL-only**.

```python
class MediaPart(BaseModel):
    kind: Literal["media"] = "media"

    modality: Literal[
        "image",
        "audio",
        "video",
        "document",
    ]

    url: HttpUrl

    # optional metadata
    mime: Optional[str] = None

    # prompt sugar only (labels, descriptions)
    prompt_hint: Optional[str] = None

    # internal correlation / replay aid
    id: Optional[str] = None
```

Notes:

- `modality` is required and semantic
- `mime` is optional, best-effort metadata
- `prompt_hint` is *explicitly* non-semantic

---

## Step grouping

Add:

```python
step_num: Optional[int]
```

Semantics:

- Groups messages produced in the same logical timestep / decision cycle
- Orthogonal to tracing and durability ledger
- The ordering of events is preserved by the ledger / replay system (deterministic replay)
- Builder supports both:
  - `next_step(...)` (advance step_num)
  - `continue_step(...)` (append more messages/parts within the current step)

Used for:
- collapsing events into transitions
- debugging / visualization

---

## Option payloads (strongly typed)

### OptionCall

```python
class OptionCallPayload(BaseModel):
    kind: Literal["option_call"] = "option_call"
    invocation_id: str  # unique per call (pairs call ↔ result)
    option_name: str
    arguments: dict[str, Any]
```

### OptionResult

```python
class OptionResultPayload(BaseModel):
    kind: Literal["option_result"] = "option_result"
    invocation_id: str  # must match the corresponding option_call
    option_name: str
    result: Any
    is_error: bool = False

    # minimal structured error (used when is_error=True)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retryable: Optional[bool] = None
```

Notes:

- Validation of `arguments` is deferred (future work)
- If arguments fail validation later, the executor/runtime returns:
  - `option_result(is_error=True, error_type=..., error_message=..., retryable=...)`
  - and does **not** invoke the target option/policy

---

## MessageBuilder

Introduce a **MessageBuilder** for ergonomic construction.

Goals:

- eliminate boilerplate
- enforce ordering and consistency
- optionally anchor to previous message / step

### Builder variants (recommended)

- `MessageBuilder.next_step(last_message_or_step)`
  - creates a builder for the next step (increments step_num)

- `MessageBuilder.continue_step(last_message_or_step)`
  - creates a builder that appends within the current step (keeps step_num)

- `OptionResultBuilder.response_to(option_call_message)`
  - constrained builder for producing an `option_result`
  - automatically copies `invocation_id`, `option_name`, and `step_num`
  - provides:
    - `success(result)`
    - `error(error_type, error_message, retryable=False)`
  - throws if you attempt to set a mismatched invocation_id / option_name

### Example usage

```python
builder = MessageBuilder.continue_step(last_message)

builder.add_option_call("search_web", {"query": "hobbit movie"})
# ... executor runs it ...
result_builder = OptionResultBuilder.response_to(builder.last_option_call)
option_result_msg = result_builder.success(results)

builder.add_image(
    "girl wearing a hat",
    url_1,
)

builder.add_image(
    "fancy dress",
    url_2,
)

builder.add_text(
    "Take the dress from the second image and apply it to the person in the first image"
)

message = builder.to_message()
```

### Builder responsibilities

- Maintain `step_num`
- Generate a unique `invocation_id` for each `option_call`
- Ensure `option_call` ↔ `option_result` pairing consistency via `invocation_id`
- Assign stable internal media ids (optional)
- Provide a predictable policy/role_hint defaulting strategy
- Produce a single `parts` message (keep builders simple; if you ever need multiple messages, use multiple builders or emit messages directly)

---

## Adapter responsibilities (explicitly *out of scope* here)

Adapters (OpenAI, Gemini, etc.) are responsible for:

- mapping `role_hint_for_llm` → provider roles
- rendering `parts` into provider-specific content/parts arrays
- inserting prompt sugar labels ("Media 1:", etc.) if desired
- pairing `option_call` + `option_result` according to provider rules
- skipping incomplete option calls

---

## Non-goals (for this refactor)

- schema validation of option arguments
- durability / ledger implementation
- provider-specific tuning knobs (e.g., OpenAI image detail)
- parallel tool-call orchestration rules

These will be layered on later.

---

## Outcome

After this refactor:

- Message semantics are clean, explicit, and strongly typed
- Media support is first-class and extensible
- Provider quirks are isolated to adapters
- The framework remains aligned with POMDP / options thinking
- Future durability work won’t require undoing this layer

