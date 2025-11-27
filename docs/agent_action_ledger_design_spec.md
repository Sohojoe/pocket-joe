# Agent Design Spec

## 1. Purpose

Define a minimal, extensible model for agent behaviour where everything is expressed as:

- **Policies**: functions that reason over a given payload and set of actions. These are the nodes.
- **Actions**: a touple with policy_name, payload and the set of actions. These are the edges.

This spec is intended to unify LLMs, humans, and heuristic code under the same interface.

---

## 2. Core Concepts

### 2.1 Policy

A **policy** is an async function that:

- That decides actions over a given payload
- Optionally calls other policies via `ctx.call()`
- Returns the set of selected actions as a list of steps

**Key principle**: Policies must be deterministic. Non-deterministic operations (LLM calls, API requests) are encapsulated as sub-policies for replay/durability.
s
Examples:

- LLM Policy (calls OpenAI/Anthropic)
- Tool policies (web search, calculator, database query)
- Orchestrator policies (coordinates multiple sub-policies)

### 2.2 Action

The **Action** dataclass is the edge that is passed into a policy. A touple with policy_name, payload and the set of actions.
The **actions** is the set of actions a policy can choose from.

In the ledger, actions are recording it two stages

- `action_call`: The request (which policy, with what payload)
- `action_result`: The response (what the policy returned)

### 2.3 Ledger

The **ledger** is an append-only sequence of `Step` objects. It is the **single source of truth**.

- `ledger`: Each policy has its own ledger. When first invoked, it starts empty (or seeded). As the policy executes and calls `ctx.call()`, results are appended. Sub-policies get a new empty ledger—they don't see parent history.

> There is no "global ledger." Each policy owns its ledger. Context must be passed via `payload`.

**Primary Purpose: Replay & Durability.** If a policy crashes, re-run it with the same ledger. The engine checks for cached `action_result` steps—if found, execution is skipped and the cached result is returned. This guarantees exactly-once semantics.

---

## 3. Data Structures

### 3.1 Step (Immutable)

Steps are **fully immutable**. We use `@dataclass(frozen=True)` to guarantee no field can be changed after creation.

```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass(frozen=True)
class Step:
    id: str                    # Unique identifier (engine-generated)
    actor: str                 # e.g. "user", "assistant", "get_weather"
    type: str                  # e.g. "text", "action_call", "action_result"
    payload: Dict[str, Any]    # JSON-serializable data
```

Notes:

- `payload` must be JSON-serializable for ledger persistence.
- Immutability ensures deterministic replay.

---

### 3.2 Ledger (Fully Immutable)

The **ledger** is represented as a frozen tuple of `Step`s. Each append returns a **new Ledger** with one or more additional steps.

This guarantees:

- append‑only semantics
- deterministic replay
- snapshot durability
- crash recovery without side‑effects

```python
from dataclasses import dataclass
from typing import Tuple, Iterable

@dataclass(frozen=True)
class Ledger:
    steps: Tuple[Step, ...] = ()

    def append(self, step: Step) -> "Ledger":
        """Return a new Ledger with one additional Step."""
        return Ledger(steps=self.steps + (step,))

    def extend(self, new_steps: Iterable[Step]) -> "Ledger":
        """Return a new Ledger with multiple additional Steps."""
        return Ledger(steps=self.steps + tuple(new_steps))

    def __iter__(self):
        return iter(self.steps)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]
```

This immutable structure allows the engine to:

- rebuild state from ledger snapshots
- replay policies safely
- skip re‑executing an action if an `action_result` already exists

---

### 3.3 Action

```python
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any

@dataclass(frozen=True)
class Action:
    policy: str          # which policy is being invoked
    payload: Dict[str, Any] = field(default_factory=dict)  # arguments / input for this policy
    actions: Set[str] = field(default_factory=set)  # sub-policies this policy can call
```

Conventions:

- `payload` is **always** a dict (no `None`).
- For the **root LLM call**, `payload` encodes the user/system input for this tick (e.g. `{ "text": "..." }`).
- For **sub-policies/tools**, `payload` is the arguments from the `action_call` step (e.g. `{ "city": "Seattle" }`).

---

### 3.4 Policy Interface

```python
from typing import List, Callable, Awaitable

# A Policy is an async function that takes an Action and a Context,
# and returns a list of Steps (the record of what it did).
Policy = Callable[[Action, Context], Awaitable[List[Step]]]
```

Policies:

- read their input arguments (`action.payload`)
- optionally inspect `action.actions` to know which other policies they *may* call
- **invoke sub-policies** via `await ctx.call(...)`
- reason of all steps via the local ledger (`ctx.get_ledger()`)
- return a list of new Steps (including optional `action_result`)

### 3.5 Context Interface

```python
class Context(Protocol):
    async def call(self, action: Action, decorators: list[Callable]) -> list[Step]: ...
    async def get_ledger(self) -> list[Step]: ...
```

Policies access configuration (API keys, etc.) via the Context. Implementation details (env vars, config files, etc.) are left to the runtime.

### 3.6 Registry

The **Registry** maps policy names to functions (and optionally metadata for tool schemas).

```python
PolicyRegistry = Dict[str, Policy]  # Simple: just name -> function

# Or with metadata for LLM tool definitions:
@dataclass
class PolicyMetadata:
    func: Policy
    description: str
    input_schema: Dict[str, Any]  # JSON Schema
```

Implementation can use decorators (like MCP) for registration.

---

## 4. Decorators

Decorators wrap policies to add behavior without modifying the policy itself.

### 4.1 Loop Wrapper

Runs a policy repeatedly until termination (e.g., no more `action_call` steps).

```python
@loop_wrapper(max_turns=10)
async def my_agent(action: Action, ctx: Context) -> List[Step]:
    # LLM decides whether to call tools or return final answer
    ...
```

The decorator:

1. Runs the policy
2. Checks if there are `action_call` steps
3. If yes, loops (up to `max_turns`)
4. If no, returns

### 4.2 Invoke Action

Executes `action_call` steps emitted by a policy.

```python
@invoke_action()
async def orchestrator(action: Action, ctx: Context) -> List[Step]:
    # Policy decides to call tools, decorator executes them
    ...
```

The decorator:

1. Inspects returned steps for `action_call`
2. For each: `await ctx.call(policy_name, payload)`
3. Appends `action_result` to ledger

**Typical pattern**: Combine both:

```python
decorators = [loop_wrapper(max_turns=5), invoke_action()]
result = await ctx.call("llm_policy", action, decorators)
```

---

## 5. Canonical Step Shapes

### 5.1 Text Step

```python
Step(
    id="...",
    actor="user",  # or "assistant", "human", etc.
    type="text",
    payload={
        "text": "What's the weather in Seattle?",
    },
)
```

### 5.2 Action Call Step

Represents a request from one policy to another.

```python
Step(
    id="...",
    actor="assistant",
    type="action_call",
    payload={
        "policy": "get_weather",
        "payload": {"city": "Seattle"},
    },
)
```

### 5.3 Action Result Step

Represents the result of executing an action/policy.

```python
Step(
    id="...",
    actor="get_weather",
    type="action_result",
    payload={"city": "Seattle", "temp_c": 12},
)
```

### 5.4 Error Result Step

Represents a recoverable error from a tool/sub-policy (Error as Data).

```python
Step(
    id="...",
    actor="get_weather",
    type="action_result",
    payload={
        "error": True,
        "code": "NOT_FOUND",
        "message": "City 'Seattlé' not found. Did you mean 'Seattle'?",
    },
)
```

The calling policy (e.g., LLM) sees this in its ledger and can decide to retry, apologize, or try a different approach.

---

## 6. Example: Search Agent

A simple ReAct-style agent that can search the web to answer questions.

### 6.1 Simple example of OpenAI completions llm api

```python
async def llm_policy(action: Action, ctx: Context) -> List[Step]:
    # 1. Map Ledger to LLM Messages
    messages = []
    for step in action.ledger:
        if step.type == "text":
            messages.append({"role": step.actor, "content": step.payload["text"]})
        elif step.type == "action_result":
            # We wrap action results as tool outputs
            messages.append({"role": "tool", "content": str(step.payload)})
        # Note: We ignore "action_call" steps as the LLM implies them via its previous output

    # 2. Map Allowed Actions to Tools
    tools = [{"type": "function", "function": {"name": name}} for name in action.actions]

    # 3. Call LLM TODO: add retry logic etc
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
    
    # 4. Map Response to Steps
    new_steps = []
    msg = response.choices[0].message
    
    if msg.content:
        new_steps.append(Step(
            id=str(uuid.uuid4()),
            actor="assistant",
            type="text",
            payload={"text": msg.content}
        ))
        
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # The Policy decides to call a tool, but does NOT execute it.
            # It appends an action_call step. The Engine/Runtime handles execution.
            new_steps.append(Step(
                id=str(uuid.uuid4()),
                actor="assistant",
                type="action_call",
                payload={
                    "policy": tc.function.name,
                    "payload": json.loads(tc.function.arguments)
                }
            ))
            
    return new_steps
```

### 6.2 Web Search Policy (Tool)

```python
@registry.register("web_search_policy")
async def web_search_policy(action: Action, ctx: Context) -> List[Step]:
    """
    Performs a web search and returns results.
    Input: action.payload = {"query": "..."}
    Output: action_result with search results
    """
    query = action.payload["query"]
    
    # Call search API (e.g., Bing, Google, Brave)
    results = await search_api.query(query)  # Mock
    
    return [
        Step(
            id="",  # Engine sets this
            actor="web_search_policy",
            type="action_result",
            payload={"text": results}
        )
    ]
```

### 6.3 Search Agent (Orchestrator)

```python
@registry.register("search_agent")
async def search_agent(action: Action, ctx: Context) -> List[Step]:
    """
    Orchestrator that gives the LLM access to web search.
    """
    # Build sub-action for LLM
    payload = action.payload
    payload['ledger'] = action.ledger  # Pass conversation history to LLM
    llm_action = Action(
        policy="llm_policy",
        payload=payload,
        actions=action.actions | {"web_search_policy"},
    )
    
    # Call LLM with decorators: loop until done, auto-execute tool calls
    steps = await ctx.call(
        action=llm_action,
        decorators=[loop_wrapper(max_turns=5), invoke_action()]
    )
    
    return steps
```

**Flow**:

1. User: "What is the latest Python version?"
2. LLM: "Let me search" → emits `action_call(web_search_policy, {"query": "latest Python version"})`
3. Decorator executes search → appends `action_result` to ledger
4. LLM sees result → emits `text("Python 3.13 was released...")`
5. No more `action_call` steps → loop terminates

---

## 7. Orchestration & Replay

### 7.1 The `Context.call` Mechanism

When a policy calls `await ctx.call(action, decorators)`:

1. Runtime creates a new scoped Context for the sub-policy
2. Check for cached `action_result` in ledger (replay/idempotency)
3. If cached: return cached steps (skip execution)
4. If not cached:
   - Execute the policy with the scoped context
   - Apply decorators (e.g., execute action_call steps)
   - Append all generated steps to master ledger
   - Return the list of steps
5. Calling policy receives the steps and can process/return them

### 7.2 Durability & Replay

**This is the core purpose of the ledger.** Before executing `ctx.call()`:

1. Check ledger for cached `action_result` matching this call.
2. If found → return cached result (skip execution).
3. If not → execute, append `action_call` + `action_result`.

**Replay**: On crash, re-run the policy with the same ledger. Completed calls return instantly.

**How matching works**: Implementation detail (by ID, position, or payload hash).

---

## 8. Multi-Turn Loops

Agents are loops: `user → LLM → tool → LLM → response → user → ...`

**Where does the loop live?** In a decorator or orchestrator policy, not the runner. The runner calls the policy once; the decorator handles iteration.

Example pattern: `@loop_wrapper(max_turns=10)` wraps a policy, running it repeatedly until no more `action_call` steps are emitted.

---

## 9. Runner

The **Runner** is the external entry point. It takes a policy name, payload, registry, and optional ledger (for replay). It creates a Context and executes the policy.

```python
async def run(
    policy_name: str,
    payload: Dict[str, Any],
    registry: PolicyRegistry,
    ledger: Ledger = Ledger(),  # For replay
) -> Ledger:
    """Returns the final ledger after execution."""
    ...
```

Policies never see the Runner. The Runner manages top-level execution and ledger persistence.

---

## 10. Foundational Principles

### 10.1 Error Handling

Two failure modes:

1. **Tool/Runtime Errors (Recoverable)**: Return `action_result` with error payload. The LLM sees the error and decides what to do (retry, apologize, etc.).
2. **Code Crashes (Unrecoverable)**: Exception propagates, job fails, ledger remains safe. On replay, the crash happens again—requires code fix.

**Guidance**: Use `assert` for invariants. If they fire, the code is wrong—let it crash.

### 10.2 Termination

Policies run **once** and return steps. The caller (or decorator) decides whether to loop. No built-in "termination" state.

---
