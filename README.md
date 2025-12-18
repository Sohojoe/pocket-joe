# PocketJoe

**LLM Agents are just agents...**

- Agents are policies
- A policy reasons over observations and chooses a batch of options
- A policy can be any mix of LLM-based, human-in-the-loop, or heuristic

## Semantics

An agent system using Reinforcement Learning theory with LLM semantics as first class

- `policy`: all code/logic/llm are policies
- `observations` - the set of observations for the policy to reason over
- `options` - additional action spaces available to the policy
- `selected_actions` - the set of concurrent actions the policy chose to take
- `Message`: a shared dataclass for `observations` and `actions` that aligns with llm semantics

### LLM semantics as platform semantics

In LLM APIs, everything is a `Message`. We adopt this as our universal unit:

- **Input:** `observations: list[Message]` (what the policy sees)
- **Output:** `selected_actions` - the policy's action space (owns its outputs)

**Key insight:** When options are provided, they expand the policy's action space. The runtime automatically invokes all option calls and injects the results back as observations.

### Everything is a Policy

**Universal Return Types:** Policies can return any JSON-serializable type - the framework automatically wraps results when called as options.

An LLM policy using the adapter pattern:

```python
@policy.tool(description="OpenAI-compatible chat completions")
async def llm_policy(
    observations: list[Message],
    options: list[OptionSchema] | None = None,
) -> list[Message]:
    """Call chat completions API and return option_call or text messages."""
    adapter = CompletionsAdapter(observations, options)
    client = CompletionsAdapter.client()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=adapter.messages,
        tools=adapter.tools or [],
    )
    return adapter.decode(response, policy="llm_policy")
```

A simple helper policy returning primitives:

```python
@policy.tool(description="Performs a web search and returns results.")
async def web_search_policy(query: str) -> str:
    """Performs a web search and returns results."""
    results = DDGS().text(query, max_results=5)
    return "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
```

A policy returning structured data with Pydantic:

```python
from pydantic import BaseModel

class TranscriptResult(BaseModel):
    title: str
    transcript: str
    thumbnail_url: str
    video_id: str
    error: str | None = None

@policy.tool(description="Transcribe YouTube video")
async def transcribe_youtube_policy(url: str) -> TranscriptResult:
    """Get video title, transcript and metadata from YouTube URL."""
    video_id = _extract_video_id(url)
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return TranscriptResult(
        title=title,
        transcript=" ".join([snippet.text for snippet in transcript]),
        thumbnail_url=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        video_id=video_id
    )
```

An orchestrator policy that coordinates LLM + search:

```python
@policy.tool(description="Orchestrates LLM with web search tool")
async def search_agent(prompt: str, max_iterations: int = 3) -> list[Message]:
    """Orchestrator that gives the LLM access to web search."""
    ctx = AppContext.get_ctx()

    system_builder = MessageBuilder(policy="system", role_hint_for_llm="system")
    system_builder.add_text("You are an AI assistant that can use tools.")
    system_message = system_builder.to_message()

    prompt_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
    prompt_builder.add_text(prompt)
    prompt_message = prompt_builder.to_message()

    history = [system_message, prompt_message]
    for _ in range(max_iterations):
        selected_actions = await ctx.llm(
            observations=history,
            options=OptionSchema.from_func([ctx.web_search])
        )
        history.extend(selected_actions)
        if not any(msg.payload and isinstance(msg.payload, OptionCallPayload) for msg in selected_actions):
            break

    return history
```

Use `AppContext` for registry (gives IDE type hints):

```python
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(llm_policy)
        self.web_search = self._bind(web_seatch_ddgs_policy)
        self.search_agent = self._bind(search_agent)
```

Enjoy:

```python
async def main():
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")

    # Get final text message (Message.__str__ extracts text automatically)
    final_msg = next((msg for msg in reversed(result) if msg.parts), '')
    print(f"\nFinal Result: {final_msg}")
```

**Why this matters:**

- **Universal Composability:** Decorate any function - it works like FastAPI/FastMCP endpoints
- **Flexible Return Types:** Return primitives (str), Pydantic models, or list[Message] for complex flows
- **Auto-wrapping:** Framework automatically wraps results in OptionResultPayload when called as options
- **Type-safe:** Full IDE support with typed context and message payloads
- **Evolution-friendly:** Start simple (primitives) → add complexity (messages) with no refactoring

A correct, simple, performant, and pythonic framework for building durable AI agents.

> "There is no flow, only Policies and Actions."

## Working with Media

`MediaPart` supports three mutually exclusive sources for image/audio/video content:

```python
from pocket_joe import MessageBuilder, MediaPart, iter_parts

# URL source - for remote images
builder = MessageBuilder(policy="agent")
builder.add_image(url="https://example.com/photo.png", mime="image/png")

# Path source - for local files (adapter handles reading)
builder.add_image_path(path="/path/to/image.png")

# Bytes source - for generated/inline content (base64-encoded internally)
builder.add_image_bytes(data=image_bytes, mime="image/png", prompt_hint="Generated cat")
```

### Iterating Over Parts

Use `iter_parts()` to iterate over all parts across multiple messages with optional type filtering:

```python
from pocket_joe import iter_parts, MediaPart, TextPart

# Find first image with inline data
first_image = next(
    (p for p in iter_parts(messages, MediaPart) if p.data_b64),
    None
)
if first_image:
    raw_bytes = first_image.get_bytes()

# Check if any images exist
has_images = any(iter_parts(messages, MediaPart))

# Get all text content
all_text = [p.text for p in iter_parts(messages, TextPart)]
```

## Getting Started

### Prerequisites

- Python 3.12+

### Installation

```bash
uv add pocket-joe
```

Or with pip:

```bash
pip install pocket-joe
```

To install with example dependencies:

```bash
uv add pocket-joe --extra examples
# or
pip install pocket-joe[examples]
```

### Development Setup

```bash
git clone https://github.com/Sohojoe/pocket-joe.git
cd pocket-joe
uv sync --dev --all-extras
```

### Running Examples

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

#### Search Agent (ReAct)

```bash
uv run python examples/search_agent.py
```

#### YouTube Summarizer

```bash
uv run python examples/youtube_summarizer.py
```

## Dev Status

Still in prerelease, things will change

Initial version

- [] Tidy up code - add partly refactored code
- [] Proper tests
- [] Implement more examples from Pocket-Flow

Durable System:

- [] Ledger - Temporal style 'at least once, only one result' replay semantic
- [] Durable Storage wrapper - For long running tasks & replay
- [] Distributed - worker model

## Background

Inspired by [PocketFlow](https://github.com/The-Pocket/PocketFlow)... I loved PocketFlow but it fell short in a couple of key areas. This is my rewrite that I can actually use.
