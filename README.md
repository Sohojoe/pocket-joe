# PocketJoe

**LLM Agents are just agents...**

- Agents are policies
- A policy reasons over observations and chooses a batch of actions
- A policy can be any mix of LLM-based, human-in-the-loop, or heuristic

## Semantics

An agent system using Reinforcement Learning theory with LLM semantics as first class

- `Policy`: all code/logic/llm are policies
- `observations` - the set of observations for the policy to reason over
- `options` - a set of optional sub policies that the policy can choose
- `selected_actions` - the set of concurrent actions the policy chose to take
- `Message`: a shared dataclass for `observations` and `actions` that aligns with llm semantics

### LLM semantics as platform semantics

In LLM APIs, everything is a `Message`. We adopt this as our universal unit:

- **Input:** `observations: list[Message]` (what the policy sees)
- **Output:** `selected_actions: list[Message]` (what the policy does)

**Key insight:** The runtime automatically invokes all option calls and injects the results back as observations. Your policy just returns requests; the platform handles execution.

### Everything is a Policy

An LLM policy that can call other policies:

```python
@policy_spec_mcp_tool(description="Calls OpenAI GPT-4 with tool support")
class OpenAILLMPolicy_v1(Policy):
    async def __call__(self, observations: list[Message], options: list[str]) -> list[Message]:
        """LLM policy that calls OpenAI GPT-4 with tool support.
        :param observations: List of Messages representing the conversation history + new input
        :param options: Set of allowed options the LLM can call (policy names that will map to tools)
        """
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=to_completions_messages(observations),
            tools=to_completions_tools(self.ctx, options))
        return map_response_to_messagess(response)
```

A simple heuristic policy:

```python
@policy_spec_mcp_tool(description="Performs web search")
class WebSearchDdgsPolicy(Policy):
    async def __call__(self, query: str) -> list[Message]:
        """
        Performs a web search and returns results.        
        :param query: The search query string to search for
        """
        results = DDGS().text(query, max_results=5)
        results_str = "\n\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
        return [Message(
            actor=self.__class__.__name__,
            type="action_result",
            payload={"content": results_str}
        )]
```

An orchestrator policy that coordinates LLM + search:

```python
@policy_spec_mcp_tool(description="Orchestrator with LLM and search")
class SearchAgent(Policy):
    ctx: "AppContext"  # Override to specify context type
    
    async def __call__(self, prompt: str) -> list[Message]:
        """
        Orchestrator that gives the LLM access to web search.
        :param prompt: The user prompt to process
        """
        system_message = Message(actor="system", type="text", 
            payload={"content": "You are an AI assistant that can use tools to help answer user questions."})
        prompt_message = Message(actor="user", type="text", payload={"content": prompt})

        history = [system_message, prompt_message]
        while True:
            selected_actions = await self.ctx.llm(observations=history, options=["web_search"])
            history.extend(selected_actions)
            if not any(msg.type == "action_call" for msg in selected_actions):
                break
        
        return history
```

Use `AppContext` for registry (gives IDE type hints):

```python
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(OpenAILLMPolicy_v1)
        self.web_search = self._bind(WebSearchDdgsPolicy)
        self.search_agent = self._bind(SearchAgent)
```

Enjoy:

```python
async def main():
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")
    print(f"\nFinal Result: {result[-1].payload['content']}")
```

**Why this matters:**

- Same interface for LLM, human, heuristic policies
- All policy parameters are optional (define what you need)
- Type-safe composition with IDE support
- Enables evolution: human → heuristic → LLM with no refactoring

A correct, simple, performant, and pythonic framework for building durable AI agents.

> "There is no flow, only Policies and Actions."

## Getting Started

### Prerequisites

- Python 3.12+
- `uv` (recommended)

### Installation

```bash
uv sync
```

### Running Examples

First, install with examples dependencies:

```bash
uv sync --extra examples
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

Intial version

- [] Tidy up code - add partly refactored code
- [] Proper tests
- [] Implement more examples from Pocket-Flow

Durable System:

- [] Ledger - Temporal style 'at least once, only one result' replay semantic
- [] Durable Storage wrapper - For long running tasks & replay
- [] Distrubuted - worker model

## Background

Inspired by [PocketFlow](https://github.com/The-Pocket/PocketFlow)... I loved PocketFlow but it fell short in a couple of key areas. This is my rewrite that I can actually use.
