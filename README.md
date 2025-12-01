# PocketJoe

**LLM Agents are just agents...**

- Agents are policies
- A policy reasons over observations and chooses a batch of actions
- A policy can be any mix of LLM-based, human-in-the-loop, or heuristic

## Semantics

An agent system using Reinforcement Learning theroy with LLM sematnics as first class

- `Policy`: all code/logic/llm are policies
- `observations` - the set of observations for the policy to reason over
- `options` - a set of optional sub policies that the policy can choose
- `selected_actions` - the set of concurrent actions the policy chose to take
- `Message`: a shared dataclass for `observations` and `actions` that alligns with llm semantics

### llm semantics to be first class platform semantics

- In llm semantics, we use `Message` as a unified dataclass for input and output
- In llm semantics, `â` (the set of actions selected)
  - single message for text, image actions are single messages
  - two messages for option call (tool/policy call)
    - the request
    - the result
- In llm semantics, the caller always invokes all option call requests. therefor we abstract that in the platform

### putting this together we have these semantics

- `Policy`: pure functions that choose a batch of actions
- `Message`: a shared dataclass for `observations` and `actions` that alligns with llm semantics

- `selected_actions = policy(observations: list[Message], options: list[str], ...) -> [Message]:`
  - `policy` - the function that implements the policy
  - `observations` - the set of observations for the policy to reason over
  - `options` - a set of optional sub policies that the policy can choose
  - `selected_actions` - the set of concurrent actions the policy chose to take

- policies mix llm, human, huristic... one interface
- each policy defines additional parameters it needs
- use `observations: list[Message]` when you need to pass observations
- use `options: list[str]` when you need to pass options
- enables evolution: human → heuristic → LLM with same interface

A correct, simple, performant, and pythonic framework for building durable AI agents.
"There is no flow, only Policies and Actions."

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
