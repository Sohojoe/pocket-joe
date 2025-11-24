# PocketJoe

A correct, simple, performant, and pythonic framework for building durable AI agents.
"There is no flow, only Policies and Actions."

## Core Concepts

* **Action**: Immutable state (Payload + History).
* **Policy**: Pure function `(Action, Context) -> Action | Result`.
* **Context**: Interface for side-effects (calling other policies/tools).

## Getting Started

### Prerequisites

* Python 3.12+
* `uv` (recommended)

### Installation

```bash
uv sync
```

### Running Examples

#### Search Agent (ReAct)

```bash
PYTHONPATH=. uv run python cookbook/search_agent.py
```

#### YouTube Summarizer

```bash
PYTHONPATH=. uv run python cookbook/youtube_summarizer.py
```

## Architecture

PocketJoe abandons the "DAG" and "Node" metaphors in favor of a functional approach. Agents are just functions (Policies) that transform state (Actions). Control flow is handled by decorators (`@loop_wrapper`, `@invoke_action`) and the Runtime.

## Background

Inspired by [PocketFlow](https://github.com/The-Pocket/PocketFlow)... I loved PocketFlow but it feel short in a couple of key areas. This is my rewrite that I can actually use.
