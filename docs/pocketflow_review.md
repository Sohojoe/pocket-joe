# PocketFlow: A Critical Review & Architectural Pivot

## What is PocketFlow?

PocketFlow is a minimalist, dependency-free Python library designed to orchestrate LLM agents. It models agent workflows as a Directed Acyclic Graph (DAG), where "Nodes" represent processing steps and "Edges" represent transitions. Its defining characteristic is its radical simplicity, constraining the core logic to roughly 100 lines of code to remain accessible and lightweight compared to heavier frameworks like LangChain.

## What do we love?

**LLM Agents are Simply Graphs:** We agree that agents are fundamentally graphs. The core abstraction of connecting discrete steps to form a workflow is the correct mental model for agentic systems.

**Dependency Zero:** We appreciate the rejection of bloat. The core logic should not be tightly coupled to specific LLM providers (OpenAI/Anthropic) or heavy vector store integrations.

**Accessibility Intent:** The goal of demystifying agents ("it's just code, not magic") is noble and necessary.

## What perspectives do we bring?

**Agents are Simply MDP Graphs:** We have 100+ years of research into agents (Markov Chains, MDP, Reinforcement Learning). An LLM is just one type of Policy.

* The Reality: In production, robust agent policies are often a complex mix of heuristics, humans, and LLMs.

* We don't need complex new semantics: We can rely on established Computer Science definitions:

  * Observation (The Payload/Input)
  * Policy (The Node/Decision Logic)
  * Action (The Edge/Side Effect)

**The "Network" Fallacy:** PocketFlow treats agent steps like synchronous HTTP requests (Trigger $\rightarrow$ Wait $\rightarrow$ Retry).

* Throughput (TPS) is the constraint: From our experience running LLM systems at scale, we know that Throughput (TPS) is the hard constraint, not latency.
* The Failure Mode: Standard "Backoff and Retry" logic during high traffic causes congestion collapse. We need Queues and Backpressure, not optimistic retries.

## Where does PocketFlow fall short in execution?

**"100 Lines" as a Liability:** The artificial constraint of 100 lines has incentivized "Code Golf" over "Engineering rigor."

* Readability: Dense, multi-statement lines make debugging difficult.
* Safety: Critical error handling and state management are sacrificed for brevity.

**The Inheritance Trap:** The codebase relies on complex inheritance trees (AsyncParallelBatchNode) to add functionality. This leads to a combinatorial explosion of classes. We prefer Composition and Decorators (e.g., a Strategy pattern).

**Mutable State Risks:** The reliance on copy.copy() and mutating self.params at runtime is fragile and thread-unsafe. Nodes should be stateless definitions; execution state should live in a separate "Ledger" or "Context."

**Syntax Sugar Overload:** Overloading operators like >> and - obscures the control flow, making the code harder for new engineers to parse and static analysis tools to check.
