# PocketFlow Type Analysis vs. PocketJoe

This document maps every class in the legacy `pocketflow` library to its equivalent pattern in `pocket-joe`.

## 1. Base Types

| Legacy Class | Responsibility | PocketJoe Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `BaseNode` | State (`params`), Graph (`successors`), Lifecycle (`prep/exec/post`) | `Policy` (Function) | State is now `Action.payload`. Graph is code logic. Lifecycle is implicit. |
| `_ConditionalTransition` | Syntax sugar for `>>` and `-` | **N/A** | We use standard Python control flow (`if/else`). |

## 2. Synchronous Nodes

| Legacy Class | Responsibility | PocketJoe Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `Node` | Sync execution + Retries | `Policy` (Async) | We are "Async All The Way". Retries handled by Runner/Context. |
| `BatchNode` | Sync iteration over list | `Policy` with loop | Just a `for` loop inside an async function. |

## 3. Synchronous Flows

| Legacy Class | Responsibility | PocketJoe Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `Flow` | Orchestration loop (Graph traversal) | `Policy` (Orchestrator) | The "loop" is just the code inside the policy calling other policies. |
| `BatchFlow` | Run Flow for multiple inputs | `Policy` with loop | Iterates inputs and calls `ctx.call("sub_policy", item)`. |

## 4. Asynchronous Nodes

| Legacy Class | Responsibility | PocketJoe Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `AsyncNode` | Async execution + Retries | `Policy` (Standard) | This is our default. All policies are async. |
| `AsyncBatchNode` | Async sequential iteration | `Policy` with `await` loop | `for item in items: await ctx.call(...)` |
| `AsyncParallelBatchNode` | Async parallel execution (`gather`) | `Policy` with `gather` | `await asyncio.gather(*[ctx.call(...)])` |

## 5. Asynchronous Flows

| Legacy Class | Responsibility | PocketJoe Equivalent | Notes |
| :--- | :--- | :--- | :--- |
| `AsyncFlow` | Async Orchestration | `Policy` (Orchestrator) | Standard Orchestrator Policy. |
| `AsyncBatchFlow` | Async sequential Flow execution | `Policy` with `await` loop | Same as `AsyncBatchNode` but calling an Orchestrator. |
| `AsyncParallelBatchFlow` | Async parallel Flow execution | `Policy` with `gather` | Same as `AsyncParallelBatchNode` but calling an Orchestrator. |

## Conclusion

We have 100% coverage.

*   **Simplicity:** We replace 11 classes with **1 Concept (Policy)**.
*   **Flexibility:** We replace rigid inheritance with **Composition (Calling functions)**.
*   **Power:** We replace custom graph engines with **Python Control Flow**.

We are good to go.
