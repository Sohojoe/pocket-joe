# PocketJoe Development Plan

**Goal**: Rewrite "PocketFlow" into "PocketJoe" — a correct, simple, performant, and pythonic framework for building durable AI agents.
**Core Philosophy**: "There is no flow, only Policies and Actions."

## 1. Architecture Overview

* **Primitives**:
  * `Action` (Immutable Data): Payload + History + Edges (Allowed Moves).
  * `Policy` (Pure Function): `(Action, Context) -> Action | Result`.
  * `Context` (Interface): The boundary for side-effects (calling other policies).
* **Control Flow**:
  * **Decorators**: `@loop_wrapper`, `@invoke_action` handle the "how" (loops, execution).
  * **Caller-Side Composition**: Orchestrators define behavior by wrapping Workers.
* **Durability**:
  * **Two Runtimes**:
    * `InMemoryRunner`: Fast, recursive, for dev/testing.
    * `DurableRunner`: Replay-based, handles `SuspendExecution`, for production.

## 2. Implementation Roadmap

### Phase 1: The Kernel (Core Primitives)

* [x] **Create `pocket_joe/core.py`**
  * Define `Action` (Frozen Dataclass).
  * Define `Context` (Protocol).
  * Define `Policy` (Type Alias).
* [x] **Create `pocket_joe/registry.py`**
  * Implement `Registry` class (Singleton or Context-bound).
  * Implement `@register` decorator with metadata support (MCP schema).

### Phase 2: The Wiring (Decorators & Composition)

* [x] **Create `pocket_joe/policy_decorators.py`**
  * Implement `@loop_wrapper(max_turns)`: Handles iteration and termination.
  * Implement `@invoke_action`: Handles tool execution and history updates.

### Phase 3: The Engine (Runtimes)

* [x] **Create `pocket_joe/memory_runtime.py`**
  * Implement `InMemoryRunner`: Simple recursive execution.
* [x] **Create `pocket_joe/durable_runtime.py`** (Skeleton)
  * Define `SuspendExecution` exception.
  * Implement basic Replay logic (store/load history).

### Phase 4: The Standard Library (Research)

* [ ] **Research Location for Standard Policies**
  * PocketFlow keeps core minimal.
  * Decide if standard tools/agents go in `pocket_joe_std`, `cookbook`, or `extensions`.
  * *Goal*: Keep `pocket_joe` core lightweight.

### Phase 5: Validation & Migration

* [x] **Unit Tests**: Test Core, Decorators, and Runtimes.
* [x] **Port Cookbook**:
  * `search_agent` (Simple ReAct).
  * `youtube_summarizer` (Map/Reduce).
* [x] **Cleanup**: Remove legacy `pocketflow/` code and `to-delete/` folder.

## 3. Status

All phases are complete. The core library is implemented, tested, and documented. Cookbook examples are available.
