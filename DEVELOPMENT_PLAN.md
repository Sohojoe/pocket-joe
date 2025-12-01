# PocketJoe Development Plan

## Refactor

### Semantics

llm agents are just agents...

- Agents are policies
- A policy reasons over observations and chooses a batch of actions
- A policy can be any mix of LLM-based, human-in-the-loop, or heuristic

#### llm semantics to be first class platform semantics

- In llm semantics, we use `Message` as a unified dataclass for input and output
- In llm semantics, `â` (the set of actions selected)
  - single message for text, image actions are single messages
  - two messages for option call (tool/policy call)
    - the request
    - the result
- In llm semantics, the caller always invokes all option call requests. therefor we abstract that in the platform

#### putting this together we have these semantics

- `Policy`: pure functions that choose a batch of actions
- `Message`: a shared dataclass for `observations` and `actions` that alligns with llm semantics

- `selected_actions = policy(observations: list[Message], options: list[str], ...) -> [Message]:`
  - `policy` - the function that implements the policy
  - `observations` - the set of observations for the policy to reason over
  - `options` - a set of optional sub policies that the policy can choose
  - `selected_actions` - the set of concurrent actions the policy chose to take

- policies mix llm, human, huristic... one interface
- all policies have signature: `policy(ctx, observations, options, **kwargs)`
  - each policy defines additional parameters it needs
  - use `observations: list[Message]` when you need to pass observations
  - use `options: list[str]` when you need to pass options
  - enables evolution: human → heuristic → LLM with same interface

#### formal RL definition

In this section we formally define

- It's a PoMDP using the Options framework
- Notation: `â = π(observation, Ω)` where `â ⊆ A = A_primitive ∪ Ω`
  - `π` is the policy function
  - `observation` is the observation (new inputs and/or history)
  - `Ω` is the set of available options (sub-policies)
  - `A_primitive` is the set of primitive actions (built into policy)
  - `â` is the batch of actions selected

### We rename Step as Message

we rename Step as Message... because message is first class llm semantic; and step was missleading when passed into the the policy as it included

### We loose Action

- policy -> now called directly via ctx.llm_policy() etc
- payload -> history/state passed explicitly as a param: ctx.llm_policy(history)
- actions -> action_space passed explicitly as a param: ctx.llm_policy(history, action_space)

### Revised Policy defenition

All policies (LLM, heuristic, human) share this signature:

```python
async def policy(
    ctx: Context,
    observations: list[Message],  # Context chosen by caller
    options: list[str] | None,     # Available sub-policies
    **kwargs                       # Policy-specific parameters
) -> list[Message]:                # Batch of selected actions (with results)
```

``` python
from typing import Protocol
from collections.abc import Awaitable
class Policy(Protocol):
    """
    A policy that reasons over observations and selects a batch of actions.
    
    All policies share this base signature:
        ctx: Context - for invoking other policies
        observations: list[Message] - context to reason over
        options: list[str] | None - available sub-policies
        **kwargs - policy-specific parameters
    
    Returns a batch of selected actions (including execution results for options).
    """
    async def __call__(
        self,
        ctx: Context,
        observations: list[Message],
        options: list[str] | None = None,
        **kwargs
    ) -> list[Message]: ...
```

### we manage history explicitly

- ledger is abstracted away
- history appended and passed around using llm semantics

### Context is the registry

- Context replaces the Registry class
- User defines a Context class that inherits from `BaseContext`
- Policies are explicitly bound in the Context `__init__`
- Runner provides the `_bind` implementation (via `create_bind_function`)

Benefits

- Clean & Explicit: IDE autocomplete works (ctx.llm, ctx.search)
- Version Control: Create different context versions (v1, v2) with different policy sets
- Isolation: Restrict capabilities (user context vs admin context)
- Type Safety: Use Protocol for BoundPolicy type hints
- No String Names: Direct function references, no registry lookup

#### BaseContext (Framework)

```python
class BaseContext:
    """Framework base - hide plumbing here."""
    def __init__(self, runner):
        self._runner = runner
        self._bind = runner.create_bind_function(self)
```

#### User Context (Explicit Policy Bindings)

``` python
class MyContext(BaseContext):
    llm: BoundPolicy      # Type hints for IDE autocomplete
    search: BoundPolicy
    email: BoundPolicy
    
    def __init__(self, runner):
        super().__init__(runner)
        # Attribute name (e.g., "llm") is what gets passed as option string
        # Function can be any implementation (e.g., openai_llm_v4)
        self.llm = self._bind(openai_llm_v4)
        self.search = self._bind(search_web_policy)
        self.email = self._bind(send_email_policy)
```

#### Usage

``` python
runner = InMemoryRunner()
ctx = MyContext(runner)

# Call policies through context
# Options are strings matching ctx attribute names
result = await ctx.llm(
    observations=[...], 
    options=["search", "email"]  # Match ctx.search, ctx.email attribute names
)
```

### Runner implements _bind strategy

The Runner controls how policies are executed by providing the `_bind` implementation. This allows the same policy code to run in different execution modes without changes.

Benefits

- Same policy code runs everywhere: Local, durable, distributed
- Zero overhead option: InMemoryRunner(trace=False) returns unwrapped policies
- Clean debugging: Without tracing, debugger goes directly to policy function
- Runner controls execution: Ledger, replay, RPC - all runner concerns
- Temporal-like pattern: Same pattern as Temporal workflow execution

#### Runner Interface

```python
class BaseRunner:
    """Base runner - defines interface."""
    def create_bind_function(self, ctx: BaseContext):
        """
        Returns a _bind function that wraps policy execution.
        Different runners provide different execution strategies.
        """
        raise NotImplementedError
```

#### InMemoryRunner (Local Execution)

```python
class InMemoryRunner:
    def __init__(self, trace: bool = False):
        self.trace = trace
    
    def create_bind_function(self, ctx: BaseContext):
        if not self.trace:
            # No tracing: direct call (zero overhead)
            def _bind(policy):
                return policy  # No wrapper!
            return _bind
        
        else:
            # With tracing: minimal wrapper for instrumentation
            def _bind(policy):
                async def bound(observations, options=None, **kwargs):
                    span = tracer.start_span(policy.__name__)
                    try:
                        result = await policy(ctx, observations, options, **kwargs)
                        return result
                    finally:
                        span.end()
                return bound
            return _bind
```

#### DurableRunner (Replay/Ledger)

```python
class DurableRunner:
    def __init__(self, ledger_store):
        self.ledger = ledger_store
    
    def create_bind_function(self, ctx: BaseContext):
        def _bind(policy):
            async def bound(observations, options=None, **kwargs):
                policy_name = policy.__name__
                
                # Record action to ledger
                action_id = self.ledger.record_action(policy_name, observations, kwargs)
                
                # Check replay mode
                if self.ledger.is_replaying(action_id):
                    return self.ledger.get_cached_result(action_id)
                
                # Execute
                result = await policy(ctx, observations, options, **kwargs)
                
                # Record result
                self.ledger.record_result(action_id, result)
                return result
            return bound
        return _bind
```

#### DistributedRunner (RPC)

```python
class DistributedRunner:
    def __init__(self, rpc_client):
        self.rpc = rpc_client
    
    def create_bind_function(self, ctx: BaseContext):
        def _bind(policy):
            async def bound(observations, options=None, **kwargs):
                # Serialize and send to worker
                request = {
                    'policy': policy.__name__,
                    'observations': serialize(observations),
                    'options': options,  # Already strings, serializable
                    'kwargs': serialize(kwargs)
                }
                response = await self.rpc.call(request)
                return deserialize(response['result'])
            return bound
        return _bind
```

### Passing Options into a Policy

Options are passed as strings matching the context attribute names.

#### User Code

```python
# Pass string names matching ctx attributes
result = await ctx.llm(
    observations=my_obs,
    options=["search", "email"]  # Match ctx.search, ctx.email
)
```

#### Context Attribute Names vs Function Names

The string in `options` matches the **context attribute name**, not the underlying function name:

```python
class MyContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        # Attribute name "llm" is what appears in options=["llm"]
        # Can bind any implementation function (e.g., openai_llm_v4)
        self.llm = self._bind(openai_llm_v4)
        self.search = self._bind(tavily_search_v2)
```

#### Policy Implementation

```python
async def llm_policy(
    ctx: Context,
    observations: list[Message],
    options: list[str] | None,  # ← Strings matching ctx attribute names
    **kwargs
) -> list[Message]:
    # options = ["search", "email"]
    # To invoke: await ctx.search(observations=[...])
```

#### Benefits

- **Simple**: No magic, just strings
- **Serializable**: Strings work everywhere (RPC, ledger, replay)
- **Clear**: String name matches what user sees in `ctx.search`
- **Flexible**: Same ctx name can bind different implementations

## ToDo

[x] We rename Step as Message
[x] We loose Action
[x] Revised Policy defenition
[x] manage history explicitly
[x] Context is the registry
[x] platform calls the options
[] rename action_call/action_result to be option_xx
[x] try without done command
[x] fix AppContext type issue
[x] move invoke options to decerators/wappers
[] implement more agents
