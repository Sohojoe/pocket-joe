# Migration Plan: Function-Based Policies with FastMCP Integration

## Overview

Migrate pocket-joe from class-based policies to function-based policies with FastMCP compatibility. This enables better IDE support, explicit MCP exposure control, and compatibility with FastMCP's schema extraction.

## Design Goals

1. **Function-based policies** with `@policy` decorator (FastMCP compatible)
2. **Tool.from_function()** for schema extraction WITHOUT MCP registration
3. **Follow MCP patterns**: `policy_spec_mcp_tool` and `policy_spec_mcp_resource` wrappers
4. **Explicit MCP exposure**: via `expose_to_mcp()` helper
5. **Per-subclass context singleton**: Each AppContext subclass has its own ContextVar
6. **IDE support**: Developers get autocomplete for `ctx.agent_x()`

## Architecture Changes

### Current (Class-Based)
```python
class OpenAILLMPolicy_v1(Policy):
    async def __call__(self, observations, options=None):
        # Implementation
        pass

class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(OpenAILLMPolicy_v1)
```

### Target (Function-Based)
```python
@policy_spec_mcp_tool(description="LLM with tool support")
async def llm(observations: list[Message], options: list[str] | None = None) -> list[Message]:
    ctx = AppContext.get_ctx()
    # Implementation
    pass

class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(llm)  # Binds function, not class
```

### Key Technical Components

#### 1. ContextVar-Based Context Management
```python
from contextvars import ContextVar

class BaseContext:
    _ctx_var: ContextVar['BaseContext'] = ContextVar('pocket_joe_context')

    @classmethod
    def get_ctx(cls) -> 'BaseContext':
        return cls._ctx_var.get()

    def _set_ctx(self):
        return self._ctx_var.set(self)

    def _reset_ctx(self, token):
        self._ctx_var.reset(token)
```

#### 2. Tool Metadata Extraction
```python
from fastmcp.tools import Tool

def policy(description: str | None = None):
    def decorator(func):
        tool = Tool.from_function(func, description=description)
        func._tool_metadata = tool
        return func
    return decorator
```

#### 3. PolicyProxy for Call Routing
```python
class PolicyProxy:
    def __init__(self, func, name, runner, tool_metadata=None):
        self.func = func
        self.name = name
        self.runner = runner
        self.tool_metadata = tool_metadata
        # Preserve metadata for IDE
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    async def __call__(self, **kwargs):
        return await self.runner.execute(self.name, kwargs)
```

#### 4. Runner Context Management
```python
class InMemoryRunner:
    async def execute(self, policy_name: str, kwargs: dict):
        func = self._registry[policy_name]
        token = self._context._set_ctx()
        try:
            return await func(**kwargs)
        finally:
            self._context._reset_ctx(token)
```

## Implementation Plan

### Phase 1: Core Infrastructure

#### Task 1: Implement ContextVar-based BaseContext
**File**: `pocket_joe/core.py:36-66`

- Add `_ctx_var: ContextVar` class variable
- Implement `get_ctx()` classmethod
- Add `_set_ctx()` and `_reset_ctx()` methods
- Update `__init__` to link runner ↔ context

**Changes**:
```python
class BaseContext:
    _ctx_var: ContextVar['BaseContext'] = ContextVar('pocket_joe_context')

    def __init__(self, runner):
        self.runner = runner
        self._registry = {}
        runner._registry = self._registry
        runner._context = self

    @classmethod
    def get_ctx(cls) -> 'BaseContext':
        return cls._ctx_var.get()

    def _set_ctx(self):
        return self._ctx_var.set(self)

    def _reset_ctx(self, token):
        self._ctx_var.reset(token)
```

#### Task 2: Create @policy Decorator Using Tool.from_function()
**File**: `pocket_joe/decorators.py` (new)

- Use FastMCP's `Tool.from_function()` for schema extraction
- Store metadata on function without wrapping
- Preserve original function for normal calls

**Implementation**:
```python
from fastmcp.tools import Tool
from typing import Callable

def policy(description: str | None = None):
    """Decorator that extracts FastMCP metadata without MCP registration"""
    def decorator(func: Callable):
        tool = Tool.from_function(func, description=description)
        func._tool_metadata = tool
        func._tool_name = tool.name
        func._tool_description = tool.description
        func._tool_schema = tool.input_schema
        return func
    return decorator
```

#### Task 3: Create policy_spec_mcp_tool and policy_spec_mcp_resource Wrappers
**File**: `pocket_joe/policy_spec_mcp.py` (update)

- Replace current implementation
- Follow MCP `mcp.tool` and `mcp.resource` patterns
- Support both tool and resource types

**Implementation**:
```python
from fastmcp.tools import Tool
from fastmcp.resources import Resource

def policy_spec_mcp_tool(description: str | None = None):
    """Wrapper for tool-type policies (follows mcp.tool pattern)"""
    def decorator(func):
        tool = Tool.from_function(func, description=description)
        func._tool_metadata = tool
        func._policy_type = "tool"
        return func
    return decorator

def policy_spec_mcp_resource(description: str | None = None):
    """Wrapper for resource-type policies (follows mcp.resource pattern)"""
    def decorator(func):
        resource = Resource.from_function(func, description=description)
        func._resource_metadata = resource
        func._policy_type = "resource"
        return func
    return decorator
```

#### Task 4: Create PolicyProxy Class
**File**: `pocket_joe/policy_proxy.py` (new)

- Route calls through runner
- Preserve function metadata for IDE support
- Store Tool metadata for MCP exposure

**Implementation**:
```python
import inspect
from typing import Callable, Any
from fastmcp.tools import Tool
from .core import Message

class PolicyProxy:
    """Proxy that routes policy calls through runner while preserving metadata"""

    def __init__(self, func: Callable, name: str, runner: Any, tool_metadata: Tool | None = None):
        self.func = func
        self.name = name
        self.runner = runner
        self.tool_metadata = tool_metadata

        # Preserve metadata for IDE support
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__signature__ = inspect.signature(func)

        # Preserve Tool metadata
        if tool_metadata:
            self._tool_name = tool_metadata.name
            self._tool_description = tool_metadata.description
            self._tool_schema = tool_metadata.input_schema

    async def __call__(self, **kwargs) -> list[Message]:
        """Route call through runner"""
        return await self.runner.execute(self.name, kwargs)
```

#### Task 5: Update _bind() Method
**File**: `pocket_joe/core.py:41-47`

- Accept functions instead of classes
- Extract Tool metadata from decorated functions
- Return PolicyProxy instance

**Changes**:
```python
from .policy_proxy import PolicyProxy

class BaseContext:
    def _bind(self, func: Callable):
        """Bind a policy function with FastMCP metadata"""
        name = func.__name__
        self._registry[name] = func

        # Extract FastMCP-generated metadata
        tool_metadata = getattr(func, '_tool_metadata', None)

        return PolicyProxy(func, name, self.runner, tool_metadata)

    # Remove old get_policy() method or update for function-based
```

#### Task 6: Update InMemoryRunner
**File**: `pocket_joe/memory_runtime.py:84-93`

- Remove or update `_bind_strategy()` (now handled by PolicyProxy)
- Add `execute()` method with context management
- Integrate with `invoke_options_wrapper`

**Changes**:
```python
class InMemoryRunner:
    async def execute(self, policy_name: str, kwargs: dict) -> list[Message]:
        """Execute policy with context management"""
        func = self._registry[policy_name]

        # Set context for this execution
        token = self._context._set_ctx()
        try:
            # Apply wrappers
            wrapped = invoke_options_wrapper_for_func(func, self._context)
            result = await wrapped(**kwargs)
            return result
        finally:
            self._context._reset_ctx(token)
```

#### Task 7: Update invoke_options_wrapper for Functions
**File**: `pocket_joe/policy_wrappers.py:65-71`

- Create function-compatible wrapper
- Replace or augment existing class-based wrapper

**Changes**:
```python
def invoke_options_wrapper_for_func(func: Callable, ctx: BaseContext):
    """Wrapper for function-based policies"""
    async def wrapped(**kwargs):
        selected_actions = await func(**kwargs)
        option_results = await _call_options_in_parallel(ctx, selected_actions)
        return selected_actions + option_results
    return wrapped
```

### Phase 2: Policy Migration (Atomic - All at Once)

⚠️ **CRITICAL**: This phase must be done atomically. Cannot mix class-based and function-based policies.

#### Task 8: Convert ALL Policies to Functions

##### 8.1: OpenAILLMPolicy_v1
**File**: `examples/utils/llm_policies.py`

**Current**:
```python
class OpenAILLMPolicy_v1(Policy):
    async def __call__(self, observations, options=None):
        # Implementation
```

**Target**:
```python
@policy_spec_mcp_tool(description="LLM with tool support")
async def llm(
    observations: list[Message],
    options: list[str] | None = None
) -> list[Message]:
    ctx = AppContext.get_ctx()

    # Get tools for the specified options
    tools = get_tools_from_context(ctx, options or [])
    openai_tools = [mcp_to_openai_tool(tool) for tool in tools]

    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=to_completions_messages(observations),
        tools=openai_tools
    )
    return map_response_to_messages(response)
```

##### 8.2: WebSearchDdgsPolicy
**File**: `examples/utils/search_web_policies.py`

**Current**:
```python
class WebSeatchDdgsPolicy(Policy):
    async def __call__(self, query: str):
        # Implementation
```

**Target**:
```python
@policy_spec_mcp_tool(description="Performs web search using DuckDuckGo")
async def web_search(query: str) -> list[Message]:
    results = DDGS().text(query, max_results=5)
    # Implementation
    return [Message(...)]
```

##### 8.3: SearchAgent
**File**: `examples/search_agent.py:13-58`

**Current**:
```python
@policy_spec_mcp_tool(description="Orchestrator with LLM and search")
class SearchAgent(Policy):
    ctx: "AppContext"
    async def __call__(self, prompt: str, max_iterations: int = 3):
        # Implementation
```

**Target**:
```python
@policy_spec_mcp_tool(description="Orchestrator with LLM and search")
async def search_agent(
    prompt: str,
    max_iterations: int = 3
) -> list[Message]:
    ctx = AppContext.get_ctx()

    system_message = Message(...)
    prompt_message = Message(...)
    history = [system_message, prompt_message]

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        selected_actions = await ctx.llm(observations=history, options=["web_search"])
        history.extend(selected_actions)
        if not any(msg.type == "action_call" for msg in selected_actions):
            break

    return history
```

##### 8.4: TranscribeYouTubePolicy
**File**: `examples/utils/transcribe_youtube_policy.py`

Convert following the same pattern as above.

##### 8.5: Update AppContext Bindings
**File**: `examples/search_agent.py:51-58`

**Current**:
```python
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(OpenAILLMPolicy_v1)
        self.web_search = self._bind(WebSeatchDdgsPolicy)
        self.search_agent = self._bind(SearchAgent)
```

**Target**:
```python
class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(llm)
        self.web_search = self._bind(web_search)
        self.search_agent = self._bind(search_agent)
```

### Phase 3: MCP Integration Helpers

#### Task 9: Add MCP Conversion Helpers
**File**: `pocket_joe/mcp_helpers.py` (new)

- Convert MCP Tool schema to OpenAI format
- Convert MCP Tool schema to Anthropic format
- Extract tools from context by name

**Implementation**:
```python
from fastmcp.tools import Tool
from .core import BaseContext

def mcp_to_openai_tool(tool: Tool) -> dict:
    """Convert MCP/FastMCP tool to OpenAI format"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    }

def mcp_to_anthropic_tool(tool: Tool) -> dict:
    """Convert MCP/FastMCP tool to Anthropic format"""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema
    }

def get_tools_from_context(ctx: BaseContext, policy_names: list[str]) -> list[Tool]:
    """Extract Tool objects for given policies"""
    tools = []
    for name in policy_names:
        func = ctx._registry.get(name)
        if func and hasattr(func, '_tool_metadata'):
            tools.append(func._tool_metadata)
    return tools
```

#### Task 10: Add expose_to_mcp() Helper
**File**: `pocket_joe/core.py`

- Explicitly expose selected policies to MCP server
- Use pre-built Tool metadata
- Follow FastMCP registration patterns

**Changes**:
```python
from fastmcp import FastMCP

class BaseContext:
    def expose_to_mcp(self, mcp: FastMCP, policy_names: list[str]):
        """Explicitly expose policies to MCP server"""
        for name in policy_names:
            func = self._registry[name]

            if hasattr(func, '_tool_metadata'):
                # Use pre-built Tool metadata
                tool = func._tool_metadata
                # Add to MCP's tool manager directly
                mcp._tool_manager._tools[name] = tool
            else:
                # Fallback to standard mcp.tool()
                mcp.tool(func)
```

#### Task 11: Update LLM Policy to Use MCP Conversion
**File**: `examples/utils/llm_policies.py`

- Use `get_tools_from_context()` to get Tool metadata
- Use `mcp_to_openai_tool()` for conversion
- Replace manual schema construction

**Changes**: See Task 8.1 implementation

### Phase 4: Examples & Documentation

#### Task 12: Update search_agent.py Example
**File**: `examples/search_agent.py:60-72`

**Current**:
```python
async def main():
    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")
```

**Target**: (Minimal changes, mostly works as-is)
```python
async def main():
    print("--- Starting Search Agent Demo ---")

    runner = InMemoryRunner()
    ctx = AppContext(runner)
    result = await ctx.search_agent(prompt="What is the latest Python version?")

    print(f"\nFinal Result: {result[-1].payload['content']}")
    print("--- Demo Complete ---")
```

#### Task 13: Create MCP Server Example
**File**: `examples/mcp_server_example.py` (new)

- Demonstrate explicit MCP exposure
- Show internal vs external policies
- Document the pattern

**Implementation**:
```python
import asyncio
from fastmcp import FastMCP
from pocket_joe import BaseContext, InMemoryRunner
from examples.utils import llm, web_search, search_agent

class AppContext(BaseContext):
    def __init__(self, runner):
        super().__init__(runner)
        self.llm = self._bind(llm)
        self.web_search = self._bind(web_search)
        self.search_agent = self._bind(search_agent)

async def main():
    runner = InMemoryRunner()
    ctx = AppContext(runner)

    # Setup MCP server
    mcp = FastMCP(name="pocket-joe")

    # Explicitly expose only external-facing policies
    ctx.expose_to_mcp(mcp, ["web_search", "search_agent"])
    # Note: 'llm' is NOT exposed - internal only

    print("MCP server running with exposed tools: web_search, search_agent")
    mcp.run()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Task 14: Update Tests
**File**: `tests/test_core.py`

Add test coverage for:
- `@policy` decorator extracts Tool metadata correctly
- `PolicyProxy` preserves function signature and metadata
- `get_ctx()` works in concurrent executions
- `ContextVar` isolation between different contexts
- MCP conversion helpers produce correct schemas
- `expose_to_mcp()` registers tools correctly

**Test Examples**:
```python
async def test_policy_decorator_extracts_metadata():
    @policy(description="Test policy")
    async def test_func(arg: str) -> list[Message]:
        return []

    assert hasattr(test_func, '_tool_metadata')
    assert test_func._tool_name == "test_func"
    assert test_func._tool_description == "Test policy"

async def test_context_isolation():
    runner1 = InMemoryRunner()
    ctx1 = AppContext(runner1)

    runner2 = InMemoryRunner()
    ctx2 = AppContext(runner2)

    # Test that contexts don't interfere
    # ...
```

#### Task 15: Cleanup & Documentation
**Files**: Multiple

##### 15.1: Clean up context.py
**File**: `pocket_joe/context.py`
- Remove all commented-out code
- Remove old class-based patterns
- Keep only if needed for backwards compatibility

##### 15.2: Update __init__.py Exports
**File**: `pocket_joe/__init__.py`
- Export new decorators
- Export MCP helpers
- Export PolicyProxy if needed

**Changes**:
```python
from .core import Message, BaseContext
from .decorators import policy
from .policy_spec_mcp import policy_spec_mcp_tool, policy_spec_mcp_resource
from .mcp_helpers import mcp_to_openai_tool, mcp_to_anthropic_tool, get_tools_from_context
from .memory_runtime import InMemoryRunner
from .policy_proxy import PolicyProxy
```

##### 15.3: Update README.md
**File**: `README.md`

Add sections:
- Quick start with function-based policies
- How to use `@policy` decorator
- Context access via `get_ctx()`
- MCP integration patterns
- Migration guide from old pattern

##### 15.4: Create Migration Guide
**File**: `docs/MIGRATION.md` (new)

Document:
- Why we migrated to function-based
- Step-by-step conversion process
- Before/after examples
- Breaking changes
- Troubleshooting common issues

## Data Flow Diagrams

### Registry Structure
```
self._registry = {
    "web_search": <function web_search>,      # Has ._tool_metadata
    "llm": <function llm>,                    # Has ._tool_metadata
    "search_agent": <function search_agent>   # Has ._tool_metadata
}
```

### Tool Metadata Flow
```
@policy(description="...")
    ↓
Tool.from_function() extracts schema
    ↓
Metadata stored on func._tool_metadata
    ↓
_bind() extracts and passes to PolicyProxy
    ↓
PolicyProxy preserves for MCP exposure
    ↓
expose_to_mcp() uses metadata directly
```

### Context Flow
```
ctx.search_agent(prompt="...")
    ↓
PolicyProxy.__call__(**kwargs)
    ↓
runner.execute("search_agent", kwargs)
    ↓
_set_ctx() → run function → _reset_ctx()
    ↓
Inside function: AppContext.get_ctx()
```

### Execution Flow
```
User Code:
  result = await ctx.search_agent(prompt="test")
    ↓
PolicyProxy routes to:
  runner.execute("search_agent", {"prompt": "test"})
    ↓
Runner sets context:
  token = ctx._set_ctx()
    ↓
Function executes:
  async def search_agent(prompt):
      ctx = AppContext.get_ctx()  # Gets the context
      await ctx.llm(...)           # Nested policy call
    ↓
Runner resets context:
  ctx._reset_ctx(token)
    ↓
Result returned to user
```

## Critical Dependencies

1. **Task 8 must be atomic**: Cannot mix class-based and function-based patterns
2. **invoke_options_wrapper must work with functions** before Task 8
3. **All decorators must be implemented** before converting policies
4. **Context management must be working** before migration

## Testing Strategy

### Unit Tests
- Decorator functionality
- PolicyProxy behavior
- ContextVar isolation
- Tool metadata extraction
- MCP conversion helpers

### Integration Tests
- Full policy execution flow
- Nested policy calls
- Concurrent context usage
- MCP server integration

### Migration Validation
- All examples run successfully
- No regression in functionality
- Performance is maintained or improved

## Rollback Plan

If migration fails:
1. Revert to previous commit
2. All changes are in a feature branch
3. Class-based pattern still available in git history

## Success Criteria

- [ ] All policies converted to function-based
- [ ] All examples work with new pattern
- [ ] All tests pass
- [ ] MCP integration works
- [ ] Documentation updated
- [ ] No performance regression
- [ ] IDE autocomplete works for all policies

## Timeline & Milestones

### Milestone 1: Infrastructure (Tasks 1-7)
Foundation for function-based policies

### Milestone 2: Migration (Task 8)
Convert all policies atomically

### Milestone 3: Integration (Tasks 9-11)
MCP helpers and conversions

### Milestone 4: Polish (Tasks 12-15)
Examples, tests, documentation

## Notes

- Per-subclass singleton via ContextVar allows multiple context types
- FastMCP's `Tool.from_function()` enables schema extraction without registration
- Explicit MCP exposure gives control over what LLMs can access
- Function-based approach improves IDE support and developer experience
