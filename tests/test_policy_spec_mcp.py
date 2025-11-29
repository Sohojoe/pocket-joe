"""Tests for policy_spec_mcp decorators and parameter unpacking."""
import asyncio
import pytest
from pocket_joe.core import Action, Context, Step
from pocket_joe.policy_spec_mcp import policy_spec, policy_spec_mcp_tool, policy_spec_mcp_resource
from pocket_joe.registry import Registry
from pocket_joe.memory_runtime import InMemoryRunner


@pytest.mark.asyncio
async def test_tool_with_required_parameters():
    """Test tool policy with required parameters gets them unpacked correctly."""
    
    @policy_spec_mcp_tool(description='Search tool')
    async def search_tool(action: Action, ctx: Context, query: str, limit: int = 10) -> list[Step]:
        return [Step(
            actor='search_tool',
            type='action_result',
            payload={'results': f'Found results for: {query} (limit={limit})'}
        )]

    registry = Registry(search_tool)
    runner = InMemoryRunner(registry)
    
    # Simulate an action_call step from LLM
    action_call_step = Step(
        actor='assistant',
        type='action_call',
        payload={
            'policy': 'search_tool',
            'payload': {'query': 'Python 3.12', 'limit': 5}
        }
    )
    
    action = Action(
        policy='search_tool',
        payload=[action_call_step]
    )
    
    result = await runner.execute(action)
    
    assert len(result) == 1
    assert result[0].actor == 'search_tool'
    assert result[0].type == 'action_result'
    assert 'Python 3.12' in result[0].payload['results']
    assert 'limit=5' in result[0].payload['results']


@pytest.mark.asyncio
async def test_tool_with_default_parameters():
    """Test tool policy with default parameters works when not provided."""
    
    @policy_spec_mcp_tool(description='Search tool')
    async def search_tool(action: Action, ctx: Context, query: str, limit: int = 10) -> list[Step]:
        return [Step(
            actor='search_tool',
            type='action_result',
            payload={'results': f'Found results for: {query} (limit={limit})'}
        )]

    registry = Registry(search_tool)
    runner = InMemoryRunner(registry)
    
    # Only provide required param
    action_call_step = Step(
        actor='assistant',
        type='action_call',
        payload={
            'policy': 'search_tool',
            'payload': {'query': 'Python 3.12'}  # limit omitted, should use default
        }
    )
    
    action = Action(
        policy='search_tool',
        payload=[action_call_step]
    )
    
    result = await runner.execute(action)
    
    assert len(result) == 1
    assert 'limit=10' in result[0].payload['results']  # Default value used


@pytest.mark.asyncio
async def test_missing_required_parameter_raises_error():
    """Test that missing required parameters raise validation error."""
    
    @policy_spec_mcp_tool(description='Search tool')
    async def search_tool(action: Action, ctx: Context, query: str) -> list[Step]:
        return [Step(
            actor='search_tool',
            type='action_result',
            payload={'results': f'Found: {query}'}
        )]

    registry = Registry(search_tool)
    runner = InMemoryRunner(registry)
    
    # Missing required 'query' param
    action_call_step = Step(
        actor='assistant',
        type='action_call',
        payload={
            'policy': 'search_tool',
            'payload': {}  # Empty!
        }
    )
    
    action = Action(
        policy='search_tool',
        payload=[action_call_step]
    )
    
    with pytest.raises(ValueError, match="Missing required parameter 'query'"):
        await runner.execute(action)


@pytest.mark.asyncio
async def test_llm_policy_with_no_extra_params():
    """Test LLM policy with no extra params (uses ctx.get_ledger() and action.actions)."""
    
    @policy_spec_mcp_tool(description='LLM policy')
    async def llm_policy(action: Action, ctx: Context) -> list[Step]:
        ledger = ctx.get_ledger()
        tools = action.actions
        return [Step(
            actor='assistant',
            type='text',
            payload={'content': f'Processed {len(ledger)} steps with {len(tools)} tools'}
        )]

    registry = Registry(llm_policy)
    runner = InMemoryRunner(registry)
    
    # LLM doesn't use action_call format, just regular invocation
    user_step = Step(
        actor='user',
        type='text',
        payload={'content': 'Hello'}
    )
    
    action = Action(
        policy='llm_policy',
        payload=[user_step],
        actions={'search', 'calculate'}
    )
    
    result = await runner.execute(action)
    
    assert len(result) == 1
    assert result[0].actor == 'assistant'
    assert result[0].type == 'text'
    assert 'Processed 1 steps with 2 tools' in result[0].payload['content']


@pytest.mark.asyncio
async def test_registry_mcp_kind_filtering():
    """Test registry correctly filters policies by MCP kind."""
    
    @policy_spec_mcp_tool(description='A tool')
    async def tool1(action: Action, ctx: Context) -> list[Step]:
        return []

    @policy_spec_mcp_resource(description='A resource')
    async def resource1(action: Action, ctx: Context) -> list[Step]:
        return []

    @policy_spec(mcp_kind='none_mcp', description='Internal only')
    async def internal1(action: Action, ctx: Context) -> list[Step]:
        return []

    registry = Registry(tool1, resource1, internal1)

    tools = registry.mcp_tools()
    resources = registry.mcp_resources()
    internal = registry.internal_only()
    
    assert list(tools.keys()) == ['tool1']
    assert list(resources.keys()) == ['resource1']
    assert list(internal.keys()) == ['internal1']
