"""Tests for pocket_joe.core module."""

import pytest
from pocket_joe.core import Message, BaseContext


class TestMessage:
    """Test Message dataclass."""
    
    def test_message_creation(self):
        """Test basic Message creation."""
        msg = Message(
            actor="user",
            type="text",
            payload={"content": "hello"}
        )
        
        assert msg.actor == "user"
        assert msg.type == "text"
        assert msg.payload == {"content": "hello"}
        assert msg.tool_id is None
        assert msg.id == ""
    
    def test_message_with_tool_id(self):
        """Test Message with tool_id."""
        msg = Message(
            actor="assistant",
            type="action_call",
            payload={"function": "get_weather"},
            tool_id="tool_123"
        )
        
        assert msg.tool_id == "tool_123"
    
    def test_message_immutability(self):
        """Test that Message is immutable (frozen)."""
        msg = Message(actor="user", type="text", payload={"content": "test"})
        
        with pytest.raises(Exception):  # FrozenInstanceError
            msg.actor = "assistant"
    
    def test_message_replace(self):
        """Test that model_copy() works for creating modified copies."""
        msg1 = Message(actor="user", type="text", payload={"content": "hello"})
        msg2 = msg1.model_copy(update={"payload": {"content": "goodbye"}})
        
        assert msg1.payload == {"content": "hello"}
        assert msg2.payload == {"content": "goodbye"}
        assert msg1.actor == msg2.actor
        assert msg1.type == msg2.type


class TestBaseContext:
    """Test BaseContext class."""
    
    def test_context_creation(self):
        """Test BaseContext creation with runner."""
        class MockRunner:
            pass
        
        runner = MockRunner()
        ctx = BaseContext(runner)
        
        assert ctx._runner is runner
    
    def test_bind_policy(self):
        """Test _bind method works with function-based policies."""
        from pocket_joe import policy
        
        @policy.tool(description="Test policy")
        async def test_policy(arg: str) -> list[Message]:
            """Test policy function"""
            return [Message(actor="test", type="test", payload={"arg": arg})]
        
        from pocket_joe import InMemoryRunner
        runner = InMemoryRunner()
        ctx = BaseContext(runner)
        
        # Bind the function-based policy
        bound = ctx._bind(test_policy)
        
        # Verify the bound callable has the policy function attached
        assert hasattr(bound, '__policy_func__')
        assert bound.__policy_func__ is test_policy
        
        # Verify it has tool metadata
        assert hasattr(test_policy, '_tool_metadata')
        assert hasattr(test_policy, '_option_schema')
    
    @pytest.mark.asyncio
    async def test_bind_and_execute(self):
        """Test that bound policies can be executed."""
        from pocket_joe import policy, InMemoryRunner
        
        @policy.tool(description="Echo policy")
        async def echo_policy(message: str) -> list[Message]:
            """Echo the message back"""
            return [Message(actor="echo", type="text", payload={"content": message})]
        
        runner = InMemoryRunner()
        ctx = BaseContext(runner)
        
        # Bind and execute
        bound = ctx._bind(echo_policy)
        result = await bound(message="hello")
        
        assert len(result) == 1
        assert result[0].payload["content"] == "hello"
    
    def test_get_policy_success(self):
        """Test get_policy retrieves the policy function."""
        from pocket_joe import policy, InMemoryRunner
        
        @policy.tool(description="Test policy")
        async def test_policy() -> list[Message]:
            return []
        
        runner = InMemoryRunner()
        ctx = BaseContext(runner)
        
        # Bind the policy
        ctx._bind(test_policy)
        
        # Retrieve the policy function by name
        retrieved = ctx.get_policy('test_policy')
        assert retrieved is test_policy
    
    def test_get_policy_not_found(self):
        """Test get_policy raises ValueError if policy doesn't exist."""
        from pocket_joe import InMemoryRunner
        
        runner = InMemoryRunner()
        ctx = BaseContext(runner)
        
        with pytest.raises(ValueError, match="Bound policy not found"):
            ctx.get_policy('nonexistent')
    
    def test_bind_duplicate_name(self):
        """Test that binding a policy with duplicate name raises ValueError."""
        from pocket_joe import policy, InMemoryRunner
        
        @policy.tool(description="First policy")
        async def duplicate_name() -> list[Message]:
            return []
        
        @policy.tool(description="Second policy", name="duplicate_name")
        async def another_policy() -> list[Message]:
            return []
        
        runner = InMemoryRunner()
        ctx = BaseContext(runner)
        
        # First bind should succeed
        ctx._bind(duplicate_name)
        
        # Second bind with same name should fail
        with pytest.raises(ValueError, match="Duplicate policy name"):
            ctx._bind(another_policy)
