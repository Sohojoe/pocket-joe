"""Tests for pocket_joe.core module."""

import pytest
from dataclasses import replace
from pocket_joe.core import Message, Policy, BaseContext


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
        """Test that replace() works for creating modified copies."""
        msg1 = Message(actor="user", type="text", payload={"content": "hello"})
        msg2 = replace(msg1, payload={"content": "goodbye"})
        
        assert msg1.payload == {"content": "hello"}
        assert msg2.payload == {"content": "goodbye"}
        assert msg1.actor == msg2.actor
        assert msg1.type == msg2.type


class TestPolicy:
    """Test Policy base class."""
    
    def test_policy_requires_context(self):
        """Test that Policy requires a context."""
        # Create a mock context
        class MockRunner:
            pass
        
        ctx = BaseContext(MockRunner())
        
        # Policy can be instantiated with context
        policy = Policy(ctx)
        assert policy.ctx is ctx
    
    @pytest.mark.asyncio
    async def test_policy_call_not_implemented(self):
        """Test that base Policy.__call__ raises NotImplementedError."""
        class MockRunner:
            pass
        
        ctx = BaseContext(MockRunner())
        policy = Policy(ctx)
        
        with pytest.raises(NotImplementedError):
            await policy()


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
        """Test _bind method delegates to runner's strategy."""
        class MockPolicy(Policy):
            async def __call__(self):
                return [Message(actor="test", type="test", payload={})]
        
        class MockRunner:
            def _bind_strategy(self, policy_class, context):
                # Return a bound instance
                instance = policy_class(context)
                return instance
        
        runner = MockRunner()
        ctx = BaseContext(runner)
        
        bound = ctx._bind(MockPolicy)
        assert isinstance(bound, MockPolicy)
        assert bound.ctx is ctx
        assert hasattr(bound, '__policy_class__')
        assert bound.__policy_class__ is MockPolicy
    
    def test_get_policy_success(self):
        """Test get_policy retrieves the policy class."""
        class MockPolicy(Policy):
            async def __call__(self):
                return []
        
        class MockRunner:
            def _bind_strategy(self, policy_class, context):
                instance = policy_class(context)
                return instance
        
        runner = MockRunner()
        ctx = BaseContext(runner)
        
        # Bind and attach to context
        bound = ctx._bind(MockPolicy)
        ctx.test_policy = bound
        
        # Retrieve the policy class
        policy_class = ctx.get_policy('test_policy')
        assert policy_class is MockPolicy
    
    def test_get_policy_not_found(self):
        """Test get_policy raises AttributeError if policy doesn't exist."""
        class MockRunner:
            pass
        
        runner = MockRunner()
        ctx = BaseContext(runner)
        
        with pytest.raises(AttributeError):
            ctx.get_policy('nonexistent')
    
    def test_get_policy_no_class_stored(self):
        """Test get_policy raises ValueError if __policy_class__ not found."""
        class MockRunner:
            pass
        
        runner = MockRunner()
        ctx = BaseContext(runner)
        
        # Manually attach something without __policy_class__
        ctx.fake = lambda: None
        
        with pytest.raises(ValueError, match="Policy class not found"):
            ctx.get_policy('fake')
