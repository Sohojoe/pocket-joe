"""Tests for pocket_joe.core module."""

import pytest
from pocket_joe.core import Message, BaseContext
from pocket_joe import MessageBuilder, OptionCallPayload, OptionResultPayload, TextPart


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation_with_parts(self):
        """Test basic Message creation with parts."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("hello")
        msg = builder.to_message()

        assert msg.policy == "user"
        assert msg.parts is not None
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].text == "hello"
        assert msg.role_hint_for_llm == "user"

    def test_message_with_option_call(self):
        """Test Message with option_call payload."""
        msg = Message(
            policy="assistant",
            role_hint_for_llm="assistant",
            payload=OptionCallPayload(
                invocation_id="inv_123",
                option_name="get_weather",
                arguments={"city": "SF"}
            )
        )

        assert msg.payload is not None
        assert isinstance(msg.payload, OptionCallPayload)
        assert msg.payload.invocation_id == "inv_123"
        assert msg.payload.option_name == "get_weather"

    def test_message_immutability(self):
        """Test that Message is immutable (frozen)."""
        builder = MessageBuilder(policy="user")
        builder.add_text("test")
        msg = builder.to_message()

        with pytest.raises(Exception):  # FrozenInstanceError
            msg.policy = "assistant"  # type: ignore

    def test_message_replace(self):
        """Test that model_copy() works for creating modified copies."""
        builder = MessageBuilder(policy="user")
        builder.add_text("hello")
        msg1 = builder.to_message()
        msg2 = msg1.model_copy(update={"step_num": 5})

        assert msg1.step_num is None
        assert msg2.step_num == 5
        assert msg1.policy == msg2.policy

    def test_message_validation_parts_and_payload(self):
        """Test that Message cannot have both parts and payload."""
        with pytest.raises(ValueError, match="cannot have both parts and payload"):
            Message(
                policy="test",
                parts=[TextPart(text="hello")],
                payload=OptionCallPayload(
                    invocation_id="123",
                    option_name="test",
                    arguments={}
                )
            )

    def test_message_validation_needs_parts_or_payload(self):
        """Test that Message must have either parts or payload."""
        with pytest.raises(ValueError, match="must have either parts or payload"):
            Message(policy="test")


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
        from pocket_joe import policy, MessageBuilder

        @policy.tool(description="Test policy")
        async def test_policy(arg: str) -> list[Message]:
            """Test policy function"""
            builder = MessageBuilder(policy="test_policy")
            builder.add_text(arg)
            return [builder.to_message()]

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
        from pocket_joe import policy, InMemoryRunner, MessageBuilder

        @policy.tool(description="Echo policy")
        async def echo_policy(message: str) -> list[Message]:
            """Echo the message back"""
            builder = MessageBuilder(policy="echo_policy")
            builder.add_text(message)
            return [builder.to_message()]

        runner = InMemoryRunner()
        ctx = BaseContext(runner)

        # Bind and execute
        bound = ctx._bind(echo_policy)
        result = await bound(message="hello")

        assert len(result) == 1
        assert result[0].parts is not None

    @pytest.mark.asyncio
    async def test_bind_and_execute_primitive_return(self):
        """Test that bound policies with primitive returns work correctly."""
        from pocket_joe import policy, InMemoryRunner

        @policy.tool(description="Get greeting")
        async def get_greeting(name: str) -> str:
            """Return a greeting"""
            return f"Hello, {name}!"

        runner = InMemoryRunner()
        ctx = BaseContext(runner)

        # Bind and execute
        bound = ctx._bind(get_greeting)
        result = await bound(name="Alice")

        # Should return the string directly
        assert result == "Hello, Alice!"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_bind_and_execute_dict_return(self):
        """Test that bound policies with dict returns work correctly."""
        from pocket_joe import policy, InMemoryRunner
        from typing import Any

        @policy.tool(description="Get user data")
        async def get_user(user_id: int) -> dict[str, Any]:
            """Return user data"""
            return {"id": user_id, "name": "Alice"}

        runner = InMemoryRunner()
        ctx = BaseContext(runner)

        # Bind and execute
        bound = ctx._bind(get_user)
        result = await bound(user_id=123)

        # Should return the dict directly
        assert result == {"id": 123, "name": "Alice"}
        assert isinstance(result, dict)
    
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
