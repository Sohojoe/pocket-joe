"""Unit tests for OpenAI LLM policies."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
import uuid

from pocket_joe import (
    Message,
    MessageBuilder,
    TextPart,
    OptionSchema,
    OptionCallPayload,
    OptionResultPayload,
)

# Import the functions to test
from examples.utils.llm_policies import (
    observations_to_completions_messages,
    options_to_completions_tools,
    completions_response_to_messages,
    openai_llm_policy_v1,
)


class TestObservationsToCompletionsMessages:
    """Test conversion from pocket-joe Messages to OpenAI chat completions format."""

    def test_text_only_message(self):
        """Test converting a text-only message."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("Hello, world!")
        msg = builder.to_message()

        result = observations_to_completions_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_multiple_text_parts(self):
        """Test message with multiple text parts gets joined."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("First part")
        builder.add_text("Second part")
        msg = builder.to_message()

        result = observations_to_completions_messages([msg])

        assert len(result) == 1
        assert result[0]["content"] == "First part Second part"

    def test_assistant_message(self):
        """Test converting an assistant message."""
        builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
        builder.add_text("I can help with that!")
        msg = builder.to_message()

        result = observations_to_completions_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I can help with that!"

    def test_option_call_with_result(self):
        """Test converting option_call and option_result messages."""
        invocation_id = str(uuid.uuid4())

        # Create option_call message
        call_msg = Message(
            id=str(uuid.uuid4()),
            policy="assistant",
            role_hint_for_llm="assistant",
            payload=OptionCallPayload(
                invocation_id=invocation_id,
                option_name="get_weather",
                arguments={"city": "San Francisco"}
            )
        )

        # Create option_result message
        result_msg = Message(
            id=str(uuid.uuid4()),
            policy="get_weather",
            role_hint_for_llm="tool",
            payload=OptionResultPayload(
                invocation_id=invocation_id,
                option_name="get_weather",
                result="Sunny, 72°F"
            )
        )

        result = observations_to_completions_messages([call_msg, result_msg])

        assert len(result) == 2

        # Check assistant message with tool_calls
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["id"] == invocation_id
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "San Francisco"}

        # Check tool response message
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == invocation_id
        assert result[1]["content"] == "Sunny, 72°F"

    def test_option_call_without_result_skipped(self):
        """Test that option_call without matching result is skipped."""
        call_msg = Message(
            id=str(uuid.uuid4()),
            policy="assistant",
            role_hint_for_llm="assistant",
            payload=OptionCallPayload(
                invocation_id="incomplete",
                option_name="test",
                arguments={}
            )
        )

        result = observations_to_completions_messages([call_msg])

        # Should be empty since no matching result
        assert len(result) == 0

    def test_complex_result_serialization(self):
        """Test that complex (non-string) results are JSON serialized."""
        invocation_id = str(uuid.uuid4())

        call_msg = Message(
            id=str(uuid.uuid4()),
            policy="assistant",
            payload=OptionCallPayload(
                invocation_id=invocation_id,
                option_name="get_data",
                arguments={}
            )
        )

        result_msg = Message(
            id=str(uuid.uuid4()),
            policy="get_data",
            role_hint_for_llm="tool",
            payload=OptionResultPayload(
                invocation_id=invocation_id,
                option_name="get_data",
                result={"temperature": 72, "conditions": "sunny"}
            )
        )

        result = observations_to_completions_messages([call_msg, result_msg])

        # Tool result should be JSON serialized
        assert result[1]["role"] == "tool"
        assert json.loads(result[1]["content"]) == {"temperature": 72, "conditions": "sunny"}

    def test_mixed_messages(self):
        """Test a realistic conversation with mixed message types."""
        invocation_id = str(uuid.uuid4())

        # User message
        user_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        user_builder.add_text("What's the weather in SF?")
        user_msg = user_builder.to_message()

        # Assistant decides to call tool
        call_msg = Message(
            id=str(uuid.uuid4()),
            policy="assistant",
            payload=OptionCallPayload(
                invocation_id=invocation_id,
                option_name="get_weather",
                arguments={"city": "SF"}
            )
        )

        # Tool result
        result_msg = Message(
            id=str(uuid.uuid4()),
            policy="get_weather",
            role_hint_for_llm="tool",
            payload=OptionResultPayload(
                invocation_id=invocation_id,
                option_name="get_weather",
                result="Sunny, 72°F"
            )
        )

        # Assistant final response
        assistant_builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
        assistant_builder.add_text("It's sunny and 72°F in San Francisco!")
        assistant_msg = assistant_builder.to_message()

        result = observations_to_completions_messages([
            user_msg, call_msg, result_msg, assistant_msg
        ])

        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "assistant"


class TestOptionsToCompletionsTools:
    """Test conversion from OptionSchema to OpenAI tools format."""

    def test_empty_options(self):
        """Test with no options."""
        result = options_to_completions_tools(None)
        assert result == []

        result = options_to_completions_tools([])
        assert result == []

    def test_single_option(self):
        """Test converting a single option."""
        option = OptionSchema(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )

        result = options_to_completions_tools([option])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather for a city"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_multiple_options(self):
        """Test converting multiple options."""
        options = [
            OptionSchema(
                name="tool1",
                description="First tool",
                parameters={"type": "object", "properties": {}}
            ),
            OptionSchema(
                name="tool2",
                description="Second tool",
                parameters={"type": "object", "properties": {}}
            )
        ]

        result = options_to_completions_tools(options)

        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["type"] == "function"
        assert result[1]["function"]["name"] == "tool2"


class TestCompletionsResponseToMessages:
    """Test conversion from OpenAI response to pocket-joe Messages."""

    def test_text_response(self):
        """Test converting a text-only response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello, how can I help you?"
        mock_response.choices[0].message.tool_calls = None

        result = completions_response_to_messages(mock_response, policy="openai")

        assert len(result) == 1
        assert result[0].policy == "openai"
        assert result[0].role_hint_for_llm == "assistant"
        assert result[0].parts is not None
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].text == "Hello, how can I help you?"

    def test_tool_call_response(self):
        """Test converting a tool call response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None

        # Mock tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "SF"}'

        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        result = completions_response_to_messages(mock_response, policy="openai")

        assert len(result) == 1
        assert result[0].payload is not None
        assert isinstance(result[0].payload, OptionCallPayload)
        assert result[0].payload.invocation_id == "call_123"
        assert result[0].payload.option_name == "get_weather"
        assert result[0].payload.arguments == {"city": "SF"}

    def test_text_and_tool_call_response(self):
        """Test response with both text and tool call."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Let me check the weather for you."

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "SF"}'

        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        result = completions_response_to_messages(mock_response, policy="openai")

        assert len(result) == 2
        # Text message
        assert result[0].parts is not None
        assert result[0].parts[0].text == "Let me check the weather for you."
        # Tool call message
        assert result[1].payload is not None
        assert result[1].payload.option_name == "get_weather"

    def test_multiple_tool_calls(self):
        """Test response with multiple tool calls."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None

        # Create multiple tool calls
        tool_call_1 = Mock()
        tool_call_1.id = "call_1"
        tool_call_1.function = Mock()
        tool_call_1.function.name = "tool1"
        tool_call_1.function.arguments = '{}'

        tool_call_2 = Mock()
        tool_call_2.id = "call_2"
        tool_call_2.function = Mock()
        tool_call_2.function.name = "tool2"
        tool_call_2.function.arguments = '{}'

        mock_response.choices[0].message.tool_calls = [tool_call_1, tool_call_2]

        result = completions_response_to_messages(mock_response, policy="openai")

        assert len(result) == 2
        assert result[0].payload.invocation_id == "call_1"
        assert result[1].payload.invocation_id == "call_2"


class TestOpenAILLMPolicyV1:
    """Test the main openai_llm_policy_v1 function."""

    @pytest.mark.asyncio
    async def test_policy_execution(self):
        """Test basic policy execution with mocked OpenAI."""
        with patch('examples.utils.llm_policies.AsyncOpenAI') as mock_openai_class:
            # Setup mock
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            # Create test input
            builder = MessageBuilder(policy="user", role_hint_for_llm="user")
            builder.add_text("Hello")
            observations = [builder.to_message()]

            # Execute
            result = await openai_llm_policy_v1(observations, [])

            # Verify
            assert len(result) == 1
            assert result[0].parts is not None
            assert result[0].parts[0].text == "Test response"

            # Verify OpenAI was called correctly
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]['model'] == "gpt-4"
            assert len(call_args[1]['messages']) == 1

    @pytest.mark.asyncio
    async def test_policy_with_tools(self):
        """Test policy with tool/function calling."""
        with patch('examples.utils.llm_policies.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            # Mock a tool call response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = None

            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{}'

            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            builder = MessageBuilder(policy="user", role_hint_for_llm="user")
            builder.add_text("Test")
            observations = [builder.to_message()]

            options = [
                OptionSchema(
                    name="test_tool",
                    description="Test tool",
                    parameters={"type": "object", "properties": {}}
                )
            ]

            # Execute
            result = await openai_llm_policy_v1(observations, options)

            # Verify tools were passed
            call_args = mock_client.chat.completions.create.call_args
            assert 'tools' in call_args[1]
            assert len(call_args[1]['tools']) == 1
            assert call_args[1]['tools'][0]['function']['name'] == 'test_tool'

            # Verify result
            assert len(result) == 1
            assert result[0].payload is not None
            assert result[0].payload.option_name == "test_tool"

    @pytest.mark.asyncio
    async def test_policy_with_conversation_history(self):
        """Test policy with multiple messages in conversation history."""
        with patch('examples.utils.llm_policies.AsyncOpenAI') as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Response"
            mock_response.choices[0].message.tool_calls = None

            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            # Create conversation history
            user_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
            user_builder.add_text("First message")
            user_msg = user_builder.to_message()

            assistant_builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
            assistant_builder.add_text("First response")
            assistant_msg = assistant_builder.to_message()

            user_builder_2 = MessageBuilder(policy="user", role_hint_for_llm="user")
            user_builder_2.add_text("Second message")
            user_msg_2 = user_builder_2.to_message()

            observations = [user_msg, assistant_msg, user_msg_2]

            # Execute
            await openai_llm_policy_v1(observations, [])

            # Verify all messages were sent
            call_args = mock_client.chat.completions.create.call_args
            assert len(call_args[1]['messages']) == 3
            assert call_args[1]['messages'][0]['content'] == "First message"
            assert call_args[1]['messages'][1]['content'] == "First response"
            assert call_args[1]['messages'][2]['content'] == "Second message"
