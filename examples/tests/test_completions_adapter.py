"""Unit tests for CompletionsAdapter."""

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

from examples.utils.completions_adapter import CompletionsAdapter


class TestCompletionsAdapterEncodeMessages:
    """Test encoding pocket-joe Messages to chat completions format."""

    def test_text_only_message(self):
        """Test converting a text-only message."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("Hello, world!")
        msg = builder.to_message()

        adapter = CompletionsAdapter([msg])

        assert len(adapter.messages) == 1
        assert adapter.messages[0]["role"] == "user"
        assert adapter.messages[0]["content"] == "Hello, world!"

    def test_multiple_text_parts(self):
        """Test message with multiple text parts gets joined."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("First part")
        builder.add_text("Second part")
        msg = builder.to_message()

        adapter = CompletionsAdapter([msg])

        assert len(adapter.messages) == 1
        assert adapter.messages[0]["content"] == "First part Second part"

    def test_assistant_message(self):
        """Test converting an assistant message."""
        builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
        builder.add_text("I can help with that!")
        msg = builder.to_message()

        adapter = CompletionsAdapter([msg])

        assert len(adapter.messages) == 1
        assert adapter.messages[0]["role"] == "assistant"
        assert adapter.messages[0]["content"] == "I can help with that!"

    def test_option_call_with_result(self):
        """Test converting option_call and option_result messages."""
        invocation_id = str(uuid.uuid4())

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

        adapter = CompletionsAdapter([call_msg, result_msg])

        assert len(adapter.messages) == 2
        assert adapter.messages[0]["role"] == "assistant"
        assert "tool_calls" in adapter.messages[0]
        assert adapter.messages[0]["tool_calls"][0]["id"] == invocation_id
        assert adapter.messages[1]["role"] == "tool"
        assert adapter.messages[1]["tool_call_id"] == invocation_id

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

        adapter = CompletionsAdapter([call_msg])
        assert len(adapter.messages) == 0


class TestCompletionsAdapterEncodeTools:
    """Test encoding OptionSchema to chat completions tools format."""

    def test_empty_options(self):
        """Test with no options."""
        adapter = CompletionsAdapter([], None)
        assert adapter.tools is None

        adapter = CompletionsAdapter([], [])
        assert adapter.tools is None

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

        adapter = CompletionsAdapter([], [option])

        assert adapter.tools is not None
        assert len(adapter.tools) == 1
        assert adapter.tools[0]["type"] == "function"
        assert adapter.tools[0]["function"]["name"] == "get_weather"

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

        adapter = CompletionsAdapter([], options)

        assert len(adapter.tools) == 2
        assert adapter.tools[0]["function"]["name"] == "tool1"
        assert adapter.tools[1]["function"]["name"] == "tool2"


class TestCompletionsAdapterDecode:
    """Test decoding chat completions response to pocket-joe Messages."""

    def test_text_response(self):
        """Test converting a text-only response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None

        adapter = CompletionsAdapter([])
        result = adapter.decode(mock_response, policy="test")

        assert len(result) == 1
        assert result[0].policy == "test"
        assert result[0].parts[0].text == "Hello!"

    def test_tool_call_response(self):
        """Test converting a tool call response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "SF"}'

        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        adapter = CompletionsAdapter([])
        result = adapter.decode(mock_response, policy="test")

        assert len(result) == 1
        assert result[0].payload.invocation_id == "call_123"
        assert result[0].payload.option_name == "get_weather"
        assert result[0].payload.arguments == {"city": "SF"}


class TestCompletionsAdapterClient:
    """Test the static client factory."""

    def test_client_returns_async_openai(self):
        """Test that client() returns an AsyncOpenAI instance."""
        with patch('examples.utils.completions_adapter.AsyncOpenAI') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client

            result = CompletionsAdapter.client(api_key="test")

            mock_class.assert_called_once_with(api_key="test")
            assert result == mock_client
