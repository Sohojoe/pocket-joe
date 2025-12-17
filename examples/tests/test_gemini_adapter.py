"""Unit tests for GeminiAdapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid

from pocket_joe import (
    Message,
    MessageBuilder,
    TextPart,
    OptionSchema,
    OptionCallPayload,
    OptionResultPayload,
)

from examples.utils.gemini_adapter import GeminiAdapter


class TestGeminiAdapterEncodeContents:
    """Test encoding pocket-joe Messages to Gemini Content format."""

    def test_text_only_message(self):
        """Test converting a text-only message."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("Hello, world!")
        msg = builder.to_message()

        adapter = GeminiAdapter([msg])

        assert len(adapter.contents) == 1
        assert adapter.contents[0].role == "user"
        assert len(adapter.contents[0].parts) == 1
        assert adapter.contents[0].parts[0].text == "Hello, world!"

    def test_multiple_text_parts(self):
        """Test message with multiple text parts."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("First part")
        builder.add_text("Second part")
        msg = builder.to_message()

        adapter = GeminiAdapter([msg])

        assert len(adapter.contents) == 1
        assert len(adapter.contents[0].parts) == 2
        assert adapter.contents[0].parts[0].text == "First part"
        assert adapter.contents[0].parts[1].text == "Second part"

    def test_assistant_mapped_to_model(self):
        """Test that assistant role is mapped to 'model'."""
        builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
        builder.add_text("I can help!")
        msg = builder.to_message()

        adapter = GeminiAdapter([msg])

        assert adapter.contents[0].role == "model"

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

        adapter = GeminiAdapter([call_msg, result_msg])

        assert len(adapter.contents) == 2
        # Function call (model role)
        assert adapter.contents[0].role == "model"
        assert adapter.contents[0].parts[0].function_call is not None
        assert adapter.contents[0].parts[0].function_call.name == "get_weather"
        # Function response (user role in Gemini)
        assert adapter.contents[1].role == "user"
        assert adapter.contents[1].parts[0].function_response is not None

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

        adapter = GeminiAdapter([call_msg])
        assert len(adapter.contents) == 0


class TestGeminiAdapterEncodeTools:
    """Test encoding OptionSchema to Gemini Tool format."""

    def test_empty_options(self):
        """Test with no options."""
        adapter = GeminiAdapter([], None)
        assert adapter.tools is None

        adapter = GeminiAdapter([], [])
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

        adapter = GeminiAdapter([], [option])

        assert adapter.tools is not None
        assert len(adapter.tools) == 1
        assert hasattr(adapter.tools[0], 'function_declarations')
        assert adapter.tools[0].function_declarations[0].name == "get_weather"

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

        adapter = GeminiAdapter([], options)

        assert adapter.tools is not None
        assert len(adapter.tools) == 1  # Gemini bundles all into one Tool
        assert len(adapter.tools[0].function_declarations) == 2


class TestGeminiAdapterDecode:
    """Test decoding Gemini response to pocket-joe Messages."""

    def test_text_response(self):
        """Test converting a text-only response."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        
        mock_part = Mock()
        mock_part.text = "Hello from Gemini!"
        mock_part.inline_data = None
        mock_part.function_call = None
        mock_response.candidates[0].content.parts = [mock_part]

        adapter = GeminiAdapter([])
        result = adapter.decode(mock_response, policy="test")

        assert len(result) == 1
        assert result[0].policy == "test"
        assert result[0].parts is not None
        text_parts = [p for p in result[0].parts if isinstance(p, TextPart)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello from Gemini!"

    def test_function_call_response(self):
        """Test converting a function call response."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        
        mock_part = Mock()
        mock_part.text = None
        mock_part.inline_data = None
        mock_part.function_call = Mock()
        mock_part.function_call.name = "get_weather"
        mock_part.function_call.args = {"city": "SF"}
        mock_response.candidates[0].content.parts = [mock_part]

        adapter = GeminiAdapter([])
        result = adapter.decode(mock_response, policy="test")

        assert len(result) == 1
        assert result[0].payload is not None
        assert isinstance(result[0].payload, OptionCallPayload)
        assert result[0].payload.option_name == "get_weather"
        assert result[0].payload.arguments == {"city": "SF"}


class TestGeminiAdapterClient:
    """Test the static client factory."""

    def test_client_with_api_key(self):
        """Test that client() creates a genai.Client."""
        with patch('examples.utils.gemini_adapter.genai.Client') as mock_class:
            mock_client = Mock()
            mock_class.return_value = mock_client

            result = GeminiAdapter.client(api_key="test-key")

            mock_class.assert_called_once_with(api_key="test-key")
            assert result == mock_client

    def test_client_uses_env_var(self):
        """Test that client() falls back to GOOGLE_API_KEY env var."""
        with patch('examples.utils.gemini_adapter.genai.Client') as mock_class, \
             patch.dict('os.environ', {'GOOGLE_API_KEY': 'env-key'}):
            mock_client = Mock()
            mock_class.return_value = mock_client

            result = GeminiAdapter.client()

            mock_class.assert_called_once_with(api_key="env-key")

    def test_client_raises_without_key(self):
        """Test that client() raises when no API key available."""
        with patch.dict('os.environ', {}, clear=True):
            # Remove GOOGLE_API_KEY if it exists
            import os
            if 'GOOGLE_API_KEY' in os.environ:
                del os.environ['GOOGLE_API_KEY']
            
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiAdapter.client()
