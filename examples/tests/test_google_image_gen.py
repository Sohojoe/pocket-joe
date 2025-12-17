"""Unit tests for Google Gemini image generation policy."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Any
import uuid

from pocket_joe import (
    Message,
    MessageBuilder,
    TextPart,
    MediaPart,
    OptionSchema,
    OptionCallPayload,
    OptionResultPayload,
)

# Import the functions to test
from examples.utils.google_image_gen import (
    observations_to_gemini_messages,
    options_to_gemini_tools,
    gemini_response_to_messages,
    google_gemini_policy_v1,
)


class TestObservationsToGeminiMessages:
    """Test conversion from pocket-joe Messages to Gemini format."""

    def test_text_only_message(self):
        """Test converting a text-only message."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("Hello, world!")
        msg = builder.to_message()

        result = observations_to_gemini_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["parts"]) == 1
        assert result[0]["parts"][0]["text"] == "Hello, world!"

    def test_text_with_image_message(self):
        """Test converting a message with text and image (interleaved)."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("What's in this image?")
        builder.add_image(url="https://example.com/image.jpg", mime="image/jpeg")
        msg = builder.to_message()

        result = observations_to_gemini_messages([msg])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0]["text"] == "What's in this image?"
        # Image is currently converted to text placeholder (TODO in implementation)
        assert "Image:" in result[0]["parts"][1]["text"]

    def test_multiple_text_parts(self):
        """Test message with multiple text parts."""
        builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        builder.add_text("First part")
        builder.add_text("Second part")
        msg = builder.to_message()

        result = observations_to_gemini_messages([msg])

        assert len(result) == 1
        assert len(result[0]["parts"]) == 2
        assert result[0]["parts"][0]["text"] == "First part"
        assert result[0]["parts"][1]["text"] == "Second part"

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

        result = observations_to_gemini_messages([call_msg, result_msg])

        assert len(result) == 2

        # Check function call
        assert result[0]["role"] == "model"
        assert "function_call" in result[0]["parts"][0]
        assert result[0]["parts"][0]["function_call"]["name"] == "get_weather"
        assert result[0]["parts"][0]["function_call"]["args"] == {"city": "San Francisco"}

        # Check function response
        assert result[1]["role"] == "function"
        assert "function_response" in result[1]["parts"][0]
        assert result[1]["parts"][0]["function_response"]["name"] == "get_weather"
        assert result[1]["parts"][0]["function_response"]["response"]["result"] == "Sunny, 72°F"

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

        result = observations_to_gemini_messages([call_msg])

        # Should be empty since no matching result
        assert len(result) == 0

    def test_role_mapping(self):
        """Test that role_hint_for_llm is correctly mapped."""
        user_builder = MessageBuilder(policy="user", role_hint_for_llm="user")
        user_builder.add_text("User message")
        user_msg = user_builder.to_message()

        assistant_builder = MessageBuilder(policy="assistant", role_hint_for_llm="assistant")
        assistant_builder.add_text("Assistant message")
        assistant_msg = assistant_builder.to_message()

        result = observations_to_gemini_messages([user_msg, assistant_msg])

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"


class TestOptionsToGeminiTools:
    """Test conversion from OptionSchema to Gemini function declarations."""

    def test_empty_options(self):
        """Test with no options."""
        result = options_to_gemini_tools(None)
        assert result is None

        result = options_to_gemini_tools([])
        assert result is None

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

        result = options_to_gemini_tools([option])

        # Should return a list with one Tool object
        assert result is not None
        assert len(result) == 1
        # The Tool object contains function_declarations
        assert hasattr(result[0], 'function_declarations')
        assert len(result[0].function_declarations) == 1
        assert result[0].function_declarations[0].name == "get_weather"
        assert result[0].function_declarations[0].description == "Get weather for a city"

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

        result = options_to_gemini_tools(options)

        # Should return a list with one Tool object containing multiple function declarations
        assert result is not None
        assert len(result) == 1
        assert len(result[0].function_declarations) == 2
        assert result[0].function_declarations[0].name == "tool1"
        assert result[0].function_declarations[1].name == "tool2"


class TestGeminiResponseToMessages:
    """Test conversion from Gemini response to pocket-joe Messages."""

    def test_text_response(self):
        """Test converting a text-only response."""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()

        mock_part = Mock()
        mock_part.text = "Hello, how can I help you?"
        mock_response.candidates[0].content.parts = [mock_part]

        # Remove other attributes
        del mock_part.inline_data
        del mock_part.function_call

        result = gemini_response_to_messages(mock_response, policy="gemini")

        assert len(result) == 1
        assert result[0].policy == "gemini"
        assert result[0].role_hint_for_llm == "assistant"
        assert result[0].parts is not None
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].text == "Hello, how can I help you?"

    def test_function_call_response(self):
        """Test converting a function call response."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()

        mock_part = Mock()
        mock_part.function_call = Mock()
        mock_part.function_call.name = "get_weather"
        mock_part.function_call.args = {"city": "SF"}
        mock_response.candidates[0].content.parts = [mock_part]

        del mock_part.text
        del mock_part.inline_data

        result = gemini_response_to_messages(mock_response, policy="gemini")

        assert len(result) == 1
        assert result[0].payload is not None
        assert isinstance(result[0].payload, OptionCallPayload)
        assert result[0].payload.option_name == "get_weather"
        assert result[0].payload.arguments == {"city": "SF"}

    def test_text_and_function_call_response(self):
        """Test response with both text and function call."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()

        text_part = Mock()
        text_part.text = "Let me check the weather"
        del text_part.inline_data
        del text_part.function_call

        func_part = Mock()
        func_part.function_call = Mock()
        func_part.function_call.name = "get_weather"
        func_part.function_call.args = {"city": "SF"}
        del func_part.text
        del func_part.inline_data

        mock_response.candidates[0].content.parts = [text_part, func_part]

        result = gemini_response_to_messages(mock_response, policy="gemini")

        assert len(result) == 2
        # Text message should come first
        assert result[0].parts is not None
        assert result[0].parts[0].text == "Let me check the weather"
        # Function call message
        assert result[1].payload is not None
        assert result[1].payload.option_name == "get_weather"

    def test_empty_response(self):
        """Test handling of empty response."""
        mock_response = Mock()
        mock_response.candidates = []

        result = gemini_response_to_messages(mock_response, policy="gemini")

        assert result == []

    def test_no_parts_response(self):
        """Test handling of response with no parts."""
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = None

        result = gemini_response_to_messages(mock_response, policy="gemini")

        assert result == []


class TestGoogleGeminiPolicyV1:
    """Test the main google_gemini_policy_v1 function."""

    @pytest.mark.asyncio
    async def test_policy_requires_api_key(self):
        """Test that policy raises error without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                await google_gemini_policy_v1([], [])

    @pytest.mark.asyncio
    async def test_policy_with_mock_api(self):
        """Test policy execution with mocked Gemini API."""
        # Mock the genai module
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('examples.utils.google_image_gen.genai') as mock_genai:
                # Setup mock model
                mock_model = Mock()
                mock_response = Mock()
                mock_response.candidates = [Mock()]
                mock_response.candidates[0].content = Mock()

                mock_part = Mock()
                mock_part.text = "Test response"
                del mock_part.inline_data
                del mock_part.function_call

                mock_response.candidates[0].content.parts = [mock_part]
                mock_model.generate_content = Mock(return_value=mock_response)
                mock_genai.GenerativeModel.return_value = mock_model

                # Create test input
                builder = MessageBuilder(policy="user", role_hint_for_llm="user")
                builder.add_text("Hello")
                observations = [builder.to_message()]

                # Execute
                result = await google_gemini_policy_v1(observations, [])

                # Verify
                assert len(result) == 1
                assert result[0].parts is not None
                assert result[0].parts[0].text == "Test response"

                # Verify API was called correctly
                mock_genai.configure.assert_called_once()
                mock_genai.GenerativeModel.assert_called_once()

    @pytest.mark.asyncio
    async def test_policy_with_custom_model(self):
        """Test policy with custom model parameter."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('examples.utils.google_image_gen.genai') as mock_genai:
                mock_model = Mock()
                mock_response = Mock()
                mock_response.candidates = []
                mock_model.generate_content = Mock(return_value=mock_response)
                mock_genai.GenerativeModel.return_value = mock_model

                builder = MessageBuilder(policy="user", role_hint_for_llm="user")
                builder.add_text("Test")
                observations = [builder.to_message()]

                # Execute with custom model
                await google_gemini_policy_v1(
                    observations,
                    [],
                    model="gemini-3-pro-preview"
                )

                # Verify custom model was used
                mock_genai.GenerativeModel.assert_called_once()
                call_args = mock_genai.GenerativeModel.call_args
                assert call_args[1]['model_name'] == "gemini-3-pro-preview"

    @pytest.mark.asyncio
    async def test_policy_with_tools(self):
        """Test policy with tool/function calling."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            with patch('examples.utils.google_image_gen.genai') as mock_genai:
                mock_model = Mock()
                mock_response = Mock()
                mock_response.candidates = []
                mock_model.generate_content = Mock(return_value=mock_response)
                mock_genai.GenerativeModel.return_value = mock_model

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

                # Execute with tools
                await google_gemini_policy_v1(observations, options)

                # Verify tools were passed
                mock_genai.GenerativeModel.assert_called_once()
                call_args = mock_genai.GenerativeModel.call_args
                assert 'tools' in call_args[1]
                # Tools should be a list (will contain actual Tool objects in real execution)
                assert call_args[1]['tools'] is not None
