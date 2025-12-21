"""Tests for OpenAI Image adapter."""

import base64
from unittest.mock import MagicMock
import pytest

from pocket_joe import Message, MessageBuilder, MediaPart, TextPart, iter_parts
from examples.utils.openai_image_adapter import OpenAIImageAdapter


class TestOpenAIImageAdapter:
    """Test OpenAIImageAdapter encoding and decoding."""

    def test_encode_prompt_single_message(self):
        """Test extracting prompt from a single message."""
        builder = MessageBuilder(policy="user")
        builder.add_text("A cute baby sea otter")
        messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        prompt = adapter.encode_prompt(messages)

        assert prompt == "A cute baby sea otter"

    def test_encode_prompt_multiple_messages(self):
        """Test extracting prompt from multiple messages."""
        builder1 = MessageBuilder(policy="user")
        builder1.add_text("First line")
        
        builder2 = MessageBuilder(policy="user")
        builder2.add_text("Second line")
        
        messages = [builder1.to_message(), builder2.to_message()]

        adapter = OpenAIImageAdapter()
        prompt = adapter.encode_prompt(messages)

        assert prompt == "First line\nSecond line"

    def test_encode_prompt_ignores_images(self):
        """Test that encode_prompt ignores media parts."""
        builder = MessageBuilder(policy="user")
        builder.add_text("Generate this")
        builder.add_image_bytes(data=b"fake", mime="image/png")
        builder.add_text("with style")
        messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        prompt = adapter.encode_prompt(messages)

        assert prompt == "Generate this\nwith style"

    def test_encode_images_from_bytes(self):
        """Test extracting images with data_b64 source."""
        image_data = b"fake png data"
        builder = MessageBuilder(policy="user")
        builder.add_image_bytes(data=image_data, mime="image/png")
        messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        images = adapter.encode_images(messages)

        assert len(images) == 1
        filename, data = images[0]
        assert filename == "image_0.png"
        assert data == image_data

    def test_encode_images_mime_to_extension(self):
        """Test that mime types map to correct extensions."""
        adapter = OpenAIImageAdapter()
        
        # Test JPEG
        builder = MessageBuilder(policy="user")
        builder.add_image_bytes(data=b"jpeg data", mime="image/jpeg")
        images = adapter.encode_images([builder.to_message()])
        assert images[0][0] == "image_0.jpg"

        # Test WebP
        builder2 = MessageBuilder(policy="user")
        builder2.add_image_bytes(data=b"webp data", mime="image/webp")
        images2 = adapter.encode_images([builder2.to_message()])
        assert images2[0][0] == "image_0.webp"

    def test_encode_images_multiple(self):
        """Test extracting multiple images."""
        builder = MessageBuilder(policy="user")
        builder.add_image_bytes(data=b"image1", mime="image/png")
        builder.add_image_bytes(data=b"image2", mime="image/png")
        builder.add_image_bytes(data=b"image3", mime="image/png")
        messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        images = adapter.encode_images(messages)

        assert len(images) == 3
        assert images[0][1] == b"image1"
        assert images[1][1] == b"image2"
        assert images[2][1] == b"image3"

    def test_encode_images_skips_url_source(self):
        """Test that URL-based images are skipped (would need fetch)."""
        builder = MessageBuilder(policy="user")
        builder.add_image(url="https://example.com/image.png")
        messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        images = adapter.encode_images(messages)

        assert len(images) == 0

    def test_decode_b64_response(self):
        """Test decoding response with base64 images."""
        # Create mock response
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"generated image").decode()
        mock_image.url = None
        
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        adapter = OpenAIImageAdapter()
        messages = adapter.decode(mock_response, policy="test_gen")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.policy == "test_gen"
        assert msg.role_hint_for_llm == "assistant"
        
        # Check image was decoded
        assert msg.parts is not None
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.get_bytes() == b"generated image"

    def test_decode_url_response(self):
        """Test decoding response with URL (DALL-E style)."""
        mock_image = MagicMock()
        mock_image.b64_json = None
        mock_image.url = "https://example.com/generated.png"
        
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        adapter = OpenAIImageAdapter()
        messages = adapter.decode(mock_response, policy="dalle")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.parts is not None
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.url is not None
        assert "example.com" in str(part.url)

    def test_decode_multiple_images(self):
        """Test decoding response with multiple images."""
        mock_images = []
        for i in range(3):
            mock_img = MagicMock()
            mock_img.b64_json = base64.b64encode(f"image{i}".encode()).decode()
            mock_img.url = None
            mock_images.append(mock_img)
        
        mock_response = MagicMock()
        mock_response.data = mock_images

        adapter = OpenAIImageAdapter()
        messages = adapter.decode(mock_response, policy="multi")

        assert len(messages) == 1
        msg = messages[0]
        assert msg.parts is not None
        assert len(msg.parts) == 3

    def test_decode_empty_response(self):
        """Test decoding response with no images."""
        mock_response = MagicMock()
        mock_response.data = None

        adapter = OpenAIImageAdapter()
        messages = adapter.decode(mock_response, policy="empty")

        # Should return empty list when no images
        assert len(messages) == 0


class TestOpenAIImageAdapterRoundTrip:
    """Test round-trip encoding/decoding patterns."""

    def test_generation_workflow(self):
        """Test typical generation workflow."""
        # User provides prompt
        prompt_builder = MessageBuilder(policy="user")
        prompt_builder.add_text("A sunset over mountains")
        input_messages = [prompt_builder.to_message()]

        adapter = OpenAIImageAdapter()
        prompt = adapter.encode_prompt(input_messages)
        
        assert prompt == "A sunset over mountains"

        # Simulate response
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"sunset image").decode()
        mock_image.url = None
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        output_messages = adapter.decode(mock_response, policy="image_gen")
        
        # Extract the generated image
        image = next(
            (p for p in iter_parts(output_messages, MediaPart) if p.data_b64),
            None
        )
        assert image is not None
        assert image.get_bytes() == b"sunset image"

    def test_edit_workflow(self):
        """Test typical edit workflow - extract images then decode result."""
        # User provides image + edit prompt
        builder = MessageBuilder(policy="user")
        builder.add_text("Add a rainbow")
        builder.add_image_bytes(data=b"original image", mime="image/png")
        input_messages = [builder.to_message()]

        adapter = OpenAIImageAdapter()
        prompt = adapter.encode_prompt(input_messages)
        images = adapter.encode_images(input_messages)
        
        assert prompt == "Add a rainbow"
        assert len(images) == 1
        assert images[0][1] == b"original image"

        # Simulate edited response
        mock_image = MagicMock()
        mock_image.b64_json = base64.b64encode(b"edited with rainbow").decode()
        mock_image.url = None
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        output_messages = adapter.decode(mock_response, policy="image_edit")
        
        edited = next(
            (p for p in iter_parts(output_messages, MediaPart) if p.data_b64),
            None
        )
        assert edited is not None
        assert edited.get_bytes() == b"edited with rainbow"
