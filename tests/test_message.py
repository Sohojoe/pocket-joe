"""Tests for pocket_joe.message module - MediaPart and iter_parts."""

import base64
import pytest
from pydantic import HttpUrl
from pocket_joe import (
    Message,
    MessageBuilder,
    MediaPart,
    TextPart,
    iter_parts,
)


class TestMediaPart:
    """Test MediaPart with url/path/data_b64 sources."""

    def test_url_source(self):
        """Test MediaPart with URL source."""
        part = MediaPart(
            modality="image",
            url=HttpUrl("https://example.com/image.png"),
            mime="image/png",
        )
        assert part.url is not None
        assert "example.com" in str(part.url)
        assert "image.png" in str(part.url)
        assert part.path is None
        assert part.data_b64 is None

    def test_path_source(self):
        """Test MediaPart with path source."""
        part = MediaPart(
            modality="image",
            path="/path/to/image.png",
        )
        assert part.path == "/path/to/image.png"
        assert part.url is None
        assert part.data_b64 is None

    def test_data_b64_source(self):
        """Test MediaPart with base64 data source."""
        data = b"fake image bytes"
        b64 = base64.b64encode(data).decode("utf-8")
        part = MediaPart(
            modality="image",
            data_b64=b64,
            mime="image/png",
        )
        assert part.data_b64 == b64
        assert part.url is None
        assert part.path is None
        assert part.mime == "image/png"

    def test_get_bytes(self):
        """Test get_bytes decodes data_b64."""
        data = b"hello world"
        b64 = base64.b64encode(data).decode("utf-8")
        part = MediaPart(modality="image", data_b64=b64, mime="image/png")
        
        assert part.get_bytes() == data

    def test_get_bytes_fails_without_data_b64(self):
        """Test get_bytes raises when data_b64 is not set."""
        part = MediaPart(modality="image", url=HttpUrl("https://example.com/img.png"))
        
        with pytest.raises(ValueError, match="requires data_b64"):
            part.get_bytes()

    def test_validation_no_source(self):
        """Test that MediaPart requires at least one source."""
        with pytest.raises(ValueError, match="exactly one source"):
            MediaPart(modality="image")

    def test_validation_multiple_sources(self):
        """Test that MediaPart rejects multiple sources."""
        with pytest.raises(ValueError, match="only one source"):
            MediaPart(
                modality="image",
                url=HttpUrl("https://example.com/img.png"),
                path="/path/to/img.png",
            )

    def test_validation_data_b64_requires_mime(self):
        """Test that data_b64 requires mime to be set."""
        b64 = base64.b64encode(b"data").decode("utf-8")
        with pytest.raises(ValueError, match="mime is required"):
            MediaPart(modality="image", data_b64=b64)

    def test_json_serializable(self):
        """Test that MediaPart with data_b64 is JSON serializable."""
        data = b"\x89PNG\r\n\x1a\n"  # PNG header (binary)
        b64 = base64.b64encode(data).decode("utf-8")
        part = MediaPart(modality="image", data_b64=b64, mime="image/png")
        
        # Should not raise
        json_str = part.model_dump_json()
        assert len(json_str) > 0
        assert b64 in json_str


class TestMessageBuilderMedia:
    """Test MessageBuilder media methods."""

    def test_add_image_url(self):
        """Test add_image with URL."""
        builder = MessageBuilder(policy="test")
        builder.add_image(url="https://example.com/img.png", mime="image/png")
        msg = builder.to_message()

        assert msg.parts is not None
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.modality == "image"
        assert part.url is not None
        assert "example.com" in str(part.url)

    def test_add_image_path(self):
        """Test add_image_path with local file."""
        builder = MessageBuilder(policy="test")
        builder.add_image_path(path="/path/to/image.png")
        msg = builder.to_message()

        assert msg.parts is not None
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.path == "/path/to/image.png"

    def test_add_image_bytes(self):
        """Test add_image_bytes with raw bytes."""
        data = b"fake image data"
        builder = MessageBuilder(policy="test")
        builder.add_image_bytes(data=data, mime="image/png")
        msg = builder.to_message()

        assert msg.parts is not None
        assert len(msg.parts) == 1
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.data_b64 is not None
        assert part.get_bytes() == data
        assert part.mime == "image/png"

    def test_add_image_bytes_with_prompt_hint(self):
        """Test add_image_bytes with prompt_hint."""
        builder = MessageBuilder(policy="test")
        builder.add_image_bytes(
            data=b"data",
            mime="image/png",
            prompt_hint="A generated cat image",
        )
        msg = builder.to_message()

        assert msg.parts is not None
        part = msg.parts[0]
        assert isinstance(part, MediaPart)
        assert part.prompt_hint == "A generated cat image"


class TestIterParts:
    """Test iter_parts helper function."""

    def test_iter_all_parts(self):
        """Test iterating over all parts."""
        builder1 = MessageBuilder(policy="test")
        builder1.add_text("hello")
        builder1.add_image(url="https://example.com/img.png")
        
        builder2 = MessageBuilder(policy="test")
        builder2.add_text("world")

        messages = [builder1.to_message(), builder2.to_message()]
        parts = list(iter_parts(messages))

        assert len(parts) == 3
        assert isinstance(parts[0], TextPart)
        assert isinstance(parts[1], MediaPart)
        assert isinstance(parts[2], TextPart)

    def test_iter_parts_with_type_filter(self):
        """Test filtering by part type."""
        builder = MessageBuilder(policy="test")
        builder.add_text("hello")
        builder.add_image(url="https://example.com/img.png")
        builder.add_text("world")

        messages = [builder.to_message()]

        text_parts = list(iter_parts(messages, TextPart))
        assert len(text_parts) == 2
        assert all(isinstance(p, TextPart) for p in text_parts)

        media_parts = list(iter_parts(messages, MediaPart))
        assert len(media_parts) == 1
        assert isinstance(media_parts[0], MediaPart)

    def test_iter_parts_empty_messages(self):
        """Test iter_parts with empty message list."""
        parts = list(iter_parts([]))
        assert parts == []

    def test_iter_parts_with_payload_messages(self):
        """Test iter_parts skips messages with payload (no parts)."""
        from pocket_joe import OptionCallPayload

        builder = MessageBuilder(policy="test")
        builder.add_text("hello")
        msg_with_parts = builder.to_message()

        msg_with_payload = Message(
            policy="test",
            payload=OptionCallPayload(
                invocation_id="123",
                option_name="test",
                arguments={},
            ),
        )

        messages = [msg_with_parts, msg_with_payload]
        parts = list(iter_parts(messages))

        assert len(parts) == 1
        assert isinstance(parts[0], TextPart)

    def test_iter_parts_find_first_image(self):
        """Test common pattern: find first image with data."""
        builder = MessageBuilder(policy="test")
        builder.add_text("Here's the image:")
        builder.add_image_bytes(data=b"image1", mime="image/png")
        builder.add_image_bytes(data=b"image2", mime="image/png")

        messages = [builder.to_message()]

        # Find first MediaPart with data_b64
        first_image = next(
            (p for p in iter_parts(messages, MediaPart) if p.data_b64),
            None
        )

        assert first_image is not None
        assert first_image.get_bytes() == b"image1"

    def test_iter_parts_any_pattern(self):
        """Test common pattern: check if any images exist."""
        builder = MessageBuilder(policy="test")
        builder.add_text("hello")
        builder.add_image_bytes(data=b"img", mime="image/png")

        messages = [builder.to_message()]

        has_image = any(p.data_b64 for p in iter_parts(messages, MediaPart))
        assert has_image is True

        # Without image
        builder2 = MessageBuilder(policy="test")
        builder2.add_text("no image")
        messages2 = [builder2.to_message()]

        has_image2 = any(p.data_b64 for p in iter_parts(messages2, MediaPart))
        assert has_image2 is False
