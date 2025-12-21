"""
OpenAI Image adapter for pocket-joe.
Encodes pocket-joe Messages to OpenAI Images API format, decodes responses back.

Works with gpt-image-1, gpt-image-1-mini, gpt-image-1.5, dall-e-2, dall-e-3.

Requirements: openai
Install with: pip install openai
"""
import base64
import os
from pathlib import Path
from typing import Any, Literal
import uuid

from pocket_joe import Message, MessageBuilder, MediaPart, TextPart, iter_parts

try:
    from openai import AsyncOpenAI
    from openai.types.images_response import ImagesResponse
except ImportError:
    raise ImportError(
        "openai is required for this adapter. "
        "Install with: pip install openai"
    )


class OpenAIImageAdapter:
    """Adapter for OpenAI Images API.
    
    Handles image generation and editing via the dedicated Images API endpoints.
    This is NOT a chat completions adapter - it's for the separate /v1/images/* endpoints.
    
    Usage - Generation:
        client = OpenAIImageAdapter.client()
        adapter = OpenAIImageAdapter()
        
        response = await client.images.generate(
            model="gpt-image-1.5",
            prompt="A cute baby sea otter",
            size="1024x1024",
        )
        
        messages = adapter.decode(response, policy="image_gen")
    
    Usage - Editing:
        adapter = OpenAIImageAdapter()
        image_files = adapter.encode_images(messages)  # Extract images for upload
        
        response = await client.images.edit(
            model="gpt-image-1.5",
            image=image_files,
            prompt="Add a hat to the otter",
        )
        
        messages = adapter.decode(response, policy="image_edit")
    """
    
    def encode_prompt(self, messages: list[Message]) -> str:
        """Extract text prompt from messages.
        
        Args:
            messages: List of Messages to extract prompt from
            
        Returns:
            Combined text from all TextParts
        """
        texts = [p.text for p in iter_parts(messages, TextPart)]
        return "\n".join(texts)
    
    def encode_images(
        self,
        messages: list[Message],
    ) -> list[tuple[str, bytes]]:
        """Extract images from messages for the edit endpoint.
        
        Args:
            messages: List of Messages containing images
            
        Returns:
            List of (filename, bytes) tuples for upload
        """
        images: list[tuple[str, bytes]] = []
        
        for i, part in enumerate(iter_parts(messages, MediaPart)):
            if part.modality != "image":
                continue
            
            # Get image bytes from appropriate source
            if part.data_b64:
                data = base64.b64decode(part.data_b64)
            elif part.path:
                file_path = Path(part.path)
                if not file_path.exists():
                    continue
                data = file_path.read_bytes()
            else:
                # URL source - would need to fetch, skip for now
                continue
            
            # Determine filename/extension
            ext = "png"
            if part.mime:
                mime_to_ext = {
                    "image/png": "png",
                    "image/jpeg": "jpg",
                    "image/webp": "webp",
                }
                ext = mime_to_ext.get(part.mime, "png")
            
            filename = f"image_{i}.{ext}"
            images.append((filename, data))
        
        return images
    
    def decode(
        self,
        response: ImagesResponse,
        policy: str = "openai_image",
    ) -> list[Message]:
        """Decode OpenAI Images API response to pocket-joe Messages.
        
        Args:
            response: ImagesResponse from images.generate or images.edit
            policy: Policy name for the output messages
            
        Returns:
            List of Messages containing generated images (empty if no images)
        """
        if not response.data:
            return []
        
        builder = MessageBuilder(policy=policy, role_hint_for_llm="assistant")
        
        for image_data in response.data:
            if image_data.b64_json:
                # GPT image models always return base64
                data = base64.b64decode(image_data.b64_json)
                builder.add_image_bytes(data=data, mime="image/png")
            elif image_data.url:
                # DALL-E 2/3 can return URLs
                builder.add_image(url=image_data.url, mime="image/png")
        
        return [builder.to_message()]
    
    @staticmethod
    def client(**kwargs) -> AsyncOpenAI:
        """Create an AsyncOpenAI client.
        
        Args:
            **kwargs: Passed to AsyncOpenAI constructor
            
        Returns:
            AsyncOpenAI client instance
        """
        return AsyncOpenAI(**kwargs)
