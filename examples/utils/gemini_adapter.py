"""
Gemini adapter for pocket-joe.
Encodes pocket-joe Messages to Gemini format, decodes responses back.

Requirements: google-genai
Install with: pip install google-genai
"""
import base64
import os
from pathlib import Path
from typing import Any
import uuid

from pocket_joe import Message, OptionSchema
from pocket_joe import (
    TextPart,
    MediaPart,
    OptionCallPayload,
    OptionResultPayload,
    MessageBuilder,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required for this adapter. "
        "Install with: pip install google-genai"
    )


class GeminiAdapter:
    """Adapter for Google Gemini API.
    
    Encodes pocket-joe Messages to Gemini Content format and decodes
    Gemini responses back to Messages. Does not abstract the API call.
    
    Usage:
        client = GeminiAdapter.client()
        adapter = GeminiAdapter(observations, options)
        
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=adapter.contents,
            config=types.GenerateContentConfig(temperature=0.7, tools=adapter.tools),
        )
        
        messages = adapter.decode(response, policy="my_agent")
    """
    
    def __init__(
        self,
        observations: list[Message],
        options: list[OptionSchema] | None = None,
    ):
        """Initialize adapter with observations and options.
        
        Args:
            observations: List of Messages representing conversation history
            options: Optional list of OptionSchema for tool/function calling
        """
        self.contents = self._encode_contents(observations)
        self.tools = self._encode_tools(options)
    
    def decode(
        self,
        response: types.GenerateContentResponse,
        policy: str = "gemini",
    ) -> list[Message]:
        """Decode Gemini response to pocket-joe Messages.
        
        Args:
            response: GenerateContentResponse from Gemini API
            policy: Policy name for the output messages
            
        Returns:
            List of Messages containing text/images and/or option_call messages
        """
        return self._decode_response(response, policy)
    
    @staticmethod
    def client(api_key: str | None = None) -> genai.Client:
        """Create a Gemini client.
        
        Args:
            api_key: Optional API key, defaults to GOOGLE_API_KEY env var
            
        Returns:
            genai.Client instance
        """
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
        return genai.Client(api_key=key)
    
    # --- Internal encoding/decoding ---
    
    def _encode_contents(self, in_msgs: list[Message]) -> list[types.Content]:
        """Convert pocket-joe Messages to Gemini Content format."""
        # Build mapping of invocation_id -> option_result
        tool_results: dict[str, Message] = {}
        for msg in in_msgs:
            if msg.payload and isinstance(msg.payload, OptionResultPayload):
                tool_results[msg.payload.invocation_id] = msg

        contents: list[types.Content] = []

        for msg in in_msgs:
            # Handle parts messages (text + media)
            if msg.parts:
                parts: list[types.Part] = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        parts.append(types.Part.from_text(text=part.text))
                    elif isinstance(part, MediaPart):
                        gemini_part = self._encode_media_part(part)
                        if gemini_part:
                            parts.append(gemini_part)

                role = "user" if msg.role_hint_for_llm == "user" else "model"
                contents.append(types.Content(role=role, parts=parts))

            # Handle option_call messages
            elif msg.payload and isinstance(msg.payload, OptionCallPayload):
                call_payload = msg.payload
                invocation_id = call_payload.invocation_id

                # Only include if we have the corresponding result (complete pair)
                if invocation_id not in tool_results:
                    continue

                # Function call from model
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(
                        name=call_payload.option_name,
                        args=call_payload.arguments
                    )]
                ))

                # Function response
                result_msg = tool_results[invocation_id]
                result_payload = result_msg.payload
                if isinstance(result_payload, OptionResultPayload):
                    result = result_payload.result
                    if isinstance(result, str):
                        response = {"result": result}
                    else:
                        response = result if isinstance(result, dict) else {"result": result}

                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=call_payload.option_name,
                            response=response
                        )]
                    ))

        return contents
    
    def _encode_tools(
        self,
        options: list[OptionSchema] | None,
    ) -> list[types.Tool] | None:
        """Convert OptionSchema list to Gemini Tool objects."""
        if not options:
            return None

        function_declarations: list[types.FunctionDeclaration] = []
        for option in options:
            func_decl = types.FunctionDeclaration(
                name=option.name,
                description=option.description or "",
                parameters_json_schema=option.parameters if option.parameters else None,
            )
            function_declarations.append(func_decl)

        return [types.Tool(function_declarations=function_declarations)]
    
    def _encode_media_part(self, part: MediaPart) -> types.Part | None:
        """Convert a MediaPart to a Gemini Part.
        
        Handles all three source types: url, path, data_b64.
        """
        if part.modality != "image":
            # For non-image modalities, fall back to placeholder text
            source = part.url or part.path or "[inline data]"
            return types.Part.from_text(text=f"[{part.modality.title()}: {source}]")
        
        # Handle image modality
        if part.data_b64:
            # Inline base64 data
            data = base64.b64decode(part.data_b64)
            mime = part.mime or "image/png"
            return types.Part.from_bytes(data=data, mime_type=mime)
        
        if part.path:
            # Load from local file
            file_path = Path(part.path)
            if not file_path.exists():
                return types.Part.from_text(text=f"[Image not found: {part.path}]")
            data = file_path.read_bytes()
            # Infer mime from extension if not provided
            mime = part.mime
            if not mime:
                suffix = file_path.suffix.lower()
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }
                mime = mime_map.get(suffix, "image/png")
            return types.Part.from_bytes(data=data, mime_type=mime)
        
        if part.url:
            # Remote URL - Gemini accepts URLs directly via from_uri
            url_str = str(part.url)
            mime = part.mime or "image/jpeg"
            return types.Part.from_uri(file_uri=url_str, mime_type=mime)
        
        return None

    def _decode_response(
        self,
        response: types.GenerateContentResponse,
        policy: str,
    ) -> list[Message]:
        """Convert Gemini response to pocket-joe Messages."""
        new_messages: list[Message] = []

        if not response.candidates:
            return new_messages

        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            return new_messages

        has_content_parts = False
        builder = MessageBuilder(policy=policy, role_hint_for_llm="assistant")

        for part in candidate.content.parts:
            # Handle text parts
            if part.text:
                builder.add_text(part.text)
                has_content_parts = True

            # Handle inline images in response
            elif part.inline_data and part.inline_data.data:
                mime_type = part.inline_data.mime_type or "image/png"
                builder.add_image_bytes(data=part.inline_data.data, mime=mime_type)
                has_content_parts = True

            # Handle function calls
            elif part.function_call and part.function_call.name:
                fc = part.function_call
                new_messages.append(Message(
                    id=str(uuid.uuid4()),
                    policy=policy,
                    role_hint_for_llm="assistant",
                    payload=OptionCallPayload(
                        invocation_id=str(uuid.uuid4()),
                        option_name=fc.name,  # type: ignore[arg-type]
                        arguments=dict(fc.args) if fc.args else {}
                    )
                ))

        if has_content_parts:
            new_messages.insert(0, builder.to_message())

        return new_messages
