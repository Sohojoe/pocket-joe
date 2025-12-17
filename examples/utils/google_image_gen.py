"""
Google Gemini image generation policy for PocketJoe.
This is NOT part of the core pocket-joe package.
Users should copy and customize this for their needs.

Requirements: google-generativeai
"""
import json
import os
from typing import Any
import uuid

from pocket_joe import Message, policy, OptionSchema
from pocket_joe import (
    TextPart,
    MediaPart,
    OptionCallPayload,
    OptionResultPayload,
    MessageBuilder,
)

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google-generativeai is required for this policy. "
        "Install it with: pip install google-generativeai"
    )


def observations_to_gemini_messages(in_msgs: list[Message]) -> list[dict[str, Any]]:
    """Convert pocket-joe Message list to Gemini API message format.

    Adapts framework messages to Google's Gemini format, properly handling
    the interweaving of text and images within messages.

    Args:
        in_msgs: List of Messages from the conversation history

    Returns:
        List of dicts in Gemini format with roles and parts
    """
    # Build mapping of invocation_id -> option_result
    tool_results = dict[str, Message]()
    for msg in in_msgs:
        if msg.payload and isinstance(msg.payload, OptionResultPayload):
            tool_results[msg.payload.invocation_id] = msg

    messages = []
    for msg in in_msgs:
        # Handle parts messages (text + media) - key for interweaving!
        if msg.parts:
            # Gemini supports interleaved content parts
            parts = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts.append({"text": part.text})
                elif isinstance(part, MediaPart):
                    # For images, we need to handle URLs or inline data
                    # Gemini expects inline_data format for images
                    if part.modality == "image":
                        # Note: In production, you'd fetch the image from the URL
                        # and convert to base64. For now, we'll include URL in text
                        parts.append({
                            "text": f"[Image: {part.url}]"
                        })
                        # TODO: Implement proper image fetching and conversion
                        # parts.append({
                        #     "inline_data": {
                        #         "mime_type": part.mime or "image/jpeg",
                        #         "data": base64_image_data
                        #     }
                        # })

            role = "user" if msg.role_hint_for_llm == "user" else "model"
            messages.append({"role": role, "parts": parts})

        # Handle option_call messages
        elif msg.payload and isinstance(msg.payload, OptionCallPayload):
            call_payload = msg.payload
            invocation_id = call_payload.invocation_id

            # Only include if we have the corresponding result (complete pair)
            if invocation_id not in tool_results:
                continue

            # Gemini function calling format
            messages.append({
                "role": "model",
                "parts": [{
                    "function_call": {
                        "name": call_payload.option_name,
                        "args": call_payload.arguments
                    }
                }]
            })

            result_msg = tool_results[invocation_id]
            result_payload = result_msg.payload
            if isinstance(result_payload, OptionResultPayload):
                # Serialize result
                result = result_payload.result
                if isinstance(result, str):
                    response = {"result": result}
                else:
                    response = result if isinstance(result, dict) else {"result": result}

                messages.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": call_payload.option_name,
                            "response": response
                        }
                    }]
                })

    return messages


def options_to_gemini_tools(options: list[OptionSchema] | None) -> list[Any] | None:
    """Convert OptionSchema list to Gemini function declarations.

    Args:
        options: List of OptionSchema objects containing tool metadata

    Returns:
        List of genai.types.Tool objects for Gemini, or None if no options
    """
    if not options:
        return None

    from google.generativeai.types import FunctionDeclaration, Tool  # type: ignore[attr-defined]

    function_declarations = []
    for option in options:
        # Create FunctionDeclaration for each option
        # Type ignores needed due to incomplete type stubs in google-generativeai
        func_decl = FunctionDeclaration(  # type: ignore[call-arg]
            name=option.name,
            description=option.description,  # type: ignore[arg-type]
            parameters=option.parameters if option.parameters else None
        )
        function_declarations.append(func_decl)

    # Wrap in a Tool object
    return [Tool(function_declarations=function_declarations)]  # type: ignore[call-arg]


def gemini_response_to_messages(
    response: Any,
    policy: str = "google_gemini"
) -> list[Message]:
    """Convert Gemini API response to pocket-joe Messages.

    Handles both text/image content and function calls, properly interweaving
    them using MessageBuilder.

    Args:
        response: GenerateContentResponse from Gemini API
        policy: Policy name for the messages

    Returns:
        List of Messages containing text/images and/or option_call messages
    """
    new_messages = []

    if not response.candidates:
        return new_messages

    candidate = response.candidates[0]

    if not candidate.content or not candidate.content.parts:
        return new_messages

    # Track if we have text/media parts vs function calls
    has_content_parts = False
    builder = MessageBuilder(policy=policy, role_hint_for_llm="assistant")

    for part in candidate.content.parts:
        # Handle text parts
        if hasattr(part, 'text') and part.text:
            builder.add_text(part.text)
            has_content_parts = True

        # Handle inline images in response (if Gemini ever returns them)
        elif hasattr(part, 'inline_data') and part.inline_data:
            # Convert inline data to data URL
            mime_type = part.inline_data.mime_type
            data = part.inline_data.data
            data_url = f"data:{mime_type};base64,{data}"
            builder.add_image(url=data_url, mime=mime_type)
            has_content_parts = True

        # Handle function calls
        elif hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            new_messages.append(Message(
                id=str(uuid.uuid4()),
                policy=policy,
                role_hint_for_llm="assistant",
                payload=OptionCallPayload(
                    invocation_id=str(uuid.uuid4()),
                    option_name=fc.name,
                    arguments=dict(fc.args)
                )
            ))

    # Add the content message if we accumulated any parts
    if has_content_parts:
        new_messages.insert(0, builder.to_message())

    return new_messages


@policy.tool(description="Calls Google Gemini with multimodal support (text + images)")
async def google_gemini_policy_v1(
    observations: list[Message],
    options: list[OptionSchema],
    model: str = "gemini-2.5-flash-image"
) -> list[Message]:
    """Gemini policy with support for interweaving text and images.

    This policy properly handles messages that contain both text and images,
    sending them in the correct format to Gemini and converting responses
    back to pocket-joe Messages.

    Args:
        observations: List of Messages representing conversation history + new input
        options: Set of allowed options the LLM can call
        model: Gemini model to use. Options:
            - "gemini-2.5-flash-image" (default, optimized for images)
            - "gemini-2.5-flash" (best price-performance)
            - "gemini-2.5-flash-lite" (ultra fast)
            - "gemini-2.5-pro" (advanced reasoning)
            - "gemini-3-pro-preview" (most intelligent)
            - "gemini-2.0-flash-exp" (experimental)

    Returns:
        List of Messages containing text/image responses and/or option_call messages
    """
    # Initialize Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set")

    genai.configure(api_key=api_key)  # type: ignore[attr-defined]

    # Convert messages and tools
    messages = observations_to_gemini_messages(observations)
    tools = options_to_gemini_tools(options)

    # Create model with or without tools
    model_name = model

    if tools:
        model = genai.GenerativeModel(  # type: ignore[attr-defined]
            model_name=model_name,
            tools=tools
        )
    else:
        model = genai.GenerativeModel(model_name=model_name)  # type: ignore[attr-defined]

    # Generate response
    response = model.generate_content(  # type: ignore[attr-defined]
        messages,
        generation_config={"temperature": 0.7}
    )

    # Convert response back to messages
    new_messages = gemini_response_to_messages(response, policy="google_gemini")

    return new_messages
