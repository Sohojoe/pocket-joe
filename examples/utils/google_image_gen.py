"""
Google Gemini policy for PocketJoe using the new google-genai SDK.
This is NOT part of the core pocket-joe package.
Users should copy and customize this for their needs.

Requirements: google-genai (NOT the deprecated google-generativeai)
Install with: pip install google-genai
"""
import base64
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
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required for this policy. "
        "Install it with: pip install google-genai"
    )


def observations_to_gemini_contents(
    in_msgs: list[Message],
) -> list[types.Content]:
    """Convert pocket-joe Message list to Gemini API Content format.

    Adapts framework messages to Google's Gemini format, properly handling
    the interweaving of text and images within messages.

    Args:
        in_msgs: List of Messages from the conversation history

    Returns:
        List of types.Content objects for the new genai SDK
    """
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
                    if part.modality == "image":
                        # For URLs, include as text reference for now
                        # TODO: Fetch and convert to bytes for inline_data
                        parts.append(types.Part.from_text(text=f"[Image: {part.url}]"))

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


def options_to_gemini_tools(
    options: list[OptionSchema] | None,
) -> list[types.Tool] | None:
    """Convert OptionSchema list to Gemini Tool objects.

    Args:
        options: List of OptionSchema objects containing tool metadata

    Returns:
        List of types.Tool objects for Gemini, or None if no options
    """
    if not options:
        return None

    function_declarations: list[types.FunctionDeclaration] = []
    for option in options:
        # Use parameters_json_schema for JSON schema dicts
        func_decl = types.FunctionDeclaration(
            name=option.name,
            description=option.description or "",
            parameters_json_schema=option.parameters if option.parameters else None,
        )
        function_declarations.append(func_decl)

    return [types.Tool(function_declarations=function_declarations)]


def gemini_response_to_messages(
    response: types.GenerateContentResponse,
    policy_name: str = "google_gemini",
) -> list[Message]:
    """Convert Gemini API response to pocket-joe Messages.

    Handles both text/image content and function calls.

    Args:
        response: GenerateContentResponse from Gemini API
        policy_name: Policy name for the messages

    Returns:
        List of Messages containing text/images and/or option_call messages
    """
    new_messages: list[Message] = []

    if not response.candidates:
        return new_messages

    candidate = response.candidates[0]

    if not candidate.content or not candidate.content.parts:
        return new_messages

    # Track if we have text/media parts vs function calls
    has_content_parts = False
    builder = MessageBuilder(policy=policy_name, role_hint_for_llm="assistant")

    for part in candidate.content.parts:
        # Handle text parts
        if part.text:
            builder.add_text(part.text)
            has_content_parts = True

        # Handle inline images in response
        elif part.inline_data and part.inline_data.data:
            mime_type = part.inline_data.mime_type or "image/jpeg"
            # Convert bytes to base64 data URL
            data_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
            data_url = f"data:{mime_type};base64,{data_b64}"
            builder.add_image(url=data_url, mime=mime_type)
            has_content_parts = True

        # Handle function calls
        elif part.function_call and part.function_call.name:
            fc = part.function_call
            new_messages.append(Message(
                id=str(uuid.uuid4()),
                policy=policy_name,
                role_hint_for_llm="assistant",
                payload=OptionCallPayload(
                    invocation_id=str(uuid.uuid4()),
                    option_name=fc.name,  # type: ignore[arg-type]  # Checked above
                    arguments=dict(fc.args) if fc.args else {}
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
    model: str = "gemini-2.5-flash",
) -> list[Message]:
    """Gemini policy with support for interweaving text and images.

    Uses the new google-genai SDK with proper async support.

    Args:
        observations: List of Messages representing conversation history + new input
        options: Set of allowed options the LLM can call
        model: Gemini model to use. Options:
            - "gemini-2.5-flash" (default, best price-performance)
            - "gemini-2.5-flash-lite" (ultra fast)
            - "gemini-2.5-pro" (advanced reasoning)
            - "gemini-2.0-flash" (previous gen)

    Returns:
        List of Messages containing text/image responses and/or option_call messages
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable must be set")

    # Create client
    client = genai.Client(api_key=api_key)

    # Convert messages and tools
    contents = observations_to_gemini_contents(observations)
    tools = options_to_gemini_tools(options)

    # Build config
    if tools:
        config = types.GenerateContentConfig(
            temperature=0.7,
            tools=tools,  # type: ignore[arg-type]  # List variance issue
            # Disable automatic function calling - we handle it ourselves
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
    else:
        config = types.GenerateContentConfig(temperature=0.7)

    # Async call - properly releases event loop while waiting
    # Cast contents to satisfy type checker (runtime accepts list[Content])
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,  # type: ignore[arg-type]
        config=config,
    )

    # Convert response back to messages
    new_messages = gemini_response_to_messages(response, policy_name="google_gemini")

    return new_messages
