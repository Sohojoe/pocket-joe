"""
Reference implementations of LLM policies for PocketJoe.
These are NOT part of the core pocket-joe package.
Users should copy and customize these for their needs.

Requirements: openai, anthropic
"""
import json
from typing import Any
from collections.abc import Callable
import uuid

from pocket_joe import Message, policy, OptionSchema, BaseContext
from pocket_joe import (
    TextPart,
    OptionCallPayload,
    OptionResultPayload,
    MessageBuilder,
)
from openai import AsyncOpenAI


def observations_to_completions_messages(in_msgs: list[Message]) -> list[dict[str, Any]]:
    """Convert pocket-joe Message list to chat completions API message format.

    Adapts framework messages to OpenAI's chat completions format.
    Only serializes complete option_call+option_result pairs (same invocation_id).

    Args:
        in_msgs: List of Messages from the conversation history

    Returns:
        List of dicts in chat completions format with roles and content
    """

    # Build mapping of invocation_id -> option_result
    tool_results = dict[str, Message]()
    for msg in in_msgs:
        if msg.payload and isinstance(msg.payload, OptionResultPayload):
            tool_results[msg.payload.invocation_id] = msg

    messages = []
    for msg in in_msgs:
        # Handle parts messages (text + media)
        if msg.parts:
            # Extract text parts
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            content = " ".join(p.text for p in text_parts)

            role = msg.role_hint_for_llm or "assistant"
            messages.append({"role": role, "content": content})

        # Handle option_call messages
        elif msg.payload and isinstance(msg.payload, OptionCallPayload):
            call_payload = msg.payload
            invocation_id = call_payload.invocation_id

            # Only include if we have the corresponding result (complete pair)
            if invocation_id not in tool_results:
                continue  # Skip incomplete calls

            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "id": invocation_id,
                    "function": {
                        "name": call_payload.option_name,
                        "arguments": json.dumps(call_payload.arguments)
                    }
                }],
            })

            result_msg = tool_results[invocation_id]
            result_payload = result_msg.payload
            if isinstance(result_payload, OptionResultPayload):
                # Serialize result - use JSON for complex types, str for primitives
                result = result_payload.result
                if isinstance(result, str):
                    content = result
                else:
                    content = json.dumps(result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": invocation_id,
                    "content": content
                })

    return messages

def options_to_completions_tools(options: list[OptionSchema] | None) -> list[dict]:
    """Convert OptionSchema list to chat completions tool format.
    
    Args:
        options: List of OptionSchema objects containing tool metadata
        
    Returns:
        List of tool dicts in chat completions format: {type: "function", function: {...}}
    """
    tools = []
    if not options:
        return tools
    for option in options:
        # openAI mapping
        tools.append({
            "type": "function",
            "function": option.model_dump()
        })
    return tools

def completions_response_to_messages(response: Any, policy: str = "openai_llm") -> list[Message]:
    """Convert chat completions API response to pocket-joe Messages.

    Args:
        response: ChatCompletion response object from OpenAI API
        policy: Policy name for the messages (defaults to "openai_llm")

    Returns:
        List of Messages containing text responses and/or option_call messages
    """
    new_messages = []
    msg = response.choices[0].message

    if msg.content:
        builder = MessageBuilder(policy=policy, role_hint_for_llm="assistant")
        builder.add_text(msg.content)
        new_messages.append(builder.to_message())

    if msg.tool_calls:
        for tc in msg.tool_calls:
            new_messages.append(Message(
                id=str(uuid.uuid4()),
                policy=policy,
                role_hint_for_llm="assistant",
                payload=OptionCallPayload(
                    invocation_id=tc.id,
                    option_name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                )
            ))

    return new_messages

@policy.tool(description="Calls LLM with tool support")
async def openai_llm_policy_v1(observations: list[Message], options: list[OptionSchema]) -> list[Message]:
    """LLM policy that calls OpenAI GPT-4 with tool support.

    Args:
        observations: List of Messages representing the conversation history + new input
        options: Set of allowed options the LLM can call
    Returns:
        List of Messages containing text responses and/or action_call messages for tools
    """

    messages = observations_to_completions_messages(observations)
    tools = options_to_completions_tools(options)
    openai = AsyncOpenAI()
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore
        tools=tools  # type: ignore
    )
    
    new_messages = completions_response_to_messages(response)
            
    return new_messages