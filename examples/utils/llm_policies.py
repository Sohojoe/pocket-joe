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

from pocket_joe import Message, policy
from pocket_joe import OptionSchema
from openai import AsyncOpenAI

from pocket_joe import BaseContext


def observations_to_completions_messages(in_msgs: list[Message]) -> list[dict[str, Any]]:
    """Convert pocket-joe Message list to chat completions API message format.
    
    Args:
        in_msgs: List of Messages from the conversation history
        
    Returns:
        List of dicts in chat completions format with roles and content
    """

    
    tool_results = dict[str, Message]()
    for msg in in_msgs:
        if msg.type == "action_result":
            if not msg.tool_id:
                raise ValueError(f"action_result message missing tool_id: {msg}")
            tool_results[msg.tool_id] = msg
    
    messages = []
    for msg in in_msgs:
        if msg.type == "text":
            messages.append({"role": msg.actor, "content": msg.payload["content"]})
        elif msg.type == "action_call":
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "id": msg.tool_id,
                    "function": {
                        "name": msg.payload["policy"],
                        "arguments": json.dumps(msg.payload["payload"])
                    }
                }],
            })
            if not msg.tool_id:
                raise ValueError(f"action_result message missing tool_id: {msg}")
            result = tool_results[msg.tool_id]
            messages.append({
                "role": "tool",
                "tool_call_id": msg.tool_id,
                "content": json.dumps(result.payload)
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

def completions_response_to_messages(response: Any) -> list[Message]:
    """Convert chat completions API response to pocket-joe Messages.
    
    Args:
        response: ChatCompletion response object from OpenAI API
        
    Returns:
        List of Messages containing text responses and/or action_call messages
    """
    new_messages = []
    msg = response.choices[0].message
    
    if msg.content:
        new_messages.append(Message(
            id=str(uuid.uuid4()),
            actor="assistant",
            type="text",
            payload={"content": msg.content}
        ))
        
    if msg.tool_calls:
        for tc in msg.tool_calls:
            new_messages.append(Message(
                id=str(uuid.uuid4()),
                actor="assistant",
                type="action_call",
                tool_id=tc.id,
                payload={
                    "policy": tc.function.name,
                    "payload": json.loads(tc.function.arguments)
                }
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