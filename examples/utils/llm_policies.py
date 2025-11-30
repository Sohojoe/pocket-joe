"""
Reference implementations of LLM policies for PocketJoe.
These are NOT part of the core pocket-joe package.
Users should copy and customize these for their needs.

Requirements: openai, anthropic
"""
import json
from typing import Any
import uuid

from pocket_joe import Message, policy_spec_mcp_tool
from openai import AsyncOpenAI

from pocket_joe import BaseContext
from pocket_joe.core import Policy
from pocket_joe.policy_spec_mcp import get_policy_spec 


def ledger_to_llm_messages(in_msgs: list[Message]) -> list[dict[str, Any]]:
    
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

def actions_to_tools(ctx: BaseContext, options: list[str] | None) -> list[dict]:
    """Convert policy names to OpenAI tool schemas using registry metadata."""
    tools = []
    if not options:
        return tools
    for name in options:
        policy_class = ctx.get_policy(name)
        if not policy_class:
            raise ValueError(f"Policy '{name}' not found in context. Check binding.")
        meta = get_policy_spec(policy_class)
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta.description,
                "parameters": meta.input_schema
            }
        })
    return tools

def map_response_to_messagess(response: Any) -> list[Message]:
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

@policy_spec_mcp_tool(
    description="Calls OpenAI GPT-4 with tool support",
)
class OpenAILLMPolicy_v1(Policy):
    async def __call__(
        self,
        observations: list[Message],
        options: list[str],
    ) -> list[Message]:
        """LLM policy that calls OpenAI GPT-4 with tool support.
        :param observations: List of Messages representing the conversation history + new input
        :param options: Set of allowed options the LLM can call (policy names that will map to tools)
        """

        # 1. Map Ledger to LLM Messages
        messages = ledger_to_llm_messages(observations)

        # 2. Map Allowed Actions to Tools
        tools = actions_to_tools(self.ctx, options)

        # 3. Call LLM TODO: add retry logic etc
        openai = AsyncOpenAI()
        response = await openai.chat.completions.create(
            model="gpt-4",
            messages=messages,  # type: ignore
            tools=tools  # type: ignore
        )
        
        # 4. Map Response to Messages
        new_messages = map_response_to_messagess(response)
                
        return new_messages