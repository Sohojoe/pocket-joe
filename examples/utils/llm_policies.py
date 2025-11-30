"""
Reference implementations of LLM policies for PocketJoe.
These are NOT part of the core pocket-joe package.
Users should copy and customize these for their needs.

Requirements: openai, anthropic
"""
import json
from typing import Any
import uuid

from pocket_joe import Action, Context, Message, policy_spec_mcp_tool
from openai import AsyncOpenAI 


def ledger_to_llm_messages(ledger: list[Message]) -> list[dict[str, Any]]:
    messages = []
    for msg in ledger:
        if msg.type == "text":
            messages.append({"role": msg.actor, "content": msg.payload["content"]})
        elif msg.type == "action_result":
            # Tool messages require tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": msg.id,
                "content": str(msg.payload)
            })
        # Note: We ignore "action_call" messages as the LLM implies them via its previous output
    return messages

def actions_to_tools(actions: set[str], ctx: Context) -> list[dict]:
    """Convert policy names to OpenAI tool schemas using registry metadata."""
    registry = ctx.get_registry()
    tools = []
    for name in actions:
        policy = registry.get(name)
        if policy:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": policy.meta.description,
                    "parameters": policy.meta.input_schema
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
                payload={
                    "policy": tc.function.name,
                    "payload": json.loads(tc.function.arguments)
                }
            ))
            
    return new_messages

@policy_spec_mcp_tool(
    description="Calls OpenAI GPT-4 with tool support",
)
async def openai_llm_policy_v1(action: Action, ctx: Context) -> list[Message]:


    # 1. Map Ledger to LLM Messages
    messages = ledger_to_llm_messages(ctx.get_ledger())

    # 2. Map Allowed Actions to Tools
    tools = actions_to_tools(action.actions, ctx)

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