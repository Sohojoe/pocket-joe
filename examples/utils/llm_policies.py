"""
Reference implementations of LLM policies for PocketJoe.
These are NOT part of the core pocket-joe package.
Users should copy and customize these for their needs.

Requirements: openai, anthropic
"""
import json
from typing import Any
import uuid

from pocket_joe import Action, Context, Step, policy_spec
from openai import AsyncOpenAI 


def ledger_to_llm_messages(ledger: list[Step]) -> list[dict[str, Any]]:
    messages = []
    for step in ledger:
        if step.type == "text":
            messages.append({"role": step.actor, "content": step.payload["text"]})
        elif step.type == "action_result":
            # Tool messages require tool_call_id
            messages.append({
                "role": "tool",
                "tool_call_id": step.id,
                "content": str(step.payload)
            })
        # Note: We ignore "action_call" steps as the LLM implies them via its previous output
    return messages

def actions_to_tools(actions: set[str],) -> list[dict]:
    tools = []
    for name in actions:
        # TODO get the full tool spec from registry
        tools.append({
            "type": "function",
            "function": {
                "name": name
            }
        })
    return tools

def map_response_to_steps(response: Any) -> list[Step]:
    new_steps = []
    msg = response.choices[0].message
    
    if msg.content:
        new_steps.append(Step(
            id=str(uuid.uuid4()),
            actor="assistant",
            type="text",
            payload={"text": msg.content}
        ))
        
    if msg.tool_calls:
        for tc in msg.tool_calls:
            new_steps.append(Step(
                id=str(uuid.uuid4()),
                actor="assistant",
                type="action_call",
                payload={
                    "policy": tc.function.name,
                    "payload": json.loads(tc.function.arguments)
                }
            ))
            
    return new_steps

@policy_spec(
    description="Calls OpenAI GPT-4 with tool support",
    input_schema={"ledger": "List[Step]", "edges": "Set[str]"}
)
async def openai_llm_policy_v1(action: Action, ctx: Context) -> list[Step]:

    # 1. Map Ledger to LLM Messages
    messages = action.payload['messages']

    # 2. Map Allowed Actions to Tools
    tools = actions_to_tools(action.actions)

    # 3. Call LLM TODO: add retry logic etc
    openai = AsyncOpenAI()
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore
        tools=tools  # type: ignore
    )
    
    # 4. Map Response to Steps
    new_steps = map_response_to_steps(response)
            
    return new_steps