import asyncio
from typing import Callable, Any
from .core import Message, Policy, BaseContext
from dataclasses import replace


async def _call_options_in_parallel(ctx: BaseContext, messages: list[Message]) -> list[Message]:
    async def execute_option(option: Message) -> list[Message]:
        """Execute a single action_call option."""
        payload_dict = option.payload
        policy_name = str(payload_dict.get("policy"))
        
        args = payload_dict.get("payload")
        if not isinstance(args, dict):
            raise TypeError(
                f"Policy '{policy_name}'.payload must be a dict[str, Any], "
                f"got {type(args).__name__}: {args}"
            )
        
        func = getattr(ctx, policy_name)
        selected_actions = await func(**args)
        final_actions: list[Message] = []
        for msg in selected_actions:
            if not isinstance(msg, Message):
                raise TypeError(
                    f"Policy '{policy_name}' must return list[Message], "
                    f"got {type(msg).__name__}: {msg}"
                )
            if option.type == "action_call":
                msg = replace(msg,
                              type = "action_result", # ensure type is action_result
                              tool_id=option.tool_id  # Propagate tool_id to action_result
                              )  
            final_actions.append(msg)

        return final_actions

    # Find all uncompleted action_call messages
    completed_ids = {
        msg.tool_id for msg in messages
        if msg.type == "action_result"
    }

    options = [
        msg for msg in messages 
        if msg.type == "action_call" 
        and msg.tool_id not in completed_ids
    ]
    
    if not options:
        return []

    # Execute all substeps in parallel and wait for completion
    # Exceptions will propagate up the stack
    option_selected_actions = await asyncio.gather(
        *[execute_option(option) for option in options]
    )
    
    # Flatten results
    all_option_selected_actions = []
    for result in option_selected_actions:
        all_option_selected_actions.extend(result)
    return all_option_selected_actions

def invoke_options_wrapper(policy_instance: Policy, ctx: BaseContext):
    """Returns a wrapped callable that executes options in parallel."""
    async def wrapped(**kwargs):
        selected_actions = await policy_instance(**kwargs)
        option_results = await _call_options_in_parallel(ctx, selected_actions)
        return selected_actions + option_results
    return wrapped