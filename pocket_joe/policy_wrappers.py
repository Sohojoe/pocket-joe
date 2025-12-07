import asyncio
from typing import Any
from collections.abc import Callable
from .core import Message, BaseContext


async def _call_options_in_parallel(ctx: BaseContext, messages: list[Message]) -> list[Message]:
    """Execute action_call messages in parallel and return their results.
    
    Args:
        ctx: The context containing bound policies
        messages: List of messages that may contain action_call messages
        
    Returns:
        List of action_result messages from executing the action_calls
    """

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
        
        func = ctx.get_policy(policy_name)
        selected_actions = await func(**args)
        final_actions: list[Message] = []
        for msg in selected_actions:
            if not isinstance(msg, Message):
                raise TypeError(
                    f"Policy '{policy_name}' must return list[Message], "
                    f"got {type(msg).__name__}: {msg}"
                )
            if option.type == "action_call":
                msg = msg.model_copy(update={
                    "type": "action_result",  # ensure type is action_result
                    "tool_id": option.tool_id  # Propagate tool_id to action_result
                })
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

def invoke_options_wrapper_for_func(func: Callable, ctx: BaseContext):
    """Returns a wrapped callable that executes options in parallel for function-based policies.
    
    Args:
        func: The policy function to wrap
        ctx: The context containing bound policies
        
    Returns:
        Wrapped async function that executes the policy and its options in parallel
    """
    async def wrapped(**kwargs):
        selected_actions = await func(**kwargs)
        option_results = await _call_options_in_parallel(ctx, selected_actions)
        return selected_actions + option_results
    return wrapped

# proposals for additional wrappers:
# # Type alias for clarity
# WrapperFactory = Callable[[Policy, BaseContext], Callable[..., Awaitable[list[Message]]]]

# # Example: tracing_wrapper
# def tracing_wrapper(policy_instance: Policy, ctx: BaseContext):
#     """Wrapper that traces execution."""
#     async def wrapped(**kwargs):
#         span = tracer.start_span(policy_instance.__class__.__name__)
#         try:
#             return await policy_instance(**kwargs)
#         finally:
#             span.end()
#     return wrapped

# # Example: retry_wrapper
# def retry_wrapper_factory(max_retries: int = 3):
#     """Returns a wrapper configured with max_retries."""
#     def wrapper(policy_instance: Policy, ctx: BaseContext):
#         async def wrapped(**kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return await policy_instance(**kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         raise
#                     await asyncio.sleep(2 ** attempt)
#         return wrapped
#     return wrapper