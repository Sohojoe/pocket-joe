import asyncio
from typing import Any
from collections.abc import Callable
from .core import Message, BaseContext, OptionCallPayload, OptionResultPayload, OptionResultBuilder


async def _call_options_in_parallel(ctx: BaseContext, messages: list[Message]) -> list[Message]:
    """Execute option_call messages in parallel and return their results.

    Args:
        ctx: The context containing bound policies
        messages: List of messages that may contain option_call messages

    Returns:
        List of option_result messages from executing the option_calls
    """

    async def execute_option(option: Message) -> list[Message]:
        """Execute a single option_call and wrap result in OptionResultPayload."""
        if not isinstance(option.payload, OptionCallPayload):
            raise TypeError(f"Expected OptionCallPayload, got {type(option.payload)}")

        call_payload = option.payload
        policy_name = call_payload.option_name
        args = call_payload.arguments

        if not isinstance(args, dict):
            raise TypeError(
                f"Policy '{policy_name}' arguments must be a dict[str, Any], "
                f"got {type(args).__name__}: {args}"
            )

        func = ctx.get_policy(policy_name)
        result = await func(**args)

        # Wrap result in OptionResultPayload
        wrapped = Message(
            policy=policy_name,
            payload=OptionResultPayload(
                invocation_id=call_payload.invocation_id,
                option_name=policy_name,
                result=result
            )
        )
        return [wrapped]

    # Find all uncompleted option_call messages
    completed_ids = {
        msg.payload.invocation_id for msg in messages
        if msg.payload and isinstance(msg.payload, OptionResultPayload)
    }

    options = [
        msg for msg in messages
        if msg.payload and isinstance(msg.payload, OptionCallPayload)
        and msg.payload.invocation_id not in completed_ids
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

        # Only process options if result is list[Message]
        if isinstance(selected_actions, list) and selected_actions and isinstance(selected_actions[0], Message):
            option_results = await _call_options_in_parallel(ctx, selected_actions)
            return selected_actions + option_results

        # Otherwise return as-is (primitives, dicts, etc.)
        return selected_actions
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