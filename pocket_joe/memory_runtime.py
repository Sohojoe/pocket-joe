import asyncio
from typing import Any, List, Callable
from dataclasses import replace
from .core import BaseContext, Policy, Message
# from .core import Action, Context, Message
# from .registry import Registry
# from .policy_spec_mcp import unpack_params

# class InMemoryContext(Context):
#     def __init__(self, runner: 'InMemoryRunner', ledger: list[Message]):
#         self.runner = runner
#         self.ledger = ledger

#     async def call(self, action: Action, decorators: List[Callable] = []) -> Any:
#         return await self.runner.execute(action, decorators)
    
#     def get_ledger(self) -> list[Message]:
#         return self.ledger
    
#     def get_registry(self) -> Registry:
#         return self.runner.registry

# class InMemoryRunner:
#     def __init__(self, registry: Registry):
#         self.registry = registry

#     async def execute(self, action: Action, decorators: List[Callable] = []) -> Any:
#         registered_policy = self.registry.get(action.policy)
#         if not registered_policy:
#             raise ValueError(f"Unknown policy: {action.policy}")
        
#         policy_func = registered_policy.func
#         policy_meta = registered_policy.meta
        
#         # Apply decorators (Outer wraps Inner)
#         # decorators=[loop, invoke] -> loop(invoke(policy))
#         wrapped_func = policy_func
#         for dec in reversed(decorators):
#             wrapped_func = dec(wrapped_func)
            
#         ctx = InMemoryContext(self, action.payload)
        
#         # Unpack parameters from the last message in action.payload if it's an action_call
#         # Convention: action_call messages have payload = {"policy": "...", "payload": {...params}}
#         params: dict[str, Any] = {}
#         if action.payload:
#             last_message = action.payload[-1]
#             if last_message.type == "action_call" and isinstance(last_message.payload, dict):
#                 tool_params = last_message.payload.get("payload", {})
#                 if isinstance(tool_params, dict):
#                     params = unpack_params(policy_meta, tool_params)
        
#         return await wrapped_func(action, ctx, **params)

# class InMemoryRunner:
#     def __init__(self, trace: bool = False):
#         self.trace = trace
    
#     def create_bind_function(self, ctx: BaseContext):
#         if not self.trace:
#             # No tracing: direct call (zero overhead)
#             def _bind(policy):
#                 return policy  # No wrapper!
#             return _bind
        
#         else:
#             # With tracing: minimal wrapper for instrumentation
#             def _bind(policy):
#                 async def bound(observations, options=None, **kwargs):
#                     span = tracer.start_span(policy.__name__)
#                     try:
#                         result = await policy(ctx, observations, options, **kwargs)
#                         return result
#                     finally:
#                         span.end()
#                 return bound
#             return _bind

async def call_options_in_parallel(ctx: BaseContext, messages: List[Message]) -> List[Message]:
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

class InMemoryRunner:
    def _bind_strategy(self, policy: type[Policy], ctx: BaseContext):
        """Bind a policy to the context - simple closure with ctx captured."""
        async def bound(**kwargs):
            instance = policy(ctx)
            selected_actions = await instance(**kwargs)

            # Execute all action_calls in parallel
            option_selected_actions = await call_options_in_parallel(ctx, selected_actions)
            
            return selected_actions + option_selected_actions


        return bound

