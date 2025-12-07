import asyncio
from typing import Any, List, Callable

from .core import BaseContext, Message

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



class InMemoryRunner:
    def _bind_strategy(self, policy: Callable, ctx: BaseContext):
        """Bind a policy function to the context.
        
        Args:
            policy: The policy function to bind
            ctx: The context instance
            
        Returns:
            Async function that wraps the policy with options execution
        """
        from .policy_wrappers import invoke_options_wrapper_for_func
        
        async def bound(**kwargs):
            wrapped = invoke_options_wrapper_for_func(policy, ctx)
            selected_actions = await wrapped(**kwargs)
            return selected_actions

        return bound

