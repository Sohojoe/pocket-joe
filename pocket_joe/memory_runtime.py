from typing import Any, List, Callable
from pocket_joe.core import Action, Context, Message
from pocket_joe.registry import Registry
from pocket_joe.policy_spec_mcp import unpack_params

class InMemoryContext(Context):
    def __init__(self, runner: 'InMemoryRunner', ledger: list[Message]):
        self.runner = runner
        self.ledger = ledger

    async def call(self, action: Action, decorators: List[Callable] = []) -> Any:
        return await self.runner.execute(action, decorators)
    
    def get_ledger(self) -> list[Message]:
        return self.ledger
    
    def get_registry(self) -> Registry:
        return self.runner.registry

class InMemoryRunner:
    def __init__(self, registry: Registry):
        self.registry = registry

    async def execute(self, action: Action, decorators: List[Callable] = []) -> Any:
        registered_policy = self.registry.get(action.policy)
        if not registered_policy:
            raise ValueError(f"Unknown policy: {action.policy}")
        
        policy_func = registered_policy.func
        policy_meta = registered_policy.meta
        
        # Apply decorators (Outer wraps Inner)
        # decorators=[loop, invoke] -> loop(invoke(policy))
        wrapped_func = policy_func
        for dec in reversed(decorators):
            wrapped_func = dec(wrapped_func)
            
        ctx = InMemoryContext(self, action.payload)
        
        # Unpack parameters from the last message in action.payload if it's an action_call
        # Convention: action_call messages have payload = {"policy": "...", "payload": {...params}}
        params: dict[str, Any] = {}
        if action.payload:
            last_message = action.payload[-1]
            if last_message.type == "action_call" and isinstance(last_message.payload, dict):
                tool_params = last_message.payload.get("payload", {})
                if isinstance(tool_params, dict):
                    params = unpack_params(policy_meta, tool_params)
        
        return await wrapped_func(action, ctx, **params)
