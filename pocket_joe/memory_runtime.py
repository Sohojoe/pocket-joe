from typing import Any, List, Callable
from pocket_joe.core import Action, Context
from pocket_joe.registry import Registry

class InMemoryContext:
    def __init__(self, runner: 'InMemoryRunner'):
        self.runner = runner

    async def call(self, policy_name: str, action: Action, decorators: List[Callable] = []) -> Any:
        return await self.runner.execute(policy_name, action, decorators)

class InMemoryRunner:
    def __init__(self, registry: Registry = None):
        self.registry = registry or Registry.get_instance()

    async def execute(self, policy_name: str, action: Action, decorators: List[Callable] = []) -> Any:
        policy_metadata = self.registry.get(policy_name)
        if not policy_metadata:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        policy_func = policy_metadata.func
        
        # Apply decorators (Outer wraps Inner)
        # decorators=[loop, invoke] -> loop(invoke(policy))
        wrapped_func = policy_func
        for dec in reversed(decorators):
            wrapped_func = dec(wrapped_func)
            
        ctx = InMemoryContext(self)
        return await wrapped_func(action, ctx)
