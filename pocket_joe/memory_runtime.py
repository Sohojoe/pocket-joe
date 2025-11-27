from typing import Any, List, Callable
from pocket_joe.core import Action, Context, Step
from pocket_joe.registry import Registry

class InMemoryContext(Context):
    def __init__(self, runner: 'InMemoryRunner', ledger: list[Step]):
        self.runner = runner
        self.ledger = ledger

    async def call(self, action: Action, decorators: List[Callable] = []) -> Any:
        return await self.runner.execute(action, decorators)
    
    def get_ledger(self) -> list[Step]:
        return self.ledger

class InMemoryRunner:
    def __init__(self, registry: Registry):
        self.registry = registry

    async def execute(self, action: Action, decorators: List[Callable] = []) -> Any:
        policy_metadata = self.registry.get(action.policy)
        if not policy_metadata:
            raise ValueError(f"Unknown policy: {action.policy}")
        
        policy_func = policy_metadata.func
        
        # Apply decorators (Outer wraps Inner)
        # decorators=[loop, invoke] -> loop(invoke(policy))
        wrapped_func = policy_func
        for dec in reversed(decorators):
            wrapped_func = dec(wrapped_func)
            
        ctx = InMemoryContext(self, [])
        return await wrapped_func(action, ctx)
