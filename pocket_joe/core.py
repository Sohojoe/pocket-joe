from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Protocol
from collections.abc import Iterable


def policy_spec(description: str = "", input_schema: dict[str, Any] | None = None):
    """
    Decorator that attaches metadata to a policy function.
    
    Usage:
        @policy_spec(description="Does something")
        async def my_policy(action, ctx):
            ...
    
    The name is inferred from the function name.
    """
    def decorator(func: "Policy"):
        func.__policy_name__ = func.__name__
        func.__policy_description__ = description
        func.__policy_input_schema__ = input_schema or {}
        return func
    return decorator


@dataclass(frozen=True)
class Step:
    id: str                    # Unique identifier (engine-generated)
    actor: str                 # e.g. "user", "assistant", "get_weather"
    type: str                  # e.g. "text", "action_call", "action_result"
    payload: dict[str, Any]    # JSON-serializable data

# @dataclass(frozen=True)
# class Ledger:
#     steps: tuple[Step, ...] = ()

#     def append(self, step: Step) -> "Ledger":
#         """Return a new Ledger with one additional Step."""
#         return Ledger(steps=self.steps + (step,))

#     def extend(self, new_steps: Iterable[Step]) -> "Ledger":
#         """Return a new Ledger with multiple additional Steps."""
#         return Ledger(steps=self.steps + tuple(new_steps))

#     def __iter__(self):
#         return iter(self.steps)

#     def __len__(self):
#         return len(self.steps)

#     def __getitem__(self, idx):
#         return self.steps[idx]

@dataclass(frozen=True)
class Action:
    policy: str          # which policy is being invoked
    payload: dict[str, Any] = field(default_factory=dict)  # arguments / input for this policy
    actions: set[str] = field(default_factory=set)   # policies this policy can call

class Context(Protocol):
    """
    Interface for policies to invoke other policies (actions).
    Handles ledger recording, replay/idempotency, and durability.
    """
    async def call(self, action: Action, decorators: list[Callable] | None = None) -> list[Step]: ...
    def get_ledger(self) -> tuple[Step, ...]: ...
    # def get_config(self, key: str, default: Any = None) -> Any: ...

# A Policy is an async function that takes an Action and a Context,
# and returns a list of Steps (the record of what it did).
Policy = Callable[[Action, Context], Awaitable[list[Step]]]