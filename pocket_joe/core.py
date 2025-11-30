from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Protocol
from collections.abc import Iterable


@dataclass(frozen=True)
class Message:
    actor: str                 # e.g. "user", "assistant", "get_weather"
    type: str                  # e.g. "text", "action_call", "action_result"
    payload: dict[str, Any]    # JSON-serializable data
    id: str = ""               # Unique identifier (engine-generated)

@dataclass(frozen=True)
class Action:
    policy: str          # which policy is being invoked
    payload: list[Message] = field(default_factory=list)  # arguments / input for this policy
    actions: set[str] = field(default_factory=set)   # policies this policy can call

class Context(Protocol):
    """
    Interface for policies to invoke other policies (actions).
    Handles ledger recording, replay/idempotency, and durability.
    """
    async def call(self, action: Action, decorators: list[Callable] | None = None) -> list[Message]: ...
    def get_ledger(self) -> list[Message]: ...
    def get_registry(self) -> Any: ...
    # def get_config(self, key: str, default: Any = None) -> Any: ...

# A Policy is an async function that takes an Action and a Context,
# and returns a list of Messages (the record of what it did).
Policy = Callable[[Action, Context], Awaitable[list[Message]]]