from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Protocol, Tuple, List, Union
import uuid

@dataclass(frozen=True)
class Action:
    """
    The fundamental unit of state in PocketJoe.
    Immutable. Contains the payload, execution history, and allowed next moves (edges).
    """
    payload: Any
    edges: Tuple[str, ...] = field(default_factory=tuple) 
    history: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class Context(Protocol):
    """
    The interface for the execution environment.
    Allows Policies to call other Policies (side-effects) in a durable way.
    """
    async def call(self, policy_name: str, action: Action, decorators: List[Callable] = []) -> Any: ...

# A Policy is a pure function that takes an Action and Context, and returns a Result or new Action.
Policy = Callable[[Action, Context], Any]
