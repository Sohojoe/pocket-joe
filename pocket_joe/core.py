from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Protocol, TYPE_CHECKING
from collections.abc import Iterable

# from pocket_joe.context import BaseContext


@dataclass(frozen=True)
class Message:
    actor: str                 # e.g. "user", "assistant", "get_weather"
    type: str                  # e.g. "text", "action_call", "action_result"
    payload: dict[str, Any]    # JSON-serializable data
    tool_id: str | None = None  # Optional tool identifier
    id: str = ""               # Unique identifier (engine-generated)

# A Policy is an async function that takes an Action and a Context,
# and returns a list of Messages (the record of what it did).
class Policy:
    """Policy base class - implement __call__
    all params are now optional
        observations: list[Message] | None = None, 
        options: list[str] | None = None,
    ctx is injected when binding to AppContext
    """
    ctx: "BaseContext"
    
    async def __call__(
        self
    ) -> list[Message]:
        raise NotImplementedError
    
    def __init__(self, ctx: "BaseContext"):
        self.ctx = ctx


class BaseContext:
    """Framework base - hide plumbing here."""
    def __init__(self, runner):
        self._runner = runner

    def _bind[T: Policy](self, policy: type[T]) -> T:
        """Bind a policy to this context using runner's strategy.
        Returns an instance of the policy type for proper type inference."""
        bound = self._runner._bind_strategy(policy, self)
        # Store reference to original policy class on the bound function
        bound.__policy_class__ = policy  # type: ignore
        return bound  # type: ignore

    def get_policy(self, name: str) -> type[Policy]:
        """Get the raw policy class by the attribute name on the context.
        
        Args:
            name: The attribute name (e.g., 'llm' to get OpenAILLMPolicy_v1)
        
        Returns:
            The policy class
            
        Raises:
            ValueError: If the policy class is not found.
        """
        bound_policy = getattr(self, name)  # Raises AttributeError if not found
        # Get the policy class stored on the bound function
        policy_class = getattr(bound_policy, '__policy_class__', None)
        if not policy_class:
            raise ValueError(f"Policy class not found for bound policy '{name}', check binding.")   
        return policy_class