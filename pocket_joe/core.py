from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Protocol, TYPE_CHECKING, ClassVar, TypeVar
from collections.abc import Iterable
from contextvars import ContextVar

T = TypeVar('T', bound='BaseContext')

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
    """Framework base - hide plumbing here.

    Each subclass gets its own ContextVar for isolation.
    """
    _ctx_var: ClassVar[ContextVar['BaseContext']] = None

    def __init_subclass__(cls):
        """Create a separate ContextVar for each subclass"""
        super().__init_subclass__()
        cls._ctx_var = ContextVar(f'{cls.__module__}.{cls.__name__}_context')

    def __init__(self, runner):
        if self._ctx_var is None:
            # For BaseContext itself (if instantiated directly)
            self.__class__._ctx_var = ContextVar(f'{self.__class__.__module__}.{self.__class__.__name__}_context')

        self._runner = runner
        # Set context once during initialization
        self._ctx_var.set(self)

    @classmethod
    def get_ctx(cls: type[T]) -> T:
        """Get the current context from contextvar

        Returns the context instance of the actual subclass type.
        For example, AppContext.get_ctx() returns AppContext instance.
        """
        if cls._ctx_var is None:
            raise RuntimeError(f"{cls.__name__} context not initialized")
        return cls._ctx_var.get()

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