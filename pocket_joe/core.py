from typing import Any, Callable, Awaitable, Protocol, TYPE_CHECKING, ClassVar, TypeVar
from collections.abc import Iterable
from contextvars import ContextVar
from pydantic import BaseModel, ConfigDict

# Import new message types
from .message import (
    Message,
    TextPart,
    MediaPart,
    Part,
    OptionCallPayload,
    OptionResultPayload,
    Payload,
    MessageBuilder,
    OptionResultBuilder,
)

T = TypeVar('T', bound='BaseContext')
F = TypeVar('F', bound=Callable[..., Awaitable[list['Message']]])


class OptionSchema(BaseModel):
    """Schema for a tool that can be called.
    
    This captures the function signature metadata needed to expose
    tools to LLMs or other agents.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None
    parameters: dict[str, Any]
    
    def __hash__(self) -> int:
        """Hash based on JSON serialization of the model"""
        return hash(self.model_dump_json())
    
    @classmethod
    def from_func(cls, functions: list[Callable]) -> list['OptionSchema']:
        """Extract schema from bound function"""
        options = []
        for func in functions:
            policy_func = getattr(func, '__policy_func__', func)
            option_schema = getattr(policy_func, '_option_schema', None)
            if not option_schema:
                raise ValueError(f"Function missing @policy.tool _option_schema metadata")
            options.append(option_schema)
        return options
    
    @classmethod
    def from_func_single(cls, function: Callable) -> 'OptionSchema':
        options = cls.from_func([function])
        if len(options) != 1:
            raise ValueError(f"Expected single option, got {len(options)}")
        return options[0]

class BaseContext:
    """Framework base - hide plumbing here.

    Each subclass gets its own ContextVar for isolation.
    """
    _ctx_var: ClassVar[ContextVar['BaseContext'] | None] = None

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
        self._ctx_var.set(self)  # type: ignore[union-attr]
        self._option_to_bound_policy: dict[str, Callable] = {}

    @classmethod
    def get_ctx(cls: type[T]) -> T:
        """Get the current context from contextvar

        Returns the context instance of the actual subclass type.
        For example, AppContext.get_ctx() returns AppContext instance.
        """
        if cls._ctx_var is None:
            raise RuntimeError(f"{cls.__name__} context not initialized")
        return cls._ctx_var.get()  # type: ignore[return-value]

    def _bind[F](self, policy: F) -> F:
        """Bind a policy function to this context using runner's strategy.
        
        Args:
            policy: The policy function decorated with @policy.tool or @policy.resource
            
        Returns:
            A bound callable that routes calls through the runner
            
        Raises:
            ValueError: If a policy with the same name is already bound
        """
        bound = self._runner._bind_strategy(policy, self)
        # Store reference to original policy function on the bound function
        bound.__policy_func__ = policy  # type: ignore
        option_schema = OptionSchema.from_func_single(policy) # type: ignore
        # check for duplicates
        if option_schema.name in self._option_to_bound_policy:
            raise ValueError(f"Duplicate policy name '{option_schema.name}' detected during binding.")
        self._option_to_bound_policy[option_schema.name] = bound
        return bound  # type: ignore

    def get_policy(self, policy_name: str) -> Callable:
        """Get the raw policy function by the attribute name on the context.
        
        Args:
            policy_name: The name of the policy to retrieve.
        
        Returns:
            The policy function
            
        Raises:
            ValueError: If the policy function is not found.
        """
        bound_policy = self._option_to_bound_policy.get(policy_name)
        if not bound_policy:
            raise ValueError(f"Bound policy not found for option '{policy_name}'")
        # Get the policy function stored on the bound function
        policy_func = getattr(bound_policy, '__policy_func__', None)
        if not policy_func:
            raise ValueError(f"Policy function not found for bound policy '{policy_name}', check binding.")   
        return policy_func