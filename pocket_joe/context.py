# from .core import 

# class Context(Protocol):
#     """
#     Interface for policies to invoke other policies (actions).
#     Handles ledger recording, replay/idempotency, and durability.
#     """
#     # async def call(self, action: Action, decorators: list[Callable] | None = None) -> list[Message]: ...
#     # def get_ledger(self) -> list[Message]: ...
#     # def get_registry(self) -> Any: ...
#     # def get_config(self, key: str, default: Any = None) -> Any: ...

from .core import Policy


# class BaseContext:
#     """Framework base - hide plumbing here."""
#     def __init__(self, runner):
#         self._runner = runner
#         self._policy_mapping: dict[str, type[Policy]] = {}

#     def _bind(self, policy: type[Policy]):
#         """Bind a policy to this context using runner's strategy."""
#         bound = self._runner._bind_strategy(policy, self)
#         # For classes, use the class name; for functions, use __name__
#         # policy_name = policy.__class__.__name__ if isinstance(policy, Policy) else policy.__name__
#         policy_name = policy.__name__
#         self._policy_mapping[policy_name] = policy
#         return bound

#     def get_policy(self, name: str) -> type[Policy]:
#         """Get the raw policy by its context attribute name."""
#         bound_policy = getattr(self, name)
#         policy = self._policy_mapping[bound_policy.__name__]
#         return policy