from dataclasses import dataclass
from typing import Any, Dict

from .core import Policy
from .policy_spec_mcp import PolicyMetadata as SpecPolicyMetadata, PolicyKind


@dataclass(frozen=True)
class RegisteredPolicy:
    """
    What the registry stores for each policy:
    - func: the actual callable
    - meta: the PolicyMetadata attached by @policy_spec / @policy_spec_mcp_*
    """
    func: Policy
    meta: SpecPolicyMetadata


class Registry:
    _instance: "Registry | None" = None

    def __init__(self, *policies: Policy):
        """
        Initialize registry with zero or more @policy_spec decorated functions.
        Each policy is registered using the name from its PolicyMetadata.
        """
        self._policies: Dict[str, RegisteredPolicy] = {}
        for p in policies:
            self.register_policy(p)

    # ---- singleton helper -------------------------------------------------

    @classmethod
    def get_instance(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ---- registration ------------------------------------------------------

    def register_policy(self, func: Policy, alias: str | None = None) -> Policy:
        """
        Register a policy function that has been decorated with @policy_spec,
        @policy_spec_mcp_tool, or @policy_spec_mcp_resource.

        We expect the decorator to have attached a `__policy__` attribute
        containing a SpecPolicyMetadata instance.
        """
        meta: SpecPolicyMetadata | None = getattr(func, "__policy__", None)
        if meta is None:
            raise ValueError(
                f"Policy function {func.__name__} is not decorated with @policy_spec"
            )

        name = alias or meta.name
        self._policies[name] = RegisteredPolicy(func=func, meta=meta)
        return func

    # ---- lookup ------------------------------------------------------------

    def get(self, name: str) -> RegisteredPolicy | None:
        return self._policies.get(name)

    def all(self) -> dict[str, RegisteredPolicy]:
        return dict(self._policies)

    # ---- MCP-specific views -----------------------------------------------

    def mcp_tools(self) -> dict[str, RegisteredPolicy]:
        """
        Policies that should be exposed as MCP tools (kind == 'tool').
        """
        return {
            name: rp
            for name, rp in self._policies.items()
            if rp.meta.kind == "tool"
        }

    def mcp_resources(self) -> dict[str, RegisteredPolicy]:
        """
        Policies that should be exposed as MCP resources (kind == 'resource').
        """
        return {
            name: rp
            for name, rp in self._policies.items()
            if rp.meta.kind == "resource"
        }

    def internal_only(self) -> dict[str, RegisteredPolicy]:
        """
        Policies that are not surfaced to MCP (kind == 'none_mcp').
        """
        return {
            name: rp
            for name, rp in self._policies.items()
            if rp.meta.kind == "none_mcp"
        }

# Convenience global, if you want it:
# _registry = Registry.get_instance()
# register = _registry.register
# get_policy = _registry.get
