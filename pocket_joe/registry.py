from dataclasses import dataclass
from typing import Any
from pocket_joe.core import Policy

@dataclass
class PolicyMetadata:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Policy

class Registry:
    _instance = None
    
    def __init__(self, *policies: Policy):
        """
        Initialize registry with zero or more @policy_spec decorated functions.
        Each policy is registered using its __policy_name__ attribute.
        
        Usage:
            registry = Registry(policy1, policy2, policy3)
        """
        self._policies: dict[str, PolicyMetadata] = {}
        for func in policies:
            self.register_policy(func)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Registry()
        return cls._instance
    
    def register_policy(self, func: Policy, alias: str | None = None):
        """
        Register a policy function that has @policy_spec metadata.
        Runtime can override the name with an alias.
        
        Usage:
            registry.register_policy(llm_policy)  # uses __policy_name__
            registry.register_policy(llm_policy, alias="llm")  # overrides name
        """
        if not hasattr(func, "__policy_name__"):
            raise ValueError(
                f"Function {func.__name__} is missing @policy_spec metadata. "
                f"All policies must use @policy_spec decorator."
            )
        
        name = alias or func.__policy_name__
        self._policies[name] = PolicyMetadata(
            name=name,
            description=func.__policy_description__,
            input_schema=func.__policy_input_schema__,
            func=func
        )
        return func

    def get(self, name: str) -> PolicyMetadata | None:
        return self._policies.get(name)

# # Global instance for convenience
# _registry = Registry.get_instance()
# register = _registry.register
# get_policy = _registry.get