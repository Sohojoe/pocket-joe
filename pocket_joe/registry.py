from dataclasses import dataclass
from typing import Dict, Any, Optional
from pocket_joe.core import Policy

@dataclass
class PolicyMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any]
    func: Policy

class Registry:
    _instance = None
    
    def __init__(self):
        self._policies: Dict[str, PolicyMetadata] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Registry()
        return cls._instance

    def register(self, name: str, description: str = "", input_schema: Dict[str, Any] = {}):
        def decorator(func: Policy):
            self._policies[name] = PolicyMetadata(name, description, input_schema, func)
            return func
        return decorator

    def get(self, name: str) -> Optional[PolicyMetadata]:
        return self._policies.get(name)

# Global instance for convenience
_registry = Registry.get_instance()
register = _registry.register
get_policy = _registry.get
