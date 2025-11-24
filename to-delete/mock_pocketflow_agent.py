from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Optional, Callable, Union
import asyncio
import uuid


# ==========================================
# 1. Core Abstractions (Refined)
# ==========================================

@dataclass
class Action:
    payload: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class Context(Protocol):
    async def call(self, policy_name: str, payload: Any, tools: Optional[List[str]] = None) -> Any: ...

Policy = Callable[[Action, Context], Any]

# ==========================================
# 2. The "MCP-Like" Registry
# ==========================================

@dataclass
class PolicyMetadata:
    name: str
    description: str
    input_schema: Dict[str, Any] # JSON Schema
    func: Policy

REGISTRY: Dict[str, PolicyMetadata] = {}

def register(name: str, description: str, input_schema: Dict[str, Any] = {}):
    def decorator(func: Policy):
        REGISTRY[name] = PolicyMetadata(name, description, input_schema, func)
        return func
    return decorator

# ==========================================
# 3. Common Worker Policies (The Tools)
# ==========================================


@register(
    name="llm_policy",
    description="Calls an LLM with optional tools.",
    input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}}
)
async def llm_policy(action: Action, ctx: Context) -> Any:

    prompt = action.payload.get("prompt")
    tools = action.payload.get("tools", []) # List of policy names

    # wrap with retry logic, backoff, etc. as needed

    # call LLM with prompt and tool schema

    # get results
    return "LLM Result"


