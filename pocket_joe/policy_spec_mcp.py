import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, TypeVar, get_type_hints, get_origin, get_args
from .core import Policy

T = TypeVar('T')

PolicyKind = Literal["tool", "resource", "none_mcp"]  # how MCP should see it


@dataclass(frozen=True)
class PolicyMetadata:
    name: str
    description: str
    kind: PolicyKind          # "tool", "resource", or "none_mcp"
    input_schema: dict[str, Any]
    annotations: dict[str, Any]

def get_policy_spec(policy: type[Policy]) -> PolicyMetadata:
    """Get the policy metadata spec for this policy class.
    
    Returns:
        PolicyMetadata if decorated with @policy_spec
        
    Raises:
        ValueError: If policy spec not found
    """
    spec = getattr(policy, '__policy_spec__', None)
    if not spec:
        raise ValueError(f"Policy class '{policy.__name__}' missing policy spec")
    return spec

def _python_type_to_json_schema(t: Any) -> dict[str, Any]:
    origin = get_origin(t)
    args = get_args(t)

    # Union / Optional
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}

    if t is str:
        return {"type": "string"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is bool:
        return {"type": "boolean"}

    if origin in (list, tuple):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    return {"type": "object"}


def _infer_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    # If it's a class, inspect __call__ method
    target = func
    if inspect.isclass(func):
        target = func.__call__
    
    sig = inspect.signature(target)
    try:
        hints = get_type_hints(target)
    except NameError:
        # Fall back to annotations if forward references can't be resolved
        hints = getattr(target, '__annotations__', {})
    
    # Extract parameter descriptions from docstring
    param_descriptions: dict[str, str] = {}
    doc = getattr(target, '__doc__', None) or getattr(func, '__doc__', None)
    if doc:
        import re
        # Parse docstring for :param name: description or Args: section
        param_pattern = r':param\s+(\w+):\s*(.+?)(?=\n|$|:param|:return)'
        matches = re.finditer(param_pattern, doc, re.MULTILINE)
        for match in matches:
            param_descriptions[match.group(1)] = match.group(2).strip()

    properties: dict[str, Any] = {}
    required: list[str] = []

    # Convention: skip 'self' and 'ctx' parameters
    # For classes with __call__: skip 'self'
    all_params = list(sig.parameters.values())
    params = all_params[1:] if inspect.isclass(func) else all_params

    for p in params:
        name = p.name
        ann = hints.get(name, Any)
        schema = _python_type_to_json_schema(ann)
        
        # all parameters must have descriptions for MCP
        if name not in param_descriptions:
            raise KeyError(f"Parameter '{name}' missing in docstring for function '{target.__name__}'")

        # Add description if available
        schema["description"] = param_descriptions[name]
        
        properties[name] = schema

        # required if no default and not Optional/Union[*, None]
        is_optional = False
        origin = get_origin(ann)
        args = get_args(ann)
        if origin is Union and any(a is type(None) for a in args):  # noqa: E721
            is_optional = True

        if p.default is inspect._empty and not is_optional:
            required.append(name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


def policy_spec(
    *,
    mcp_kind: PolicyKind = "none_mcp",
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Callable[[T], T]:
    """
    Base decorator: everything is a policy.
    MCP is just a projection controlled by `mcp_kind`.
    """
    def decorator(func: T) -> T:
        schema = input_schema or _infer_input_schema(func)  # type: ignore
        func_name = getattr(func, '__name__', func.__class__.__name__ if hasattr(func, '__class__') else 'unknown')
        meta = PolicyMetadata(
            name=name or func_name,
            description=description,
            kind=mcp_kind,
            input_schema=schema,
            annotations=annotations or {},
        )
        func.__policy_spec__ = meta  # type: ignore
        return func

    return decorator

def policy_spec_mcp_tool(
    *,
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Callable[[T], T]:
    return policy_spec(
        mcp_kind="tool",
        name=name,
        description=description,
        input_schema=input_schema,
        annotations=annotations,
    )


def policy_spec_mcp_resource(
    *,
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Callable[[T], T]:
    return policy_spec(
        mcp_kind="resource",
        name=name,
        description=description,
        input_schema=input_schema,
        annotations=annotations,
    )


def unpack_params(meta: PolicyMetadata, params_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Extract and validate parameters from params_dict based on input_schema.
    
    Args:
        meta: PolicyMetadata containing the input_schema
        params_dict: Raw parameters (e.g., from step.payload["payload"])
    
    Returns:
        Dict of parameters ready to unpack as **kwargs
    
    Raises:
        ValueError: If required parameters are missing
    """
    schema = meta.input_schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Extract params that match the schema
    unpacked: dict[str, Any] = {}
    for param_name in properties:
        if param_name in params_dict:
            unpacked[param_name] = params_dict[param_name]
        elif param_name in required:
            raise ValueError(
                f"Missing required parameter '{param_name}' for policy '{meta.name}'"
            )
    
    return unpacked

