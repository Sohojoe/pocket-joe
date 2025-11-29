import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_type_hints, get_origin, get_args

PolicyKind = Literal["tool", "resource", "none_mcp"]  # how MCP should see it


@dataclass(frozen=True)
class PolicyMetadata:
    name: str
    description: str
    kind: PolicyKind          # "tool", "resource", or "none_mcp"
    input_schema: dict[str, Any]
    annotations: dict[str, Any]


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
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    # Extract parameter descriptions from docstring
    param_descriptions: dict[str, str] = {}
    if func.__doc__:
        import re
        # Parse docstring for :param name: description or Args: section
        param_pattern = r':param\s+(\w+):\s*(.+?)(?=\n|$|:param|:return)'
        matches = re.finditer(param_pattern, func.__doc__, re.MULTILINE)
        for match in matches:
            param_descriptions[match.group(1)] = match.group(2).strip()

    properties: dict[str, Any] = {}
    required: list[str] = []

    # Convention: first two params are (action, ctx); rest are "inputs"
    params = list(sig.parameters.values())[2:]

    for p in params:
        name = p.name
        ann = hints.get(name, Any)
        schema = _python_type_to_json_schema(ann)
        
        # Add description if available
        if name in param_descriptions:
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Base decorator: everything is a policy.
    MCP is just a projection controlled by `mcp_kind`.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        schema = input_schema or _infer_input_schema(func)
        meta = PolicyMetadata(
            name=name or func.__name__,
            description=description,
            kind=mcp_kind,
            input_schema=schema,
            annotations=annotations or {},
        )
        func.__policy__ = meta
        return func

    return decorator

def policy_spec_mcp_tool(
    *,
    name: str | None = None,
    description: str = "",
    input_schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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

