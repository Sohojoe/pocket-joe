"""Policy decorators that wrap FastMCP's Tool and Resource extractors.

Usage:
    from pocket_joe import policy

    @policy.tool(description="A tool that performs web search")
    async def web_search(query: str) -> list[Message]:
        ...

    @policy.resource(uri="config://settings")
    async def get_settings() -> str:
        ...
"""

from typing import Callable, Any, Literal, TypeAlias
from fastmcp.tools import Tool
from fastmcp.resources import Resource
from mcp.types import Annotations, Icon
from mcp.types import CallToolResult, ContentBlock, Icon, TextContent, ToolAnnotations
from fastmcp.utilities.types import (
    Audio,
    File,
    Image,
    NotSet,
    NotSetT,
    create_function_without_params,
    find_kwarg_by_type,
    get_cached_typeadapter,
    replace_type,
)
ToolResultSerializerType: TypeAlias = Callable[[Any], str]

from pocket_joe.core import OptionSchema


class PolicyDecorators:
    """Namespace for policy decorators that mirror FastMCP's API"""

    @staticmethod
    def tool(
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        icons: list[Icon] | None = None,
        tags: set[str] | None = None,
        annotations: ToolAnnotations | None = None,
        exclude_args: list[str] | None = None,
        output_schema: dict[str, Any] | Literal[False] | NotSetT | None = NotSet,
        serializer: ToolResultSerializerType | None = None,
        meta: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> Callable:
        """Decorator that extracts FastMCP Tool metadata without MCP registration.

        This mirrors @mcp.tool but stores metadata on the function instead of
        registering with an MCP server.

        Args:
            name: Tool name (defaults to function name)
            title: Tool title for display purposes
            description: Tool description
            icons: List of icons for the tool
            tags: Set of tags for categorization
            annotations: Tool-specific annotations
            exclude_args: List of argument names to exclude from schema
            output_schema: Schema for tool output (dict, False, or NotSet)
            serializer: Function to serialize tool results to string
            meta: Additional metadata dictionary
            enabled: Whether the tool is enabled

        Returns:
            Decorated function with _tool_metadata attached
        """
        def decorator(func: Callable) -> Callable:
            # Use FastMCP to extract schema
            tool = Tool.from_function(
                func,
                name=name,
                title=title,
                description=description,
                icons=icons,
                tags=tags,
                annotations=annotations,
                exclude_args=exclude_args,
                output_schema=output_schema,
                serializer=serializer,
                meta=meta,
                enabled=enabled,
                )

            # Store the Tool object - it has all FastMCP metadata
            func._tool_metadata = tool
            func._policy_type = "tool"
            func._option_schema = OptionSchema(
                name = tool.name,
                description = tool.description,
                parameters = tool.parameters
            )

            # Return original function (not wrapped)
            return func

        return decorator

    @staticmethod
    def resource(
        uri: str | Any,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        icons: list[Icon] | None = None,
        mime_type: str | None = None,
        tags: set[str] | None = None,
        enabled: bool | None = None,
        annotations: Annotations | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator that extracts FastMCP Resource metadata without MCP registration.

        This mirrors @mcp.resource but stores metadata on the function instead of
        registering with an MCP server.

        Args:
            uri: Resource URI template
            name: Resource name (defaults to function name)
            title: Resource title for display purposes
            description: Resource description
            icons: List of icons for the resource
            mime_type: MIME type of the resource
            tags: Set of tags for categorization
            enabled: Whether the resource is enabled
            annotations: Resource-specific annotations
            meta: Additional metadata dictionary

        Returns:
            Decorated function with _resource_metadata attached
        """
        def decorator(func: Callable) -> Callable:
            # Use FastMCP to extract resource metadata
            resource = Resource.from_function(
                func,
                uri=uri,
                name=name,
                title=title,
                description=description,
                icons=icons,
                mime_type=mime_type,
                tags=tags,
                enabled=enabled,
                annotations=annotations,
                meta=meta,
            )

            # Store the Resource object - it has all FastMCP metadata
            func._resource_metadata = resource
            func._policy_type = "resource"
            func._option_schema = OptionSchema(
                name = resource.name,
                description = resource.description,
                parameters = {}
            )
            # Return original function (not wrapped)
            return func

        return decorator


# Create singleton instance for import
policy = PolicyDecorators()
