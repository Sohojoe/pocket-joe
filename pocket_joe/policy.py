"""Policy decorators that wrap FastMCP's Tool and Resource extractors.

Usage:
    from pocket_joe import policy

    @policy.tool(description="Performs web search")
    async def web_search(query: str) -> list[Message]:
        ...

    @policy.resource(uri="config://settings")
    async def get_settings() -> str:
        ...
"""

from typing import Callable, Any
from fastmcp.tools import Tool
from fastmcp.resources import Resource


class PolicyDecorators:
    """Namespace for policy decorators that mirror FastMCP's API"""

    @staticmethod
    def tool(
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any
    ) -> Callable:
        """Decorator that extracts FastMCP Tool metadata without MCP registration.

        This mirrors @mcp.tool but stores metadata on the function instead of
        registering with an MCP server.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description
            **kwargs: Additional arguments passed to Tool.from_function()

        Returns:
            Decorated function with _tool_metadata attached
        """
        def decorator(func: Callable) -> Callable:
            # Use FastMCP to extract schema
            tool = Tool.from_function(func, name=name, description=description, **kwargs)

            # Store the Tool object - it has all FastMCP metadata
            func._tool_metadata = tool
            func._policy_type = "tool"

            # Return original function (not wrapped)
            return func

        return decorator

    @staticmethod
    def resource(
        uri: str | None = None,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        **kwargs: Any
    ) -> Callable:
        """Decorator that extracts FastMCP Resource metadata without MCP registration.

        This mirrors @mcp.resource but stores metadata on the function instead of
        registering with an MCP server.

        Args:
            uri: Resource URI template
            name: Resource name (defaults to function name)
            description: Resource description
            mime_type: MIME type of the resource
            **kwargs: Additional arguments passed to Resource.from_function()

        Returns:
            Decorated function with _resource_metadata attached
        """
        def decorator(func: Callable) -> Callable:
            # Use FastMCP to extract resource metadata
            resource = Resource.from_function(
                func,
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                **kwargs
            )

            # Store the Resource object - it has all FastMCP metadata
            func._resource_metadata = resource
            func._policy_type = "resource"

            # Return original function (not wrapped)
            return func

        return decorator


# Create singleton instance for import
policy = PolicyDecorators()

__all__ = ['policy', 'PolicyDecorators']
