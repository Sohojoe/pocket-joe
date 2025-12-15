"""Tests for policy decorators (policy.tool and policy.resource)"""

import pytest
from pocket_joe import policy, Message


class TestPolicyToolDecorator:
    """Test @policy.tool decorator"""

    def test_tool_decorator_basic(self):
        """Test basic @policy.tool decoration"""
        @policy.tool(description="Test tool")
        async def test_func(query: str) -> list[Message]:
            return []

        assert hasattr(test_func, '_tool_metadata')
        assert hasattr(test_func, '_policy_type')
        assert test_func._policy_type == "tool"

    def test_tool_metadata_extraction(self):
        """Test that FastMCP metadata is extracted correctly"""
        @policy.tool(description="Performs web search")
        async def web_search(query: str, max_results: int = 5) -> list[Message]:
            """Search the web"""
            return []

        tool = web_search._tool_metadata
        assert tool.name == "web_search"
        assert tool.description == "Performs web search"
        assert "query" in tool.parameters["properties"]
        assert "max_results" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    def test_tool_with_custom_name(self):
        """Test @policy.tool with custom name"""
        @policy.tool(name="custom_name", description="Custom")
        async def original_name(arg: str) -> list[Message]:
            return []

        assert original_name._tool_metadata.name == "custom_name"

    def test_tool_preserves_function(self):
        """Test that decorator doesn't wrap the function"""
        @policy.tool(description="Test")
        async def my_func(x: int) -> list[Message]:
            """My docstring"""
            return [Message(actor="test", type="text", payload={"x": x})]

        # Function should be callable normally
        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring"

    @pytest.mark.asyncio
    async def test_tool_function_callable(self):
        """Test that decorated function can be called normally"""
        @policy.tool(description="Echo tool")
        async def echo(message: str) -> str:
            return message

        result = await echo(message="hello")
        assert result == "hello"


class TestPolicyResourceDecorator:
    """Test @policy.resource decorator"""

    def test_resource_decorator_basic(self):
        """Test basic @policy.resource decoration"""
        @policy.resource(uri="config://test", description="Test resource")
        async def test_resource() -> str:
            return "test"

        assert hasattr(test_resource, '_resource_metadata')
        assert hasattr(test_resource, '_policy_type')
        assert test_resource._policy_type == "resource"

    def test_resource_metadata_extraction(self):
        """Test that FastMCP resource metadata is extracted correctly"""
        @policy.resource(
            uri="config://app/settings",
            description="Application settings",
            mime_type="application/json"
        )
        async def app_settings() -> str:
            return '{"key": "value"}'

        resource = app_settings._resource_metadata
        assert resource.name == "app_settings"
        assert resource.description == "Application settings"

    def test_resource_with_custom_name(self):
        """Test @policy.resource with custom name"""
        @policy.resource(
            uri="data://custom",
            name="custom_name",
            description="Custom"
        )
        async def original_name() -> str:
            return "data"

        assert original_name._resource_metadata.name == "custom_name"

    def test_resource_preserves_function(self):
        """Test that decorator doesn't wrap the function"""
        @policy.resource(uri="test://data", description="Test")
        async def my_resource() -> str:
            """My resource docstring"""
            return "data"

        assert my_resource.__name__ == "my_resource"
        assert my_resource.__doc__ == "My resource docstring"

    @pytest.mark.asyncio
    async def test_resource_function_callable(self):
        """Test that decorated resource function can be called normally"""
        @policy.resource(uri="config://test", description="Test config")
        async def get_config() -> str:
            return '{"setting": "value"}'

        result = await get_config()
        assert result == '{"setting": "value"}'


class TestPolicyDecoratorIntegration:
    """Test integration between policy decorators and the framework"""

    def test_tool_metadata_accessible(self):
        """Test that tool metadata is accessible for binding"""
        @policy.tool(description="Integration test")
        async def integration_tool(param: str) -> list[Message]:
            return []

        # Verify metadata can be accessed as expected by _bind()
        assert hasattr(integration_tool, '_tool_metadata')
        tool = integration_tool._tool_metadata
        assert tool.name == "integration_tool"
        assert tool.description == "Integration test"

    def test_multiple_decorations(self):
        """Test decorating multiple functions"""
        @policy.tool(description="Tool 1")
        async def tool1(x: int) -> list[Message]:
            return []

        @policy.tool(description="Tool 2")
        async def tool2(y: str) -> list[Message]:
            return []

        assert tool1._tool_metadata.name == "tool1"
        assert tool2._tool_metadata.name == "tool2"
        assert tool1._tool_metadata.description == "Tool 1"
        assert tool2._tool_metadata.description == "Tool 2"
