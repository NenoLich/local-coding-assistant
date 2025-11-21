"""Tests for the tool registry system."""

import pytest

from local_coding_assistant.tools.tool_registry import register_tool
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolPermission,
    ToolSource,
    ToolTag,
)


@pytest.fixture
def tool_registry():
    """Fixture that provides a fresh tool registry for each test."""
    from local_coding_assistant.tools.tool_registry import _TOOL_REGISTRY

    # Save the original registry
    original_registry = _TOOL_REGISTRY.copy()
    # Clear it for testing
    _TOOL_REGISTRY.clear()

    yield _TOOL_REGISTRY

    # Restore the original registry
    _TOOL_REGISTRY.clear()
    _TOOL_REGISTRY.update(original_registry)


def test_tool_registration(tool_registry):
    """Test basic tool registration."""
    # Define a test tool
    tool_name = "test_tool_" + str(id(tool_registry))  # Unique name for this test

    @register_tool(
        name=tool_name,
        description="A test tool",
        category=ToolCategory.UTILITY,
        tags=[ToolTag.UTILITY],
        permissions=[ToolPermission.SANDBOX],
        is_async=False,
        supports_streaming=False,
    )
    class TestTool:
        """A test tool implementation."""

        class Input:
            pass

        class Output:
            pass

        def run(self, input_data):
            return {}

    # Check that the tool was registered
    assert tool_name in tool_registry

    # Check the registration details
    registration = tool_registry[tool_name]
    assert registration.name == tool_name
    assert registration.description == "A test tool"
    assert registration.category == ToolCategory.UTILITY
    assert ToolTag.UTILITY in registration.tags
    assert ToolPermission.SANDBOX in registration.permissions
    assert registration.is_async is False
    assert registration.supports_streaming is False
    assert registration.source == ToolSource.BUILTIN


def test_auto_naming(tool_registry):
    """Test automatic tool naming from class name."""
    # Use a unique class name to avoid conflicts
    class_name = f"AnotherTestTool_{id(tool_registry)}"

    # Create the class dynamically to ensure a unique name
    TestTool = type(
        class_name,
        (),
        {
            "__doc__": "Another test tool with auto-naming.",
            "Input": type("Input", (), {}),
            "Output": type("Output", (), {}),
            "run": lambda self, input_data: {},
        },
    )

    # Register the tool with auto-naming
    register_tool()(TestTool)

    # The registry should now contain the automatically named tool
    expected_name = f"another_test_tool_{id(tool_registry)}".lower()
    assert expected_name in tool_registry

    # Verify the registration details
    registration = tool_registry[expected_name]
    assert registration.name == expected_name
    assert registration.description == "Another test tool with auto-naming."
    assert registration.category is None  # Not specified
    assert registration.is_async is False
    assert registration.supports_streaming is False


def test_async_detection(tool_registry):
    """Test automatic async detection."""
    import asyncio

    # Use a unique class name to avoid conflicts
    class_name = f"AsyncTestTool_{id(tool_registry)}"

    # Create the class dynamically to ensure a unique name
    async def async_run(self, input_data):
        await asyncio.sleep(0)
        return {}

    AsyncTestTool = type(
        class_name,
        (),
        {
            "__doc__": "An async test tool.",
            "Input": type("Input", (), {}),
            "Output": type("Output", (), {}),
            "run": async_run,
        },
    )

    # Register the tool with auto-naming
    register_tool()(AsyncTestTool)

    # The registry should now contain the automatically named tool
    expected_name = f"async_test_tool_{id(tool_registry)}".lower()
    assert expected_name in tool_registry

    # Verify the registration details
    registration = tool_registry[expected_name]
    assert registration.is_async is True
    assert registration.name == expected_name


def test_validation(tool_registry):
    """Test validation of tool registration."""
    with pytest.raises(ValueError):
        # Create a unique class name to avoid conflicts
        class_name = f"InvalidTool_{id(tool_registry)}"

        # Create the class dynamically
        InvalidTool = type(
            class_name,
            (),
            {
                "Input": type("Input", (), {}),
                "Output": type("Output", (), {}),
                "run": lambda self, input_data: {},
            },
        )

        # Register with invalid category
        register_tool(category="invalid_category")(InvalidTool)


def test_get_tools_by_category(tool_registry):
    """Test getting tools by category."""
    from local_coding_assistant.tools.tool_registry import get_tools_by_category

    # Create a unique tool name to avoid conflicts
    tool_name = f"test_math_tool_{id(tool_registry)}"

    # First, register a math tool
    @register_tool(
        name=tool_name, category=ToolCategory.MATH, description="A test math tool"
    )
    class TestMathTool:
        class Input:
            pass

        class Output:
            pass

        def run(self, input_data):
            return {}

    # Now get tools by category
    math_tools = get_tools_by_category(ToolCategory.MATH)

    # Should find our test math tool
    assert tool_name in math_tools
    assert math_tools[tool_name].category == ToolCategory.MATH


def test_get_tools_with_permission(tool_registry):
    """Test getting tools by permission."""
    from local_coding_assistant.tools.tool_registry import get_tools_with_permission

    # Create a unique tool name to avoid conflicts
    tool_name = f"network_tool_{id(tool_registry)}"

    # Register a tool with network permission
    @register_tool(
        name=tool_name,
        permissions=[ToolPermission.NETWORK],
        description="A tool that needs network access",
    )
    class NetworkTool:
        class Input:
            pass

        class Output:
            pass

        def run(self, input_data):
            return {}

    # Test getting tools with network permission
    network_tools = get_tools_with_permission(ToolPermission.NETWORK)

    # Should find our network tool
    assert tool_name in network_tools
    assert ToolPermission.NETWORK in network_tools[tool_name].permissions
