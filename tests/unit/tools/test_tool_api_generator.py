"""Unit tests for ToolAPIGenerator class."""

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.tool_api_generator import ToolAPIGenerator
from local_coding_assistant.tools.tool_manager import ToolRuntime
from local_coding_assistant.tools.types import ToolCategory, ToolInfo, ToolSource


# Test tool classes with different parameter configurations
class TestToolWithParameters:
    """Test tool with various parameter types."""

    class Input:
        required_int: int
        optional_str: str = "default"
        with_default: bool = True

    class Output:
        result: str

    def run(
        self,
        required_int: int,
        optional_str: str = "default",
        with_default: bool = True,
    ) -> str:
        return f"{required_int}-{optional_str}-{with_default}"


class TestToolWithNestedParams:
    """Test tool with nested parameter structures."""

    class Input:
        class Nested:
            value: int

        nested: Nested
        items: list[str]

    class Output:
        result: str

    def run(self, nested: dict[str, Any], items: list[str]) -> str:
        return f"{nested['value']}-{'-'.join(items)}"


class TestSyncTool:
    """A test sync tool."""

    class Input:
        value: int

    class Output:
        result: int

    def run(self, value: int) -> int:
        return value * 2


class TestAsyncTool:
    """A test async tool."""

    class Input:
        value: int

    class Output:
        result: int

    async def run(self, value: int) -> int:
        return value * 3


@pytest.fixture
def mock_tool_runtime():
    """Create a mock ToolRuntime instance with customizable parameters."""

    def create_runtime(
        tool_class, is_async=False, supports_streaming=False, parameters: dict = None
    ):
        runtime = MagicMock(spec=ToolRuntime)
        runtime.info = ToolInfo(
            name=tool_class.__name__.lower().replace("tool", ""),
            description=f"{tool_class.__name__} description",
            category=ToolCategory.UTILITY,
            source=ToolSource.BUILTIN,
            tool_class=tool_class,
            parameters=parameters or {},
        )
        # Set the module path as an attribute since it's not part of ToolInfo
        runtime.info._module = "test_tools"
        runtime.run_is_async = is_async
        runtime.supports_streaming = supports_streaming
        runtime.execute.return_value = {"result": "test"}
        return runtime

    return create_runtime


def test_get_parameters_basic():
    """Test getting parameters for a tool with basic parameter types."""
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean", "default": True},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    }

    tool_info = ToolInfo(
        name="test_tool",
        description="Test tool",
        category=ToolCategory.UTILITY,
        source=ToolSource.BUILTIN,
        tool_class=TestToolWithParameters,
        parameters=parameters,
    )

    params = ToolAPIGenerator._get_parameters(tool_info)
    assert "name: str" in params
    assert "age: int" in params
    assert "active: bool = True" in params
    assert "tags: list[str]" in params


def test_get_parameters_with_nested_objects():
    """Test getting parameters with nested object types."""
    parameters = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "timeout": {"type": "number"},
                    "retries": {"type": "integer", "default": 3},
                },
            }
        },
    }

    tool_info = ToolInfo(
        name="test_nested",
        description="Test nested params",
        category=ToolCategory.UTILITY,
        source=ToolSource.BUILTIN,
        tool_class=TestToolWithNestedParams,
        parameters=parameters,
    )

    params = ToolAPIGenerator._get_parameters(tool_info)
    assert "config: dict" in params
    assert "timeout: float" not in params  # Nested properties should be flattened


def test_get_parameters_with_default_values():
    """Test parameter extraction with various default values."""
    parameters = {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "default": True},
            "count": {"type": "integer", "default": 0},
            "name": {"type": "string", "default": "test"},
            "empty": {"type": "string", "default": ""},
            "none_val": {"type": "string", "default": None},
        },
    }

    tool_info = ToolInfo(
        name="test_defaults",
        description="Test default values",
        category=ToolCategory.UTILITY,
        source=ToolSource.BUILTIN,
        tool_class=TestToolWithParameters,
        parameters=parameters,
    )

    params = ToolAPIGenerator._get_parameters(tool_info)
    assert "enabled: bool = True" in params
    assert "count: int = 0" in params
    assert 'name: str = "test"' in params
    assert 'empty: str = ""' in params
    assert "none_val: str = None" in params


def test_get_parameters_no_parameters():
    """Test parameter extraction when no parameters are defined."""
    tool_info = ToolInfo(
        name="no_params",
        description="No parameters",
        category=ToolCategory.UTILITY,
        source=ToolSource.BUILTIN,
        tool_class=TestToolWithParameters,
        parameters={},  # Empty parameters
    )

    params = ToolAPIGenerator._get_parameters(tool_info)
    assert params == []


def test_get_parameters_with_unsupported_types():
    """Test parameter extraction with unsupported JSON schema types."""
    parameters = {
        "type": "object",
        "properties": {
            "binary": {"type": "binary"},  # Not in TYPE_MAPPING
            "date": {"type": "date"},  # Not in TYPE_MAPPING
            "custom": {},  # No type specified
        },
    }

    tool_info = ToolInfo(
        name="test_unsupported",
        description="Test unsupported types",
        category=ToolCategory.UTILITY,
        source=ToolSource.BUILTIN,
        tool_class=TestToolWithParameters,
        parameters=parameters,
    )

    params = ToolAPIGenerator._get_parameters(tool_info)
    assert "binary: Any" in params
    assert "date: Any" in params
    assert "custom: Any" in params


def test_generate_api_basic(mock_tool_runtime, tmp_path):
    """Test basic API generation with a sync tool."""
    # Setup
    tools = {"test_sync": mock_tool_runtime(TestSyncTool, is_async=False)}

    # Execute
    generator = ToolAPIGenerator(output_dir=str(tmp_path))
    output_path = generator.generate(tools)

    # Verify
    assert os.path.exists(output_path)
    assert os.path.isfile(output_path)

    # Check the generated code
    with open(output_path, encoding="utf-8") as f:
        content = f.read()

    assert "from sandbox_tools import TestSyncTool" in content
    assert "def test_sync" in content
    assert "return tool.run(" in content


def test_generate_api_async(mock_tool_runtime, tmp_path):
    """Test API generation with an async tool."""
    # Setup
    tools = {"test_async": mock_tool_runtime(TestAsyncTool, is_async=True)}

    # Execute
    generator = ToolAPIGenerator(output_dir=str(tmp_path))
    output_path = generator.generate(tools)

    # Verify
    with open(output_path, encoding="utf-8") as f:
        content = f.read()

    # For async tools, we don't generate a streaming function by default
    # Only generate streaming function if supports_streaming=True
    assert "async def stream_test_async" not in content
    assert "return loop.run_until_complete(test_async_async())" in content


def test_generate_api_streaming(mock_tool_runtime, tmp_path):
    """Test API generation with a streaming tool."""
    # Setup
    tools = {
        "test_async": mock_tool_runtime(
            TestAsyncTool, is_async=True, supports_streaming=True
        )
    }

    # Execute
    generator = ToolAPIGenerator(output_dir=str(tmp_path))
    output_path = generator.generate(tools)

    # Verify
    with open(output_path) as f:
        content = f.read()

    # Check that streaming function is generated for tools that support it
    assert "async def stream_test_async" in content
    assert "async for chunk in tool.stream():" in content


def test_generate_api_multiple_tools(mock_tool_runtime, tmp_path):
    """Test API generation with multiple tools."""
    # Setup
    tools = {
        "test_sync": mock_tool_runtime(TestSyncTool, is_async=False),
        "test_async": mock_tool_runtime(TestAsyncTool, is_async=True),
    }

    # Execute
    generator = ToolAPIGenerator(output_dir=str(tmp_path))
    output_path = generator.generate(tools)

    # Verify
    with open(output_path) as f:
        content = f.read()

    assert "from sandbox_tools import TestSyncTool" in content
    assert "from sandbox_tools import TestAsyncTool" in content
    assert "def test_sync" in content
    assert "async def test_async" in content


def test_generate_api_error_handling(mock_tool_runtime, tmp_path):
    """Test error handling during API generation."""
    # Test with invalid tool (None)
    with pytest.raises(ToolRegistryError):
        generator = ToolAPIGenerator()
        generator.generate({"test": None}, output_dir=str(tmp_path))


def test_generate_tool_function_with_parameters(mock_tool_runtime, tmp_path):
    """Test that tool functions are generated with correct parameters."""
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 1},
            "enabled": {"type": "boolean", "default": True},
        },
        "required": ["name"],
    }

    tools = {
        "test_tool": mock_tool_runtime(TestToolWithParameters, parameters=parameters)
    }

    generator = ToolAPIGenerator(output_dir=str(tmp_path))
    output_path = generator.generate(tools)

    with open(output_path) as f:
        content = f.read()

    # Check parameter definitions
    assert "name: str" in content
    assert "count: int = 1" in content
    assert "enabled: bool = True" in content

    # Check function signature
    assert (
        "def test_tool(name: str, count: int = 1, enabled: bool = True) -> Any:"
        in content
    )
    assert (
        "async def test_tool_async(name: str, count: int = 1, enabled: bool = True) -> Any:"
        in content
    )
