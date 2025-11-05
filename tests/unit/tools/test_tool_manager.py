"""Comprehensive tests for the enhanced ToolManager."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.base import Tool
from local_coding_assistant.tools.builtin import SumTool
from local_coding_assistant.tools.tool_manager import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfo,
    ToolManager,
    ToolRegistration,
)


class MockTool(Tool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"

    class Input(BaseModel):
        value: str

    class Output(BaseModel):
        result: str

    def run(self, payload: Input) -> Output:
        return self.Output(result=f"processed_{payload.value}")


class InvalidTool:
    """Tool without required methods for testing error cases."""

    name = "invalid_tool"


class MockTool2(Tool):
    """Second mock tool for testing."""

    name = "mock_tool2"
    description = "A second mock tool for testing"

    class Input(BaseModel):
        value: str

    class Output(BaseModel):
        result: str

    def run(self, payload: Input) -> Output:
        return self.Output(result=f"processed_{payload.value}")


class TestToolManagerInitialization:
    """Test ToolManager initialization and basic setup."""

    def test_initialization(self):
        """Test that ToolManager initializes correctly."""
        manager = ToolManager()

        assert len(manager) == 0
        assert manager._execution_stats == {}
        assert manager._tools == []
        assert manager._by_name == {}
        assert manager._tool_info == {}

    def test_constructor_logging(self):
        """Test that constructor logs initialization."""
        # Import the module first to ensure it's loaded
        from local_coding_assistant.tools import tool_manager

        with patch.object(tool_manager, "logger") as mock_logger:
            _manager = ToolManager()
            mock_logger.info.assert_called_once_with("ToolManager initialized")


class TestToolRegistration:
    def test_register_tool_class(self):
        """Test registering a tool class (auto-instantiation)."""
        manager = ToolManager()

        manager.register_tool(MockTool)

        assert len(manager) == 1
        assert "mock_tool" in manager._by_name
        assert isinstance(manager.get("mock_tool"), MockTool)

    def test_register_tool_with_category(self):
        """Test registering a tool with a category."""
        manager = ToolManager()
        tool = MockTool()

        manager.register_tool(tool, category="test")

        tool_info = manager.get_tool_info("mock_tool")
        assert tool_info is not None
        assert tool_info.category == "test"
        assert tool_info.name == "mock_tool"
        assert tool_info.class_name == "MockTool"

    def test_register_duplicate_tool_raises_error(self):
        """Test that registering a duplicate tool raises an error."""
        manager = ToolManager()
        tool = MockTool()

        manager.register_tool(tool)

        with pytest.raises(ToolRegistryError, match="already registered"):
            manager.register_tool(tool)

    def test_register_tool_invalid_name_raises_error(self):
        """Test that registering a tool with invalid name raises an error."""
        manager = ToolManager()

        # Create a tool with non-string name
        class BadTool:
            name = 123  # Invalid name type

        with pytest.raises(ToolRegistryError, match="Tool name must be a string"):
            manager.register_tool(BadTool())

    def test_register_tool_missing_run_method_raises_error(self):
        """Test that registering a tool without run method raises an error."""
        manager = ToolManager()

        # Tool that inherits from Tool but doesn't properly implement run method
        class IncompleteTool(Tool):
            name = "incomplete"
            description = "Incomplete tool"

            def run(self, payload):
                # This run method exists but doesn't properly implement the abstract method
                # The validation should catch that it's not a proper override
                raise NotImplementedError("run method not implemented")

        with pytest.raises(
            ToolRegistryError,
            match=r"Tool 'incomplete' has not implemented the required 'run' method",
        ):
            manager.register_tool(IncompleteTool())

    def test_register_tool_instantiation_failure(self):
        """Test handling of tool instantiation failures."""
        manager = ToolManager()

        # Tool class that fails to instantiate
        class FailingTool:
            def __init__(self):
                raise ValueError("Cannot instantiate")

        with pytest.raises(ToolRegistryError, match="Failed to instantiate tool class"):
            manager.register_tool(FailingTool)

    def test_register_builtin_tool(self):
        """Test registering the builtin SumTool."""
        manager = ToolManager()

        manager.register_tool(SumTool())

        assert len(manager) == 1
        assert "sum" in manager._by_name

        tool_info = manager.get_tool_info("sum")
        assert tool_info is not None
        assert tool_info.name == "sum"
        assert tool_info.has_input_validation is True
        assert tool_info.has_output_validation is True


class TestToolExecution:
    """Test tool execution functionality."""

    def test_run_tool_success(self):
        """Test successful tool execution."""
        manager = ToolManager()
        manager.register_tool(MockTool())

        result = manager.run_tool("mock_tool", {"value": "test"})

        assert result == {"result": "processed_test"}

    def test_run_tool_with_validation(self):
        """Test tool execution with input/output validation."""
        manager = ToolManager()
        manager.register_tool(SumTool())

        result = manager.run_tool("sum", {"a": 2, "b": 3})

        assert result == {"sum": 5}

    def test_run_tool_unknown_tool_raises_error(self):
        """Test that running unknown tool raises an error."""
        manager = ToolManager()

        with pytest.raises(ToolRegistryError, match="Unknown tool"):
            manager.run_tool("unknown_tool", {})

    def test_run_tool_invalid_input_raises_error(self):
        """Test that invalid input raises an error."""
        manager = ToolManager()
        manager.register_tool(SumTool())

        with pytest.raises(ToolRegistryError, match="Invalid input"):
            manager.run_tool("sum", {"a": "invalid", "b": 3})

    def test_run_tool_invalid_output_raises_error(self):
        """Test that invalid output raises an error."""
        manager = ToolManager()

        # Create a tool that returns invalid output
        class BadOutputTool(Tool):
            name = "bad_output"

            class Input(BaseModel):
                value: str

            class Output(BaseModel):
                result: int  # Expects int but we'll return string

            def run(self, payload: Input) -> Output:
                # Return a dict that doesn't match the Output schema
                return {"result": "not_an_int"}  # type: ignore

        manager.register_tool(BadOutputTool())

        with pytest.raises(ToolRegistryError, match="Invalid output"):
            manager.run_tool("bad_output", {"value": "test"})

    def test_run_tool_execution_stats_tracking(self):
        """Test that execution statistics are tracked."""
        manager = ToolManager()
        manager.register_tool(MockTool())

        # Run tool multiple times
        manager.run_tool("mock_tool", {"value": "test1"})
        manager.run_tool("mock_tool", {"value": "test2"})
        manager.run_tool("mock_tool", {"value": "test3"})

        stats = manager.get_execution_stats()
        assert stats["mock_tool"] == 3

    def test_run_tool_execution_time_tracking(self):
        """Test that execution time is tracked."""
        manager = ToolManager()
        manager.register_tool(MockTool())

        # Patch the time module at the module level used by ToolManager
        with patch("local_coding_assistant.tools.tool_manager.time") as mock_time:
            # Set up the time mock to return increasing values
            mock_time.time.side_effect = [0.0, 0.001]  # Start and end times

            # Also patch the logger to prevent it from actually logging
            with patch(
                "local_coding_assistant.tools.tool_manager.logger.info"
            ) as mock_logger:
                result = manager.run_tool("mock_tool", {"value": "test"})

                # Verify the result
                assert result == {"result": "processed_test"}

                # Verify the logger was called with the expected message
                mock_logger.assert_called_once()
                log_message = mock_logger.call_args[0][0]
                assert (
                    "Tool 'mock_tool' executed successfully in 1.00 ms" in log_message
                )


class TestToolInformation:
    """Test tool information and listing functionality."""

    def test_get_tool_info(self):
        """Test getting tool information."""
        manager = ToolManager()
        manager.register_tool(SumTool(), category="math")

        info = manager.get_tool_info("sum")

        assert info is not None
        assert info.name == "sum"
        assert info.category == "math"
        assert info.has_input_validation is True
        assert info.has_output_validation is True

    def test_get_tool_info_not_found(self):
        """Test getting info for non-existent tool."""
        manager = ToolManager()

        info = manager.get_tool_info("nonexistent")
        assert info is None

    def test_list_tools_all(self):
        """Test listing all tools."""
        manager = ToolManager()
        manager.register_tool(SumTool(), category="math")
        manager.register_tool(MockTool(), category="test")

        tools = manager.list_tools()

        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"sum", "mock_tool"}

    def test_list_tools_by_category(self):
        """Test listing tools by category."""
        manager = ToolManager()
        manager.register_tool(SumTool(), category="math")
        manager.register_tool(MockTool(), category="test")
        manager.register_tool(MockTool2(), category="test")  # Another in same category

        math_tools = manager.list_tools(category="math")
        test_tools = manager.list_tools(category="test")

        assert len(math_tools) == 1
        assert len(test_tools) == 2
        assert math_tools[0].name == "sum"
        assert all(tool.category == "test" for tool in test_tools)


class TestBackwardCompatibility:
    """Test backward compatibility with ToolRegistry interface."""

    def test_legacy_register_method(self):
        """Test legacy register method."""
        manager = ToolManager()
        tool = MockTool()

        # Should work the same as register_tool
        manager.register(tool)

        assert len(manager) == 1
        assert "mock_tool" in manager._by_name

    def test_legacy_get_method(self):
        """Test legacy get method."""
        manager = ToolManager()
        manager.register_tool(MockTool())

        tool = manager.get("mock_tool")
        assert tool is not None
        assert isinstance(tool, MockTool)

    def test_legacy_invoke_method(self):
        """Test legacy invoke method."""
        manager = ToolManager()
        manager.register_tool(SumTool())

        # Should work the same as run_tool
        result = manager.invoke("sum", {"a": 2, "b": 3})

        assert result == {"sum": 5}

    def test_legacy_list_method(self):
        """Test legacy list method."""
        manager = ToolManager()
        manager.register_tool(SumTool(), category="math")

        tools_list = manager.list_tools()

        assert len(tools_list) == 1
        assert tools_list[0].name == "sum"
        assert tools_list[0].class_name == "SumTool"

    def test_iteration_compatibility(self):
        """Test that iteration still works."""
        manager = ToolManager()
        tools = [MockTool(), SumTool()]
        for tool in tools:
            manager.register_tool(tool)

        iterated_tools = list(manager)
        assert len(iterated_tools) == 2


class TestPydanticModels:
    """Test the pydantic models used in ToolManager."""

    def test_tool_registration_model(self):
        """Test ToolRegistration pydantic model."""
        registration = ToolRegistration(
            name="test_tool",
            tool_class=MockTool,
            description="Test tool",
            category="testing",
        )

        assert registration.name == "test_tool"
        assert registration.tool_class == MockTool
        assert registration.description == "Test tool"
        assert registration.category == "testing"

    def test_tool_execution_request_model(self):
        """Test ToolExecutionRequest pydantic model."""
        request = ToolExecutionRequest(tool_name="test_tool", payload={"key": "value"})

        assert request.tool_name == "test_tool"
        assert request.payload == {"key": "value"}

    def test_tool_execution_response_model(self):
        """Test ToolExecutionResponse pydantic model."""
        response = ToolExecutionResponse(
            tool_name="test_tool",
            success=True,
            result={"output": "value"},
            execution_time_ms=1.5,
        )

        assert response.tool_name == "test_tool"
        assert response.success is True
        assert response.result == {"output": "value"}
        assert response.execution_time_ms == 1.5

    def test_tool_info_model(self):
        """Test ToolInfo pydantic model."""
        info = ToolInfo(
            name="test_tool",
            class_name="TestTool",
            description="A test tool",
            category="testing",
            has_input_validation=True,
            has_output_validation=False,
        )

        assert info.name == "test_tool"
        assert info.class_name == "TestTool"
        assert info.description == "A test tool"
        assert info.category == "testing"
        assert info.has_input_validation is True
        assert info.has_output_validation is False


class TestErrorHandling:
    """Test error handling and logging."""

    def test_registration_error_logging(self):
        """Test that registration errors are properly logged."""
        manager = ToolManager()

        with patch("local_coding_assistant.tools.tool_manager.logger") as mock_logger:
            with pytest.raises(ToolRegistryError):
                manager.register_tool(InvalidTool())

            mock_logger.error.assert_called()

    def test_execution_error_logging(self):
        """Test that execution errors are properly logged."""
        manager = ToolManager()

        with patch("local_coding_assistant.tools.tool_manager.logger") as mock_logger:
            with pytest.raises(
                ToolRegistryError, match="Unknown tool: nonexistent_tool"
            ):
                manager.run_tool("nonexistent_tool", {})

            # Check that debug log was called with the tool execution attempt
            mock_logger.debug.assert_called_once()
            debug_message = mock_logger.debug.call_args[0][0]
            assert "Executing tool 'nonexistent_tool'" in debug_message

    def test_success_logging(self):
        """Test that successful operations are logged."""
        manager = ToolManager()

        with patch("local_coding_assistant.tools.tool_manager.logger") as mock_logger:
            manager.register_tool(MockTool())

            mock_logger.info.assert_called()

    def test_execution_success_logging(self):
        """Test that successful executions are logged."""
        manager = ToolManager()
        manager.register_tool(MockTool())

        with patch("local_coding_assistant.tools.tool_manager.logger") as mock_logger:
            manager.run_tool("mock_tool", {"value": "test"})

            # Should log debug for execution start and info for success
            assert mock_logger.debug.call_count >= 1
            assert mock_logger.info.call_count >= 1


class TestIntegration:
    """Integration tests for ToolManager functionality."""

    def test_full_workflow_builtin_tool(self):
        """Test full workflow with builtin tool."""
        manager = ToolManager()

        # Register tool
        manager.register_tool(SumTool())

        # Get tool info
        info = manager.get_tool_info("sum")
        assert info is not None

        # Execute tool
        result = manager.run_tool("sum", {"a": 5, "b": 10})

        # Verify result
        assert result == {"sum": 15}

        # Check stats
        stats = manager.get_execution_stats()
        assert stats["sum"] == 1

    def test_multiple_tools_different_categories(self):
        """Test managing multiple tools across different categories."""
        manager = ToolManager()

        # Register tools in different categories
        manager.register_tool(SumTool(), category="math")
        manager.register_tool(MockTool(), category="test")

        # List by category
        math_tools = manager.list_tools(category="math")
        test_tools = manager.list_tools(category="test")

        assert len(math_tools) == 1
        assert len(test_tools) == 1
        assert math_tools[0].name == "sum"
        assert test_tools[0].name == "mock_tool"

        # Execute both tools
        sum_result = manager.run_tool("sum", {"a": 1, "b": 2})
        mock_result = manager.run_tool("mock_tool", {"value": "hello"})

        assert sum_result == {"sum": 3}
        assert mock_result == {"result": "processed_hello"}
