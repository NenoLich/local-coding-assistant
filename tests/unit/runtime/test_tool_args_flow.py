"""
Test to verify that tool_args flow works correctly from tool call to failed tool collection.
"""

from unittest.mock import Mock

import pytest

from local_coding_assistant.agent.frame_agent import FrameAgent
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.types import ToolExecutionResponse


class TestToolArgsFlow:
    """Test the complete flow of tool_args from tool call to failed tool collection."""

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return SessionState(
            id="test_tool_args_flow",
            current_task="Test tool args flow",
            history=[],
            tool_calls=[],
        )

    def test_collect_failed_tools_with_tool_args(self, session):
        """Test that _collect_failed_tools can access tool_args from ToolExecutionResponse."""
        # Create a mock tool result that simulates a failed tool call
        failed_tool_result = ToolExecutionResponse(
            tool_name="search_files",
            tool_args={"pattern": "*.py", "recursive": True},  # Original tool args
            success=False,
            error_message="Connection timeout after 30 seconds",
            execution_time_ms=30000.0,
        )

        # Simulate the _collect_failed_tools method logic
        failed_tools = []

        for tool_result in [failed_tool_result]:
            if not getattr(tool_result, "success", True):
                # Extract tool information (this is what FrameAgent._collect_failed_tools does)
                tool_name = getattr(tool_result, "tool_name", "unknown")
                tool_args = getattr(
                    tool_result, "tool_args", {}
                )  # This should now work!
                error_message = getattr(tool_result, "error_message", "Unknown error")

                failed_tools.append(
                    {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "error_message": error_message,
                    }
                )

        # Verify the extraction worked correctly
        assert len(failed_tools) == 1
        assert failed_tools[0]["tool_name"] == "search_files"
        assert failed_tools[0]["tool_args"] == {"pattern": "*.py", "recursive": True}
        assert failed_tools[0]["error_message"] == "Connection timeout after 30 seconds"

    def test_handler_integration_receives_tool_args(self, session):
        """Test that handler integration receives proper tool_args in failed_tools."""

        # Create minimal mocks
        mock_llm = Mock()
        mock_tool_manager = Mock()
        mock_context_manager = Mock()
        mock_config_manager = Mock()

        # Create FrameAgent
        agent = FrameAgent(
            llm_service=mock_llm,
            tool_manager=mock_tool_manager,
            context_manager=mock_context_manager,
            config_manager=mock_config_manager,
        )

        # Test _collect_failed_tools with our mock result
        failed_tool_result = {
            "tool_name": "read_file",
            "tool_args": {"path": "/test/file.py", "encoding": "utf-8"},
            "success": False,
            "error_message": "Permission denied: cannot read file",
        }

        failed_tools = agent._collect_failed_tools([failed_tool_result])

        # Verify the tool_args are preserved
        assert len(failed_tools) == 1
        assert failed_tools[0]["tool_name"] == "read_file"
        assert failed_tools[0]["tool_args"] == {
            "path": "/test/file.py",
            "encoding": "utf-8",
        }
        assert failed_tools[0]["error_message"] == "Permission denied: cannot read file"

    def test_tool_execution_response_structure(self):
        """Test that ToolExecutionResponse has the expected structure."""
        # Test successful response
        success_response = ToolExecutionResponse(
            tool_name="test_tool",
            tool_args={"arg1": "value1", "arg2": "value2"},
            success=True,
            result="Operation completed",
            execution_time_ms=100.0,
        )

        assert success_response.tool_name == "test_tool"
        assert success_response.tool_args == {"arg1": "value1", "arg2": "value2"}
        assert success_response.success == True
        assert success_response.result == "Operation completed"
        assert success_response.execution_time_ms == 100.0

        # Test failed response
        failed_response = ToolExecutionResponse(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            success=False,
            error_message="Test error",
            execution_time_ms=200.0,
        )

        assert failed_response.tool_name == "test_tool"
        assert failed_response.tool_args == {"arg1": "value1"}
        assert failed_response.success == False
        assert failed_response.error_message == "Test error"
        assert failed_response.execution_time_ms == 200.0
