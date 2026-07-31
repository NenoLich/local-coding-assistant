"""
Mock tool responses for testing handler integration scenarios.
"""

from dataclasses import dataclass
from typing import Any

from local_coding_assistant.tools.types import ToolExecutionResponse


@dataclass
class MockToolResponse:
    """Factory for creating mock tool responses with specific error characteristics."""

    @staticmethod
    def success_response(
        result: Any = {"files_found": ["file1.py", "file2.py"], "count": 2},
        tool_name: str = "search_files",
        execution_time_ms: int = 150,
    ) -> ToolExecutionResponse:
        """Create a successful tool response."""
        return ToolExecutionResponse(
            success=True,
            result=result,
            tool_name=tool_name,
            tool_args={"pattern": "*.py", "max_results": 10},
            execution_time_ms=execution_time_ms,
            error_message=None,
        )

    @staticmethod
    def network_error_response(
        tool_name: str = "search_files",
        error_message: str = "Connection timeout after 30 seconds",
    ) -> ToolExecutionResponse:
        """Create a network error tool response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"pattern": "*.py", "max_results": 10},
            execution_time_ms=30000,
            error_message=error_message,
        )

    @staticmethod
    def tool_not_found_response(
        tool_name: str = "search_files",
        error_message: str = "Tool 'search_files' is not exposed",
    ) -> ToolExecutionResponse:
        """Create a tool not found error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"pattern": "*.py", "max_results": 10},
            execution_time_ms=10,
            error_message=error_message,
        )

    @staticmethod
    def syntax_error_response(
        tool_name: str = "search_files",
        error_message: str = "Invalid argument: 'pattern' must be a string",
    ) -> ToolExecutionResponse:
        """Create a syntax/validation error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"pattern": "*.py", "max_results": 10},
            execution_time_ms=50,
            error_message=error_message,
        )

    @staticmethod
    def rate_limit_error_response(
        tool_name: str = "search_files",
        error_message: str = "Rate limit exceeded. Try again in 60 seconds",
    ) -> ToolExecutionResponse:
        """Create a rate limit error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"pattern": "*.py", "max_results": 10},
            execution_time_ms=100,
            error_message=error_message,
        )

    @staticmethod
    def permission_error_response(
        tool_name: str = "read_file",
        error_message: str = "Permission denied: cannot access file '/etc/passwd'",
    ) -> ToolExecutionResponse:
        """Create a permission error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"file_path": "example.py"},
            execution_time_ms=25,
            error_message=error_message,
        )

    @staticmethod
    def timeout_error_response(
        tool_name: str = "execute_command",
        error_message: str = "Command execution timed out after 120 seconds",
    ) -> ToolExecutionResponse:
        """Create a timeout error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"command": "ls -la"},
            execution_time_ms=120000,
            error_message=error_message,
        )

    @staticmethod
    def authentication_error_response(
        tool_name: str = "api_call",
        error_message: str = "Authentication failed: invalid API key",
    ) -> ToolExecutionResponse:
        """Create an authentication error response."""
        return ToolExecutionResponse(
            success=False,
            result=None,
            tool_name=tool_name,
            tool_args={"api_key": "invalid"},
            execution_time_ms=500,
            error_message=error_message,
        )


class MockToolCall:
    """Factory for creating mock tool calls."""

    @staticmethod
    def search_files_call(
        pattern: str = "*.py", max_results: int = 10
    ) -> dict[str, Any]:
        """Create a search_files tool call."""
        return {
            "id": "call_search_123",
            "name": "search_files",
            "arguments": {"pattern": pattern, "max_results": max_results},
        }

    @staticmethod
    def read_file_call(file_path: str = "example.py") -> dict[str, Any]:
        """Create a read_file tool call."""
        return {
            "id": "call_read_456",
            "name": "read_file",
            "arguments": {"file_path": file_path},
        }

    @staticmethod
    def execute_command_call(command: str = "ls -la") -> dict[str, Any]:
        """Create an execute_command tool call."""
        return {
            "id": "call_exec_789",
            "name": "execute_command",
            "arguments": {"command": command},
        }
