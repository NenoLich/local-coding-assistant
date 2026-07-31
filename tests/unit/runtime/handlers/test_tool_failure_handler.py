"""
Unit tests for ToolFailureHandler.
"""

from unittest.mock import Mock

import pytest

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
)
from local_coding_assistant.runtime.handlers.tool_failure_handler import (
    ToolFailureHandler,
)


class TestToolFailureHandler:
    """Unit tests for ToolFailureHandler."""

    @pytest.fixture
    def handler(self):
        """Create a ToolFailureHandler instance."""
        return ToolFailureHandler()

    @pytest.fixture
    def retryable_tool_failure_context(self):
        """Create context with retryable tool failures."""
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout after 30 seconds",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            }
        ]

        return HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
        )

    @pytest.fixture
    def non_retryable_tool_failure_context(self):
        """Create context with non-retryable tool failures."""
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": 123},  # Invalid type
                "error_message": "Invalid argument: 'pattern' must be a string",
                "error_type": "SYNTAX_ERROR",
                "retry_strategy": "NO_RETRY",
                "can_retry": False,
            }
        ]

        return HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
        )

    @pytest.fixture
    def mixed_tool_failure_context(self):
        """Create context with both retryable and non-retryable failures."""
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout after 30 seconds",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            },
            {
                "tool_name": "read_file",
                "tool_args": {"file_path": "/etc/passwd"},
                "error_message": "Permission denied: cannot access file '/etc/passwd'",
                "error_type": "PERMISSION",
                "retry_strategy": "NO_RETRY",
                "can_retry": False,
            },
        ]

        return HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
        )

    @pytest.mark.asyncio
    async def test_retryable_tool_failure_continues(
        self, handler, retryable_tool_failure_context
    ):
        """Test that retryable tool failures generate continuation."""
        result = await handler.handle(retryable_tool_failure_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path is not None
        assert result.adjusted_llm_options is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_non_retryable_tool_failure_stops(
        self, handler, non_retryable_tool_failure_context
    ):
        """Test that non-retryable tool failures stop execution."""
        result = await handler.handle(non_retryable_tool_failure_context)

        assert result.status == ExecutionStatus.FAILED
        assert result.should_retry == False
        assert result.template_path is None
        assert result.adjusted_llm_options is None
        assert "search_files" in result.error_message

    @pytest.mark.asyncio
    async def test_mixed_tool_failures_continues_with_retryable(
        self, handler, mixed_tool_failure_context
    ):
        """Test that mixed failures continue if at least one tool can retry."""
        result = await handler.handle(mixed_tool_failure_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path is not None
        assert result.adjusted_llm_options is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded_stops(
        self, handler, retryable_tool_failure_context
    ):
        """Test that execution stops when max attempts exceeded."""
        # Update context to exceed max attempts
        retryable_tool_failure_context.attempt_count = 2
        retryable_tool_failure_context.max_attempts = 2

        result = await handler.handle(retryable_tool_failure_context)

        assert result.status == ExecutionStatus.FAILED
        assert result.should_retry == False
        assert result.template_path is None
        assert result.adjusted_llm_options is None
        assert "Exceeded maximum retry attempts" in result.error_message

    @pytest.mark.asyncio
    async def test_empty_failed_tools_stops(self, handler):
        """Test that empty failed_tools list stops execution."""
        context = HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=[],
        )

        result = await handler.handle(context)

        assert result.status == ExecutionStatus.FAILED
        assert result.should_retry == False
        assert result.template_path is None

    @pytest.mark.asyncio
    async def test_none_failed_tools_stops(self, handler):
        """Test that None failed_tools stops execution."""
        context = HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

        result = await handler.handle(context)

        assert result.status == ExecutionStatus.FAILED
        assert result.should_retry == False
        assert result.template_path is None

    @pytest.mark.asyncio
    async def test_template_path_content(self, handler, retryable_tool_failure_context):
        """Test that template_path is set correctly for retryable failures."""
        result = await handler.handle(retryable_tool_failure_context)

        assert result.template_path == "handlers/tool_retry.jinja2"

    @pytest.mark.asyncio
    async def test_mixed_failures_prompt_only_retryable(
        self, handler, mixed_tool_failure_context
    ):
        """Test that mixed failures with retryable tools set template_path."""
        result = await handler.handle(mixed_tool_failure_context)

        assert result.template_path == "handlers/tool_retry.jinja2"

    @pytest.mark.asyncio
    async def test_multiple_retryable_tools_in_prompt(self, handler):
        """Test that multiple retryable tools set template_path."""
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            },
            {
                "tool_name": "read_file",
                "tool_args": {"file_path": "test.py"},
                "error_message": "File not found",
                "error_type": "FILE_NOT_FOUND",
                "retry_strategy": "RETRY_IMMEDIATELY",
                "can_retry": True,
            },
        ]

        context = HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
        )

        result = await handler.handle(context)

        assert result.template_path == "handlers/tool_retry.jinja2"

    @pytest.mark.asyncio
    async def test_no_adjusted_llm_options(
        self, handler, retryable_tool_failure_context
    ):
        """Test that tool failures don't generate LLM options adjustments."""
        result = await handler.handle(retryable_tool_failure_context)

        assert result.adjusted_llm_options is None

    @pytest.mark.asyncio
    async def test_error_message_formatting(
        self, handler, non_retryable_tool_failure_context
    ):
        """Test that error messages are properly formatted."""
        result = await handler.handle(non_retryable_tool_failure_context)

        assert result.status == ExecutionStatus.FAILED
        assert result.error_message is not None
        assert "search_files" in result.error_message
        assert isinstance(result.error_message, str)
