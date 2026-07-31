"""
Unit tests for ParsingErrorHandler.
"""

from unittest.mock import Mock

import pytest

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
)
from local_coding_assistant.runtime.handlers.parsing_error_handler import (
    ParsingErrorHandler,
)


class TestParsingErrorHandler:
    """Unit tests for ParsingErrorHandler."""

    @pytest.fixture
    def handler(self):
        """Create a ParsingErrorHandler instance."""
        return ParsingErrorHandler()

    @pytest.fixture
    def parsing_error_context(self):
        """Create context with parsing error."""
        return HandlerContext(
            session=Mock(),
            llm_response=Mock(),
            reasoning="Some reasoning content",
            reasoning_tokens=100,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            error_type="parsing_error",
            error_message="Invalid JSON in tool arguments",
            raw_response='{"tool_calls": [{"name": "search_files", "arguments": "invalid json"}]}',
            raw_tool_calls=[{"name": "search_files", "arguments": "invalid json"}],
        )

    @pytest.mark.asyncio
    async def test_handle_parsing_error_retries(self, handler, parsing_error_context):
        """Test that parsing errors generate retry."""
        result = await handler.handle(parsing_error_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path == "handlers/parsing_retry.jinja2"
        assert result.adjusted_llm_options is None
        assert result.error_message is None
        assert result.handler_context is not None

    @pytest.mark.asyncio
    async def test_handler_context_includes_error_details(
        self, handler, parsing_error_context
    ):
        """Test that handler context includes error details for template."""
        result = await handler.handle(parsing_error_context)

        context = result.handler_context
        assert context["error_type"] == "parsing_error"
        assert context["message"] == "Invalid JSON in tool arguments"
        assert (
            context["raw_response"]
            == '{"tool_calls": [{"name": "search_files", "arguments": "invalid json"}]}'
        )
        assert context["reasoning"] == "Some reasoning content"
        assert context["raw_tool_calls"] == [
            {"name": "search_files", "arguments": "invalid json"}
        ]

    @pytest.mark.asyncio
    async def test_handle_with_none_fields(self, handler):
        """Test handling with None fields in context."""
        context = HandlerContext(
            session=Mock(),
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            error_type="parsing_error",
            error_message=None,
            raw_response=None,
            raw_tool_calls=None,
        )

        result = await handler.handle(context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path == "handlers/parsing_retry.jinja2"

        # Context should still be populated with None values
        handler_context = result.handler_context
        assert handler_context["error_type"] == "parsing_error"
        assert handler_context["message"] is None
        assert handler_context["raw_response"] is None
        assert handler_context["reasoning"] is None
        assert handler_context["raw_tool_calls"] is None
