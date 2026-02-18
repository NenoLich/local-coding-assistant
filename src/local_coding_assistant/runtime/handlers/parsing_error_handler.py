"""Handler for parsing errors.

This module handles LLM content errors where tool arguments cannot be parsed
or the response contains malformed JSON that prevents proper tool execution.
"""

from __future__ import annotations

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.handlers.partial_handler import (
    PartialResponseHandler,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.parsing_error_handler")


class ParsingErrorHandler(PartialResponseHandler):
    """Handler for parsing errors."""

    async def handle(self, context: HandlerContext) -> HandlerOutput:
        """Handle parsing error by creating structured retry prompt."""
        logger.info("Handling parsing error with %d attempts", context.attempt_count)

        # Create detailed error context for template rendering
        handler_context_update = {
            "error_type": "parsing_error",
            "message": context.error_message,
            "raw_response": context.raw_response,
            "reasoning": context.reasoning,
            "raw_tool_calls": context.raw_tool_calls,
        }

        # Always retry parsing errors
        return HandlerOutput(
            status=ExecutionStatus.PARTIAL,
            template_path="handlers/parsing_retry.jinja2",
            adjusted_llm_options=None,
            should_retry=True,
            error_message=None,
            handler_context=handler_context_update,
        )
