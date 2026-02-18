"""
Integration module for partial response handlers.

This module provides integration utilities for using partial response
handlers in different execution contexts (frame_agent and runtime_manager).
"""

from typing import Any

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.handlers.parsing_error_handler import (
    ParsingErrorHandler,
)
from local_coding_assistant.runtime.handlers.partial_handler import (
    PartialResponseHandler,
)
from local_coding_assistant.runtime.handlers.tool_error_classifier import (
    ToolErrorClassifier,
)
from local_coding_assistant.runtime.handlers.tool_failure_handler import (
    ToolFailureHandler,
)
from local_coding_assistant.runtime.handlers.truncation_handler import TruncationHandler
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.handler_integration")

# Handler mapping for easy extension
HANDLER_MAPPING: dict[str, type[PartialResponseHandler]] = {
    "truncation": TruncationHandler,
    "tool_failures": ToolFailureHandler,
    "parsing_error": ParsingErrorHandler,
}


class HandlerIntegration:
    """Integration utilities for partial response handlers."""

    def __init__(self):
        self.handler_mapping = HANDLER_MAPPING
        self.error_classifier = ToolErrorClassifier()

    async def handle_partial_response(
        self,
        execution_status: ExecutionStatus,
        handler_context: HandlerContext,
    ) -> HandlerOutput:
        """Handle partial response by routing to appropriate handler."""

        # Return SUCCESS for non-partial responses without handler context
        if execution_status != ExecutionStatus.PARTIAL or not handler_context:
            return HandlerOutput(
                status=ExecutionStatus.SUCCESS,
                template_path=None,
                adjusted_llm_options=None,
                should_retry=False,
                error_message=None,
            )

        # Route to appropriate handler based on error type
        logger.info(
            "Handling partial response: %s (iteration %d)",
            handler_context.error_type,
            handler_context.current_iteration,
        )

        # Check attempt limits
        if handler_context.attempt_count >= handler_context.max_attempts:
            return HandlerOutput(
                status=ExecutionStatus.FAILED,
                template_path=None,
                adjusted_llm_options=None,
                should_retry=False,
                error_message=f"Handler attempts exhausted for: {handler_context.error_type}",
            )

        try:
            if handler_context.error_type is None:
                raise ValueError("Could not define error type for handler integration")
            handler_class = self.handler_mapping.get(handler_context.error_type)
            if handler_class is None:
                raise ValueError("Could not define error type for handler integration")
            handler_instance = handler_class()
            output = await handler_instance.handle(context=handler_context)
            return output
        except Exception as exc:
            logger.warning(
                f"Could not use handler. Returning placeholder. Error: {exc!s}"
            )

            return HandlerOutput(
                status=ExecutionStatus.FAILED,
                template_path=None,
                adjusted_llm_options=None,
                should_retry=False,
                error_message=f"Unhandled partial error type: {handler_context.error_type}",
            )

    def classify_tool_error(
        self, tool_name: str, tool_args: dict[str, Any], error: str | Exception
    ) -> dict[str, Any]:
        """Classify a tool error and return structured failure info."""
        tool_error = self.error_classifier.classify_error(tool_name, tool_args, error)
        return self.error_classifier.create_failure_info(tool_error)

    def should_continue_execution(self, handler_output: HandlerOutput) -> bool:
        """Determine if execution should continue based on handler output."""
        return handler_output.should_retry and handler_output.template_path is not None

    def get_adjusted_llm_options(
        self, base_options: dict[str, Any], handler_output: HandlerOutput
    ) -> dict[str, Any]:
        """Get adjusted LLM options from handler output."""
        if handler_output.adjusted_llm_options:
            adjusted = base_options.copy()
            adjusted.update(handler_output.adjusted_llm_options)
            return adjusted
        return base_options
