"""
Handler for tool failure scenarios.

This module implements handling of partial responses due to tool failures,
with analysis of failure patterns and retry logic.
"""

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.handlers.partial_handler import (
    PartialResponseHandler,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.tool_failure_handler")


class ToolFailureHandler(PartialResponseHandler):
    """Handle partial responses due to tool failures."""

    async def handle(self, context: HandlerContext) -> HandlerOutput:
        """Handle tool failures by analyzing and deciding on retry strategy."""
        # Check if max attempts exceeded
        if context.attempt_count >= context.max_attempts:
            return HandlerOutput(
                status=ExecutionStatus.FAILED,
                template_path=None,
                adjusted_llm_options=None,
                should_retry=False,
                error_message="Exceeded maximum retry attempts for tool failure handling",
            )

        # Use structured failed_tools data if available
        failed_tools = context.failed_tools or []

        logger.info("Handling %d tool failures", len(failed_tools))

        # Decide if retry is possible or escalation needed
        retryable_tools = [
            tool for tool in failed_tools if tool.get("can_retry", False)
        ]

        # Determine retry strategy from tool errors
        retry_strategies = [
            tool.get("retry_strategy")
            for tool in retryable_tools
            if tool.get("retry_strategy")
        ]
        primary_strategy = (
            retry_strategies[0] if retry_strategies else "retry_immediate"
        )

        if retryable_tools:
            handler_context_update = {
                "failed_tools": retryable_tools,
            }
            return HandlerOutput(
                status=ExecutionStatus.PARTIAL,  # Use PARTIAL for all handler interventions
                template_path="handlers/tool_retry.jinja2",  # Use template_path instead of continuation_prompt
                adjusted_llm_options=None,
                should_retry=True,
                error_message=None,
                retry_strategy=primary_strategy,
                handler_context=handler_context_update,
            )
        else:
            return HandlerOutput(
                status=ExecutionStatus.FAILED,  # No retryable tools, fail
                template_path=None,
                adjusted_llm_options=None,
                should_retry=False,
                error_message=f"Tool failures exceeded retry limit: {[f.get('tool_name', 'Unknown tool') for f in failed_tools]}",
            )
