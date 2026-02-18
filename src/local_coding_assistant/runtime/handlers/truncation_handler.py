"""
Handler for token limit truncations.

This module implements intelligent handling of responses that are truncated
due to reaching max_tokens limits, with token-aware strategies.
"""

from typing import Any

from local_coding_assistant.runtime.execution_types import (
    ContinuationStrategy,
    ExecutionStatus,
)
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.handlers.partial_handler import (
    PartialResponseHandler,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.truncation_handler")


class TruncationHandler(PartialResponseHandler):
    """Handle token limit truncations."""

    async def handle(self, context: HandlerContext) -> HandlerOutput:
        """Handle truncation by determining strategy and preparing continuation."""
        strategy = self._determine_strategy(context)
        truncated_reasoning = self._prepare_reasoning(context.reasoning, strategy)

        # Adjust LLM options for retry
        adjusted_options = self._build_adjusted_options(context, strategy)

        logger.info("Handling truncation with strategy: %s", strategy.value)

        handler_context_update = {
            "reasoning": truncated_reasoning,
            "finish_reason": "length",
        }

        return HandlerOutput(
            status=ExecutionStatus.PARTIAL,  # Use PARTIAL for all handler interventions
            template_path="handlers/truncation.jinja2",  # Use template_path instead of continuation_prompt
            adjusted_llm_options=adjusted_options,
            should_retry=True,
            error_message=None,
            handler_context=handler_context_update,
        )

    def _determine_strategy(self, context: HandlerContext) -> ContinuationStrategy:
        """Determine best continuation strategy based on context."""
        reasoning_tokens = context.reasoning_tokens or 0

        # If reasoning tokens are significant portion of total, continue it
        if reasoning_tokens > 1000:  # Heuristic threshold
            return ContinuationStrategy.CONTINUE_REASONING

        # If we have substantial reasoning, try extending
        if context.reasoning and len(context.reasoning.strip()) > 100:
            return ContinuationStrategy.EXTEND_AND_CONTINUE

        # Default to restart with minimal context
        return ContinuationStrategy.RESTART_REASONING

    def _prepare_reasoning(
        self, reasoning: str | None, strategy: ContinuationStrategy
    ) -> str | None:
        """Prepare reasoning content based on strategy."""
        if not reasoning:
            return None

        if strategy == ContinuationStrategy.CONTINUE_REASONING:
            # Keep last 50% + describe what happened
            reasoning_len = len(reasoning)
            keep_chars = reasoning_len // 2
            return reasoning[-keep_chars:] if keep_chars < reasoning_len else reasoning

        elif strategy == ContinuationStrategy.RESTART_REASONING:
            # Keep very minimal, describe what happened and ask LLM not to reason too much
            lines = reasoning.split("\n")
            # Keep first 2-3 lines as context
            minimal_context = "\n".join(lines[:3])
            return f"Previous reasoning was truncated. Context: {minimal_context}\n\nPlease continue without extensive reasoning."

        elif strategy == ContinuationStrategy.EXTEND_AND_CONTINUE:
            # Keep full reasoning
            return reasoning

        return None

    def _build_adjusted_options(
        self, context: HandlerContext, strategy: ContinuationStrategy
    ) -> dict[str, Any]:
        """Build adjusted LLM options based on strategy."""
        base_options = {}

        if strategy == ContinuationStrategy.EXTEND_AND_CONTINUE:
            # Increase max_tokens by 50%
            current_max = (
                context.llm_response.metadata.get("max_tokens")
                if context.llm_response and context.llm_response.metadata
                else 1000
            )
            base_options["llm.max_tokens"] = (
                int(current_max * 1.5) if current_max else 1500
            )

        return base_options
