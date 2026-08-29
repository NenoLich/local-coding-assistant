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
        truncated_content = self._prepare_content(context.raw_response, strategy)
        truncated_tool_calls = self._prepare_tool_calls(
            context.raw_tool_calls, strategy
        )

        # Adjust LLM options for retry
        adjusted_options = self._build_adjusted_options(context, strategy)

        logger.info("Handling truncation with strategy: %s", strategy.value)

        handler_context_update = {
            "reasoning": truncated_reasoning,
            "raw_response": truncated_content,
            "raw_tool_calls": truncated_tool_calls,
            "finish_reason": "length",
        }

        # Use simple continuation template for CONTINUE_WITH_HISTORY strategy
        template_path = (
            "handlers/truncation_continue.jinja2"
            if strategy == ContinuationStrategy.CONTINUE_WITH_HISTORY
            else "handlers/truncation.jinja2"
        )

        return HandlerOutput(
            status=ExecutionStatus.PARTIAL,  # Use PARTIAL for all handler interventions
            template_path=template_path,
            adjusted_llm_options=adjusted_options,
            should_retry=True,
            error_message=None,
            handler_context=handler_context_update,
        )

    def _determine_strategy(self, context: HandlerContext) -> ContinuationStrategy:
        """Determine best continuation strategy based on context."""
        reasoning_tokens = context.reasoning_tokens or 0

        # Check for incomplete tool calls first (highest priority)
        if context.raw_tool_calls and self._has_incomplete_tool_calls(
            context.raw_tool_calls
        ):
            return ContinuationStrategy.CONTINUE_TOOL_CALLS

        # Check for truncated content without reasoning - use history-based continuation
        if context.raw_response and reasoning_tokens == 0:
            return ContinuationStrategy.CONTINUE_WITH_HISTORY

        # Check for truncated content with reasoning
        if context.raw_response and self._is_content_truncated(context.raw_response):
            return ContinuationStrategy.CONTINUE_CONTENT

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

    def _prepare_content(
        self, content: str | None, strategy: ContinuationStrategy
    ) -> str | None:
        """Prepare content based on strategy."""
        if not content:
            return None

        if strategy == ContinuationStrategy.CONTINUE_CONTENT:
            # Keep last 70% of content to continue from
            content_len = len(content)
            keep_chars = int(content_len * 0.7)
            return content[-keep_chars:] if keep_chars < content_len else content

        elif strategy == ContinuationStrategy.CONTINUE_TOOL_CALLS:
            # Keep minimal content context for tool call continuation
            lines = content.split("\n")
            minimal_context = "\n".join(lines[-5:]) if len(lines) > 5 else content
            return minimal_context

        elif strategy in (
            ContinuationStrategy.RESTART_REASONING,
            ContinuationStrategy.EXTEND_AND_CONTINUE,
        ):
            # Keep full content for these strategies
            return content

        return None

    def _prepare_tool_calls(
        self, tool_calls: list[dict[str, Any]] | None, strategy: ContinuationStrategy
    ) -> list[dict[str, Any]] | None:
        """Prepare tool calls based on strategy."""
        if not tool_calls:
            return None

        if strategy == ContinuationStrategy.CONTINUE_TOOL_CALLS:
            # Keep incomplete tool calls for continuation
            return self._filter_incomplete_tool_calls(tool_calls)

        elif strategy == ContinuationStrategy.CONTINUE_CONTENT:
            # Keep all tool calls as context
            return tool_calls

        elif strategy in (
            ContinuationStrategy.RESTART_REASONING,
            ContinuationStrategy.EXTEND_AND_CONTINUE,
        ):
            # Keep all tool calls for these strategies
            return tool_calls

        return None

    def _has_incomplete_tool_calls(self, tool_calls: list[dict[str, Any]]) -> bool:
        """Check if there are incomplete tool calls."""
        for tool_call in tool_calls:
            # Check if tool call is missing required fields or appears truncated
            if not tool_call.get("function") or not tool_call.get("function", {}).get(
                "name"
            ):
                return True
            # Check if arguments are incomplete (unclosed braces, etc.)
            arguments = tool_call.get("function", {}).get("arguments", "")
            if arguments and self._is_json_incomplete(arguments):
                return True
        return False

    def _is_json_incomplete(self, json_str: str) -> bool:
        """Check if JSON string appears incomplete."""
        json_str = json_str.strip()
        if not json_str:
            return True
        # Check for unbalanced braces
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        if open_braces != close_braces:
            return True
        # Check for unbalanced brackets
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        if open_brackets != close_brackets:
            return True
        # Check if ends abruptly (no closing quote)
        if json_str.endswith('"') and json_str.count('"') % 2 != 0:
            return True
        return False

    def _is_content_truncated(self, content: str) -> bool:
        """Check if content appears truncated."""
        content = content.strip()
        # Check for incomplete sentences
        if not content.endswith((".", "!", "?", "}", "]", ")", ">", '"')):
            return True
        # Check for unbalanced code blocks
        if content.count("```") % 2 != 0:
            return True
        # Check for unbalanced braces in code
        if content.count("{") != content.count("}"):
            return True
        if content.count("[") != content.count("]"):
            return True
        if content.count("(") != content.count(")"):
            return True
        return False

    def _filter_incomplete_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter to keep only incomplete tool calls."""
        incomplete_calls = []
        for tool_call in tool_calls:
            arguments = tool_call.get("function", {}).get("arguments", "")
            if self._is_json_incomplete(arguments):
                incomplete_calls.append(tool_call)
        return incomplete_calls if incomplete_calls else tool_calls

    def _build_adjusted_options(
        self, context: HandlerContext, strategy: ContinuationStrategy
    ) -> dict[str, Any]:
        """Build adjusted LLM options based on strategy."""
        base_options = {}

        if strategy in (
            ContinuationStrategy.EXTEND_AND_CONTINUE,
            ContinuationStrategy.CONTINUE_CONTENT,
            ContinuationStrategy.CONTINUE_TOOL_CALLS,
        ):
            # Increase max_tokens by 50% for content/tool call continuation
            current_max = (
                context.llm_response.metadata.get("max_tokens")
                if context.llm_response and context.llm_response.metadata
                else 1000
            )
            base_options["llm.max_tokens"] = (
                int(current_max * 1.5) if current_max else 1500
            )

        return base_options
