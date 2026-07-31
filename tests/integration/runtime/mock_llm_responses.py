"""
Mock LLM responses for testing handler integration scenarios.
"""

from dataclasses import dataclass
from typing import Any

from local_coding_assistant.agent.llm import LLMResult


@dataclass
class MockLLMResponse:
    """Factory for creating mock LLM responses with specific characteristics."""

    @staticmethod
    def truncated_response(
        content: str = "This is a partial response that was cut off due to token limits. The original response would have contained much more detailed information about topic being discussed, but unfortunately it was truncated before completion.",
        reasoning: str = "The user is asking about a complex topic that requires detailed explanation. I need to provide comprehensive coverage including historical context, technical details, and practical examples. This will require significant reasoning to organize information properly and ensure all aspects are covered thoroughly.",
        reasoning_tokens: int = 500,
        total_tokens: int = 1000,
        model: str = "test-model",
        provider: str = "test-provider",
        finish_reason: str = "length",
    ) -> LLMResult:
        """Create a truncated LLM response."""
        return LLMResult(
            content=content,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            tool_calls=[],
        )

    @staticmethod
    def minimal_reasoning_truncated() -> LLMResult:
        """Truncated response with minimal reasoning content."""
        return MockLLMResponse.truncated_response(
            reasoning="Brief explanation needed.", reasoning_tokens=50
        )

    @staticmethod
    def moderate_reasoning_truncated() -> LLMResult:
        """Truncated response with moderate reasoning content."""
        return MockLLMResponse.truncated_response(
            reasoning="The user needs a detailed explanation of concept with examples and practical applications. I should structure this to cover main points clearly while providing enough depth for understanding.",
            reasoning_tokens=1200,
        )

    @staticmethod
    def substantial_reasoning_truncated() -> LLMResult:
        """Truncated response with substantial reasoning content."""
        return MockLLMResponse.truncated_response(
            reasoning="This is a complex multi-faceted question that requires comprehensive analysis. I need to break down the problem into its core components, analyze each aspect thoroughly, consider various perspectives and approaches, provide historical context, explain technical details, give practical examples, address potential edge cases, and summarize key takeaways. This requires extensive reasoning to ensure all aspects are covered properly and response is both comprehensive and well-structured.",
            reasoning_tokens=800,
        )

    @staticmethod
    def successful_response(
        content: str = "This is a complete and successful response to user's query.",
        reasoning: str = "Standard reasoning for successful response.",
        reasoning_tokens: int = 100,
        total_tokens: int = 300,
        model: str = "test-model",
        provider: str = "test-provider",
        finish_reason: str = "stop",
    ) -> LLMResult:
        """Create a successful LLM response."""
        return LLMResult(
            content=content,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            tool_calls=[],
        )

    @staticmethod
    def blocked_response(
        content: str = "I cannot provide a response to this request.",
        reasoning: str = "The request violates safety guidelines.",
        reasoning_tokens: int = 50,
        total_tokens: int = 100,
        model: str = "test-model",
        provider: str = "test-provider",
        finish_reason: str = "content_filter",
    ) -> LLMResult:
        """Create a blocked LLM response."""
        return LLMResult(
            content=content,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            tool_calls=[],
        )

    @staticmethod
    def tool_call_response(
        content: str = "I'll help you with that task.",
        tool_calls: list[dict[str, Any]] = None,
        reasoning: str = "The user needs help with a specific task that requires tool usage.",
        reasoning_tokens: int = 100,
        total_tokens: int = 300,
        model: str = "test-model",
        provider: str = "test-provider",
        finish_reason: str = "stop",
    ) -> LLMResult:
        """Create an LLM response with tool calls."""
        if tool_calls is None:
            tool_calls = [
                {
                    "id": "call_123",
                    "name": "search_files",
                    "arguments": {"pattern": "*.py", "max_results": 10},
                }
            ]

        return LLMResult(
            content=content,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
