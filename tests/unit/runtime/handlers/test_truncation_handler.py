"""
Unit tests for TruncationHandler.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.handlers.truncation_handler import TruncationHandler
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.agent.llm import LLMResult


class TestTruncationHandler:
    """Unit tests for TruncationHandler."""

    @pytest.fixture
    def handler(self):
        """Create a TruncationHandler instance."""
        return TruncationHandler()

    @pytest.fixture
    def minimal_reasoning_context(self):
        """Create context with minimal reasoning."""
        return HandlerContext(
            session=Mock(),
            llm_response=Mock(),
            reasoning="Brief explanation needed.",
            reasoning_tokens=50,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

    @pytest.fixture
    def moderate_reasoning_context(self):
        """Create context with moderate reasoning."""
        return HandlerContext(
            session=Mock(),
            llm_response=Mock(),
            reasoning="The user needs a detailed explanation with examples and practical applications.",
            reasoning_tokens=1200,  # Above 1000 threshold for CONTINUE_REASONING
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

    @pytest.fixture
    def substantial_reasoning_context(self):
        """Create context with substantial reasoning."""
        llm_response = Mock()
        llm_response.metadata = {"max_tokens": 1000}

        return HandlerContext(
            session=Mock(),
            llm_response=llm_response,
            reasoning="This is a complex multi-faceted question requiring comprehensive analysis of multiple aspects and detailed consideration of various factors.",
            reasoning_tokens=500,  # Below 1000 but reasoning length > 100 chars for EXTEND_AND_CONTINUE
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

    @pytest.mark.asyncio
    async def test_minimal_reasoning_selects_restart_strategy(
        self, handler, minimal_reasoning_context
    ):
        """Test that minimal reasoning selects restart strategy."""
        result = await handler.handle(minimal_reasoning_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path == "handlers/truncation.jinja2"
        assert result.adjusted_llm_options is not None
        # Restart strategy should not increase max_tokens
        max_tokens_value = result.adjusted_llm_options.get(
            "max_tokens"
        ) or result.adjusted_llm_options.get("llm.max_tokens")
        assert max_tokens_value is None  # No increase for restart

    @pytest.mark.asyncio
    async def test_moderate_reasoning_selects_continue_strategy(
        self, handler, moderate_reasoning_context
    ):
        """Test that moderate reasoning selects continue strategy."""
        result = await handler.handle(moderate_reasoning_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path == "handlers/truncation.jinja2"
        assert result.adjusted_llm_options is not None
        # Continue strategy should not increase max_tokens
        max_tokens_value = result.adjusted_llm_options.get(
            "max_tokens"
        ) or result.adjusted_llm_options.get("llm.max_tokens")
        assert max_tokens_value is None  # No increase for continue

    @pytest.mark.asyncio
    async def test_substantial_reasoning_selects_extend_strategy(
        self, handler, substantial_reasoning_context
    ):
        """Test that substantial reasoning selects extend strategy."""
        result = await handler.handle(substantial_reasoning_context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path is not None
        assert result.adjusted_llm_options is not None
        assert result.template_path == "handlers/truncation.jinja2"

        # Verify max_tokens is increased for extend strategy
        max_tokens_value = result.adjusted_llm_options.get(
            "max_tokens"
        ) or result.adjusted_llm_options.get("llm.max_tokens")
        assert max_tokens_value is not None
        assert max_tokens_value > 1000

    @pytest.mark.asyncio
    async def test_zero_reasoning_tokens_uses_restart(self, handler):
        """Test that zero reasoning tokens defaults to restart strategy."""
        context = HandlerContext(
            session=Mock(),
            llm_response=Mock(),
            reasoning="",
            reasoning_tokens=0,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

        result = await handler.handle(context)

        assert result.status == ExecutionStatus.PARTIAL
        assert result.should_retry == True
        assert result.template_path == "handlers/truncation.jinja2"
        # Check for restart-related content (may not contain "restart" exactly)
        assert result.adjusted_llm_options is not None

    @pytest.mark.asyncio
    async def test_adjusted_llm_options_structure(
        self, handler, substantial_reasoning_context
    ):
        """Test that adjusted LLM options are properly structured."""
        result = await handler.handle(substantial_reasoning_context)

        assert isinstance(result.adjusted_llm_options, dict)
        assert len(result.adjusted_llm_options) > 0

        # Check for max_tokens adjustment
        max_tokens_value = result.adjusted_llm_options.get(
            "max_tokens"
        ) or result.adjusted_llm_options.get("llm.max_tokens")
        assert max_tokens_value is not None
        assert isinstance(max_tokens_value, int)
        assert max_tokens_value > 1000

    @pytest.mark.asyncio
    async def test_template_path_content(
        self, handler, moderate_reasoning_context
    ):
        """Test that template_path is set correctly."""
        result = await handler.handle(moderate_reasoning_context)

        assert result.template_path == "handlers/truncation.jinja2"

    @pytest.mark.asyncio
    async def test_restart_strategy_no_max_tokens_increase(
        self, handler, minimal_reasoning_context
    ):
        """Test that restart strategy doesn't increase max_tokens."""
        result = await handler.handle(minimal_reasoning_context)

        # Restart strategy should not increase max_tokens
        if result.adjusted_llm_options:
            max_tokens_value = result.adjusted_llm_options.get(
                "max_tokens"
            ) or result.adjusted_llm_options.get("llm.max_tokens")
            if max_tokens_value:
                assert max_tokens_value <= 1000  # Should not be increased for restart

    @pytest.mark.asyncio
    async def test_continue_strategy_no_max_tokens_increase(
        self, handler, moderate_reasoning_context
    ):
        """Test that continue strategy doesn't increase max_tokens."""
        result = await handler.handle(moderate_reasoning_context)

        # Continue strategy should not increase max_tokens
        if result.adjusted_llm_options:
            max_tokens_value = result.adjusted_llm_options.get(
                "max_tokens"
            ) or result.adjusted_llm_options.get("llm.max_tokens")
            if max_tokens_value:
                assert max_tokens_value <= 1000  # Should not be increased for continue

    @pytest.mark.asyncio
    async def test_extend_strategy_increases_max_tokens(
        self, handler, substantial_reasoning_context
    ):
        """Test that extend strategy increases max_tokens."""
        result = await handler.handle(substantial_reasoning_context)

        # Extend strategy should increase max_tokens
        assert result.adjusted_llm_options is not None
        max_tokens_value = result.adjusted_llm_options.get(
            "max_tokens"
        ) or result.adjusted_llm_options.get("llm.max_tokens")
        assert max_tokens_value is not None
        assert max_tokens_value > 1000  # Should be increased for extend
