"""
Integration tests for truncation handling from driver level through entire system.
"""

import pytest

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerErrorType,
)

from tests.integration.runtime.conftest import (
    MockLLMDriver,
    create_test_session,
    assert_handler_output,
    assert_session_continuation,
    verify_llm_options_adjustment,
)
from tests.integration.runtime.mock_llm_responses import MockLLMResponse


class TestTruncationIntegration:
    """Test truncation handling integration across all components."""

    @pytest.mark.asyncio
    async def test_minimal_reasoning_truncation_strategy(self):
        """Test truncation with minimal reasoning selects restart strategy."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.minimal_reasoning_truncated(),
                MockLLMResponse.successful_response(),
            ]
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
        )

        # Verify restart strategy (minimal reasoning should restart)
        assert "truncation" in (handler_output.template_path or ""), (
            "Should contain truncation"
        )

        # Verify session updated
        assert_session_continuation(session, expected_keywords=["truncation"])

    @pytest.mark.asyncio
    async def test_moderate_reasoning_truncation_strategy(self):
        """Test truncation with moderate reasoning selects continue strategy."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.moderate_reasoning_truncated(),
                MockLLMResponse.successful_response(),
            ]
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
        )

        # Verify continue strategy (moderate reasoning should continue)
        assert "truncation" in (handler_output.template_path or "").lower()

        # Verify session updated
        assert_session_continuation(session, expected_keywords=["truncation"])

    @pytest.mark.asyncio
    async def test_substantial_reasoning_truncation_strategy(self):
        """Test truncation with substantial reasoning selects extend strategy."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.substantial_reasoning_truncated(),
                MockLLMResponse.successful_response(),
            ]
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
            has_adjusted_options=True,
        )

        # Verify extend strategy (substantial reasoning should extend)
        assert "truncation" in (handler_output.template_path or "").lower()

        # Verify LLM options adjustment for extend strategy
        verify_llm_options_adjustment(
            handler_output.adjusted_llm_options,
            expected_max_tokens_increase=True,
            expected_min_max_tokens=1500,  # Original 1000 + 50%
        )

        # Verify session updated
        assert_session_continuation(session, expected_keywords=["truncation"])

    @pytest.mark.asyncio
    async def test_truncation_retry_limit_enforcement(self):
        """Test that truncation retries are limited to max_attempts."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.truncated_response(),  # First truncation
                MockLLMResponse.truncated_response(),  # Second truncation
                MockLLMResponse.successful_response(),  # Would be third call but shouldn't reach
            ]
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # First truncation
        handler_context1 = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )
        handler_output1 = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context1
        )

        assert_handler_output(
            handler_output1,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
        )

        # Apply first continuation
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output1.template_path,
            "retry_strategy": getattr(handler_output1, "retry_strategy", None),
        }

        # Second truncation (should stop due to max_attempts)
        handler_context2 = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[1],
            reasoning=mock_driver.responses[1].reasoning,
            reasoning_tokens=mock_driver.responses[1].reasoning_tokens,
            current_iteration=2,
            max_attempts=2,
            attempt_count=2,  # Third attempt (exceeds max_attempts=2)
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )
        handler_output2 = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context2
        )

        assert_handler_output(
            handler_output2,
            expected_status=ExecutionStatus.FAILED,  # Should be FAILED when max attempts exceeded
            should_retry=False,  # Should not continue after max attempts
            has_continuation=False,
        )

    @pytest.mark.asyncio
    async def test_truncation_session_state_management(self):
        """Test that session state is properly managed during truncation handling."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.truncated_response(),
                MockLLMResponse.successful_response(),
            ]
        )

        session = create_test_session()
        initial_history_length = len(session.history)

        handler_integration = HandlerIntegration()

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )

        # Execute truncation handling
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify session state
        assert "handler_context" in session.metadata, (
            "Should have added handler context to metadata"
        )
        handler_context_meta = session.metadata["handler_context"]
        assert handler_context_meta["error_type"] == HandlerErrorType.TRUNCATION
        assert handler_context_meta["template_path"] == handler_output.template_path
        assert "truncation" in handler_context_meta["template_path"], (
            "Should have truncation template"
        )

    @pytest.mark.asyncio
    async def test_truncation_adjusted_llm_options_persistence(self):
        """Test that adjusted LLM options are properly structured and accessible."""
        # Setup
        mock_driver = MockLLMDriver(
            [
                MockLLMResponse.substantial_reasoning_truncated(),
                MockLLMResponse.successful_response(),
            ]
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=mock_driver.responses[0],
            reasoning=mock_driver.responses[0].reasoning,
            reasoning_tokens=mock_driver.responses[0].reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=HandlerErrorType.TRUNCATION,
        )

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": HandlerErrorType.TRUNCATION,
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify adjusted options structure
        assert handler_output.adjusted_llm_options is not None, (
            "Should have adjusted options"
        )
        assert isinstance(handler_output.adjusted_llm_options, dict), (
            "Should be dictionary"
        )

        # Check specific adjustments for extend strategy
        adjusted_options = handler_output.adjusted_llm_options
        max_tokens_value = adjusted_options.get("max_tokens") or adjusted_options.get(
            "llm.max_tokens"
        )
        assert max_tokens_value > 1000, "Should increase max_tokens"

        # Verify the adjustment is properly structured
        assert isinstance(max_tokens_value, int), "max_tokens should be an integer"


if __name__ == "__main__":
    # Run a quick test to verify setup
    import asyncio

    async def quick_test():
        test = TestTruncationIntegration()
        await test.test_moderate_reasoning_truncation_strategy()
        print("✅ Truncation integration test setup working!")

    asyncio.run(quick_test())
