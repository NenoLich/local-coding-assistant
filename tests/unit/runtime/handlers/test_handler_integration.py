"""
Unit tests for HandlerIntegration.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.session import SessionState, Message
from local_coding_assistant.agent.llm import LLMResult


class TestHandlerIntegration:
    """Unit tests for HandlerIntegration."""

    @pytest.fixture
    def handler_integration(self):
        """Create a HandlerIntegration instance."""
        return HandlerIntegration()  # ✅ No more max_attempts parameter

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return SessionState(id="test_session_123", history=[], tool_calls=[])

    @pytest.fixture
    def truncation_context(self, session):
        """Create context for truncation handling."""
        return HandlerContext(
            session=session,
            llm_response=Mock(),
            reasoning="Test reasoning content",
            reasoning_tokens=100,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=[],
        )

    @pytest.fixture
    def tool_failure_context(self, session):
        """Create context for tool failure handling."""
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            }
        ]

        return HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
        )

    async def test_handle_truncation_status(
        self, handler_integration, truncation_context
    ):
        """Test handling truncation status."""
        with patch.object(handler_integration, "handler_mapping") as mock_mapping:
            # Mock the truncation handler class and instance
            mock_truncation_handler_instance = Mock()
            mock_truncation_handler_instance.handle = AsyncMock(
                return_value=HandlerOutput(
                    status=ExecutionStatus.PARTIAL,  # ✅ New status
                    template_path="Continue reasoning",  # ✅ New field
                    adjusted_llm_options=None,
                    should_retry=True,
                    error_message=None,
                )
            )

            mock_truncation_handler_class = Mock(
                return_value=mock_truncation_handler_instance
            )
            mock_mapping.get.return_value = mock_truncation_handler_class

            # Create HandlerContext
            handler_context = HandlerContext(
                session=truncation_context.session,
                llm_response=truncation_context.llm_response,
                reasoning=truncation_context.reasoning,
                reasoning_tokens=truncation_context.reasoning_tokens,
                current_iteration=truncation_context.current_iteration,
                max_attempts=truncation_context.max_attempts,
                attempt_count=truncation_context.attempt_count,
                failed_tools=truncation_context.failed_tools,
                error_type="truncation",  # ✅ Set error_type for handler routing
            )

            result = await handler_integration.handle_partial_response(
                execution_status=ExecutionStatus.PARTIAL,  # ✅ New status
                handler_context=handler_context,  # ✅ New signature
            )

            mock_mapping.get.assert_called_once_with("truncation")
            mock_truncation_handler_class.assert_called_once()
            mock_truncation_handler_instance.handle.assert_called_once_with(
                context=handler_context
            )
            assert result.status == ExecutionStatus.PARTIAL  # ✅ New status
            assert result.should_retry == True
            assert result.template_path == "Continue reasoning"  # ✅ New field

    async def test_handle_tool_failures_status(
        self, handler_integration, tool_failure_context
    ):
        """Test handling tool failures status."""
        with patch.object(handler_integration, "handler_mapping") as mock_mapping:
            # Mock the tool failure handler class and instance
            mock_tool_failure_handler_instance = Mock()
            mock_tool_failure_handler_instance.handle = AsyncMock(
                return_value=HandlerOutput(
                    status=ExecutionStatus.PARTIAL,  # ✅ Tool failures use PARTIAL status
                    template_path="Retry failed tools",
                    adjusted_llm_options=None,
                    should_retry=True,
                    error_message=None,
                )
            )

            mock_tool_failure_handler_class = Mock(
                return_value=mock_tool_failure_handler_instance
            )
            mock_mapping.get.return_value = mock_tool_failure_handler_class

            # Create HandlerContext
            handler_context = HandlerContext(
                session=tool_failure_context.session,
                llm_response=tool_failure_context.llm_response,
                reasoning=tool_failure_context.reasoning,
                reasoning_tokens=tool_failure_context.reasoning_tokens,
                current_iteration=tool_failure_context.current_iteration,
                max_attempts=tool_failure_context.max_attempts,
                attempt_count=tool_failure_context.attempt_count,
                failed_tools=tool_failure_context.failed_tools,
                error_type="tool_failures",  # ✅ Set error_type for handler routing
            )

            result = await handler_integration.handle_partial_response(
                execution_status=ExecutionStatus.PARTIAL,  # ✅ Tool failures use PARTIAL status
                handler_context=handler_context,
            )

            mock_mapping.get.assert_called_once_with("tool_failures")
            mock_tool_failure_handler_class.assert_called_once()
            mock_tool_failure_handler_instance.handle.assert_called_once_with(
                context=handler_context
            )
            assert (
                result.status == ExecutionStatus.PARTIAL
            )  # ✅ Tool failures use PARTIAL status
            assert result.should_retry == True
            assert result.template_path == "Retry failed tools"

    @pytest.mark.asyncio
    async def test_handle_success_status_returns_success(
        self, handler_integration, session
    ):
        """Test handling success status returns success output."""
        # Create HandlerContext
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

        result = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.SUCCESS, handler_context=handler_context
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.should_retry == False
        assert result.template_path is None
        assert result.adjusted_llm_options is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_handle_blocked_status_returns_success(
        self, handler_integration, session
    ):
        """Test handling blocked status returns success output."""
        # Create HandlerContext
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
        )

        result = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.BLOCKED, handler_context=handler_context
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.should_retry == False
        assert result.template_path is None
        assert result.adjusted_llm_options is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_unknown_status_returns_success(self, handler_integration, session):
        """Test that unknown status returns success output."""
        # Create HandlerContext
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=None,  # ✅ No error_type for unknown status
        )

        result = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.FAILED,  # Not handled specifically
            handler_context=handler_context,
        )

        assert result.status == ExecutionStatus.SUCCESS
        assert result.should_retry == False
        assert result.template_path is None
        assert result.adjusted_llm_options is None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_should_continue_execution_true(self, handler_integration):
        """Test should_continue_execution returns True when both conditions are met."""
        handler_output = HandlerOutput(
            status=ExecutionStatus.PARTIAL,  # ✅ New status
            template_path="Continue",  # ✅ New field
            adjusted_llm_options=None,
            should_retry=True,
            error_message=None,
        )

        result = handler_integration.should_continue_execution(handler_output)
        assert result == True

    @pytest.mark.asyncio
    async def test_should_continue_execution_false_no_retry(self, handler_integration):
        """Test should_continue_execution returns False when should_retry is False."""
        handler_output = HandlerOutput(
            status=ExecutionStatus.PARTIAL,  # ✅ New status
            template_path="Continue",  # ✅ New field
            adjusted_llm_options=None,
            should_retry=False,
            error_message=None,
        )

        result = handler_integration.should_continue_execution(handler_output)
        assert result == False

    @pytest.mark.asyncio
    async def test_should_continue_execution_false_no_continuation(
        self, handler_integration
    ):
        """Test should_continue_execution returns False when template_path is None."""
        handler_output = HandlerOutput(
            status=ExecutionStatus.PARTIAL,  # ✅ New status
            template_path=None,  # ✅ New field
            adjusted_llm_options=None,
            should_retry=True,
            error_message=None,
        )

        result = handler_integration.should_continue_execution(handler_output)
        assert result == False

    @pytest.mark.asyncio
    async def test_should_continue_execution_false_both_false(
        self, handler_integration
    ):
        """Test should_continue_execution returns False when both conditions are False."""
        handler_output = HandlerOutput(
            status=ExecutionStatus.PARTIAL,
            template_path=None,
            adjusted_llm_options=None,
            should_retry=False,
            error_message=None,
        )

        result = handler_integration.should_continue_execution(handler_output)
        assert result == False

    @pytest.mark.asyncio
    async def test_classify_tool_error(self, handler_integration):
        """Test tool error classification."""
        with patch.object(handler_integration, "error_classifier") as mock_classifier:
            mock_classifier.classify_error.return_value = Mock(
                error_type="NETWORK",
                retry_strategy="RETRY_WITH_BACKOFF",
                can_retry=True,
            )
            mock_classifier.create_failure_info.return_value = {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            }

            result = handler_integration.classify_tool_error(
                tool_name="search_files",
                tool_args={"pattern": "*.py"},
                error="Connection timeout",
            )

            mock_classifier.classify_error.assert_called_once_with(
                "search_files", {"pattern": "*.py"}, "Connection timeout"
            )
            mock_classifier.create_failure_info.assert_called_once()

            assert result["tool_name"] == "search_files"
            assert result["error_type"] == "NETWORK"
            assert result["can_retry"] == True

    @pytest.mark.asyncio
    async def test_max_attempts_property(self):
        """Test that HandlerIntegration can be instantiated."""
        handler = HandlerIntegration()  # ✅ No more max_attempts parameter
        assert handler is not None
        assert hasattr(handler, "handler_mapping")
        assert hasattr(handler, "error_classifier")
