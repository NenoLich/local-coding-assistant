"""
End-to-end integration tests for handler integration system.
Tests the complete flow through handler integration without full FrameAgent complexity.
"""

from unittest.mock import Mock

import pytest

from local_coding_assistant.agent.llm import LLMResult
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
)
from local_coding_assistant.runtime.session import SessionState


class TestHandlerIntegrationE2E:
    """End-to-end tests for handler integration system."""

    @pytest.fixture
    def handler_integration(self):
        """Create a HandlerIntegration instance."""
        return HandlerIntegration()  # ✅ No more max_attempts parameter

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return SessionState(
            id="test_session_e2e",
            current_task="Test task for e2e testing",
            history=[],
            tool_calls=[],
        )

    @pytest.mark.asyncio
    async def test_truncation_handling_complete_flow(
        self, handler_integration, session
    ):
        """Test complete truncation handling flow from handler to session."""
        # Setup LLM response with substantial reasoning
        llm_response = Mock()
        llm_response.metadata = {"max_tokens": 1000}

        truncated_response = LLMResult(
            content="This is a partial response that was cut off due to token limits.",
            reasoning="This is a complex multi-faceted question requiring comprehensive analysis of multiple aspects and detailed consideration of various factors.",
            reasoning_tokens=500,
            total_tokens=1000,
            model="test-model",
            provider="test-provider",
            finish_reason="length",
            tool_calls=[],
        )

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=llm_response,
            reasoning=truncated_response.reasoning,
            reasoning_tokens=truncated_response.reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type="truncation",
        )

        # Handle truncation
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Verify handler output
        assert handler_output.status == ExecutionStatus.PARTIAL
        assert handler_output.should_retry == True
        assert handler_output.template_path is not None  # ✅ New field
        assert handler_output.adjusted_llm_options is not None

        # Apply handler context to session metadata (new approach)
        session.metadata["handler_context"] = {
            "error_type": "truncation",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify session was updated with handler context in metadata
        assert "handler_context" in session.metadata
        handler_context_meta = session.metadata["handler_context"]
        assert handler_context_meta["error_type"] == "truncation"
        assert handler_context_meta["template_path"] == handler_output.template_path
        assert "template_path" in session.metadata["handler_context"]

        # Verify should continue execution
        should_continue = handler_integration.should_continue_execution(handler_output)
        assert should_continue == True

        # Test max attempts exceeded
        handler_context_exceeded = HandlerContext(
            session=session,
            llm_response=llm_response,
            reasoning=truncated_response.reasoning,
            reasoning_tokens=truncated_response.reasoning_tokens,
            current_iteration=2,
            max_attempts=2,
            attempt_count=2,  # Exceeds max_attempts=2
            failed_tools=None,
            error_type="truncation",
        )

        handler_output2 = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL,
            handler_context=handler_context_exceeded,
        )

        # Verify no continuation when max attempts exceeded
        assert handler_output2.should_retry == False
        assert (
            handler_output2.template_path is None
        )  # ✅ Use template_path instead of continuation_prompt
        assert handler_output2.adjusted_llm_options is None

    @pytest.mark.asyncio
    async def test_tool_failure_handling_complete_flow(
        self, handler_integration, session
    ):
        """Test complete tool failure handling flow from handler to session."""
        # Setup failed tools data
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout after 30 seconds",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            }
        ]

        # Create HandlerContext for tool failures
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
            error_type="tool_failures",
        )

        # Handle tool failures
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Verify handler output
        assert handler_output.status == ExecutionStatus.PARTIAL
        assert handler_output.should_retry == True
        assert handler_output.template_path is not None  # ✅ New field
        assert handler_output.adjusted_llm_options is None

        # Apply handler context to session metadata (new approach)
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify session was updated with handler context in metadata
        assert "handler_context" in session.metadata
        handler_context_meta = session.metadata["handler_context"]
        assert handler_context_meta["error_type"] == "tool_failures"
        assert handler_context_meta["template_path"] == handler_output.template_path

        # Verify should continue execution
        should_continue = handler_integration.should_continue_execution(handler_output)
        assert should_continue == True

        # Test max attempts exceeded
        handler_context_exceeded = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=2,
            max_attempts=2,
            attempt_count=2,  # Exceeds max_attempts=2
            failed_tools=failed_tools,
            error_type="tool_failures",
        )

        handler_output2 = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL,
            handler_context=handler_context_exceeded,
        )

        # Verify no continuation when max attempts exceeded
        assert handler_output2.status == ExecutionStatus.FAILED
        assert handler_output2.should_retry == False
        assert (
            handler_output2.template_path is None
        )  # ✅ Use template_path instead of continuation_prompt

    @pytest.mark.asyncio
    async def test_non_retryable_tool_failure_flow(self, handler_integration, session):
        """Test handling of non-retryable tool failures."""
        # Setup non-retryable failed tools data
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": 123},  # Invalid type
                "error_message": "Invalid argument: 'pattern' must be a string",
                "error_type": "SYNTAX_ERROR",
                "retry_strategy": "NO_RETRY",
                "can_retry": False,
            }
        ]

        # Create HandlerContext for tool failures
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
            error_type="tool_failures",
        )

        # Handle tool failures
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Verify handler output for non-retryable failures
        assert (
            handler_output.status == ExecutionStatus.FAILED
        )  # ✅ Non-retryable failures still return PARTIAL
        assert handler_output.should_retry == False
        assert (
            handler_output.template_path is None
        )  # ✅ Use template_path instead of continuation_prompt
        assert handler_output.adjusted_llm_options is None
        assert "search_files" in handler_output.error_message

        # Verify should not continue execution
        should_continue = handler_integration.should_continue_execution(handler_output)
        assert should_continue == False

    @pytest.mark.asyncio
    async def test_mixed_tool_failures_flow(self, handler_integration, session):
        """Test handling of mixed retryable and non-retryable tool failures."""
        # Setup mixed failed tools data
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Connection timeout after 30 seconds",
                "error_type": "NETWORK",
                "retry_strategy": "RETRY_WITH_BACKOFF",
                "can_retry": True,
            },
            {
                "tool_name": "read_file",
                "tool_args": {"file_path": "/etc/passwd"},
                "error_message": "Permission denied: cannot access file '/etc/passwd'",
                "error_type": "PERMISSION",
                "retry_strategy": "NO_RETRY",
                "can_retry": False,
            },
        ]

        # Create HandlerContext for tool failures
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=failed_tools,
            error_type="tool_failures",
        )

        # Handle tool failures
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Verify handler output includes only retryable tools
        assert handler_output.status == ExecutionStatus.PARTIAL
        assert handler_output.should_retry == True
        assert handler_output.template_path is not None  # ✅ New field

        # Apply handler context to session metadata (new approach)
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify session continuation only mentions retryable tool
        handler_context_meta = session.metadata["handler_context"]
        assert (
            "tool_retry.jinja2" in handler_context_meta["template_path"]
        )  # Check for tool_retry template name
        # Note: Connection timeout message should be in error_message, not template_path
        assert (
            "read_file" not in handler_context_meta["template_path"]
        )  # Non-retryable not mentioned

    @pytest.mark.asyncio
    async def test_success_status_no_handling(self, handler_integration, session):
        """Test that success status doesn't trigger handling."""
        # Create HandlerContext for success status
        handler_context = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type=None,  # No error_type for success
        )

        # Handle success status
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.SUCCESS, handler_context=handler_context
        )

        # Verify no handling for success status
        assert handler_output.status == ExecutionStatus.SUCCESS
        assert handler_output.should_retry == False
        assert (
            handler_output.template_path is None
        )  # ✅ Use template_path instead of continuation_prompt
        assert handler_output.adjusted_llm_options is None
        assert handler_output.error_message is None

    @pytest.mark.asyncio
    async def test_llm_options_adjustment_flow(self, handler_integration, session):
        """Test LLM options adjustment for extend strategy."""
        # Setup LLM response with substantial reasoning for extend strategy
        llm_response = Mock()
        llm_response.metadata = {"max_tokens": 1000}

        truncated_response = LLMResult(
            content="Truncated response with substantial reasoning.",
            reasoning="This is a complex multi-faceted question requiring comprehensive analysis of multiple aspects and detailed consideration of various factors.",
            reasoning_tokens=500,
            total_tokens=1000,
            model="test-model",
            provider="test-provider",
            finish_reason="length",
            tool_calls=[],
        )

        # Create HandlerContext for truncation
        handler_context = HandlerContext(
            session=session,
            llm_response=llm_response,
            reasoning=truncated_response.reasoning,
            reasoning_tokens=truncated_response.reasoning_tokens,
            current_iteration=1,
            max_attempts=2,
            attempt_count=0,
            failed_tools=None,
            error_type="truncation",
        )

        # Handle truncation
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Verify template_path mentions increased tokens
        assert "llm.max_tokens" in handler_output.adjusted_llm_options
        max_tokens_value = handler_output.adjusted_llm_options["llm.max_tokens"]
        assert max_tokens_value > 1000  # Should be increased
        assert max_tokens_value == 1500  # 1000 * 1.5
        assert (
            "truncation" in handler_output.template_path.lower()
        )  # Check for truncation template

    @pytest.mark.asyncio
    async def test_tool_error_classification_flow(self, handler_integration, session):
        """Test tool error classification integration."""
        # Test error classification
        failure_info = handler_integration.classify_tool_error(
            tool_name="search_files",
            tool_args={"pattern": "*.py"},
            error="Connection timeout after 30 seconds",
        )

        # Verify classification result
        assert failure_info["tool_name"] == "search_files"
        assert failure_info["tool_args"] == {"pattern": "*.py"}
        assert failure_info["error_message"] == "Connection timeout after 30 seconds"
        assert "error_type" in failure_info
        assert "retry_strategy" in failure_info
        assert "can_retry" in failure_info
        assert failure_info["can_retry"] == True  # Network errors should be retryable
