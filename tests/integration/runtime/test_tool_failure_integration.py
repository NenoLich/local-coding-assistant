"""
Integration tests for tool failure handling from tool manager through entire system.
"""

import pytest

from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
)
from tests.integration.runtime.conftest import (
    MockToolManager,
    assert_handler_output,
    assert_session_continuation,
    create_test_session,
)
from tests.integration.runtime.mock_tool_responses import MockToolResponse


class TestToolFailureIntegration:
    """Test tool failure handling integration across all components."""

    @pytest.mark.asyncio
    async def test_network_error_retry_with_backoff(self):
        """Test network errors trigger retry with backoff strategy."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "search_files": [
                    MockToolResponse.network_error_response(
                        error_message="Connection timeout after 30 seconds"
                    ),
                    MockToolResponse.success_response(),  # Retry should succeed
                ]
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data (simulating executor collection)
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

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
            expected_strategy="retry",
        )

        # Verify retry prompt contains tool details
        assert handler_output.template_path is not None, "Should have template path"

        # Verify session updated
        assert_session_continuation(session)

    @pytest.mark.asyncio
    async def test_tool_not_found_immediate_retry(self):
        """Test tool not found errors trigger immediate retry."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "search_files": [
                    MockToolResponse.tool_not_found_response(
                        error_message="Tool 'search_files' is not exposed"
                    ),
                    MockToolResponse.success_response(),  # Retry should succeed
                ]
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data
        failed_tools = [
            {
                "tool_name": "search_files",
                "tool_args": {"pattern": "*.py"},
                "error_message": "Tool 'search_files' is not exposed",
                "error_type": "TOOL_NOT_FOUND",
                "retry_strategy": "RETRY_IMMEDIATELY",
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

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
            expected_strategy="retry",
        )

        # Verify retry prompt
        assert handler_output.template_path is not None, "Should have template path"

    @pytest.mark.asyncio
    async def test_syntax_error_no_retry(self):
        """Test syntax errors trigger no retry strategy."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "search_files": [
                    MockToolResponse.syntax_error_response(
                        error_message="Invalid argument: 'pattern' must be a string"
                    )
                    # No retry response expected
                ]
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data
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

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # For non-retryable failures, no handler context is applied to session

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.FAILED,  # Should be FAILED when no tools can retry
            should_retry=False,  # Should not continue for syntax errors
            has_continuation=False,  # No continuation when no retry possible
        )

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_backoff(self):
        """Test rate limit errors trigger retry with backoff."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "api_call": [
                    MockToolResponse.rate_limit_error_response(
                        error_message="Rate limit exceeded. Try again in 60 seconds"
                    ),
                    MockToolResponse.success_response(),  # Retry should succeed
                ]
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data
        failed_tools = [
            {
                "tool_name": "api_call",
                "tool_args": {"endpoint": "/api/data"},
                "error_message": "Rate limit exceeded. Try again in 60 seconds",
                "error_type": "RATE_LIMIT",
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

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,
            has_continuation=True,
            expected_strategy="retry",
        )

        # Verify retry prompt
        assert handler_output.template_path is not None, "Should have template path"

    @pytest.mark.asyncio
    async def test_multiple_tool_failures(self):
        """Test handling multiple tool failures in single response."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "search_files": [MockToolResponse.network_error_response()],
                "read_file": [MockToolResponse.permission_error_response()],
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data with multiple failures
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

        # Execute
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify
        assert_handler_output(
            handler_output,
            expected_status=ExecutionStatus.PARTIAL,
            should_retry=True,  # Should continue because at least one tool can retry
            has_continuation=True,
            expected_strategy="retry",  # Should use most permissive strategy
        )

        # Verify retry prompt contains retryable tool failures
        assert handler_output.template_path is not None, "Should have template path"

    @pytest.mark.asyncio
    async def test_tool_failure_retry_limit_enforcement(self):
        """Test that tool failure retries are limited to max_attempts."""
        # Setup
        mock_tool_manager = MockToolManager(
            {
                "search_files": [
                    MockToolResponse.network_error_response(),  # First failure
                    MockToolResponse.network_error_response(),  # Second failure
                    MockToolResponse.success_response(),  # Would be third but shouldn't reach
                ]
            }
        )

        session = create_test_session()
        handler_integration = HandlerIntegration()

        # Create failed tools data
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

        # First failure
        handler_context1 = HandlerContext(
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
            "error_type": "tool_failures",
            "template_path": handler_output1.template_path,
            "retry_strategy": getattr(handler_output1, "retry_strategy", None),
        }

        # Second failure (should stop due to max_attempts)
        handler_context2 = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=None,
            reasoning_tokens=None,
            current_iteration=2,
            max_attempts=2,
            attempt_count=2,  # Third attempt (exceeds max_attempts=2)
            failed_tools=failed_tools,
            error_type="tool_failures",
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
    async def test_tool_failure_session_state_management(self):
        """Test that session state is properly managed during tool failure handling."""
        # Setup
        mock_tool_manager = MockToolManager(
            {"search_files": [MockToolResponse.network_error_response()]}
        )

        session = create_test_session()
        initial_history_length = len(session.history)

        handler_integration = HandlerIntegration()

        # Create failed tools data
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

        # Execute tool failure handling
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
        handler_output = await handler_integration.handle_partial_response(
            execution_status=ExecutionStatus.PARTIAL, handler_context=handler_context
        )

        # Apply handler context to session metadata
        session.metadata["handler_context"] = {
            "error_type": "tool_failures",
            "template_path": handler_output.template_path,
            "retry_strategy": getattr(handler_output, "retry_strategy", None),
        }

        # Verify session state
        assert "handler_context" in session.metadata, (
            "Should have added handler context to metadata"
        )
        handler_context_meta = session.metadata["handler_context"]
        assert handler_context_meta["error_type"] == "tool_failures"
        assert handler_context_meta["template_path"] == handler_output.template_path
        assert "tool_retry.jinja2" in handler_context_meta["template_path"], (
            "Should have retry template"
        )
        assert "retry" in handler_context_meta["template_path"].lower(), (
            "Should indicate retry"
        )


if __name__ == "__main__":
    # Run a quick test to verify setup
    import asyncio

    async def quick_test():
        test = TestToolFailureIntegration()
        await test.test_network_error_retry_with_backoff()
        print("✅ Tool failure integration test setup working!")

    asyncio.run(quick_test())
