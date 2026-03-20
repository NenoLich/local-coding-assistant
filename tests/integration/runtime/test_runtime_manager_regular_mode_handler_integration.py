"""
Integration tests for RuntimeManager regular mode handler integration.

Tests the interaction between RuntimeManager and HandlerIntegration in regular mode,
specifically covering the code block that handles partial responses (lines 485-521).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from local_coding_assistant.agent.llm import LLMResult
from local_coding_assistant.runtime.events import EventType
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerOutput,
)
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.runtime.session import SessionState


class TestRuntimeManagerRegularModeHandlerIntegration:
    """Integration tests for RuntimeManager regular mode with handler integration."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        config_manager = MagicMock()
        config_manager.global_config = AppConfig(
            llm=LLMConfig(),
            runtime=RuntimeConfig(agent_mode="no_agent", persistent_sessions=False),
        )
        config_manager.load_global_config = MagicMock(
            return_value=config_manager.global_config
        )
        return config_manager

    @pytest.fixture
    def mock_llm_service_partial_response(self):
        """Mock LLM service that returns a partial response on first call."""
        llm_service = MagicMock()
        llm_service.generate = AsyncMock(
            return_value=LLMResult(
                content="This is a partial response that was truncated due to token limits.",
                reasoning="This is reasoning that explains the partial response.",
                reasoning_tokens=500,
                total_tokens=1000,
                model="test-model",
                provider="test-provider",
                finish_reason="length",
                tool_calls=[],
                metadata={"max_tokens": 1000},
            )
        )
        return llm_service

    @pytest.fixture
    def mock_tool_manager(self):
        """Mock tool manager."""
        tool_manager = MagicMock()
        tool_manager.execute_async = AsyncMock(return_value=MagicMock())
        return tool_manager

    @pytest.fixture
    def mock_handler_output_continue(self):
        """Mock handler output that should continue execution."""
        return HandlerOutput(
            status=ExecutionStatus.PARTIAL,
            should_retry=True,
            template_path="truncation.jinja2",
            adjusted_llm_options={"llm.max_tokens": 1500},
            handler_context={"retry_strategy": "extend_tokens"},
            error_message="Response truncated, extending token limit",
        )

    @pytest.fixture
    def mock_handler_output_fail(self):
        """Mock handler output that should fail execution."""
        return HandlerOutput(
            status=ExecutionStatus.FAILED,
            should_retry=False,
            template_path=None,
            adjusted_llm_options=None,
            handler_context=None,
            error_message="Maximum attempts exceeded",
        )

    @pytest.fixture
    def runtime_manager(
        self, mock_config_manager, mock_llm_service_partial_response, mock_tool_manager
    ):
        """Create RuntimeManager with mocked dependencies."""
        return RuntimeManager(
            config_manager=mock_config_manager,
            llm_service=mock_llm_service_partial_response,
            tool_manager=mock_tool_manager,
        )

    @pytest.mark.asyncio
    async def test_handler_integration_continue_execution(
        self, mock_handler_output_continue
    ):
        """Test that handler integration continues execution when should_continue returns True."""
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        mock_handler_integration = MagicMock()
        mock_handler_integration.handle_partial_response = AsyncMock(
            return_value=mock_handler_output_continue
        )
        mock_handler_integration.should_continue_execution = MagicMock(
            return_value=True
        )
        mock_handler_integration.get_adjusted_llm_options = MagicMock(
            return_value={"max_tokens": 1500}
        )

        config_manager = MagicMock()
        config_manager.global_config = AppConfig(
            llm=LLMConfig(),
            runtime=RuntimeConfig(agent_mode="no_agent", persistent_sessions=False),
        )
        config_manager.load_global_config = MagicMock(
            return_value=config_manager.global_config
        )

        llm_service = MagicMock()
        llm_service.generate = AsyncMock(
            side_effect=[
                # First call: partial response
                LLMResult(
                    content="This is a partial response that was truncated due to token limits.",
                    reasoning="This is reasoning that explains the partial response.",
                    reasoning_tokens=500,
                    total_tokens=1000,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="length",
                    tool_calls=[],
                    metadata={"max_tokens": 1000},
                ),
                # Second call: success response
                LLMResult(
                    content="This is a complete response after adjusting token limits.",
                    reasoning="This is complete reasoning.",
                    reasoning_tokens=200,
                    total_tokens=800,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="stop",
                    tool_calls=[],
                    metadata={"max_tokens": 1500},
                ),
            ]
        )

        tool_manager = MagicMock()
        tool_manager.execute_async = AsyncMock(return_value=MagicMock())

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_service=llm_service,
            tool_manager=tool_manager,
        )
        # Replace the handler integration with our mock
        runtime_manager._handler_integration = mock_handler_integration

        # Run regular mode - this should trigger the handler integration code path
        report = None
        async for event in runtime_manager.orchestrate(
            "Test query that triggers partial response"
        ):
            if event.type == EventType.TURN_COMPLETE:
                report = event.data["report"]
                break

        # Verify the report indicates the execution continued and eventually succeeded
        assert report.status == "success"
        assert report.mode == "regular"
        assert "complete response" in report.final_answer.lower()

        # Verify handler integration was called correctly
        mock_handler_integration.handle_partial_response.assert_called_once()
        call_args = mock_handler_integration.handle_partial_response.call_args

        # Verify HandlerContext was created correctly
        handler_context = call_args[1]["handler_context"]
        assert isinstance(handler_context, HandlerContext)
        assert handler_context.error_type.value == "truncation"
        assert (
            handler_context.raw_response
            == "This is a partial response that was truncated due to token limits."
        )
        assert (
            handler_context.reasoning
            == "This is reasoning that explains the partial response."
        )
        assert handler_context.reasoning_tokens == 500
        assert handler_context.max_attempts == 2
        assert handler_context.attempt_count == 1

        # Verify should_continue_execution was called
        mock_handler_integration.should_continue_execution.assert_called_once_with(
            mock_handler_output_continue
        )

        # Verify get_adjusted_llm_options was called
        mock_handler_integration.get_adjusted_llm_options.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_integration_fail_execution(self, mock_handler_output_fail):
        """Test that handler integration fails execution when should_continue returns False."""
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        mock_handler_integration = MagicMock()
        mock_handler_integration.handle_partial_response = AsyncMock(
            return_value=mock_handler_output_fail
        )
        mock_handler_integration.should_continue_execution = MagicMock(
            return_value=False
        )

        config_manager = MagicMock()
        config_manager.global_config = AppConfig(
            llm=LLMConfig(),
            runtime=RuntimeConfig(agent_mode="no_agent", persistent_sessions=False),
        )
        config_manager.load_global_config = MagicMock(
            return_value=config_manager.global_config
        )

        llm_service = MagicMock()
        llm_service.generate = AsyncMock(
            return_value=LLMResult(
                content="This is a partial response that was truncated due to token limits.",
                reasoning="This is reasoning that explains the partial response.",
                reasoning_tokens=500,
                total_tokens=1000,
                model="test-model",
                provider="test-provider",
                finish_reason="length",
                tool_calls=[],
                metadata={"max_tokens": 1000},
            )
        )

        tool_manager = MagicMock()
        tool_manager.execute_async = AsyncMock(return_value=MagicMock())

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_service=llm_service,
            tool_manager=tool_manager,
        )
        # Replace the handler integration with our mock
        runtime_manager._handler_integration = mock_handler_integration

        # Run regular mode - this should trigger the handler integration code path and fail
        report = None
        async for event in runtime_manager.orchestrate(
            "Test query that triggers partial response"
        ):
            if event.type == EventType.TURN_COMPLETE:
                report = event.data["report"]
                break

        # Verify the report indicates failure
        assert report.status == "failed"
        assert report.mode == "regular"

        # Verify handler integration was called correctly
        mock_handler_integration.handle_partial_response.assert_called_once()
        call_args = mock_handler_integration.handle_partial_response.call_args

        # Verify HandlerContext was created correctly
        handler_context = call_args[1]["handler_context"]
        assert isinstance(handler_context, HandlerContext)
        assert handler_context.error_type.value == "truncation"

        # Verify should_continue_execution was called and returned False
        mock_handler_integration.should_continue_execution.assert_called_once_with(
            mock_handler_output_fail
        )

    @pytest.mark.asyncio
    async def test_handler_context_creation(self, mock_handler_output_continue):
        """Test that HandlerContext is created with correct data from LLM response."""
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        mock_handler_integration = MagicMock()
        mock_handler_integration.handle_partial_response = AsyncMock(
            return_value=mock_handler_output_continue
        )
        mock_handler_integration.should_continue_execution = MagicMock(
            return_value=True
        )
        mock_handler_integration.get_adjusted_llm_options = MagicMock(
            return_value={"max_tokens": 1500}
        )

        config_manager = MagicMock()
        config_manager.global_config = AppConfig(
            llm=LLMConfig(),
            runtime=RuntimeConfig(agent_mode="no_agent", persistent_sessions=False),
        )
        config_manager.load_global_config = MagicMock(
            return_value=config_manager.global_config
        )

        llm_service = MagicMock()
        llm_service.generate = AsyncMock(
            side_effect=[
                # First call: partial response
                LLMResult(
                    content="This is a partial response that was truncated due to token limits.",
                    reasoning="This is reasoning that explains the partial response.",
                    reasoning_tokens=500,
                    total_tokens=1000,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="length",
                    tool_calls=[],
                    metadata={"max_tokens": 1000},
                ),
                # Second call: success response
                LLMResult(
                    content="This is a complete response after adjusting token limits.",
                    reasoning="This is complete reasoning.",
                    reasoning_tokens=200,
                    total_tokens=800,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="stop",
                    tool_calls=[],
                    metadata={"max_tokens": 1500},
                ),
            ]
        )

        tool_manager = MagicMock()
        tool_manager.execute_async = AsyncMock(return_value=MagicMock())

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_service=llm_service,
            tool_manager=tool_manager,
        )
        # Replace the handler integration with our mock
        runtime_manager._handler_integration = mock_handler_integration

        # Run the orchestration to trigger handler context creation
        report = None
        async for event in runtime_manager.orchestrate("Test query"):
            if event.type == EventType.TURN_COMPLETE:
                report = event.data["report"]
                break

        # Verify the handler integration was called
        mock_handler_integration.handle_partial_response.assert_called_once()
        call_args = mock_handler_integration.handle_partial_response.call_args

        # Extract the HandlerContext that was passed
        handler_context = call_args[1]["handler_context"]

        # Verify all expected fields are populated
        assert handler_context.session is not None
        assert isinstance(handler_context.session, SessionState)
        assert handler_context.llm_response is None  # Not set in this code path
        assert (
            handler_context.reasoning
            == "This is reasoning that explains the partial response."
        )
        assert handler_context.reasoning_tokens == 500
        assert handler_context.max_attempts == 2
        assert handler_context.attempt_count == 1  # First attempt
        assert handler_context.error_type.value == "truncation"
        assert (
            handler_context.error_message == "Unknown error"
        )  # Default when not in metadata
        assert (
            handler_context.raw_response
            == "This is a partial response that was truncated due to token limits."
        )
        assert handler_context.raw_tool_calls == []  # No tool calls in the mock

    @pytest.mark.asyncio
    async def test_llm_options_adjustment_integration(
        self, mock_handler_output_continue
    ):
        """Test that LLM options are adjusted when handler integration provides adjustments."""
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        mock_handler_integration = MagicMock()
        mock_handler_integration.handle_partial_response = AsyncMock(
            return_value=mock_handler_output_continue
        )
        mock_handler_integration.should_continue_execution = MagicMock(
            return_value=True
        )
        mock_handler_integration.get_adjusted_llm_options = MagicMock(
            return_value={"max_tokens": 1500}
        )

        config_manager = MagicMock()
        config_manager.global_config = AppConfig(
            llm=LLMConfig(),
            runtime=RuntimeConfig(agent_mode="no_agent", persistent_sessions=False),
        )
        config_manager.load_global_config = MagicMock(
            return_value=config_manager.global_config
        )

        llm_service = MagicMock()
        llm_service.generate = AsyncMock(
            side_effect=[
                # First call: partial response
                LLMResult(
                    content="This is a partial response that was truncated due to token limits.",
                    reasoning="This is reasoning that explains the partial response.",
                    reasoning_tokens=500,
                    total_tokens=1000,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="length",
                    tool_calls=[],
                    metadata={"max_tokens": 1000},
                ),
                # Second call: success response
                LLMResult(
                    content="This is a complete response after adjusting token limits.",
                    reasoning="This is complete reasoning.",
                    reasoning_tokens=200,
                    total_tokens=800,
                    model="test-model",
                    provider="test-provider",
                    finish_reason="stop",
                    tool_calls=[],
                    metadata={"max_tokens": 1500},
                ),
            ]
        )

        tool_manager = MagicMock()
        tool_manager.execute_async = AsyncMock(return_value=MagicMock())

        runtime_manager = RuntimeManager(
            config_manager=config_manager,
            llm_service=llm_service,
            tool_manager=tool_manager,
        )
        # Replace the handler integration with our mock
        runtime_manager._handler_integration = mock_handler_integration

        # Run the orchestration
        report = None
        async for event in runtime_manager.orchestrate("Test query"):
            if event.type == EventType.TURN_COMPLETE:
                report = event.data["report"]
                break

        # Verify get_adjusted_llm_options was called
        mock_handler_integration.get_adjusted_llm_options.assert_called_once()
        call_args = mock_handler_integration.get_adjusted_llm_options.call_args

        # Verify the base options were passed correctly
        base_options = call_args[1]["base_options"]  # Keyword argument
        assert "model" in base_options
        assert "temperature" in base_options
        assert "max_tokens" in base_options

        # Verify handler output was passed
        handler_output = call_args[1]["handler_output"]  # Keyword argument
        assert handler_output.adjusted_llm_options == {"llm.max_tokens": 1500}
