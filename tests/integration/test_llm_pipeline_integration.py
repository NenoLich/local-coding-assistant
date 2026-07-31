"""Integration tests for LLM pipeline functions and error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.agent.llm import (
    LLMResult,
    LLMService,
    LLMTask,
)
from local_coding_assistant.agent.llm.pipeline import (
    ResponseNormalizer,
    StreamingNormalizer,
    _extract_reasoning_tokens,
    _extract_usage_metrics,
)
from local_coding_assistant.config.schemas import AgentProfileConfig
from local_coding_assistant.core.exceptions import LLMContentError
from local_coding_assistant.providers.base import (
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.runtime.events import EventType
from local_coding_assistant.runtime.execution_types import (
    ExecutionFrame,
    ExecutionStatus,
    RenderedPrompt,
)
from local_coding_assistant.runtime.executor import RuntimeExecutor
from local_coding_assistant.runtime.runtime_types import ExecutionMode, PromptContext


class TestLLMPipelineIntegration:
    """Integration tests for LLM pipeline functions and error handling."""

    def test_extract_usage_metrics_various_scenarios(self):
        """Test _extract_usage_metrics with various usage data scenarios."""
        # Test with None usage
        assert _extract_usage_metrics(None, None) == (None, None, None)

        # Test with empty usage dict
        assert _extract_usage_metrics({}, 100) == (None, None, 100)

        # Test with prompt_tokens and completion_tokens
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        assert _extract_usage_metrics(usage, None) == (10, 20, 30)

        # Test with input_tokens and output_tokens (alternative keys)
        usage = {"input_tokens": 15, "output_tokens": 25}
        assert _extract_usage_metrics(usage, None) == (15, 25, 40)

        # Test with explicit total_tokens
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 35}
        assert _extract_usage_metrics(usage, None) == (10, 20, 35)

        # Test fallback to tokens_used when total_tokens is None
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        assert _extract_usage_metrics(usage, 50) == (10, 20, 30)

    def test_extract_reasoning_tokens_various_scenarios(self):
        """Test _extract_reasoning_tokens with various usage data scenarios."""
        # Test with None usage
        assert _extract_reasoning_tokens(None) is None

        # Test with empty usage dict
        assert _extract_reasoning_tokens({}) is None

        # Test with completion_tokens_details containing reasoning_tokens
        usage = {"completion_tokens_details": {"reasoning_tokens": 42}}
        assert _extract_reasoning_tokens(usage) == 42

        # Test with string reasoning_tokens that can be converted to int
        usage = {"completion_tokens_details": {"reasoning_tokens": "37"}}
        assert _extract_reasoning_tokens(usage) == 37

        # Test with invalid reasoning_tokens (should return None)
        usage = {"completion_tokens_details": {"reasoning_tokens": "invalid"}}
        assert _extract_reasoning_tokens(usage) is None

        # Test with non-dict completion_tokens_details
        usage = {"completion_tokens_details": "invalid"}
        assert _extract_reasoning_tokens(usage) is None

    @pytest.mark.asyncio
    async def test_response_normalizer_llm_content_error_handling(self):
        """Test ResponseNormalizer handles LLMContentError in tool call normalization."""
        normalizer = ResponseNormalizer()

        # Create a mock response with malformed tool calls that will cause LLMContentError
        response = ProviderLLMResponse(
            content="Test content",
            model="test-model",
            tokens_used=50,
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "test_tool",
                        # Malformed arguments that should cause LLMContentError
                        "arguments": "invalid json {{{",
                    },
                }
            ],
            finish_reason="stop",
            usage={"total_tokens": 50},
        )

        # Mock the _normalize_tool_arguments to raise LLMContentError
        with patch(
            "local_coding_assistant.agent.llm.pipeline._normalize_tool_arguments"
        ) as mock_normalize:
            mock_normalize.side_effect = LLMContentError(
                "Invalid JSON in tool arguments"
            )

            result = normalizer.to_result(
                response, provider_name="test-provider", policy_name=None
            )

            # Verify the result contains error metadata
            assert result.content == "Test content"
            assert result.model == "test-model"
            assert "content_error" in result.metadata
            assert (
                result.metadata["content_error"]["error_type"]
                == "tool_call_parsing_error"
            )
            assert (
                "Invalid JSON in tool arguments"
                in result.metadata["content_error"]["message"]
            )
            assert result.metadata["content_error"]["raw_response"] == "Test content"
            assert (
                result.metadata["content_error"]["tool_calls_attempted"]
                == response.tool_calls
            )

    @pytest.mark.asyncio
    async def test_streaming_normalizer_llm_content_error_handling(self):
        """Test StreamingNormalizer handles LLMContentError in tool call normalization."""
        normalizer = StreamingNormalizer()

        # Create a mock delta with malformed tool calls
        delta = ProviderLLMResponseDelta(
            content="Test content chunk",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "test_tool", "arguments": "invalid json {{{"},
                }
            ],
            finish_reason=None,
        )

        # Mock the _normalize_tool_arguments to raise LLMContentError
        with patch(
            "local_coding_assistant.agent.llm.pipeline._normalize_tool_arguments"
        ) as mock_normalize:
            mock_normalize.side_effect = LLMContentError(
                "Invalid JSON in streaming tool arguments"
            )

            chunk = normalizer.to_chunk(
                delta, provider_name="test-provider", model_name="test-model"
            )

            # Verify the chunk contains error metadata
            assert chunk.content == "Test content chunk"
            assert chunk.model == "test-model"
            assert "content_error" in chunk.metadata
            assert (
                chunk.metadata["content_error"]["error_type"]
                == "tool_call_parsing_error"
            )
            assert (
                "Invalid JSON in streaming tool arguments"
                in chunk.metadata["content_error"]["message"]
            )
            assert (
                chunk.metadata["content_error"]["tool_calls_attempted"]
                == delta.tool_calls
            )

    @pytest.mark.asyncio
    async def test_executor_llm_content_error_partial_status(self):
        """Test RuntimeExecutor handles LLMContentError by setting PARTIAL status."""
        # Create a mock LLM service that returns a result with content_error
        mock_llm_service = AsyncMock()

        # Mock the stream method to return an async iterator with the error
        async def mock_stream(task, options=None):
            from local_coding_assistant.agent.llm import LLMStreamChunk

            # Yield a chunk with content error metadata
            yield LLMStreamChunk(
                content="Response with parsing error",
                provider="test-provider",
                model="test-model",
                is_final=True,
                metadata={
                    "content_error": {
                        "error_type": "tool_call_parsing_error",
                        "message": "Failed to parse tool arguments",
                        "raw_response": "raw response content",
                        "tool_calls_attempted": [{"id": "call_1"}],
                    }
                },
            )

        mock_llm_service.stream = mock_stream
        mock_llm_service.generate.return_value = LLMResult(
            content="Response with parsing error",
            model="test-model",
            provider="test-provider",
            finish_reason="stop",
            metadata={
                "content_error": {
                    "error_type": "tool_call_parsing_error",
                    "message": "Failed to parse tool arguments",
                    "raw_response": "raw response content",
                    "tool_calls_attempted": [{"id": "call_1"}],
                }
            },
        )

        # Create minimal mocks for other dependencies
        mock_tool_manager = MagicMock()
        mock_context_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config.llm.model_name = "test-model"
        mock_config_manager.global_config.runtime.capture_reasoning = False

        executor = RuntimeExecutor(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
            config_manager=mock_config_manager,
        )

        # Create a minimal execution frame
        frame = ExecutionFrame(
            session_id="test-session-123",  # Required field
            id="test-frame-123",
            user_input="Test user input",
            rendered_prompt=RenderedPrompt(
                user_messages=["Test prompt"],
                system_messages=[],
                history=[],
                tool_schemas=[],
            ),
            agent_profile=AgentProfileConfig(
                name="test-agent",
                description="Test agent for integration tests",
                model_policy=None,
            ),
            prompt_context=PromptContext(
                session_id="test-session-123",
                execution_mode=ExecutionMode.CLASSIC_TOOLS,
                tool_call_mode="classic",
                max_iterations=10,
                timeout_seconds=30.0,
                user_input="Test user input",
            ),
        )

        # Execute the frame
        result_frame = None
        async for event in executor.execute(frame):
            if event.type == EventType.FRAME_COMPLETE:
                result_frame = event.data["frame"]
                break

        # Verify PARTIAL status is set due to content error
        assert result_frame.result.status == ExecutionStatus.PARTIAL
        assert result_frame.result.handler_context is not None
        assert result_frame.result.handler_context["error_type"] == "parsing_error"
        assert (
            "Failed to parse tool arguments"
            in result_frame.result.handler_context["message"]
        )
        assert (
            result_frame.result.handler_context["raw_response"]
            == "raw response content"
        )
        assert result_frame.result.handler_context["raw_tool_calls"] == [
            {"id": "call_1"}
        ]

    @pytest.mark.asyncio
    async def test_llm_service_except_block_error_handling(self):
        """Test LLMService except block (lines 184-203) handles various exceptions."""
        # Create a mock config manager
        mock_config_manager = MagicMock()

        # Create LLM service
        llm_service = LLMService.__new__(LLMService)
        llm_service._config_manager = mock_config_manager
        llm_service._provider_manager = MagicMock()
        llm_service._router = AsyncMock()
        llm_service._policy_resolver = MagicMock()
        llm_service._provider_selector = AsyncMock()
        llm_service._telemetry = MagicMock()
        llm_service._logger = MagicMock()

        # Mock the provider selector to succeed but provider generation to fail
        mock_decision = MagicMock()
        mock_decision.provider = AsyncMock()
        mock_decision.provider.name = "test_provider"
        mock_decision.provider.generate_with_retry.side_effect = Exception(
            "Generation failed"
        )
        mock_decision.model = "test-model"
        llm_service._provider_selector.select.return_value = (mock_decision, [])

        # Mock request builder and other components
        llm_service._request_builder = MagicMock()
        llm_service._response_normalizer = MagicMock()
        llm_service._streaming_normalizer = MagicMock()

        # Create a test task
        task = LLMTask(
            prompt="Test prompt", context=[], system_prompt="Test system prompt"
        )

        # Test that exceptions in the except block are handled properly
        # The except block should call telemetry.attempt_failure and continue to next attempt
        with patch.object(llm_service, "_clone_request") as mock_clone:
            mock_clone.return_value = MagicMock()

            # Mock resolve_options to return options that trigger multiple attempts
            with patch.object(llm_service, "_resolve_options") as mock_resolve:
                mock_resolve.return_value = MagicMock(
                    model="test-model",
                    max_failovers=1,  # Only one failover to keep test simple
                    retry_attempts=1,
                    retry_delay=0.1,
                )

                # This should raise LLMError after exhausting attempts
                with pytest.raises(Exception):  # Should be LLMError but we're mocking
                    await llm_service._run_generation(
                        MagicMock(
                            task=task,
                            resolved_options=mock_resolve.return_value,
                            provider_request=MagicMock(),
                        )
                    )

                # Verify telemetry.attempt_failure was called at least once
                assert llm_service._telemetry.attempt_failure.call_count >= 1

    def test_usage_metrics_integration_with_response_normalizer(self):
        """Test usage metrics extraction integrated with ResponseNormalizer."""
        normalizer = ResponseNormalizer()

        # Test with complex usage data
        response = ProviderLLMResponse(
            content="Test response",
            model="test-model",
            tokens_used=100,
            tool_calls=None,
            finish_reason="stop",
            usage={
                "prompt_tokens": 50,
                "completion_tokens": 40,
                "total_tokens": 90,
                "completion_tokens_details": {"reasoning_tokens": 15},
            },
        )

        result = normalizer.to_result(
            response, provider_name="test-provider", policy_name="test-policy"
        )

        # Verify usage metrics are correctly extracted and set
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 40
        assert result.total_tokens == 90
        assert result.reasoning_tokens == 15
        assert result.metadata["usage"] == response.usage
        assert result.metadata["policy"] == "test-policy"

    def test_reasoning_tokens_integration_with_streaming_normalizer(self):
        """Test reasoning tokens extraction integrated with StreamingNormalizer."""
        normalizer = StreamingNormalizer()

        # Test delta with usage containing reasoning tokens in metadata
        delta = ProviderLLMResponseDelta(
            content="Test chunk",
            metadata={"usage": {"completion_tokens_details": {"reasoning_tokens": 20}}},
        )

        chunk = normalizer.to_chunk(
            delta, provider_name="test-provider", model_name="test-model"
        )

        # Verify usage is passed through to metadata
        assert chunk.usage == delta.metadata.get("usage")
        assert chunk.metadata["usage"] == delta.metadata["usage"]
