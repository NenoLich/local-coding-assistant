"""Integration tests for LLM pipeline functions and error handling."""

from unittest.mock import patch

import pytest

from local_coding_assistant.agent.llm.pipeline import (
    ResponseNormalizer,
    StreamingNormalizer,
    _extract_reasoning_tokens,
    _extract_usage_metrics,
)
from local_coding_assistant.core.exceptions import LLMContentError
from local_coding_assistant.providers.base import (
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)


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
