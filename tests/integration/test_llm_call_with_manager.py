from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.agent.llm import (
    LLMService,
    LLMTask,
    LLMResult,
)
from local_coding_assistant.agent.llm.routing import ProviderSelector

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.config.schemas import ProviderStatus, LLMConfig
from local_coding_assistant.providers.base import ProviderLLMResponse


def make_test_llm_service():
    # Create a mock provider manager for testing
    mock_provider_manager = MagicMock()
    mock_provider_manager.list_providers.return_value = ["test_provider"]

    # Create a mock provider
    mock_provider = AsyncMock()
    mock_provider.name = "test_provider"
    mock_provider.generate_with_retry = AsyncMock(
        return_value=ProviderLLMResponse(
            content="Test response from provider",
            model="test-model",
            tokens_used=50,
            tool_calls=None,
            finish_reason="stop",
        )
    )

    mock_provider_manager.get_provider.return_value = mock_provider

    # Create LLM service with mocked provider system
    with patch("local_coding_assistant.providers.ProviderManager") as mock_pm_class:
        mock_pm_class.return_value = mock_provider_manager

        # Create a mock config manager
        mock_config_manager = MagicMock(spec=ConfigManager)
        mock_config_manager.global_config = {
            "llm": {
                "default_provider": "test_provider",
                "providers": {"test_provider": {"type": "test_provider"}},
            }
        }

        llm = LLMService.__new__(LLMService)
        llm._provider_manager = mock_provider_manager
        llm._config_manager = mock_config_manager
        llm._router = MagicMock()
        # Configure the mark_provider_success method to avoid coroutine warning
        llm._router.mark_provider_success = MagicMock()

        return llm


class TestLLMServiceIntegration:
    """Integration tests for LLMService with real async behavior."""

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self):
        """Test complete request-response cycle."""
        # Create mock provider for testing
        mock_provider = AsyncMock()
        mock_provider.name = "test_provider"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Integration test response",
                model="test-model",
                tokens_used=50,
                tool_calls=None,
                finish_reason="stop",
                usage={},
            )
        )

        # Create mock router
        mock_router = AsyncMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "test-model")
        )
        # Sync methods
        mock_router.mark_provider_success = MagicMock()
        mock_router.mark_provider_failure = MagicMock()

        # Create a mock config manager
        mock_config_manager = MagicMock(spec=ConfigManager)
        mock_config_manager.global_config.llm = LLMConfig(
            providers=[ProviderStatus(name="test_provider")]
        )

        llm = LLMService.__new__(LLMService)
        llm._router = mock_router
        llm._provider_manager = MagicMock()
        llm._config_manager = mock_config_manager
        llm._request_builder = MagicMock()
        llm._response_normalizer = MagicMock()
        llm._policy_resolver = MagicMock()
        llm._provider_selector = ProviderSelector(mock_router)
        llm._telemetry = MagicMock()
        llm._logger = MagicMock()

        # Configure mocks
        mock_request = MagicMock()
        llm._request_builder.build.return_value = mock_request

        expected_response = LLMResult(
            content="Integration test response",
            model="test-model",
            provider="test_provider",
            total_tokens=50,
        )
        llm._response_normalizer.to_result.return_value = expected_response

        request = LLMTask(
            prompt="Integration test prompt",
            context=[{"test": "data"}],
            system_prompt="You are a test assistant",
        )

        response = await llm.generate(request)

        # Verify response structure
        assert isinstance(response, LLMResult)
        assert response.content == "Integration test response"
        assert response.model == "test-model"
        assert response.total_tokens == 50

        # Verify provider was called correctly
        mock_provider.generate_with_retry.assert_called_once()
        mock_router.get_provider_for_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_tool_calls_are_executed(self):
        """Test that LLM-initiated tool calls are properly handled."""
        llm = make_test_llm_service()

        # Mock LLM response with tool call
        original_generate = llm.generate

        async def mock_generate(request):
            if "sum" in request.prompt:
                return LLMResult(
                    content="I'll calculate that for you.",
                    model="fake-model",
                    provider="test-provider",
                    total_tokens=50,
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "sum",
                                "arguments": '{"a": 10, "b": 15}',
                            },
                        }
                    ],
                )
            return await original_generate(request)

        llm.generate = mock_generate

        # Test that LLM can generate tool calls
        request = LLMTask(
            prompt="Calculate 10 + 15 using the sum tool",
            context=[],
            system_prompt="You are a helpful assistant",
        )

        response = await llm.generate(request)

        # Should have generated tool call
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "sum"
        assert response.content == "I'll calculate that for you."
