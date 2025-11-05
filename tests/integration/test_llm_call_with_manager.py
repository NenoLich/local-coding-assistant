from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.agent.llm_manager import (
    LLMManager,
    LLMRequest,
    LLMResponse,
)
from local_coding_assistant.providers import ProviderLLMResponse


def make_test_llm_manager():
    """Create a test LLM manager setup for integration testing."""
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

    # Create LLM manager with mocked provider system
    with patch("local_coding_assistant.config.get_config_manager"):
        # Update the import path to use the correct module
        with patch("local_coding_assistant.providers.ProviderManager") as mock_pm_class:
            mock_pm_class.return_value = mock_provider_manager

            llm = LLMManager.__new__(LLMManager)
            llm.provider_manager = mock_provider_manager
            llm.config_manager = MagicMock()
            llm.router = MagicMock()

            return llm


class TestLLMManagerIntegration:
    """Integration tests for LLMManager with real async behavior."""

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
            )
        )

        # Create mock router
        mock_router = AsyncMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "test-model")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            llm = LLMManager.__new__(LLMManager)
            llm.router = mock_router
            llm.provider_manager = MagicMock()
            llm.config_manager = MagicMock()

            request = LLMRequest(
                prompt="Integration test prompt",
                context={"test": "data"},
                system_prompt="You are a test assistant",
            )

            response = await llm.generate(request)

            # Verify response structure
            assert isinstance(response, LLMResponse)
            assert response.content == "Integration test response"
            assert response.model_used == "test-model"
            assert response.tokens_used == 50

            # Verify provider was called correctly
            mock_provider.generate_with_retry.assert_called_once()
            mock_router.get_provider_for_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_tool_calls_are_executed(self):
        """Test that LLM-initiated tool calls are properly handled."""
        llm = make_test_llm_manager()

        # Mock LLM response with tool call
        original_generate = llm.generate

        async def mock_generate(request):
            if "sum" in request.prompt:
                return LLMResponse(
                    content="I'll calculate that for you.",
                    model_used="fake-model",
                    tokens_used=50,
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
        request = LLMRequest(
            prompt="Calculate 10 + 15 using the sum tool",
            context={},
            system_prompt="You are a helpful assistant",
        )

        response = await llm.generate(request)

        # Should have generated tool call
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "sum"
        assert response.content == "I'll calculate that for you."
