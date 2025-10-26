"""
Unit tests for LLMManager with provider system integration.
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest, LLMResponse
from local_coding_assistant.core.exceptions import LLMError
from local_coding_assistant.providers import (
    BaseProvider,
    ProviderError,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
    ProviderManager,
    ProviderRouter,
)


class TestLLMRequest:
    """Test LLMRequest functionality."""

    def test_basic_request_creation(self):
        """Test creating a basic LLMRequest."""
        request = LLMRequest(prompt="Test prompt")
        assert request.prompt == "Test prompt"
        assert request.context == {}
        assert request.system_prompt is None
        assert request.tools == []
        assert request.tool_outputs == {}

    def test_full_request_creation(self):
        """Test creating a full LLMRequest with all fields."""
        request = LLMRequest(
            prompt="Test prompt",
            context={"user": "test"},
            system_prompt="You are a helpful assistant",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_outputs={"test": "result"},
        )
        assert request.prompt == "Test prompt"
        assert request.context == {"user": "test"}
        assert request.system_prompt == "You are a helpful assistant"
        assert len(request.tools) == 1
        assert request.tool_outputs == {"test": "result"}

    def test_to_provider_request_conversion(self):
        """Test converting LLMRequest to ProviderLLMRequest."""
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="System prompt",
            context={"env": "test"},
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
        )

        provider_request = request.to_provider_request("gpt-4")

        assert provider_request.model == "gpt-4"
        assert provider_request.stream is False
        assert provider_request.temperature == 0.7
        assert provider_request.tools == request.tools

        # Check messages format
        messages = provider_request.messages
        assert len(messages) == 3  # system + context + user prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert "Context:" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Test prompt"


class TestLLMResponse:
    """Test LLMResponse functionality."""

    def test_basic_response_creation(self):
        """Test creating a basic LLMResponse."""
        response = LLMResponse(
            content="Test response",
            model_used="gpt-4",
            tokens_used=100,
            tool_calls=[{"id": "call_1", "type": "function"}],
        )
        assert response.content == "Test response"
        assert response.model_used == "gpt-4"
        assert response.tokens_used == 100
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

    def test_response_defaults(self):
        """Test LLMResponse default values."""
        response = LLMResponse(content="test", model_used="gpt-3.5")
        assert response.tokens_used is None
        assert response.tool_calls is None


class TestLLMManagerInitialization:
    """Test LLMManager initialization and setup."""

    def test_initialization_with_defaults(self):
        """Test LLMManager initialization with default parameters."""
        with patch(
            "local_coding_assistant.config.get_config_manager"
        ) as mock_get_config:
            mock_config_manager = MagicMock()
            mock_get_config.return_value = mock_config_manager

            manager = LLMManager()

            assert manager.config_manager == mock_config_manager
            assert isinstance(manager.provider_manager, ProviderManager)
            assert isinstance(manager.router, ProviderRouter)
            assert manager._cache_ttl == 30.0 * 60.0
            mock_get_config.assert_called_once()

    def test_initialization_with_custom_parameters(self):
        """Test LLMManager initialization with custom parameters."""
        mock_config_manager = MagicMock()
        mock_provider_manager = MagicMock()

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )

        assert manager.config_manager == mock_config_manager
        assert manager.provider_manager == mock_provider_manager
        assert isinstance(manager.router, ProviderRouter)

    def test_provider_system_initialization(self):
        """Test that provider system is properly initialized."""
        with patch(
            "local_coding_assistant.config.get_config_manager"
        ) as mock_get_config:
            mock_config_manager = MagicMock()
            mock_get_config.return_value = mock_config_manager

            # Mock the ProviderManager to ensure it has the expected methods
            with patch("local_coding_assistant.agent.llm_manager.ProviderManager") as mock_provider_manager_class:
                mock_provider_manager = MagicMock()
                mock_provider_manager_class.return_value = mock_provider_manager

                _manager = LLMManager()

                # Verify provider manager was reloaded with config
                mock_provider_manager.reload.assert_called_once_with(mock_config_manager)


class TestLLMManagerGenerate:
    """Test LLMManager generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful response generation."""
        # Mock provider system
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "test_provider"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Generated response",
                model="gpt-4",
                tokens_used=100,
                tool_calls=None,
                finish_reason="stop",
            )
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router
            manager._generate_mock_response = MagicMock()

            request = LLMRequest(prompt="Test prompt")
            response = await manager.generate(request)

            # Verify correct response format
            assert isinstance(response, LLMResponse)
            assert response.content == "Generated response"
            assert response.model_used == "gpt-4"
            assert response.tokens_used == 100

            # Verify provider was marked healthy
            mock_router.mark_provider_success.assert_called_once_with("test_provider")

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self):
        """Test generate method with various parameters."""
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "openai"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Response", model="gpt-4", tokens_used=50
            )
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")
            _response = await manager.generate(
                request,
                provider="openai",
                model="gpt-4",
                policy="coding",
                overrides={"temperature": 0.5},
            )

            # Verify parameters were passed to router
            mock_router.get_provider_for_request.assert_called_once_with(
                provider_name="openai",
                model_name="gpt-4",
                role="coding",
                overrides={"temperature": 0.5},
            )

    @pytest.mark.asyncio
    async def test_generate_with_provider_error(self):
        """Test generate method handles provider errors."""
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "failing_provider"
        mock_provider.generate_with_retry = AsyncMock(
            side_effect=ProviderError("API Error")
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )
        mock_router._is_critical_error = MagicMock(return_value=False)

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")

            with pytest.raises(LLMError, match="Provider error"):
                await manager.generate(request)

    @pytest.mark.asyncio
    async def test_generate_with_critical_error_and_fallback(self):
        """Test generate method with critical error and successful fallback."""
        # First provider fails with critical error
        mock_failing_provider = AsyncMock(spec=BaseProvider)
        mock_failing_provider.name = "failing_provider"
        mock_failing_provider.generate_with_retry = AsyncMock(
            side_effect=ProviderError("Critical error")
        )
        mock_failing_provider.provider = "failing_provider"  # Add provider attribute

        # Second provider succeeds
        mock_success_provider = AsyncMock(spec=BaseProvider)
        mock_success_provider.name = "success_provider"
        mock_success_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Fallback response", model="gpt-3.5", tokens_used=75
            )
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            side_effect=[
                (mock_failing_provider, "gpt-4"),  # First call fails
                (mock_success_provider, "gpt-3.5"),  # Fallback succeeds
            ]
        )
        mock_router._is_critical_error = MagicMock(return_value=True)

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")
            response = await manager.generate(request)

            # Verify fallback response
            assert response.content == "Fallback response"
            assert response.model_used == "gpt-3.5"

            # Verify both providers were marked appropriately
            mock_router.mark_provider_failure.assert_called_once_with(
                "failing_provider", mock_failing_provider.generate_with_retry.side_effect
            )
            mock_router.mark_provider_success.assert_called_once_with(
                "success_provider"
            )

    @pytest.mark.asyncio
    async def test_generate_test_mode(self):
        """Test generate method in test mode."""
        with patch("local_coding_assistant.config.get_config_manager"):
            with patch.dict(os.environ, {"LOCCA_TEST_MODE": "true"}):
                manager = LLMManager()

                request = LLMRequest(prompt="Test prompt")
                response = await manager.generate(request)

                assert response.content.startswith("[LLMManager] Echo:")
                assert response.model_used == "mock-model"
                assert response.tokens_used == 50

    @pytest.mark.asyncio
    async def test_generate_with_tool_calls(self):
        """Test generate method with tool calls in response."""
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "test_provider"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="I'll use the tool",
                model="gpt-4",
                tokens_used=100,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_tool"},
                    }
                ],
                finish_reason="tool_calls",
            )
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Use tool")
            response = await manager.generate(request)

            assert response.content == "I'll use the tool"
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "test_tool"


class TestLLMManagerStream:
    """Test LLMManager stream method."""

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming response."""
        # Mock streaming deltas
        mock_deltas = [
            ProviderLLMResponseDelta(content="Hello", finish_reason=None),
            ProviderLLMResponseDelta(content=" world", finish_reason=None),
            ProviderLLMResponseDelta(content="!", finish_reason="stop"),
        ]

        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "test_provider"

        # Create async iterator that yields mock deltas
        async def mock_stream_with_retry(*args, **kwargs):
            for delta in mock_deltas:
                yield delta

        mock_provider.stream_with_retry = mock_stream_with_retry

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test prompt")
            chunks = []
            async for chunk in manager.stream(request):
                chunks.append(chunk)

            assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_empty_deltas(self):
        """Test streaming with some empty deltas."""
        mock_deltas = [
            ProviderLLMResponseDelta(content="", finish_reason=None),
            ProviderLLMResponseDelta(content="Hello", finish_reason=None),
            ProviderLLMResponseDelta(content="", finish_reason=None),
            ProviderLLMResponseDelta(content="world", finish_reason="stop"),
        ]

        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "test_provider"

        # Create async iterator that yields mock deltas
        async def mock_stream_with_retry(*args, **kwargs):
            for delta in mock_deltas:
                yield delta

        mock_provider.stream_with_retry = mock_stream_with_retry

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")
            chunks = []
            async for chunk in manager.stream(request):
                chunks.append(chunk)

            assert chunks == ["Hello", "world"]  # Empty deltas filtered out

    @pytest.mark.asyncio
    async def test_stream_test_mode(self):
        """Test stream method in test mode."""
        with patch("local_coding_assistant.config.get_config_manager"):
            with patch.dict(os.environ, {"LOCCA_TEST_MODE": "true"}):
                manager = LLMManager()

                request = LLMRequest(prompt="Test prompt")
                chunks = []
                async for chunk in manager.stream(request):
                    chunks.append(chunk)

                assert len(chunks) == 1
                assert chunks[0].startswith("[LLMManager] Echo:")

    @pytest.mark.asyncio
    async def test_stream_with_provider_error(self):
        """Test stream method with provider errors."""
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "failing_provider"

        # Mock stream_with_retry to return async iterator that raises ProviderError
        class FailingAsyncIterator:
            def __init__(self):
                self._started = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._started:
                    self._started = True
                    raise ProviderError("Stream error")
                raise StopAsyncIteration

        mock_provider.stream_with_retry = lambda *args, **kwargs: FailingAsyncIterator()

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )
        mock_router.is_critical_error = MagicMock(return_value=False)

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")

            with pytest.raises(LLMError, match="Stream error"):
                async for _chunk in manager.stream(request):
                    pass


class TestLLMManagerProviderStatus:
    """Test LLMManager provider status methods."""

    def test_get_provider_status_list_cache_miss(self):
        """Test provider status list when cache is empty."""
        mock_provider_manager = MagicMock()
        mock_provider_manager.list_providers.return_value = ["openai", "google"]
        mock_provider_manager.get_provider_source.side_effect = lambda name: {
            "openai": "built-in",
            "google": "built-in",
        }.get(name)

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.provider_manager = mock_provider_manager
            manager._provider_status_cache = {}
            manager._last_health_check = 0
            manager._cache_ttl = 30 * 60

            # Mock async context
            async def mock_refresh():
                manager._provider_status_cache = {
                    "openai": {"healthy": True, "models": ["gpt-4", "gpt-3.5"]},
                    "google": {
                        "healthy": False,
                        "models": [],
                        "error": "API key missing",
                    },
                }
                manager._last_health_check = time.time()

            manager._refresh_provider_status_cache = mock_refresh

            with patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")):
                # Mock asyncio.run to execute the coroutine and return None
                with patch("asyncio.run", side_effect=lambda coro: None):
                    # Manually execute the mock refresh function since it's just setting attributes
                    manager._provider_status_cache = {
                        "openai": {"healthy": True, "models": ["gpt-4", "gpt-3.5"]},
                        "google": {
                            "healthy": False,
                            "models": [],
                            "error": "API key missing",
                        },
                    }
                    # Set to old time so cache refresh is triggered
                    manager._last_health_check = 0

                    status_list = manager.get_provider_status_list()

                    assert len(status_list) == 2
                    openai_status = next(
                        s for s in status_list if s["name"] == "openai"
                    )
                    google_status = next(
                        s for s in status_list if s["name"] == "google"
                    )

                    assert openai_status["status"] == "available"
                    assert openai_status["models"] == 2
                    assert google_status["status"] == "unavailable"
                    assert google_status["error"] == "API key missing"

    def test_reload_providers(self):
        """Test provider reload functionality."""
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()

        manager = LLMManager.__new__(LLMManager)
        manager.provider_manager = mock_provider_manager
        manager.config_manager = mock_config_manager
        manager._provider_status_cache = {"old": {"data": "some_data"}}
        manager._last_health_check = time.time()

        manager.reload_providers()

        mock_provider_manager.reload.assert_called_once_with(mock_config_manager)
        assert manager._provider_status_cache == {}
        assert manager._last_health_check == 0

    @pytest.mark.asyncio
    async def test_get_provider_status(self):
        """Test detailed provider status retrieval."""
        mock_provider_manager = MagicMock()
        mock_provider_manager.list_providers.return_value = ["openai"]

        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.health_check = AsyncMock(return_value=True)
        mock_provider.get_available_models.return_value = ["gpt-4", "gpt-3.5"]

        mock_provider_manager.get_provider.return_value = mock_provider

        mock_router = MagicMock()
        mock_router._unhealthy_providers = set()

        manager = LLMManager.__new__(LLMManager)
        manager.provider_manager = mock_provider_manager
        manager.router = mock_router

        status = await manager.get_provider_status()

        assert "openai" in status
        assert status["openai"]["healthy"] is True
        assert status["openai"]["models"] == ["gpt-4", "gpt-3.5"]
        assert status["openai"]["in_unhealthy_set"] is False

        mock_provider.health_check.assert_called_once()
        mock_provider.get_available_models.assert_called_once()


class TestLLMManagerErrorHandling:
    """Test LLMManager error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_generate_with_non_provider_error(self):
        """Test generate method with non-provider errors."""
        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            side_effect=Exception("Network error")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")

            with pytest.raises(LLMError, match="LLM generation failed"):
                await manager.generate(request)

    @pytest.mark.asyncio
    async def test_stream_with_non_provider_error(self):
        """Test stream method with non-provider errors."""
        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            side_effect=Exception("Network error")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")

            with pytest.raises(LLMError, match="LLM streaming generation failed"):
                async for _chunk in manager.stream(request):
                    pass

    @pytest.mark.asyncio
    async def test_fallback_failure(self):
        """Test fallback when both primary and fallback providers fail."""
        # Both providers fail with critical errors
        mock_provider1 = AsyncMock(spec=BaseProvider)
        mock_provider1.name = "provider1"
        mock_provider1.generate_with_retry = AsyncMock(
            side_effect=ProviderError("Critical error 1")
        )
        mock_provider1.provider = "provider1"

        mock_provider2 = AsyncMock(spec=BaseProvider)
        mock_provider2.name = "provider2"
        mock_provider2.generate_with_retry = AsyncMock(
            side_effect=ProviderError("Critical error 2")
        )
        mock_provider2.provider = "provider2"

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            side_effect=[
                (mock_provider1, "gpt-4"),  # First call
                (mock_provider2, "gpt-3.5"),  # Fallback call
            ]
        )
        mock_router.is_critical_error = MagicMock(return_value=True)

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Test")

            with pytest.raises(LLMError, match="Provider error"):
                await manager.generate(request)


class TestLLMManagerIntegration:
    """Test LLMManager integration scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_generation_flow(self):
        """Test complete generation flow from request to response."""
        # Mock the entire provider system
        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "integration_provider"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Integration test response",
                model="gpt-4-integration",
                tokens_used=150,
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "test"}}
                ],
            )
        )

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            # Test with complex request
            request = LLMRequest(
                prompt="Complex integration test",
                system_prompt="You are a test assistant",
                context={"test": "integration"},
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

            response = await manager.generate(request)

            # Verify complete flow
            assert response.content == "Integration test response"
            assert response.model_used == "gpt-4-integration"
            assert response.tokens_used == 150
            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1

            # Verify provider request was created correctly
            mock_router.get_provider_for_request.assert_called_once()
            call_args = mock_router.get_provider_for_request.call_args
            # Check that the call was made with the expected keyword arguments
            assert call_args.kwargs["provider_name"] is None  # No provider override
            assert call_args.kwargs["model_name"] is None    # No model override
            assert call_args.kwargs["role"] is None         # No policy override
            assert call_args.kwargs["overrides"] is None    # No overrides

    @pytest.mark.asyncio
    async def test_streaming_end_to_end_flow(self):
        """Test complete streaming flow."""
        mock_deltas = [
            ProviderLLMResponseDelta(content="Stream", finish_reason=None),
            ProviderLLMResponseDelta(content="ing", finish_reason=None),
            ProviderLLMResponseDelta(content=" test", finish_reason=None),
            ProviderLLMResponseDelta(content=" complete", finish_reason="stop"),
        ]

        mock_provider = AsyncMock(spec=BaseProvider)
        mock_provider.name = "streaming_provider"

        # Create async iterator that yields mock deltas
        async def mock_stream_with_retry(*args, **kwargs):
            for delta in mock_deltas:
                yield delta

        mock_provider.stream_with_retry = mock_stream_with_retry

        mock_router = AsyncMock(spec=ProviderRouter)
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-4")
        )

        with patch("local_coding_assistant.config.get_config_manager"):
            manager = LLMManager.__new__(LLMManager)
            manager.router = mock_router

            request = LLMRequest(prompt="Streaming test")
            chunks = []
            async for chunk in manager.stream(request):
                chunks.append(chunk)

            assert chunks == ["Stream", "ing", " test", " complete"]
            mock_router.mark_provider_success.assert_called_once_with(
                "streaming_provider"
            )
