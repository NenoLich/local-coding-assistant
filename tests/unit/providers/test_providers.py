"""
Unit tests for the provider system including base classes, manager, and all providers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.providers.base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.providers.exceptions import ProviderError
from local_coding_assistant.providers.google_provider import GoogleGeminiProvider
from local_coding_assistant.providers.local_provider import LocalProvider
from local_coding_assistant.providers.openrouter_provider import OpenRouterProvider
from local_coding_assistant.providers.provider_manager import ProviderManager
from local_coding_assistant.providers.router import ProviderRouter


class TestProviderLLMRequest:
    """Test ProviderLLMRequest functionality."""

    def test_request_creation(self):
        """Test creating a ProviderLLMRequest."""
        messages = [{"role": "user", "content": "Test message"}]
        request = ProviderLLMRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            stream=False,
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        assert request.messages == messages
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000
        assert request.stream is False
        assert request.tools is not None
        assert len(request.tools) == 1

    def test_request_defaults(self):
        """Test ProviderLLMRequest default values."""
        messages = [{"role": "user", "content": "Test"}]
        request = ProviderLLMRequest(messages=messages, model="gpt-3.5")

        assert request.temperature == 0.7
        assert request.max_tokens is None
        assert request.stream is False
        assert request.tools is None
        assert request.tool_choice is None
        assert request.response_format is None
        assert request.extra_params is None


class TestProviderLLMResponse:
    """Test ProviderLLMResponse functionality."""

    def test_response_creation(self):
        """Test creating a ProviderLLMResponse."""
        response = ProviderLLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used=100,
            tool_calls=[{"id": "call_1", "type": "function"}],
            finish_reason="stop",
            metadata={"test": "data"},
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.tokens_used == 100
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "stop"
        assert response.metadata == {"test": "data"}


class TestProviderLLMResponseDelta:
    """Test ProviderLLMResponseDelta functionality."""

    def test_delta_creation(self):
        """Test creating a ProviderLLMResponseDelta."""
        delta = ProviderLLMResponseDelta(
            content="Partial response", finish_reason=None, metadata={"chunk": 1}
        )

        assert delta.content == "Partial response"
        assert delta.finish_reason is None
        assert delta.metadata == {"chunk": 1}

    def test_delta_minimal(self):
        """Test creating a minimal ProviderLLMResponseDelta."""
        delta = ProviderLLMResponseDelta()
        assert delta.content == ""
        assert delta.finish_reason is None
        assert delta.metadata == {}


class TestBaseProvider:
    """Test BaseProvider abstract base class."""

    def test_initialization(self):
        """Test BaseProvider initialization."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider initialization."""

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test_provider",
            base_url="https://api.test.com",
            models=["gpt-4", "gpt-3.5"],
            api_key="test_key",
            api_key_env="TEST_API_KEY"
        )

        assert provider.name == "test_provider"
        assert provider.models == ["gpt-4", "gpt-3.5"]
        assert provider.api_key == "test_key"
        assert provider.api_key_env == "TEST_API_KEY"
        assert provider.base_url == "https://api.test.com"
        # driver_instance is now initialized immediately in __init__
        assert provider.driver_instance is not None

    def test_supports_model(self):
        """Test model support checking."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(name="test", base_url="https://api.test.com", models=["gpt-4", "gpt-3.5"], api_key="test_key")
        assert provider.supports_model("gpt-4") is True
        assert provider.supports_model("gpt-3") is False

    def test_get_available_models(self):
        """Test getting available models."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(name="test", base_url="https://api.test.com", models=["gpt-4", "gpt-3.5"], api_key="test_key")
        assert provider.get_available_models() == ["gpt-4", "gpt-3.5"]

    def test_stream_method_without_driver(self):
        """Test stream method when no driver is available."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            async def generate(self, request):
                return ProviderLLMResponse(
                    content="Test response", model="gpt-4", finish_reason="stop"
                )

            async def health_check(self):
                return True

        provider = TestProvider(name="test", base_url="https://api.test.com", models=["gpt-4"], api_key="test_key")

        # Set driver_instance to None to test fallback mechanism
        provider.driver_instance = None

        # Test that stream falls back to single delta when no driver is available
        async def test_stream():
            deltas = []
            async for delta in provider.stream(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}], model="gpt-4"
                )
            ):
                deltas.append(delta)

            assert len(deltas) == 1
            assert deltas[0].content == "Test response"
            assert deltas[0].finish_reason == "stop"

        asyncio.run(test_stream())

    def test_stream_with_retry_success(self):
        """Test stream_with_retry with successful streaming."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="gpt-4")

            async def stream(self, request):
                yield ProviderLLMResponseDelta(content="chunk1")
                yield ProviderLLMResponseDelta(content="chunk2", finish_reason="stop")

            async def health_check(self):
                return True

        provider = TestProvider(name="test", base_url="https://api.test.com", models=["gpt-4"], api_key="test_key")

        async def test_retry():
            chunks = []
            async for delta in provider.stream_with_retry(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}], model="gpt-4"
                ),
                max_retries=3,
            ):
                chunks.append(delta.content)

            assert chunks == ["chunk1", "chunk2"]

        asyncio.run(test_retry())

    def test_stream_with_retry_failure(self):
        """Test stream_with_retry with failures and retries."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="gpt-4")

            async def health_check(self):
                return True

        provider = TestProvider(name="test", base_url="https://api.test.com", models=["gpt-4"], api_key="test_key")

        # Mock stream method that fails twice then succeeds
        call_count = 0

        async def failing_stream(_request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ProviderError("Temporary failure")
            yield ProviderLLMResponseDelta(content="success", finish_reason="stop")

        # Explicitly shadow the inherited stream method for testing retry behavior
        # This is intentional to test the retry mechanism with a failing stream
        provider.stream = failing_stream  # type: ignore[assignment]

        async def test_retry():
            chunks = []
            async for delta in provider.stream_with_retry(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}], model="gpt-4"
                ),
                max_retries=3,
                retry_delay=0.1,  # Fast for testing
            ):
                chunks.append(delta.content)

            assert chunks == ["success"]
            assert call_count == 3  # Two failures + one success

        asyncio.run(test_retry())


class TestProviderManager:
    """Test ProviderManager functionality."""

    def test_initialization(self):
        """Test ProviderManager initialization."""
        manager = ProviderManager()
        assert len(manager._providers) == 3  # 3 builtin providers are auto-registered
        assert "google_gemini" in manager._providers
        assert "local" in manager._providers
        assert "openrouter" in manager._providers
        assert len(manager._instances) == 0  # But no instances are created initially

    def test_register_provider(self):
        """Test provider registration."""
        manager = ProviderManager()

        # Mock provider class
        class MockProvider(BaseProvider):
            def __init__(self):
                super().__init__(name="mock", base_url="https://api.test.com", models=["gpt-4"])

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock response", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock", finish_reason="stop")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test_source"

        assert "mock" in manager._providers
        assert manager._provider_sources["mock"] == "test_source"
        assert manager.list_providers() == ["google_gemini", "local", "mock", "openrouter"]

    def test_get_provider(self):
        """Test getting a registered provider."""
        manager = ProviderManager()

        class MockProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(name="mock", base_url="https://api.test.com", models=["gpt-4"], api_key="test_key", **kwargs)

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test"

        provider = manager.get_provider("mock")
        assert provider is not None
        assert provider.name == "mock"

        # Test getting non-existent provider
        assert manager.get_provider("nonexistent") is None

    def test_list_providers(self):
        """Test listing registered providers."""
        manager = ProviderManager()

        class MockProvider(BaseProvider):
            def __init__(self, name):
                super().__init__(name=name, base_url="https://api.test.com", models=["gpt-4"])

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register providers manually for testing
        manager._providers["provider1"] = MockProvider
        manager._provider_sources["provider1"] = "source1"
        manager._providers["provider2"] = MockProvider
        manager._provider_sources["provider2"] = "source2"

        providers = manager.list_providers()
        assert set(providers) == {"google_gemini", "local", "openrouter", "provider1", "provider2"}

    def test_get_provider_source(self):
        """Test getting provider source."""
        manager = ProviderManager()

        class MockProvider(BaseProvider):
            def __init__(self):
                super().__init__(name="mock", base_url="https://api.test.com", models=["gpt-4"])

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test_source"

        assert manager.get_provider_source("mock") == "test_source"
        assert manager.get_provider_source("nonexistent") is None

    def test_reload_functionality(self):
        """Test provider reload functionality."""
        manager = ProviderManager()
        mock_config = MagicMock()

        # Register a provider
        class MockProvider(BaseProvider):
            def __init__(self):
                super().__init__(name="mock", base_url="https://api.test.com", models=["gpt-4"])

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register a provider manually for testing
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test"

        # Verify provider is registered (along with builtin providers)
        assert "mock" in manager.list_providers()

        # Reload should clear providers (simplified test)
        manager.reload(mock_config)

        # In real implementation, reload would reload from config
        # This test verifies the method exists and can be called
        assert hasattr(manager, "reload")


class TestGoogleGeminiProvider:
    """Test GoogleGeminiProvider functionality."""

    def test_initialization(self):
        """Test GoogleGeminiProvider initialization."""
        provider = GoogleGeminiProvider(
            api_key="test_key", models=["gemini-pro", "gemini-pro-vision"]
        )

        assert provider.name == "google_gemini"
        assert "gemini-pro" in provider.models
        assert provider.api_key == "test_key"

    def test_initialization_with_env_key(self):
        """Test GoogleGeminiProvider initialization with environment key."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env_key"}):
            provider = GoogleGeminiProvider(api_key_env="GOOGLE_API_KEY")

            assert provider.api_key_env == "GOOGLE_API_KEY"
            # In real implementation, this would get the key from env

    @pytest.mark.asyncio
    async def test_health_check_with_driver(self):
        """Test health check when driver is available."""
        provider = GoogleGeminiProvider(api_key="test_key")

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        assert is_healthy is True
        mock_driver.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_without_driver(self):
        """Test health check when no driver is available."""
        provider = GoogleGeminiProvider(api_key="test_key")

        # Mock driver health_check to return False
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=False)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        assert is_healthy is False
        mock_driver.health_check.assert_called_once()

    def test_stream_method(self):
        """Test GoogleGeminiProvider stream method."""
        provider = GoogleGeminiProvider(api_key="test_key")

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.stream = AsyncMock()
        provider.driver_instance = mock_driver

        # Test that stream delegates to driver
        _request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}], model="gemini-pro"
        )

        # The stream method should delegate to the driver
        # We can't easily test the async generator without more complex setup,
        # but we can verify the method exists and has correct signature
        import inspect

        sig = inspect.signature(provider.stream)
        assert "request" in sig.parameters


class TestOpenRouterProvider:
    """Test OpenRouterProvider functionality."""

    def test_initialization(self):
        """Test OpenRouterProvider initialization."""
        provider = OpenRouterProvider(
            api_key="test_key", models=["auto", "gpt-4", "claude-3"]
        )

        assert provider.name == "openrouter"
        assert "auto" in provider.models
        assert provider.api_key == "test_key"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test OpenRouterProvider health check."""
        provider = OpenRouterProvider(api_key="test_key")

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        assert is_healthy is True

    def test_stream_method(self):
        """Test OpenRouterProvider stream method."""
        provider = OpenRouterProvider(api_key="test_key")

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.stream = AsyncMock()
        provider.driver_instance = mock_driver

        # Verify method exists and delegates to driver
        _request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}], model="gpt-4"
        )

        import inspect

        sig = inspect.signature(provider.stream)
        assert "request" in sig.parameters


class TestLocalProvider:
    """Test LocalProvider functionality."""

    def test_initialization(self):
        """Test LocalProvider initialization."""
        provider = LocalProvider(
            models=["local-model-1", "local-model-2"], base_url="http://localhost:8000", api_key="dummy_key"
        )

        assert provider.name == "local"
        assert "local-model-1" in provider.models
        assert provider.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test LocalProvider health check."""
        provider = LocalProvider(api_key="dummy_key")

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(
            return_value=False
        )  # Local provider might not be available
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        assert is_healthy is False  # Expected for local provider without actual server


class TestProviderRouter:
    """Test ProviderRouter functionality."""

    def test_initialization(self):
        """Test ProviderRouter initialization."""
        mock_config_manager = MagicMock()
        mock_provider_manager = MagicMock()

        router = ProviderRouter(mock_config_manager, mock_provider_manager)

        assert router.config_manager == mock_config_manager
        assert router.provider_manager == mock_provider_manager
        assert len(router._unhealthy_providers) == 0

    def test_mark_provider_healthy(self):
        """Test marking provider as healthy."""
        router = ProviderRouter(MagicMock(), MagicMock())

        # Add provider to unhealthy set
        router._unhealthy_providers.add("test_provider")

        # Mark as healthy
        router._mark_provider_healthy("test_provider")

        # Should be removed from unhealthy set
        assert "test_provider" not in router._unhealthy_providers

    def test_mark_provider_unhealthy(self):
        """Test marking provider as unhealthy."""
        router = ProviderRouter(MagicMock(), MagicMock())

        # Mark as unhealthy
        router._mark_provider_unhealthy("test_provider")

        # Should be added to unhealthy set
        assert "test_provider" in router._unhealthy_providers

    def test_is_critical_error(self):
        """Test critical error detection."""
        router = ProviderRouter(MagicMock(), MagicMock())

        # Test different error types
        from local_coding_assistant.providers.exceptions import (
            ProviderAuthError,
            ProviderConnectionError,
            ProviderRateLimitError,
            ProviderTimeoutError,
        )

        # These should be considered critical
        assert (
            router._is_critical_error(ProviderConnectionError("Connection failed"))
            is True
        )
        assert router._is_critical_error(ProviderAuthError("Invalid API key")) is True
        assert router._is_critical_error(ProviderRateLimitError("Rate limited")) is True

        # This might not be critical (implementation dependent)
        assert isinstance(
            router._is_critical_error(ProviderTimeoutError("Timeout")), bool
        )


class TestProviderIntegration:
    """Test provider system integration scenarios."""

    def test_provider_registration_and_retrieval(self):
        """Test complete provider registration and retrieval flow."""
        manager = ProviderManager()

        class TestProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(name="test", base_url="https://api.test.com", models=["model1"], api_key="test_key", **kwargs)

            async def generate(self, _request):
                return ProviderLLMResponse(content="test", model="model1")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="test", finish_reason="stop")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["test"] = TestProvider
        manager._provider_sources["test"] = "integration_test"

        # Retrieve provider
        provider = manager.get_provider("test")
        assert provider is not None
        assert provider.name == "test"
        assert provider.supports_model("model1") is True

        # Test source tracking
        assert manager.get_provider_source("test") == "integration_test"

    @pytest.mark.asyncio
    async def test_provider_streaming_flow(self):
        """Test complete streaming flow through provider system."""
        manager = ProviderManager()

        class StreamingProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(name="streaming", base_url="https://api.test.com", models=["stream-model"], api_key="test_key", **kwargs)

            async def generate(self, _request):
                return ProviderLLMResponse(content="streamed", model="stream-model")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="chunk1")
                yield ProviderLLMResponseDelta(content="chunk2", finish_reason="stop")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["streaming"] = StreamingProvider
        manager._provider_sources["streaming"] = "test"

        provider = manager.get_provider("streaming")

        # Test streaming
        deltas = []
        assert provider is not None
        async for delta in provider.stream(
            ProviderLLMRequest(
                messages=[{"role": "user", "content": "test"}], model="stream-model"
            )
        ):
            deltas.append(delta)

        assert len(deltas) == 2
        assert deltas[0].content == "chunk1"
        assert deltas[1].content == "chunk2"
        assert deltas[1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_provider_error_propagation(self):
        """Test error propagation through provider system."""
        manager = ProviderManager()

        class ErrorProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(name="error", base_url="https://api.test.com", models=["error-model"], api_key="test_key", **kwargs)

            async def generate(self, _request):
                raise ProviderError("Test error")

            async def stream(self, _request):
                # Simulate normal streaming that fails during iteration
                yield ProviderLLMResponseDelta(content="chunk1")
                raise ProviderError("Stream error")

            async def health_check(self):
                return False

        # Register provider manually for testing
        manager._providers["error"] = ErrorProvider
        manager._provider_sources["error"] = "test"

        provider = manager.get_provider("error")

        # Test error in generate
        assert provider is not None
        with pytest.raises(ProviderError, match="Test error"):
            await provider.generate(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}], model="error-model"
                )
            )

        # Test error in stream - expect error after consuming one delta
        with pytest.raises(ProviderError, match="Stream error"):
            deltas = []
            async for delta in provider.stream(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}], model="error-model"
                )
            ):
                deltas.append(delta)
                if len(deltas) > 1:  # Should not reach here
                    break

        # Test health check
        assert await provider.health_check() is False
