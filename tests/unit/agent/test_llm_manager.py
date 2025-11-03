from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest, LLMResponse
from local_coding_assistant.config.schemas import AppConfig, LLMConfig, ProviderConfig
from local_coding_assistant.providers.base import ProviderLLMResponseDelta
from local_coding_assistant.providers.exceptions import (
    ProviderConnectionError,
)


class TestLLMConfig:
    """Test LLMConfig pydantic model validation."""
    def test_valid_config_creation(self):
        """Test creating a valid LLMConfig."""
        config = LLMConfig(
            temperature=0.7,
            max_tokens=1000,
            max_retries=3,
            retry_delay=1.0,
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.providers == []

    def test_temperature_validation(self):
        """Test temperature field validation."""
        # Valid range
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=2.0)

        # Invalid range
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens field validation."""
        # Valid values
        LLMConfig(max_tokens=1)
        LLMConfig(max_tokens=1000)

        # Invalid values
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)

        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-1)

    def test_max_retries_validation(self):
        """Test max_retries field validation."""
        # Valid values
        LLMConfig(max_retries=1)
        LLMConfig(max_retries=10)

        # Invalid values
        with pytest.raises(ValueError):
            LLMConfig(max_retries=0)

        with pytest.raises(ValueError):
            LLMConfig(max_retries=-1)

    def test_retry_delay_validation(self):
        """Test retry_delay field validation."""
        # Valid values
        LLMConfig(retry_delay=0.1)
        LLMConfig(retry_delay=10.0)

        # Invalid values
        with pytest.raises(ValueError):
            LLMConfig(retry_delay=0.0)

        with pytest.raises(ValueError):
            LLMConfig(retry_delay=-1.0)


class TestLLMRequest:
    """Test LLMRequest class."""

    def test_valid_request_creation(self):
        """Test creating a valid LLMRequest."""
        request = LLMRequest(
            prompt="Hello, world!",
            context={"user": "test"},
            system_prompt="You are a helpful assistant",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_outputs={"test": "result"},
        )
        assert request.prompt == "Hello, world!"
        assert request.context == {"user": "test"}
        assert request.system_prompt == "You are a helpful assistant"
        assert request.tools == [{"type": "function", "function": {"name": "test"}}]
        assert request.tool_outputs == {"test": "result"}

    def test_request_defaults(self):
        """Test LLMRequest default values."""
        request = LLMRequest(prompt="test prompt")
        assert request.context == {}
        assert request.system_prompt is None
        assert request.tools == []
        assert request.tool_outputs == {}

    def test_to_provider_request(self):
        """Test conversion to provider request format."""
        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="System prompt",
            context={"test": "value"},
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        provider_request = request.to_provider_request("test-model")

        assert provider_request.model == "test-model"
        assert provider_request.temperature == 0.7
        assert len(provider_request.messages) == 3  # system, context, prompt
        assert provider_request.messages[0]["role"] == "system"
        assert provider_request.messages[1]["role"] == "user"
        assert provider_request.messages[2]["role"] == "user"


class TestLLMResponse:
    """Test LLMResponse class."""

    def test_valid_response_creation(self):
        """Test creating a valid LLMResponse."""
        response = LLMResponse(
            content="Test response",
            model_used="gpt-3.5-turbo",
            tokens_used=100,
            tool_calls=[{"id": "call_1", "type": "function"}],
        )
        assert response.content == "Test response"
        assert response.model_used == "gpt-3.5-turbo"
        assert response.tokens_used == 100
        assert response.tool_calls == [{"id": "call_1", "type": "function"}]

    def test_response_defaults(self):
        """Test LLMResponse default values."""
        response = LLMResponse(content="test", model_used="gpt-4")
        assert response.tokens_used is None
        assert response.tool_calls is None


class TestProviderConfig:
    """Test ProviderConfig pydantic model."""

    def test_valid_provider_creation(self):
        """Test creating a valid ProviderConfig."""
        provider = ProviderConfig(
            name="openai",
            driver="openai_chat",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            models={"gpt-3.5-turbo": {"max_tokens": 4096}},
        )
        assert provider.name == "openai"
        assert provider.driver == "openai_chat"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key_env == "OPENAI_API_KEY"
        assert provider.models == {"gpt-3.5-turbo": {"max_tokens": 4096}}

    def test_provider_defaults(self):
        """Test ProviderConfig default values."""
        provider = ProviderConfig(
            name="test",
            driver="openai_chat",
            base_url="https://api.example.com",
            api_key_env="TEST_API_KEY",
        )
        assert provider.models == {}


class TestAppConfig:
    """Test AppConfig integration."""

    def test_valid_app_config(self):
        """Test creating a valid AppConfig."""
        provider = ProviderConfig(
            name="openai",
            driver="openai_chat",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
        )

        app_config = AppConfig(
            providers={"openai": provider},
            llm=LLMConfig(temperature=0.5),
        )

        assert app_config.providers["openai"].name == "openai"
        assert app_config.llm.temperature == 0.5


class TestLLMManager:
    """Test LLMManager functionality with provider system."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        provider_manager = MagicMock()
        provider_manager.list_providers.return_value = ["openai"]
        provider_manager.get_provider.return_value = MagicMock()
        return provider_manager

    @pytest.fixture
    def mock_provider_router(self):
        """Create a mock provider router."""
        router = MagicMock()
        router.get_provider_for_request.return_value = (MagicMock(), "gpt-3.5-turbo")
        router.mark_provider_success = MagicMock()
        router.mark_provider_failure = MagicMock()
        return router

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        config_manager = MagicMock()
        config_manager.global_config = AppConfig()
        return config_manager

    def test_initialization_with_provider_system(
        self, mock_provider_manager, mock_config_manager
    ):
        """Test LLMManager initialization with provider system."""
        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)

        assert llm.config_manager == mock_config_manager
        assert llm.provider_manager == mock_provider_manager
        assert llm.router is not None

    @pytest.mark.asyncio
    async def test_generate_success_with_provider_system(
        self, mock_provider_manager, mock_config_manager
    ):
        """Test successful response generation using provider system."""
        # Mock provider and response
        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=Mock(
                content="Generated response",
                model="gpt-3.5-turbo",
                tokens_used=100,
                tool_calls=None,
            )
        )

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(return_value=(mock_provider, "gpt-3.5-turbo"))
        mock_router.mark_provider_success = MagicMock()
        mock_router.mark_provider_failure = MagicMock()

        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)
        llm.router = mock_router

        request = LLMRequest(prompt="Test prompt")
        response = await llm.generate(request)

        # Assertions
        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.model_used == "gpt-3.5-turbo"
        assert response.tokens_used == 100
        mock_router.get_provider_for_request.assert_called_once()
        mock_provider.generate_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_tools_using_provider_system(
        self, mock_provider_manager, mock_config_manager
    ):
        """Test response generation with tool calls using provider system."""
        # Mock provider and response with tool calls
        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=Mock(
                content="I'll use the tool",
                model="gpt-3.5-turbo",
                tokens_used=75,
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"arg": "value"}',
                        },
                    }
                ],
            )
        )

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(return_value=(mock_provider, "gpt-3.5-turbo"))
        mock_router.mark_provider_success = MagicMock()
        mock_router.mark_provider_failure = MagicMock()

        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)
        llm.router = mock_router

        request = LLMRequest(
            prompt="Use the tool",
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
        )
        response = await llm.generate(request)

        assert response is not None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "call_123"
        assert response.tool_calls[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_generate_fallback_on_provider_error(
        self, mock_provider_manager, mock_config_manager
    ):
        """Test fallback behavior when primary provider fails."""
        # Mock failing provider
        mock_failing_provider = MagicMock()
        mock_failing_provider.name = "openai_primary"
        mock_failing_provider.generate_with_retry = AsyncMock(
            side_effect=ProviderConnectionError("API Error", provider="openai_primary")
        )

        # Mock fallback provider
        mock_fallback_provider = MagicMock()
        mock_fallback_provider.name = "openai_fallback"
        mock_fallback_provider.generate_with_retry = AsyncMock(
            return_value=Mock(
                content="Fallback response",
                model="gpt-3.5-turbo",
                tokens_used=50,
                tool_calls=None,
            )
        )

        # Mock router with fallback
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(side_effect=[
            (mock_failing_provider, "gpt-3.5-turbo"),  # First call (fails)
            (mock_fallback_provider, "gpt-3.5-turbo"),  # Fallback call (succeeds)
        ])
        mock_router.mark_provider_success = MagicMock()
        mock_router.mark_provider_failure = MagicMock()

        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)
        llm.router = mock_router

        request = LLMRequest(prompt="Test prompt")

        # Should succeed with fallback
        response = await llm.generate(request)
        assert response.content == "Fallback response"
        assert mock_router.get_provider_for_request.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_mock_mode(self, mock_provider_manager, mock_config_manager):
        """Test mock mode for testing."""
        import os

        # Enable mock mode
        with patch.dict(os.environ, {"LOCCA_TEST_MODE": "true"}):
            llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)

            request = LLMRequest(prompt="Test prompt")
            response = await llm.generate(request)

            assert response.content.startswith("[LLMManager] Echo:")
            assert response.model_used == "mock-model"

    @pytest.mark.asyncio
    async def test_streaming_generation(
        self, mock_provider_manager, mock_config_manager
    ):
        """Test streaming response generation."""
        # Mock provider with streaming
        mock_provider = MagicMock()
        mock_provider.name = "openai"

        # Create an async generator that returns ProviderLLMResponseDelta objects
        async def mock_stream_with_retry(request, max_retries=3, retry_delay=1.0):
            yield ProviderLLMResponseDelta(content="Hello")
            yield ProviderLLMResponseDelta(content=" world")

        # Make stream_with_retry return the async generator
        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(return_value=(mock_provider, "gpt-3.5-turbo"))

        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)
        llm.router = mock_router

        request = LLMRequest(prompt="Test streaming")

        # Collect streamed content
        chunks = []
        async for chunk in llm.stream(request):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "Hello" in chunks[0]
        assert "world" in chunks[1]

    def test_provider_status_management(self, mock_provider_manager, mock_config_manager):
        """Test provider status management."""
        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)

        # Mock provider status
        mock_provider_manager.list_providers.return_value = ["openai", "claude"]
        mock_provider_manager.get_provider_source.return_value = "config"

        status_list = llm.get_provider_status_list()

        assert len(status_list) == 2
        assert status_list[0]["name"] == "openai"
        assert status_list[1]["name"] == "claude"

    @pytest.mark.asyncio
    async def test_provider_health_check(self, mock_provider_manager, mock_config_manager):
        """Test provider health checking."""
        # Mock healthy provider
        mock_provider = MagicMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        mock_provider.get_available_models.return_value = ["gpt-3.5-turbo", "gpt-4"]

        mock_provider_manager.get_provider.return_value = mock_provider

        llm = LLMManager(config_manager=mock_config_manager, provider_manager=mock_provider_manager)

        status = await llm.get_provider_status()

        assert "openai" in status
        assert status["openai"]["healthy"] is True
        assert len(status["openai"]["models"]) == 2
