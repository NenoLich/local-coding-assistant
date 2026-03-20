from unittest.mock import MagicMock, AsyncMock

import pytest

from local_coding_assistant.agent.llm import (
    LLMService,
    LLMTask,
    LLMResult,
    LLMToolCall,
)
from local_coding_assistant.config.schemas import AppConfig, LLMConfig, ProviderConfig


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


class TestLLMTask:
    """Test LLMTask class."""

    def test_valid_task_creation(self):
        """Test creating a valid LLMTask."""
        task = LLMTask(
            prompt="Hello, world!",
            context=[{"role": "user", "message": "test"}],
            system_prompt="You are a helpful assistant",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_outputs={"test": "result"},
        )
        assert task.prompt == "Hello, world!"
        assert task.context == [{"role": "user", "message": "test"}]
        assert task.system_prompt == "You are a helpful assistant"
        assert task.tools == [{"type": "function", "function": {"name": "test"}}]
        assert task.tool_outputs == {"test": "result"}

    def test_task_defaults(self):
        """Test LLMTask default values."""
        task = LLMTask(prompt="test prompt")
        assert task.context == []
        assert task.system_prompt is None
        assert task.tools == []
        assert task.tool_outputs == {}

    def test_build_provider_request(self):
        """Test conversion to provider request format."""
        task = LLMTask(
            prompt="Test prompt",
            system_prompt="System prompt",
            context=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        provider_request = task.build_provider_request(
            model="test-model",
            stream=False,
            temperature=0.7,
        )

        assert provider_request.model == "test-model"
        assert provider_request.temperature == 0.7
        assert len(provider_request.messages) == 3  # system, context, prompt
        assert provider_request.messages[0]["role"] == "system"
        assert provider_request.messages[1]["role"] == "user"
        assert provider_request.messages[2]["role"] == "user"


class TestLLMResult:
    """Test LLMResult class."""

    def test_valid_result_creation(self):
        """Test creating a valid LLMResult."""
        tool_call = LLMToolCall(id="call_1", name="test_tool")
        result = LLMResult(
            content="Test response",
            model="gpt-3.5-turbo",
            provider="openai",
            total_tokens=100,
            tool_calls=[tool_call],
        )
        assert result.content == "Test response"
        assert result.model == "gpt-3.5-turbo"
        assert result.provider == "openai"
        assert result.total_tokens == 100
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "test_tool"

    def test_result_defaults(self):
        """Test LLMResult default values."""
        result = LLMResult(content="test", model="gpt-4", provider="openai")
        assert result.total_tokens is None
        assert result.tool_calls == []


class TestProviderConfig:
    """Test ProviderConfig pydantic model."""

    def test_valid_provider_creation(self):
        """Test creating a valid ProviderConfig."""
        # Create a model config dictionary that will be converted to ModelConfig objects
        models_config = {
            "gpt-3.5-turbo": {"supported_parameters": ["max_tokens", "temperature"]}
        }

        # Create provider config using from_dict to handle model conversion
        provider = ProviderConfig.from_dict(
            {
                "name": "openai",
                "driver": "openai_chat",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": models_config,
            }
        )

        # Verify the provider configuration
        assert provider.name == "openai"
        assert provider.driver == "openai_chat"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key_env == "OPENAI_API_KEY"

        # Verify the models were properly converted to ModelConfig objects
        assert len(provider.models) == 1
        model_config = provider.models[0]
        assert model_config.name == "gpt-3.5-turbo"
        assert "max_tokens" in model_config.supported_parameters
        assert "temperature" in model_config.supported_parameters

    def test_provider_defaults(self):
        """Test ProviderConfig default values."""
        provider = ProviderConfig(
            name="test",
            driver="openai_chat",
            base_url="https://api.example.com",
            api_key_env="TEST_API_KEY",
        )
        assert provider.models == []


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


class TestLLMService:
    """Test LLMService functionality."""

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        provider_manager = MagicMock()
        provider_manager.list_providers.return_value = ["openai"]
        provider_manager.get_provider.return_value = MagicMock()
        provider_manager.get_provider_source.return_value = "config"
        return provider_manager

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        config_manager = MagicMock()
        config_manager.global_config = AppConfig()
        return config_manager

    @pytest.fixture
    def service_with_mocks(self, mock_provider_manager, mock_config_manager):
        """Create LLMService with mocked dependencies."""
        service = LLMService(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        return service

    def test_initialization(self, mock_provider_manager, mock_config_manager):
        """LLMService wires provider components on init."""
        service = LLMService(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )

        assert service._config_manager == mock_config_manager
        assert service._provider_manager == mock_provider_manager
        assert service._provider_selector is not None

    @pytest.mark.asyncio
    async def test_provider_status_methods(self, service_with_mocks):
        """Test provider status methods."""
        service = service_with_mocks

        # Mock provider manager methods
        service._provider_manager.list_providers.return_value = ["openai", "anthropic"]
        mock_provider = MagicMock()
        mock_provider.health_check = AsyncMock(return_value=True)
        mock_provider.get_available_models = MagicMock(
            return_value=["model1", "model2"]
        )
        service._provider_manager.get_provider.return_value = mock_provider
        service._provider_manager.get_provider_source.return_value = "config"

        # Test get_provider_status_list
        status_list = service.get_provider_status_list()
        assert len(status_list) == 2
        assert status_list[0]["name"] in ["openai", "anthropic"]

        # Test get_provider_status
        status = await service.get_provider_status("openai")
        assert status["name"] == "openai"

        # Test reload_providers
        service._provider_manager.reload.reset_mock()
        service.reload_providers()
        service._provider_manager.reload.assert_called_once()
