from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.agent.llm_manager import (
    LLMConfig,
    LLMManager,
    LLMRequest,
    LLMResponse,
)
from local_coding_assistant.core.exceptions import LLMError


class TestLLMConfig:
    """Test LLMConfig pydantic model validation."""

    def test_valid_config_creation(self):
        """Test creating a valid LLMConfig."""
        config = LLMConfig(
            model_name="gpt-3.5-turbo",
            provider="openai",
            temperature=0.7,
            max_tokens=1000,
            api_key="test-key",
        )
        assert config.model_name == "gpt-3.5-turbo"
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.api_key == "test-key"

    def test_config_defaults(self):
        """Test LLMConfig default values."""
        config = LLMConfig(model_name="gpt-4")
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.api_key is None

    def test_temperature_validation(self):
        """Test temperature field validation."""
        # Valid range
        LLMConfig(model_name="test", temperature=0.0)
        LLMConfig(model_name="test", temperature=2.0)

        # Invalid range
        with pytest.raises(ValueError):
            LLMConfig(model_name="test", temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(model_name="test", temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens field validation."""
        # Valid values
        LLMConfig(model_name="test", max_tokens=1)
        LLMConfig(model_name="test", max_tokens=1000)

        # Invalid values
        with pytest.raises(ValueError):
            LLMConfig(model_name="test", max_tokens=0)

        with pytest.raises(ValueError):
            LLMConfig(model_name="test", max_tokens=-1)


class TestLLMRequest:
    """Test LLMRequest pydantic model."""

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
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tool_outputs == {"test": "result"}

    def test_request_defaults(self):
        """Test LLMRequest default values."""
        request = LLMRequest(prompt="test prompt")
        assert request.context is None
        assert request.system_prompt is None
        assert request.tools is None
        assert request.tool_outputs is None


class TestLLMResponse:
    """Test LLMResponse pydantic model."""

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
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

    def test_response_defaults(self):
        """Test LLMResponse default values."""
        response = LLMResponse(content="test", model_used="gpt-4")
        assert response.tokens_used is None
        assert response.tool_calls is None


class TestLLMManager:
    """Test LLMManager functionality."""

    def test_initialization_with_valid_config(self):
        """Test LLMManager initialization with valid config."""
        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")

        # Create comprehensive mock openai module
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()

        # Mock client
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Patch sys.modules to include our mock
        with patch.dict("sys.modules", {"openai": mock_openai}):
            llm = LLMManager(config)
            assert llm.config == config
            assert llm.client == mock_client

            # Verify OpenAI was called with the config values
            mock_openai.AsyncOpenAI.assert_called_once()
            call_args = mock_openai.AsyncOpenAI.call_args[1]
            assert call_args["api_key"] is None  # Should be None, will use env var

    def test_initialization_with_unsupported_provider(self):
        """Test LLMManager initialization fails with unsupported provider."""
        config = LLMConfig(model_name="test", provider="unsupported")
        with pytest.raises(LLMError, match="Unsupported provider"):
            LLMManager(config)

    def test_openai_client_setup_success(self):
        """Test successful OpenAI client setup."""
        # Create comprehensive mock openai module
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()

        # Mock client
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        # Set up the mock in sys.modules
        with patch.dict("sys.modules", {"openai": mock_openai}):
            config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
            llm = LLMManager(config)

            mock_openai.AsyncOpenAI.assert_called_once()
            assert llm.client == mock_client

    def test_openai_client_setup_missing_package(self):
        """Test OpenAI client setup fails when package not installed."""
        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(LLMError, match="OpenAI package not installed"):
                LLMManager(config)

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful response generation."""
        # Mock OpenAI response
        mock_content_item = MagicMock()
        mock_content_item.text = "Generated response"

        mock_output_item = MagicMock()
        mock_output_item.content = [mock_content_item]
        mock_output_item.tool_calls = None

        mock_response = MagicMock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.output_tokens = 50
        mock_response.output = [mock_output_item]

        mock_client = AsyncMock()
        mock_client.responses.create.return_value = mock_response

        # Patch the client setup to return the mock client
        with patch.object(LLMManager, "_setup_openai_client", return_value=None):
            # Also patch the client attribute directly after __init__
            config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
            llm = LLMManager(config=config)
            llm.client = mock_client  # Override the client after __init__

            request = LLMRequest(prompt="Test prompt")
            response = await llm.generate(request)

            # Assertions
            assert isinstance(response, LLMResponse)
            assert response.content == "Generated response"
            assert response.model_used == "gpt-3.5-turbo"
            assert response.tokens_used == 50
            assert response.tool_calls == []
            mock_client.responses.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_with_tools(self):
        """Test response generation with tool calls."""
        # Mock tool call response for new Responses API
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.name = "test_tool"
        mock_tool_call.arguments = '{"arg": "value"}'

        mock_response = MagicMock()
        mock_response.model = "gpt-3.5-turbo"
        mock_response.usage.output_tokens = 75

        # Create mock output item with content and tool calls
        mock_output_item = MagicMock()
        mock_content_item = MagicMock()
        mock_content_item.text = "I'll use the tool"
        mock_output_item.content = [mock_content_item]
        mock_output_item.tool_calls = [mock_tool_call]
        mock_response.output = [mock_output_item]

        mock_client = AsyncMock()
        mock_client.responses.create.return_value = mock_response

        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
        with patch.object(LLMManager, "_setup_openai_client"):
            llm = LLMManager.__new__(LLMManager)
            llm.config = config
            llm.client = mock_client

            request = LLMRequest(
                prompt="Use the tool",
                tools=[{"type": "function", "function": {"name": "test_tool"}}],
            )
            response = await llm.generate(request)

            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["id"] == "call_123"
            assert response.tool_calls[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_generate_without_client(self):
        """Test generate fails when client is not initialized."""
        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
        llm = LLMManager.__new__(LLMManager)
        llm.config = config
        llm.client = None

        request = LLMRequest(prompt="test")

        with pytest.raises(LLMError, match="LLM client not initialized"):
            await llm.generate(request)

    @pytest.mark.asyncio
    async def test_generate_openai_error(self):
        """Test handling of OpenAI API errors."""
        mock_client = AsyncMock()
        mock_client.responses.create.side_effect = Exception("API Error")

        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
        with patch.object(LLMManager, "_setup_openai_client"):
            llm = LLMManager.__new__(LLMManager)
            llm.config = config
            llm.client = mock_client

            request = LLMRequest(prompt="test")

            with pytest.raises(LLMError, match="LLM generation failed"):
                await llm.generate(request)
