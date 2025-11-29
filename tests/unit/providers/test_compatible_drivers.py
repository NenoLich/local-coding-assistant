"""Unit tests for compatible_drivers.py"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta, Choice as ChunkChoice

from local_coding_assistant.providers.base import ProviderLLMRequest, ProviderLLMResponse
from local_coding_assistant.providers.compatible_drivers import (
    OpenAIChatCompletionsDriver,
    OpenAIResponsesDriver,
)
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
)


class TestOpenAIChatCompletionsDriver:
    """Tests for OpenAIChatCompletionsDriver"""

    @pytest.fixture
    def driver(self):
        """Create a test driver instance"""
        return OpenAIChatCompletionsDriver(
            api_key="test-api-key",
            base_url="https://api.example.com",
            provider_name="test-provider"
        )

    @pytest.fixture
    def mock_client(self, driver):
        """Mock the OpenAI client"""
        with patch.object(driver, 'client', new_callable=AsyncMock) as mock_client:
            yield mock_client

    @pytest.fixture
    def test_request(self):
        """Create a test request"""
        return ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, driver, mock_client, test_request):
        """Test successful generate call"""
        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.choices = [self.MockChoice()]
                self.model = "test-model"
                self.usage = {
                    "total_tokens": 10,
                    "prompt_tokens": 5,
                    "completion_tokens": 5
                }
                self.id = "test-response-id"
                self.created = 1234567890
            
            class MockChoice:
                def __init__(self):
                    self.message = self.MockMessage()
                    self.finish_reason = "stop"
                
                class MockMessage:
                    def __init__(self):
                        self.role = "assistant"
                        self.content = "Test response"
        
        mock_client.chat.completions.create.return_value = MockResponse()

        # Call the method
        response = await driver.generate(test_request)

        # Assertions
        assert isinstance(response, ProviderLLMResponse)
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 10
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, driver, mock_client):
        """Test generate with tool calls"""
        # Prepare test request with tools
        request = ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather?"}],
            parameters={
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"]
                        }
                    }
                }]
            }
        )

        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.choices = [self.MockChoice()]
                self.model = "test-model"
                self.usage = {
                    "total_tokens": 20,
                    "prompt_tokens": 10,
                    "completion_tokens": 10
                }
                self.id = "test-response-id"
                self.created = 1234567890
            
            class MockChoice:
                def __init__(self):
                    self.message = self.MockMessage()
                    self.finish_reason = "tool_calls"
                
                class MockMessage:
                    def __init__(self):
                        self.role = "assistant"
                        self.content = None
                        self.tool_calls = [self.MockToolCall()]
                    
                    class MockToolCall:
                        def __init__(self):
                            self.id = "call_123"
                            self.type = "function"
                            self.function = self.MockFunction()
                        
                        class MockFunction:
                            def __init__(self):
                                self.name = "get_weather"
                                self.arguments = '{"location": "San Francisco"}'
                            
                            def get(self, key, default=None):
                                if key == "name":
                                    return self.name
                                elif key == "arguments":
                                    return self.arguments
                                return default
        
        mock_client.chat.completions.create.return_value = MockResponse()

        # Call the method
        response = await driver.generate(request)

        # Assertions
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_stream_success(self, driver, mock_client, test_request):
        """Test successful streaming"""
        # Create a proper mock chunk class
        class MockChunk:
            def __init__(self):
                self.id = "test-chunk-123"
                self.created = 1234567890
                self.model = "test-model"
                self.choices = [self.MockChoice()]
            
            class MockChoice:
                def __init__(self):
                    self.delta = self.MockDelta()
                    self.finish_reason = None
                
                class MockDelta:
                    def __init__(self):
                        self.content = "Hello"
        
        # Create an async generator for the mock response
        async def mock_stream():
            yield MockChunk()
        
        mock_client.chat.completions.create.return_value = mock_stream()

        # Call the method
        chunks = []
        async for chunk in driver.stream(test_request):
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert chunks[0].metadata["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_health_check_success(self, driver, mock_client):
        """Test successful health check"""
        mock_client.models.list.return_value = MagicMock()
        assert await driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, driver, mock_client):
        """Test failed health check"""
        mock_client.models.list.side_effect = Exception("API error")
        assert await driver.health_check() is False

    @pytest.mark.parametrize("status_code,error_class,error_message_suffix", [
        (401, ProviderAuthError, "Invalid API key (HTTP 401)"),
        (429, ProviderRateLimitError, "Rate limit exceeded (HTTP 429)"),
        (500, ProviderError, "Server error (HTTP 500)"),
    ])
    @pytest.mark.asyncio
    async def test_error_handling(self, driver, mock_client, test_request, status_code, error_class, error_message_suffix):
        """Test error handling for different status codes"""
        # Create a proper HTTPError with status code
        from httpx import HTTPStatusError, Request, Response
        
        request = Request("POST", "https://api.example.com/chat/completions")
        response = Response(status_code=status_code, request=request, text="Error message")
        error = HTTPStatusError("Test error", request=request, response=response)
        
        mock_client.chat.completions.create.side_effect = error

        # Assert the appropriate exception is raised with the correct message
        with pytest.raises(error_class) as exc_info:
            await driver.generate(test_request)
            
        # Verify the error message contains the expected suffix
        error_message = str(exc_info.value)
        assert error_message_suffix in error_message


class TestOpenAIResponsesDriver:
    """Tests for OpenAIResponsesDriver"""

    @pytest.fixture
    def driver(self):
        """Create a test driver instance"""
        return OpenAIResponsesDriver(
            api_key="test-api-key",
            base_url="https://api.example.com",
            provider_name="test-provider"
        )

    @pytest.fixture
    def mock_client(self, driver):
        """Mock the OpenAI client"""
        with patch.object(driver, 'client', new_callable=AsyncMock) as mock_client:
            yield mock_client

    @pytest.fixture
    def test_request(self):
        """Create a test request"""
        return ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, driver, mock_client, test_request):
        """Test successful generate call"""
        # Mock response
        class MockResponse:
            def __init__(self):
                self.output_text = "Test response"
                self.finish_reason = "stop"
                self.model = "test-model"
                self.id = "test-response-id"
                self.created = 1234567890
                self.usage = {
                    "total_tokens": 10,
                    "prompt_tokens": 5,
                    "completion_tokens": 5
                }
        
        mock_client.responses.create.return_value = MockResponse()

        # Call the method
        response = await driver.generate(test_request)

        # Assertions
        assert isinstance(response, ProviderLLMResponse)
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 10
        mock_client.responses.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, driver, mock_client):
        """Test generate with tool calls"""
        # Prepare test request with tools
        request = ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }]
        )

        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.output_text = ""  # Must be a string, not None
                self.output = [{
                    "type": "function_call",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}'
                    }
                }]
                self.finish_reason = "tool_calls"
                self.model = "test-model"
                self.id = "test-response-id"
                self.created = 1234567890
                self.usage = {
                    "total_tokens": 20,
                    "prompt_tokens": 10,
                    "completion_tokens": 10
                }
        
        mock_client.responses.create.return_value = MockResponse()

        # Call the method
        response = await driver.generate(request)

        # Assertions
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_stream_success(self, driver, mock_client, test_request):
        """Test successful streaming"""
        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.output_text = "Hello"
        mock_chunk.finish_reason = None
        mock_chunk.id = "test-chunk-123"
        mock_chunk.created = 1234567890
        mock_chunk.model = "test-model"
        
        # Create an async generator for the mock response
        async def mock_stream():
            yield mock_chunk
            
        mock_client.responses.create.return_value = mock_stream()

        # Call the method
        chunks = []
        async for chunk in driver.stream(test_request):
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        assert chunks[0].metadata["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_health_check_success(self, driver, mock_client):
        """Test successful health check"""
        mock_client.models.list.return_value = MagicMock()
        assert await driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, driver, mock_client):
        """Test failed health check"""
        mock_client.models.list.side_effect = Exception("API error")
        assert await driver.health_check() is False

    @pytest.mark.parametrize("status_code,error_class,error_message_suffix", [
        (401, ProviderAuthError, "Invalid API key (HTTP 401)"),
        (429, ProviderRateLimitError, "Rate limit exceeded (HTTP 429)"),
        (500, ProviderError, "Server error (HTTP 500)"),
    ])
    @pytest.mark.asyncio
    async def test_error_handling(self, driver, mock_client, test_request, status_code, error_class, error_message_suffix):
        """Test error handling for different status codes"""
        # Create a proper HTTPError with status code
        from httpx import HTTPStatusError, Request, Response
        
        request = Request("POST", "https://api.example.com/responses")
        response = Response(status_code=status_code, request=request, text="Error message")
        error = HTTPStatusError("Test error", request=request, response=response)
        
        mock_client.responses.create.side_effect = error

        # Assert the appropriate exception is raised with the correct message
        with pytest.raises(error_class) as exc_info:
            await driver.generate(test_request)
            
        # Verify the error message contains the expected suffix
        error_message = str(exc_info.value)
        assert error_message_suffix in error_message

    def test_format_tools_for_responses_api(self, driver):
        """Test tool formatting for Responses API"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"}
                }
            }
        ]
        
        formatted = driver._format_tools_for_responses_api(tools)
        assert len(formatted) == 1
        assert formatted[0]["name"] == "get_weather"
        assert formatted[0]["type"] == "function"


