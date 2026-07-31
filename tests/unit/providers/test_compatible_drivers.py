"""Unit tests for compatible_drivers.py"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.providers.base import (
    OptionalParameters,
    ProviderLLMRequest,
    ProviderLLMResponse,
)
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
            provider_name="test-provider",
        )

    @pytest.fixture
    def mock_acompletion(self):
        """Mock litellm's acompletion function"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.acompletion",
            new_callable=AsyncMock,
        ) as mock:
            yield mock

    @pytest.fixture
    def test_request(self):
        """Create a test request"""
        return ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, driver, mock_acompletion, test_request):
        """Test successful generate call"""

        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.choices = [self.MockChoice()]
                self.model = "openai/test-model"
                self.usage = MagicMock()
                self.usage.total_tokens = 10
                self.usage.prompt_tokens = 5
                self.usage.completion_tokens = 5
                self.usage.model_dump = MagicMock(
                    return_value={
                        "total_tokens": 10,
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                    }
                )
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

        mock_acompletion.return_value = MockResponse()

        # Call the method
        response = await driver.generate(test_request)

        # Assertions
        assert isinstance(response, ProviderLLMResponse)
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 10
        mock_acompletion.assert_called_once()
        # Verify model prefix was added
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "openai/test-model"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, driver, mock_acompletion):
        """Test generate with tool calls"""
        # Prepare test request with tools
        request = ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather?"}],
            parameters=OptionalParameters(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ]
            ),
        )

        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.choices = [self.MockChoice()]
                self.model = "openai/test-model"
                self.usage = MagicMock()
                self.usage.total_tokens = 20
                self.usage.prompt_tokens = 10
                self.usage.completion_tokens = 10
                self.usage.model_dump = MagicMock(
                    return_value={
                        "total_tokens": 20,
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                    }
                )
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

        mock_acompletion.return_value = MockResponse()

        # Call the method
        response = await driver.generate(request)

        # Assertions
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_stream_success(self, driver, mock_acompletion, test_request):
        """Test successful streaming"""

        # Create a proper mock chunk class
        class MockChunk:
            def __init__(self):
                self.id = "test-chunk-123"
                self.created = 1234567890
                self.model = "openai/test-model"
                self.choices = [self.MockChoice()]
                self.usage = None

            class MockChoice:
                def __init__(self):
                    self.delta = self.MockDelta()
                    self.finish_reason = None

                class MockDelta:
                    def __init__(self):
                        self.content = "Hello"
                        self.tool_calls = None
                        self.reasoning = None

        # Create an async generator for the mock response
        async def mock_stream():
            yield MockChunk()

        mock_acompletion.return_value = mock_stream()

        # Call the method
        chunks = []
        async for chunk in driver.stream(test_request):
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
        # The metadata model is the prefixed model name from litellm
        assert chunks[0].metadata["model"] == "openai/test-model"

    @pytest.mark.asyncio
    async def test_health_check_success(self, driver):
        """Test successful health check"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.get_valid_models",
            return_value=["model1"],
        ):
            assert await driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, driver):
        """Test failed health check"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.get_valid_models",
            side_effect=Exception("API error"),
        ):
            assert await driver.health_check() is False

    @pytest.mark.parametrize(
        "status_code,error_class,error_message_suffix",
        [
            (401, ProviderAuthError, "Invalid API key (HTTP 401)"),
            (429, ProviderRateLimitError, "Rate limit exceeded (HTTP 429)"),
            (500, ProviderError, "Server error (HTTP 500)"),
        ],
    )
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        driver,
        mock_acompletion,
        test_request,
        status_code,
        error_class,
        error_message_suffix,
    ):
        """Test error handling for different status codes"""
        # Create a proper HTTPError with status code
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://api.example.com/chat/completions")
        response = Response(
            status_code=status_code, request=request, text="Error message"
        )
        error = HTTPStatusError("Test error", request=request, response=response)

        mock_acompletion.side_effect = error

        # Assert the appropriate exception is raised with the correct message
        with pytest.raises(error_class) as exc_info:
            await driver.generate(test_request)

        # Verify the error message contains the expected suffix
        error_message = str(exc_info.value)
        assert error_message_suffix in error_message

    @pytest.mark.parametrize(
        "error_message,expected_exception",
        [
            ("Authentication failed 401", ProviderAuthError),
            ("Rate limit exceeded 429", ProviderRateLimitError),
            ("Some other error", ProviderError),
        ],
    )
    @pytest.mark.asyncio
    async def test_string_based_error_handling(
        self, driver, mock_acompletion, test_request, error_message, expected_exception
    ):
        """Test error handling based on string content when no status code"""
        # Mock litellm to raise a plain exception with error message
        mock_acompletion.side_effect = Exception(error_message)

        # Assert the appropriate exception is raised
        with pytest.raises(expected_exception) as exc_info:
            await driver.generate(test_request)

        # Verify the error message contains the original error
        error_str = str(exc_info.value)
        assert error_message in error_str

    def test_model_prefix_strategy(self, driver):
        """Test that model prefix is correctly added for chat completions"""
        request = ProviderLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        payload = driver._build_chat_payload(request)
        assert payload["model"] == "openai/gpt-4"

    def test_extra_body_parameter_handling(self, driver):
        """Test that extra_body parameters are correctly passed to litellm"""
        request = ProviderLLMRequest(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            parameters=OptionalParameters(
                extra_body={"thinking_config": {"include_thought_signature": True}}
            ),
        )
        payload = driver._build_chat_payload(request)
        assert "extra_body" in payload
        assert (
            payload["extra_body"]["thinking_config"]["include_thought_signature"]
            is True
        )

    def test_format_messages_basic(self, driver):
        """Test basic message formatting."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        formatted = driver._format_messages(messages)

        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "Hi there"

    def test_format_messages_with_tool_calls(self, driver):
        """Test message formatting with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll help",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": {"arg": "value"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Tool result",
            },
        ]

        formatted = driver._format_messages(messages)

        assert len(formatted) == 2
        # Assistant message
        assert formatted[0]["role"] == "assistant"
        assert formatted[0]["content"] == "I'll help"
        assert (
            formatted[0]["tool_calls"][0]["function"]["arguments"] == '{"arg": "value"}'
        )
        # Tool message
        assert formatted[1]["role"] == "tool"
        assert formatted[1]["tool_call_id"] == "call_1"
        assert formatted[1]["content"] == "Tool result"

    def test_format_messages_tool_content_string(self, driver):
        """Test that tool message content is converted to string."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "Result"}],
            },
        ]

        formatted = driver._format_messages(messages)

        assert formatted[0]["content"] == "Result"
        assert isinstance(formatted[0]["content"], str)

    def test_extract_error_details_edge_cases(self, driver):
        """Test _extract_error_details with various status_code scenarios (lines 289-295)"""

        # Test case 1: status_code is None
        class MockException1(Exception):
            def __init__(self):
                self.status_code = None
                self.response = "mock_response"

        e1 = MockException1()
        status_code, response = driver._extract_error_details(e1)
        assert status_code is None
        assert response == "mock_response"

        # Test case 2: status_code is valid string
        class MockException2(Exception):
            def __init__(self):
                self.status_code = "400"
                self.response = "mock_response"

        e2 = MockException2()
        status_code, response = driver._extract_error_details(e2)
        assert status_code == 400
        assert response == "mock_response"

        # Test case 3: status_code is invalid string
        class MockException3(Exception):
            def __init__(self):
                self.status_code = "invalid"
                self.response = "mock_response"

        e3 = MockException3()
        status_code, response = driver._extract_error_details(e3)
        assert status_code is None
        assert response == "mock_response"

        # Test case 4: status_code in response object
        class MockResponse:
            def __init__(self):
                self.status_code = 500

        class MockException4(Exception):
            def __init__(self):
                self.response = MockResponse()

        e4 = MockException4()
        status_code, response = driver._extract_error_details(e4)
        assert status_code == 500
        assert response == e4.response

        # Test case 5: status_code in response is None
        class MockResponse5:
            def __init__(self):
                self.status_code = None

        class MockException5(Exception):
            def __init__(self):
                self.response = MockResponse5()

        e5 = MockException5()
        status_code, response = driver._extract_error_details(e5)
        assert status_code is None
        assert response == e5.response

        # Test case 6: No status_code or response
        class MockException6(Exception):
            pass

        e6 = MockException6()
        status_code, response = driver._extract_error_details(e6)
        assert status_code is None
        assert response is None


class TestOpenAIResponsesDriver:
    """Tests for OpenAIResponsesDriver"""

    @pytest.fixture
    def driver(self):
        """Create a test driver instance"""
        return OpenAIResponsesDriver(
            api_key="test-api-key",
            base_url="https://api.example.com",
            provider_name="test-provider",
        )

    @pytest.fixture
    def mock_acompletion(self):
        """Mock litellm's acompletion function"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.acompletion",
            new_callable=AsyncMock,
        ) as mock:
            yield mock

    @pytest.fixture
    def test_request(self):
        """Create a test request"""
        return ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_generate_success(self, driver, mock_acompletion, test_request):
        """Test successful generate call"""

        # Mock response
        class MockResponse:
            def __init__(self):
                self.output_text = "Test response"
                self.output = []
                self.finish_reason = "stop"
                self.model = "openai/responses/test-model"
                self.id = "test-response-id"
                self.created = 1234567890
                self.usage = MagicMock()
                self.usage.total_tokens = 10
                self.usage.prompt_tokens = 5
                self.usage.completion_tokens = 5
                self.usage.model_dump = MagicMock(
                    return_value={
                        "total_tokens": 10,
                        "prompt_tokens": 5,
                        "completion_tokens": 5,
                    }
                )

        mock_acompletion.return_value = MockResponse()

        # Call the method
        response = await driver.generate(test_request)

        # Assertions
        assert isinstance(response, ProviderLLMResponse)
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.finish_reason == "stop"
        assert response.tokens_used == 10
        mock_acompletion.assert_called_once()
        # Verify model prefix was added
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs["model"] == "openai/responses/test-model"

    @pytest.mark.asyncio
    async def test_generate_with_tools(self, driver, mock_acompletion):
        """Test generate with tool calls"""
        # Prepare test request with tools
        request = ProviderLLMRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather?"}],
            parameters=OptionalParameters(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ]
            ),
        )

        # Create a proper mock response class
        class MockResponse:
            def __init__(self):
                self.output_text = ""
                self.output = [
                    {
                        "type": "function_call",
                        "call_id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ]
                self.finish_reason = "tool_calls"
                self.model = "openai/responses/test-model"
                self.id = "test-response-id"
                self.created = 1234567890
                self.usage = MagicMock()
                self.usage.total_tokens = 20
                self.usage.prompt_tokens = 10
                self.usage.completion_tokens = 10
                self.usage.model_dump = MagicMock(
                    return_value={
                        "total_tokens": 20,
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                    }
                )

        mock_acompletion.return_value = MockResponse()

        # Call the method
        response = await driver.generate(request)

        # Assertions
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"
        assert response.finish_reason == "tool_calls"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_stream_success(self, driver, mock_acompletion, test_request):
        """Test successful streaming"""
        # Mock streaming response - create proper Responses API events
        event1 = MagicMock()
        event1.type = "response.output_text.delta"
        event1.delta = "Hello"
        event1.sequence_number = 1
        event1.item_id = "item1"
        event1.output_index = 0
        event1.content_index = 0

        event2 = MagicMock()
        event2.type = "response.output_text.delta"
        event2.delta = " world"
        event2.sequence_number = 2
        event2.item_id = "item1"
        event2.output_index = 0
        event2.content_index = 1

        # Completion event
        completion_event = MagicMock()
        completion_event.type = "response.completed"
        completion_event.sequence_number = 3
        completion_event.response = MagicMock()
        completion_event.response.output = []
        completion_event.response.id = "test-response-id"
        completion_event.response.usage = MagicMock()
        completion_event.response.usage.model_dump = MagicMock(
            return_value={"total_tokens": 10}
        )

        async def mock_stream():
            yield event1
            yield event2
            yield completion_event

        mock_acompletion.return_value = mock_stream()

        # Call the method
        chunks = []
        async for chunk in driver.stream(test_request):
            chunks.append(chunk)

        # Assertions
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].finish_reason == "completed"

    @pytest.mark.asyncio
    async def test_health_check_success(self, driver):
        """Test successful health check"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.get_valid_models",
            return_value=["model1"],
        ):
            assert await driver.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, driver):
        """Test failed health check"""
        with patch(
            "local_coding_assistant.providers.compatible_drivers.get_valid_models",
            side_effect=Exception("API error"),
        ):
            assert await driver.health_check() is False

    @pytest.mark.parametrize(
        "status_code,error_class,error_message_suffix",
        [
            (401, ProviderAuthError, "Invalid API key (HTTP 401)"),
            (429, ProviderRateLimitError, "Rate limit exceeded (HTTP 429)"),
            (500, ProviderError, "Server error (HTTP 500)"),
        ],
    )
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        driver,
        mock_acompletion,
        test_request,
        status_code,
        error_class,
        error_message_suffix,
    ):
        """Test error handling for different status codes"""
        # Create a proper HTTPError with status code
        from httpx import HTTPStatusError, Request, Response

        request = Request("POST", "https://api.example.com/responses")
        response = Response(
            status_code=status_code, request=request, text="Error message"
        )
        error = HTTPStatusError("Test error", request=request, response=response)

        mock_acompletion.side_effect = error

        # Assert the appropriate exception is raised with the correct message
        with pytest.raises(error_class) as exc_info:
            await driver.generate(test_request)

        # Verify the error message contains the expected suffix
        error_message = str(exc_info.value)
        assert error_message_suffix in error_message

    @pytest.mark.parametrize(
        "error_message,expected_exception",
        [
            ("Authentication failed 401", ProviderAuthError),
            ("Rate limit exceeded 429", ProviderRateLimitError),
            ("Some other error", ProviderError),
        ],
    )
    @pytest.mark.asyncio
    async def test_string_based_error_handling(
        self, driver, mock_acompletion, test_request, error_message, expected_exception
    ):
        """Test error handling based on string content when no status code"""
        # Mock litellm to raise a plain exception with error message
        mock_acompletion.side_effect = Exception(error_message)

        # Assert the appropriate exception is raised
        with pytest.raises(expected_exception) as exc_info:
            await driver.generate(test_request)

        # Verify the error message contains the original error
        error_str = str(exc_info.value)
        assert error_message in error_str

    def test_model_prefix_strategy_responses(self, driver):
        """Test that model prefix is correctly added for Responses API"""
        request = ProviderLLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )
        payload = driver._build_responses_payload(request)
        assert payload["model"] == "openai/responses/gpt-4"

    def test_extra_body_parameter_handling_responses(self, driver):
        """Test that extra_body parameters are correctly passed to litellm for Responses API"""
        request = ProviderLLMRequest(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            parameters=OptionalParameters(
                extra_body={"thinking_config": {"include_thought_signature": True}}
            ),
        )
        payload = driver._build_responses_payload(request)
        assert "extra_body" in payload
        assert (
            payload["extra_body"]["thinking_config"]["include_thought_signature"]
            is True
        )

    def test_format_messages_basic(self, driver):
        """Test basic message formatting for Responses API."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        formatted = driver._format_messages(messages)

        assert len(formatted) == 2
        assert formatted[0]["type"] == "message"
        assert formatted[0]["content"] == "Hello"
        assert formatted[1]["type"] == "message"
        assert formatted[1]["content"] == "Hi there"

    def test_format_messages_with_tool_calls(self, driver):
        """Test message formatting with tool calls for Responses API."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll help",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": {"arg": "value"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "text", "text": "Tool result"}],
            },
        ]

        formatted = driver._format_messages(messages)

        assert len(formatted) == 2
        # Assistant message
        assert formatted[0]["type"] == "message"
        assert formatted[0]["content"] == "I'll help"
        assert formatted[0]["tool_calls"] == messages[0]["tool_calls"]
        # Tool message
        assert formatted[1]["type"] == "function_call_output"
        assert formatted[1]["call_id"] == "call_1"
        assert formatted[1]["output"] == "Tool result"

    def test_format_messages_tool_content_string(self, driver):
        """Test tool message content extraction from list."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            },
        ]

        formatted = driver._format_messages(messages)

        assert formatted[0]["output"] == "Part 1Part 2"

    def test_format_messages_skips_system(self, driver):
        """Test that system messages are skipped in input formatting."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]

        formatted = driver._format_messages(messages)

        assert len(formatted) == 1
        assert formatted[0]["content"] == "Hello"

    def test_format_tools_for_responses_api(self, driver):
        """Test tool formatting for Responses API"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object"},
                },
            }
        ]

        formatted = driver._format_tools_for_responses_api(tools)
        assert len(formatted) == 1
        assert formatted[0]["name"] == "get_weather"
        assert formatted[0]["type"] == "function"

    def test_extract_reasoning_output_list(self, driver):
        """Test _extract_reasoning with output list containing reasoning (lines 564-572)"""

        # Test case 1: reasoning_text with string content
        class MockResponse1:
            def __init__(self):
                self.output = [
                    {"type": "reasoning_text", "content": "This is reasoning"}
                ]

        response1 = MockResponse1()
        reasoning = driver._extract_reasoning(response1)
        assert reasoning == "This is reasoning"

        # Test case 2: reasoning with list content
        class MockResponse2:
            def __init__(self):
                self.output = [
                    {
                        "type": "reasoning",
                        "content": [
                            {"text": "First part"},
                            {"text": "Second part"},
                            {"other": "ignored"},
                        ],
                    }
                ]

        response2 = MockResponse2()
        reasoning = driver._extract_reasoning(response2)
        assert reasoning == "First partSecond part"

        # Test case 3: reasoning with non-list content
        class MockResponse3:
            def __init__(self):
                self.output = [{"type": "reasoning", "content": "Simple reasoning"}]

        response3 = MockResponse3()
        reasoning = driver._extract_reasoning(response3)
        assert reasoning == "Simple reasoning"

        # Test case 4: no reasoning in output
        class MockResponse4:
            def __init__(self):
                self.output = [{"type": "text", "content": "Some text"}]

        response4 = MockResponse4()
        reasoning = driver._extract_reasoning(response4)
        assert reasoning is None

        # Test case 5: empty output
        class MockResponse5:
            def __init__(self):
                self.output = []

        response5 = MockResponse5()
        reasoning = driver._extract_reasoning(response5)
        assert reasoning is None
