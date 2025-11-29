"""Unit tests for the base provider implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec
import pytest
from local_coding_assistant.providers.base import (
    BaseProvider,
    BaseDriver,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
    OptionalParameters,
)
from local_coding_assistant.providers.exceptions import ProviderAuthError, ProviderValidationError
from local_coding_assistant.config.schemas import ModelConfig


class TestBaseDriver:
    """Tests for the BaseDriver class."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock BaseDriver instance for testing."""
        # Create a mock that implements the abstract methods
        from local_coding_assistant.providers.base import BaseDriver
        
        class MockDriver(BaseDriver):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._generate_mock = AsyncMock()
                self._health_check_mock = AsyncMock(return_value=True)
                self._stream_mock = AsyncMock()
                self.api_key = "test-api-key"
                self.base_url = "http://test-url.com"
                
            async def generate(self, request):
                return await self._generate_mock(request)
                
            async def health_check(self):
                return await self._health_check_mock()
                
            async def stream(self, request):
                # Call the mock and get the async generator
                result = self._stream_mock(request)
                # If it's a coroutine, await it to get the async generator
                if asyncio.iscoroutine(result):
                    result = await result
                # Iterate through the async generator
                async for delta in result:
                    yield delta
                    
            async def generate_with_retry(self, request, max_retries=3, retry_delay=1.0):
                return await BaseDriver.generate_with_retry(self, request, max_retries, retry_delay)
                
            async def stream_with_retry(self, request, max_retries=3, retry_delay=1.0):
                async for delta in BaseDriver.stream_with_retry(self, request, max_retries, retry_delay):
                    yield delta
        
        return MockDriver(api_key="test-api-key", base_url="http://test-url.com")

    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self, mock_driver):
        """Test that generate_with_retry returns the response on success."""
        # Mock the generate method to return a successful response
        mock_response = ProviderLLMResponse(content="Test response", model="test-model")
        mock_driver._generate_mock.return_value = mock_response

        # Create a test request
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )

        # Call the method with retry
        response = await mock_driver.generate_with_retry(request, max_retries=2)

        # Assert the response is as expected
        assert response == mock_response
        mock_driver._generate_mock.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_generate_with_retry_failure(self, mock_driver):
        """Test that generate_with_retry raises the last error after all retries."""
        # Mock the generate method to raise an exception
        error = Exception("Test error")
        mock_driver._generate_mock.side_effect = error

        # Create a test request
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )

        # Call the method with retry and expect it to raise the error
        with pytest.raises(Exception) as exc_info:
            await mock_driver.generate_with_retry(request, max_retries=2, retry_delay=0.1)

        # Assert the error is as expected
        assert str(exc_info.value) == str(error)
        assert mock_driver._generate_mock.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_stream_with_retry_success(self, mock_driver):
        """Test that stream_with_retry yields deltas and doesn't retry on success."""
        # Create test deltas
        delta1 = ProviderLLMResponseDelta(content="Hello")
        delta2 = ProviderLLMResponseDelta(content=" World")

        # Mock the stream method to return an async generator
        async def mock_stream(_):
            yield delta1
            yield delta2

        mock_driver._stream_mock.side_effect = mock_stream

        # Create a test request
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )

        # Call the method with retry
        deltas = []
        async for delta in mock_driver.stream_with_retry(request, max_retries=2):
            deltas.append(delta)

        # Assert the deltas are as expected
        assert deltas == [delta1, delta2]

    @pytest.mark.asyncio
    async def test_stream_with_retry_failure(self, mock_driver):
        """Test that stream_with_retry raises the last error after all retries."""
        # Mock the stream method to raise an exception
        error = Exception("Test error")
        mock_driver._stream_mock.side_effect = error

        # Create a test request
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )

        # Call the method with retry and expect it to raise the error
        with pytest.raises(Exception) as exc_info:
            async for _ in mock_driver.stream_with_retry(request, max_retries=2, retry_delay=0.1):
                pass

        # Assert the error is as expected
        assert str(exc_info.value) == str(error)
        assert mock_driver._stream_mock.call_count == 3  # Initial + 2 retries


class TestBaseProvider:
    """Tests for the BaseProvider class."""

    @pytest.fixture
    def mock_env_manager(self):
        """Create a mock EnvManager instance."""
        mock = MagicMock()
        mock.get_env.return_value = "env-api-key"
        return mock

    @pytest.fixture
    def mock_driver(self):
        """Create a mock BaseDriver instance."""
        mock = create_autospec(
            BaseDriver,
            instance=True,
            spec_set=True,
            generate=AsyncMock(),
            health_check=AsyncMock(return_value=True),
            stream=AsyncMock()
        )
        return mock

    @pytest.fixture
    def provider(self, mock_env_manager, mock_driver):
        """Create a test provider instance with mocks."""
        # Create a test provider class that implements the abstract method
        class TestProvider(BaseProvider):
            def _create_driver_instance(self):
                return mock_driver

        # Create model configs for testing
        model_configs = [
            ModelConfig(
                name="test-model",
                supported_parameters=["temperature", "top_p"]
            ),
            ModelConfig(
                name="another-model",
                supported_parameters=["temperature", "max_tokens"]
            )
        ]

        return TestProvider(
            name="test-provider",
            base_url="http://test.com",
            env_manager=mock_env_manager,
            api_key="test-api-key",
            models=model_configs,
        )

    def test_get_api_key_from_env(self, mock_env_manager):
        """Test getting API key from environment variable."""
        # Create a test provider class that implements the abstract method
        class TestProvider(BaseProvider):
            def _create_driver_instance(self):
                return MagicMock()
                
        # Create a test provider with API key from env
        provider = TestProvider(
            name="test-provider",
            base_url="http://test.com",
            env_manager=mock_env_manager,
            api_key_env="TEST_API_KEY",
            models=[],
        )
        
        # The _get_api_key method should get the key from the environment
        assert provider._get_api_key() == "env-api-key"
        # The method is called during initialization and in _get_api_key
        assert mock_env_manager.get_env.call_count >= 1
        assert any(call("TEST_API_KEY") in mock_env_manager.get_env.mock_calls for call in mock_env_manager.get_env.mock_calls)

    @pytest.mark.asyncio
    async def test_handle_auth_error_with_env_key(self, provider, mock_env_manager, mock_driver):
        """Test handling auth error with fallback to environment key."""
        # Set up the test
        provider.api_key = "old-key"
        provider.api_key_env = "TEST_API_KEY"
    
        # Create a test error
        error = ProviderAuthError("Invalid API key")
        
        # Configure the mock to raise an error on the first call and return a mock driver on the second
        mock_driver.health_check.side_effect = [error, True]
        
        # Mock the _create_driver_instance method
        with patch.object(provider, '_create_driver_instance', return_value=mock_driver):
            # Call the method
            await provider._handle_auth_error(error)
            
            # Should have tried to get the key from env
            mock_env_manager.get_env.assert_called_with("TEST_API_KEY")
            
            # Should have updated the API key and recreated the driver
            assert provider.api_key == "env-api-key"
    
    def test_handle_auth_error_no_env_key(self, provider, mock_env_manager):
        """Test handling auth error with no environment key fallback."""
        # Set up the test with no API key env var
        provider.api_key = "test-key"
        provider.api_key_env = None
        
        # Create a test error
        error = ProviderAuthError("Invalid API key")
        
        # Call the method and expect the same error
        with pytest.raises(ProviderAuthError) as exc_info:
            asyncio.run(provider._handle_auth_error(error))
            
        # Should not have tried to get the key from env
        mock_env_manager.get_env.assert_not_called()
        assert exc_info.value == error

    def test_validate_request_success(self, provider):
        """Test successful request validation."""
        # Mock the model config
        model_config = ModelConfig(
            name="test-model",
            parameters=["temperature", "max_tokens"],
        )
        provider.models = [model_config]
        
        # Create a valid request
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
            parameters=OptionalParameters(
                temperature=0.7,
                max_tokens=100,
            )
        )
        
        # Should not raise an exception
        provider.validate_request(request)


    def test_validate_request_unknown_model(self, provider):
        """Test request validation with unknown model."""
        # Set up the test with no models
        provider.models = []
        
        # Create a request with an unknown model
        request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="unknown-model",
        )
        
        # Should raise a validation error
        with pytest.raises(ProviderValidationError) as exc_info:
            provider.validate_request(request)
            
        assert "Model 'unknown-model' is not supported by provider 'test-provider'" in str(exc_info.value)
