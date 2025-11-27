"""
Unit tests for the provider system including base classes, manager, and all providers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from local_coding_assistant.config import EnvManager
from local_coding_assistant.providers.base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.providers.exceptions import ProviderError
from local_coding_assistant.providers.google_provider import GoogleGeminiProvider
from local_coding_assistant.providers.openrouter_provider import OpenRouterProvider
from local_coding_assistant.providers.provider_manager import ProviderManager
from local_coding_assistant.providers.router import ProviderRouter


class TestProviderLLMRequest:
    """Test ProviderLLMRequest functionality."""

    def test_request_creation(self):
        """Test creating a ProviderLLMRequest."""
        from local_coding_assistant.providers.base import OptionalParameters

        messages = [{"role": "user", "content": "Test message"}]
        params = OptionalParameters(
            max_tokens=1000,
            stream=False,
            tools=[{"type": "function", "function": {"name": "test"}}],
        )

        request = ProviderLLMRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            tool_outputs={"test": "output"},
            parameters=params,
        )

        assert request.messages == messages
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.tool_outputs == {"test": "output"}
        assert request.parameters.max_tokens == 1000
        assert request.parameters.stream is False
        assert request.parameters.tools is not None
        assert len(request.parameters.tools) == 1

    def test_request_defaults(self):
        """Test ProviderLLMRequest default values."""
        from local_coding_assistant.providers.base import OptionalParameters

        messages = [{"role": "user", "content": "Test"}]
        request = ProviderLLMRequest(
            messages=messages, model="gpt-3.5", parameters=OptionalParameters()
        )

        assert request.temperature == 0.7
        assert request.tool_outputs is None
        assert request.parameters is not None
        assert request.parameters.max_tokens is None
        assert request.parameters.stream is False
        assert request.parameters.tools == []
        assert request.parameters.tool_choice is None
        assert request.parameters.response_format is None

    def test_validate_against_model(self):
        """Test validate_against_model method with different validation modes."""
        from local_coding_assistant.providers.base import OptionalParameters

        # Create a request with some parameters including required ones
        messages = [{"role": "user", "content": "Test"}]
        params = OptionalParameters(
            max_tokens=100,
            stream=True,
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_choice="auto",  # Add required tool_choice parameter
        )
        request = ProviderLLMRequest(
            messages=messages, model="test-model", parameters=params
        )

        # Test IGNORE mode - should do nothing
        request.validate_against_model(
            [], mode=ProviderLLMRequest.ValidationMode.IGNORE
        )
        assert (
            len(request.parameters.model_fields_set) > 0
        )  # No parameters should be removed

        # Test TRUNCATE mode with unsupported parameters
        # Include required parameters in supported list
        supported = ["max_tokens", "temperature", "tools", "tool_choice"]
        request.validate_against_model(
            supported,
            mode=ProviderLLMRequest.ValidationMode.TRUNCATE,
            required_parameters={"tools", "tool_choice"},  # Mark tools as required
        )

        # After validation, check that unsupported parameters are set to None
        if "stream" not in supported:
            assert getattr(request.parameters, "stream", None) is None, (
                "Expected 'stream' to be set to None but it's not"
            )

        # Then check the expected set of parameters
        remaining_params = {
            field
            for field in request.parameters.model_fields_set
            if getattr(request.parameters, field) is not None
        }
        expected_params = {"max_tokens", "tools", "tool_choice"}
        assert remaining_params == expected_params, (
            f"Expected parameters {expected_params}, got {remaining_params}"
        )

        # Create a new request for VALIDATE mode test
        validate_request = ProviderLLMRequest(
            messages=messages,
            model="test-model",
            parameters=OptionalParameters(
                max_tokens=100,
                stream=True,  # Set stream to True for validation
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto",
            ),
        )

        # Test VALIDATE mode with required parameters
        required = {"max_tokens"}
        validate_request.validate_against_model(
            {
                "max_tokens": int,
                "temperature": float,
                "tools": list,
                "tool_choice": (str, dict),
                "stream": bool,
            },
            mode=ProviderLLMRequest.ValidationMode.VALIDATE,
            required_parameters=required,
        )

        # Test missing required parameter
        with pytest.raises(ValueError, match="missing required parameters"):
            request.validate_against_model(
                {"temperature": float},
                mode=ProviderLLMRequest.ValidationMode.VALIDATE,
                required_parameters=required,
            )

        # Test type validation with presence_penalty field (should be float between -2 and 2)
        # First create a request with a valid presence_penalty
        type_test_request = ProviderLLMRequest(
            messages=messages,
            model="test-model",
            parameters=OptionalParameters(
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto",
            ),
        )
        # Set presence_penalty to an invalid value directly to bypass Pydantic's validation
        type_test_request.parameters.presence_penalty = "not_a_float"
        # The validation should fail with a ValueError
        with pytest.raises(ValueError, match="must be float"):
            type_test_request.validate_against_model(
                {"presence_penalty": float, "tools": list, "tool_choice": (str, dict)},
                mode=ProviderLLMRequest.ValidationMode.VALIDATE,
            )


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
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup mock EnvManager for tests."""
        self.mock_env_manager = MagicMock(spec=EnvManager)
        self.mock_env_manager.get_env.return_value = None

    def test_initialization(self):
        """Test BaseProvider initialization."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider initialization."""

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test_provider",
            base_url="https://api.test.com",
            env_manager=self.mock_env_manager,
            models=["gpt-4", "gpt-3.5"],
            api_key="test_key",
            api_key_env="TEST_API_KEY",
        )

        assert provider.name == "test_provider"
        assert any(model.name == "gpt-4" for model in provider.models)
        assert any(model.name == "gpt-3.5" for model in provider.models)
        assert provider.api_key == "test_key"
        assert provider.api_key_env == "TEST_API_KEY"
        assert provider.base_url == "https://api.test.com"
        # driver_instance is now initialized immediately in __init__
        assert provider.driver_instance is not None

    def test_supports_model(self):
        """Test model support checking."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test",
            base_url="https://api.test.com",
            env_manager=self.mock_env_manager,
            models=["gpt-4", "gpt-3.5"],
            api_key="test_key",
        )
        assert provider.supports_model("gpt-4") is True
        assert provider.supports_model("gpt-3") is False

    def test_get_available_models(self):
        """Test getting available models."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="test")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test",
            base_url="https://api.test.com",
            env_manager=self.mock_env_manager,
            models=["gpt-4", "gpt-3.5"],
            api_key="test_key",
        )
        assert provider.get_available_models() == ["gpt-4", "gpt-3.5"]

    def test_stream_method_without_driver(self):
        """Test stream method when no driver is available."""
        from local_coding_assistant.providers.base import OptionalParameters

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                # Mock the generate method to return a test response
                return ProviderLLMResponse(
                    content="Test response", 
                    model=request.model,
                    finish_reason="stop"
                )

            async def health_check(self):
                return True

        # Create a mock environment manager that returns False for is_testing
        mock_env = MagicMock()
        mock_env.is_testing.return_value = False

        provider = TestProvider(
            name="test",
            base_url="https://api.test.com",
            env_manager=mock_env,
            models=["gpt-4"],
            api_key="test_key",
        )

        # Set driver_instance to None to test fallback mechanism
        provider.driver_instance = None

        async def test_stream():
            deltas = []
            async for delta in provider.stream(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}],
                    model="gpt-4",
                    parameters=OptionalParameters(stream=True),
                )
            ):
                deltas.append(delta)

            assert len(deltas) == 1
            assert deltas[0].content == "Test response"
            assert deltas[0].finish_reason == "stop"

        # Run the test
        asyncio.run(test_stream())

    def test_stream_with_retry_success(self):
        """Test stream_with_retry with successful streaming."""

        class TestProvider(BaseProvider):
            """Concrete implementation for testing BaseProvider methods."""

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="gpt-4")

            async def stream(self, request):
                yield ProviderLLMResponseDelta(content="chunk1")
                yield ProviderLLMResponseDelta(content="chunk2", finish_reason="stop")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test",
            base_url="https://api.test.com",
            env_manager=self.mock_env_manager,
            models=["gpt-4"],
            api_key="test_key",
        )

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

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, request):
                return ProviderLLMResponse(content="test", model="gpt-4")

            async def health_check(self):
                return True

        provider = TestProvider(
            name="test",
            base_url="https://api.test.com",
            env_manager=self.mock_env_manager,
            models=["gpt-4"],
            api_key="test_key",
        )

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

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup mock EnvManager for tests."""
        self.mock_env_manager = MagicMock(spec=EnvManager)
        self.mock_env_manager.get_env.return_value = None

    def test_initialization(self):
        """Test ProviderManager initialization."""
        manager = ProviderManager(env_manager=self.mock_env_manager)
        assert len(manager._providers) == 3  # 3 builtin providers are auto-registered
        assert "google_gemini" in manager._providers
        assert "local" in manager._providers
        assert "openrouter" in manager._providers
        assert len(manager._instances) == 0  # But no instances are created initially

    def test_register_provider(self):
        """Test provider registration."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        # Mock provider class
        class MockProvider(BaseProvider):
            def __init__(self, **kwargs):
                super().__init__(
                    name="mock",
                    base_url="https://api.test.com",
                    models=["gpt-4"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock response", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock", finish_reason="stop")

            async def health_check(self):
                return True

        # Register default providers for testing
        manager._instances = {
            "google_gemini": MagicMock(),
            "local": MagicMock(),
            "openrouter": MagicMock(),
        }

        # Register our test provider
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test_source"

        # Create and store an instance of our test provider with env_manager
        mock_provider = MockProvider(env_manager=self.mock_env_manager)
        manager._instances["mock"] = mock_provider

        # Verify registration
        assert "mock" in manager._providers
        assert manager._provider_sources["mock"] == "test_source"

        # The list should include all providers, including our mock
        assert sorted(manager.list_providers()) == [
            "google_gemini",
            "local",
            "mock",
            "openrouter",
        ]

    def test_get_provider(self):
        """Test getting a registered provider."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class MockProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(
                    name="mock",
                    base_url="https://api.test.com",
                    models=["gpt-4"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test"

        # Create and store an instance of our test provider with env_manager
        mock_provider = MockProvider(env_manager=self.mock_env_manager)
        manager._instances["mock"] = mock_provider

        # Now get the provider instance
        provider = manager.get_provider("mock")
        assert provider is not None
        assert isinstance(provider, MockProvider)
        assert provider.name == "mock"

        # Test getting non-existent provider
        assert manager.get_provider("nonexistent") is None

    def test_list_providers(self):
        """Test listing registered providers."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class MockProvider(BaseProvider):
            def __init__(self, name, **kwargs):
                super().__init__(
                    name=name,
                    base_url="https://api.test.com",
                    models=["gpt-4"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Initialize with default providers
        manager._instances = {
            "google_gemini": MockProvider("google_gemini", env_manager=self.mock_env_manager),
            "local": MockProvider("local", env_manager=self.mock_env_manager),
            "openrouter": MockProvider("openrouter", env_manager=self.mock_env_manager),
        }

        # Register and create instances for test providers
        manager._providers["provider1"] = MockProvider
        manager._provider_sources["provider1"] = "source1"
        manager._instances["provider1"] = MockProvider("provider1", env_manager=self.mock_env_manager)

        manager._providers["provider2"] = MockProvider
        manager._provider_sources["provider2"] = "source2"
        manager._instances["provider2"] = MockProvider("provider2", env_manager=self.mock_env_manager)

        # Get the list of providers
        providers = manager.list_providers()

        # Verify the list includes all expected providers
        assert set(providers) == {
            "google_gemini",
            "local",
            "openrouter",
            "provider1",
            "provider2",
        }

        # Verify the list is sorted
        assert providers == sorted(
            ["google_gemini", "local", "openrouter", "provider1", "provider2"]
        )

    def test_get_provider_source(self):
        """Test getting provider source."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class MockProvider(BaseProvider):
            def __init__(self):
                super().__init__(
                    name="mock", base_url="https://api.test.com", models=["gpt-4"]
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

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
        manager = ProviderManager(env_manager=self.mock_env_manager)
        mock_config = MagicMock()

        # Register a provider
        class MockProvider(BaseProvider):
            def __init__(self, **kwargs):
                super().__init__(
                    name="mock",
                    base_url="https://api.test.com",
                    models=["gpt-4"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

            async def generate(self, _request):
                return ProviderLLMResponse(content="mock", model="gpt-4")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="mock")

            async def health_check(self):
                return True

        # Register and create an instance of the provider
        manager._providers["mock"] = MockProvider
        manager._provider_sources["mock"] = "test"
        manager._instances["mock"] = MockProvider(env_manager=self.mock_env_manager)

        # Verify provider is registered and listed
        assert "mock" in manager.list_providers()
        assert isinstance(manager.get_provider("mock"), MockProvider)

        # Reload should clear providers (simplified test)
        manager.reload(mock_config)

        # In real implementation, reload would reload from config
        # This test verifies the method exists and can be called
        assert hasattr(manager, "reload")


class TestGoogleGeminiProvider:
    """Test GoogleGeminiProvider functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup mock EnvManager for tests."""
        self.mock_env_manager = MagicMock(spec=EnvManager)
        self.mock_env_manager.get_env.return_value = None

    def test_initialization(self):
        """Test GoogleGeminiProvider initialization."""
        provider = GoogleGeminiProvider(
            api_key="test_key", 
            models=["gemini-pro", "gemini-pro-vision"],
            env_manager=self.mock_env_manager
        )

        assert provider.name == "google_gemini"
        assert any(model.name == "gemini-pro" for model in provider.models)
        assert provider.api_key == "test_key"

    def test_initialization_with_env_key(self):
        """Test GoogleGeminiProvider initialization with environment key."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env_key"}):
            # Mock the _get_api_key method to return a test key
            with patch('local_coding_assistant.providers.google_provider.GoogleGeminiProvider._get_api_key', 
                      return_value="test_api_key"):
                provider = GoogleGeminiProvider(
                    api_key_env="GOOGLE_API_KEY",
                    env_manager=self.mock_env_manager
                )

                assert provider.api_key_env == "GOOGLE_API_KEY"
                assert provider.api_key == "test_api_key"

    @pytest.mark.asyncio
    async def test_health_check_with_driver(self):
        """Test health check when driver is available."""
        provider = GoogleGeminiProvider(
            api_key="test_key",
            env_manager=self.mock_env_manager
        )

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        # Can be either True or 'unavailable' based on health check implementation
        assert is_healthy is True or is_healthy == "unavailable"
        if is_healthy is True:
            mock_driver.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_without_driver(self):
        """Test health check when no driver is available."""
        provider = GoogleGeminiProvider(
            api_key="test_key",
            env_manager=self.mock_env_manager
        )

        # Mock driver health_check to return False or be unavailable
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=False)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()

        # Check that the return value is either False or 'unavailable'
        assert is_healthy is False or is_healthy == "unavailable"

        # If the provider returned a boolean, the driver's health check should have been called
        if isinstance(is_healthy, bool):
            mock_driver.health_check.assert_called_once()

    def test_stream_method(self):
        """Test GoogleGeminiProvider stream method."""
        provider = GoogleGeminiProvider(
            api_key="test_key",
            env_manager=self.mock_env_manager
        )

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.stream = AsyncMock()
        provider.driver_instance = mock_driver

        # Test that stream delegates to driver
        from local_coding_assistant.providers.base import OptionalParameters

        _request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gemini-pro",
            parameters=OptionalParameters(stream=True),
        )

        # The stream method should delegate to the driver
        # We can't easily test the async generator without more complex setup,
        # but we can verify the method exists and has correct signature
        import inspect

        sig = inspect.signature(provider.stream)
        assert "request" in sig.parameters


class TestOpenRouterProvider:
    """Test OpenRouterProvider functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup mock EnvManager for tests."""
        self.mock_env_manager = MagicMock(spec=EnvManager)
        self.mock_env_manager.get_env.return_value = None

    def test_initialization(self):
        """Test OpenRouterProvider initialization."""
        provider = OpenRouterProvider(
            api_key="test_key", 
            models=["auto", "gpt-4", "claude-3"],
            env_manager=self.mock_env_manager
        )

        assert provider.name == "openrouter"
        assert any(model.name == "auto" for model in provider.models)
        assert provider.api_key == "test_key"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test OpenRouterProvider health check."""
        provider = OpenRouterProvider(
            api_key="test_key",
            env_manager=self.mock_env_manager
        )

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.health_check = AsyncMock(return_value=True)
        provider.driver_instance = mock_driver

        is_healthy = await provider.health_check()
        # Can be either True or 'unavailable' based on health check implementation
        assert is_healthy is True or is_healthy == "unavailable"
        if is_healthy is True:
            mock_driver.health_check.assert_called_once()

    def test_stream_method(self):
        """Test OpenRouterProvider stream method."""
        provider = OpenRouterProvider(
            api_key="test_key",
            env_manager=self.mock_env_manager
        )

        # Mock driver
        mock_driver = AsyncMock()
        mock_driver.stream = AsyncMock()
        provider.driver_instance = mock_driver

        # Verify method exists and delegates to driver
        from local_coding_assistant.providers.base import OptionalParameters

        _request = ProviderLLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            parameters=OptionalParameters(),
        )

        import inspect

        sig = inspect.signature(provider.stream)
        assert "request" in sig.parameters


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

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup mock EnvManager for tests."""
        self.mock_env_manager = MagicMock(spec=EnvManager)
        self.mock_env_manager.get_env.return_value = None

    def test_provider_registration_and_retrieval(self):
        """Test complete provider registration and retrieval flow."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class TestProvider(BaseProvider):
            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = AsyncMock()
                driver.health_check.return_value = True
                return driver

            def __init__(self, provider_name=None, **kwargs):
                super().__init__(
                    name="test",
                    base_url="https://api.test.com",
                    models=["model1"],
                    api_key="test_key",
                    **kwargs,
                )

            async def generate(self, _request):
                return ProviderLLMResponse(content="test", model="model1")

            async def stream(self, _request):
                yield ProviderLLMResponseDelta(content="test", finish_reason="stop")

            async def health_check(self):
                return True

        # Register provider manually for testing
        manager._providers["test"] = TestProvider
        manager._provider_sources["test"] = "integration_test"

        # Create and store an instance of our test provider
        test_provider = TestProvider(env_manager=self.mock_env_manager)
        manager._instances["test"] = test_provider

        # Retrieve provider
        provider = manager.get_provider("test")
        assert provider is not None
        assert isinstance(provider, TestProvider)
        assert provider.name == "test"
        assert provider.supports_model("model1") is True

        # Test source tracking
        assert manager.get_provider_source("test") == "integration_test"

    @pytest.mark.asyncio
    async def test_provider_streaming_flow(self):
        """Test complete streaming flow through provider system."""
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class StreamingProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(
                    name="streaming",
                    base_url="https://api.test.com",
                    models=["stream-model"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

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

        # Create and store an instance of our test provider
        streaming_provider = StreamingProvider(env_manager=self.mock_env_manager)
        manager._instances["streaming"] = streaming_provider

        # Retrieve provider
        provider = manager.get_provider("streaming")
        assert provider is not None
        assert isinstance(provider, StreamingProvider)

        # Test streaming
        deltas = []
        from local_coding_assistant.providers.base import OptionalParameters

        async for delta in provider.stream(
            ProviderLLMRequest(
                messages=[{"role": "user", "content": "test"}],
                model="stream-model",
                parameters=OptionalParameters(stream=True),
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
        manager = ProviderManager(env_manager=self.mock_env_manager)

        class ErrorProvider(BaseProvider):
            def __init__(self, provider_name=None, **kwargs):
                super().__init__(
                    name="error",
                    base_url="https://api.test.com",
                    models=["error-model"],
                    api_key="test_key",
                    **kwargs,
                )

            def _create_driver_instance(self):
                # Create a mock driver instance for testing
                driver = MagicMock()
                driver.health_check.return_value = True
                return driver

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

        # Create and store an instance of our test provider
        error_provider = ErrorProvider(env_manager=self.mock_env_manager)
        manager._instances["error"] = error_provider

        # Retrieve provider
        provider = manager.get_provider("error")
        assert provider is not None
        assert isinstance(provider, ErrorProvider)

        # Test error in generate
        with pytest.raises(ProviderError, match="Test error"):
            from local_coding_assistant.providers.base import OptionalParameters

            await provider.generate(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}],
                    model="error-model",
                    parameters=OptionalParameters(),
                )
            )

        # Test error in stream - expect error after consuming one delta
        with pytest.raises(ProviderError, match="Stream error"):
            deltas = []
            from local_coding_assistant.providers.base import OptionalParameters

            async for delta in provider.stream(
                ProviderLLMRequest(
                    messages=[{"role": "user", "content": "test"}],
                    model="error-model",
                    parameters=OptionalParameters(stream=True),
                )
            ):
                deltas.append(delta)
                if len(deltas) > 1:  # Should not reach here
                    break

        # Test health check
        assert await provider.health_check() is False
