"""
Unit tests for LLMManager with provider system integration.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest, LLMResponse
from local_coding_assistant.core.exceptions import LLMError
from local_coding_assistant.providers.base import ProviderLLMResponseDelta


class MockAsyncGenerator:
    """Mock async generator for testing streaming responses."""

    def __init__(self, deltas):
        self.deltas = deltas
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.deltas):
            raise StopAsyncIteration
        delta = self.deltas[self.index]
        self.index += 1
        return delta


class TestLLMManagerStreaming:
    """Test LLMManager streaming functionality."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        mock = MagicMock()
        mock.global_config = {"providers": {}, "models": {}}
        mock.env_manager = MagicMock()
        return mock

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_stream_method_exists(self, mock_config_manager):
        """Test that stream method exists and has correct signature."""
        # Initialize with the mock config manager
        manager = LLMManager(config_manager=mock_config_manager)

        # Verify method exists
        assert hasattr(manager, "stream")
        assert callable(manager.stream)

        # Test method signature
        sig = inspect.signature(manager.stream)
        assert "request" in sig.parameters

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming response."""

        # Mock provider manager and router
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()

        # Mock provider with streaming
        mock_provider = MagicMock()
        mock_provider.name = "openai"

        # Define an async generator for the mock
        async def mock_stream_with_retry(*args, **kwargs):
            for local_chunk in [
                ProviderLLMResponseDelta(content="Hello"),
                ProviderLLMResponseDelta(content=" world"),
                ProviderLLMResponseDelta(content="!"),
            ]:
                yield local_chunk

        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-3.5-turbo")
        )

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        manager.router = mock_router

        request = LLMRequest(prompt="Test prompt")
        chunks = []
        async for chunk in manager.stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks == ["Hello", " world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_tools(self):
        """Test streaming with tools in request."""

        # Mock provider manager and router
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()

        # Mock provider with streaming
        mock_provider = MagicMock()
        mock_provider.name = "openai"

        # Define an async generator for the mock
        async def mock_stream_with_retry(*args, **kwargs):
            for local_chunk in [
                ProviderLLMResponseDelta(content="I'll use the tool"),
            ]:
                yield local_chunk

        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-3.5-turbo")
        )

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        manager.router = mock_router

        request = LLMRequest(
            prompt="Use tool",
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
        )

        chunks = []
        async for chunk in manager.stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0] == "I'll use the tool"

    @pytest.mark.asyncio
    async def test_stream_openai_error(self):
        """Test streaming with provider errors."""
        from local_coding_assistant.providers.exceptions import ProviderError

        # Mock provider manager and router
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()

        # Mock failing provider with proper async generator
        mock_provider = AsyncMock()
        mock_provider.name = "openai"

        # Create an async generator that raises the error
        async def mock_stream_with_retry(*args, **kwargs):
            raise ProviderError("API Error", provider="openai")

        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-3.5-turbo")
        )

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        manager.router = mock_router

        request = LLMRequest(prompt="Test")

        with pytest.raises(LLMError, match="LLM streaming generation failed"):
            async for chunk in manager.stream(request):
                pass


class TestLLMManagerMethodConsistency:
    """Test consistency between generate and stream methods."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        mock = MagicMock()
        mock.global_config = {"providers": {}, "models": {}}
        return mock

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_stream_method_signature_matches_generate(self, mock_config_manager):
        """Test that stream method has similar signature to generate."""
        manager = LLMManager(config_manager=mock_config_manager)

        # Get method signatures
        generate_sig = inspect.signature(manager.generate)
        stream_sig = inspect.signature(manager.stream)

        # Check that they have the same parameters (except for return type)
        generate_params = [
            p
            for p in generate_sig.parameters.values()
            if p.name != "return" and p.name != "self"
        ]
        stream_params = [
            p
            for p in stream_sig.parameters.values()
            if p.name != "return" and p.name != "self"
        ]

        assert len(generate_params) == len(stream_params)
        for gp, sp in zip(generate_params, stream_params):
            assert gp.name == sp.name
            assert gp.kind == sp.kind
            assert gp.default == sp.default

    @pytest.mark.asyncio
    async def test_both_methods_use_same_config_resolution(
        self, mock_config_manager, mock_provider_manager
    ):
        """Test that both methods use the same config resolution."""
        # This test is not applicable to v2 architecture which uses provider system
        # The v2 manager resolves providers differently
        pass

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that error handling is consistent between methods."""
        from local_coding_assistant.providers.exceptions import ProviderError

        # Mock provider manager and router
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()

        # Mock failing provider
        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.generate_with_retry = AsyncMock(
            side_effect=ProviderError("API Error", provider="openai")
        )

        # Properly mock the async generator function
        async def mock_stream_with_retry(*args, **kwargs):
            raise ProviderError("API Error", provider="openai")

        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-3.5-turbo")
        )

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        manager.router = mock_router

        # Test both methods raise LLMError for similar failures
        # Generate should raise LLMError
        with pytest.raises(LLMError):
            await manager.generate(LLMRequest(prompt="test"))

        # Stream should also raise LLMError
        with pytest.raises(LLMError):
            async for chunk in manager.stream(LLMRequest(prompt="test")):
                pass


class TestLLMManagerIntegrationUpdates:
    """Test integration with updated systems."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        mock = MagicMock()
        mock.global_config = {"providers": {}, "models": {}}
        return mock

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        return MagicMock()

    def test_ainvoke_method_exists(self, mock_config_manager):
        """Test that ainvoke method exists as alias."""
        manager = LLMManager(config_manager=mock_config_manager)
        assert hasattr(manager, "ainvoke")
        assert callable(manager.ainvoke)

        # ainvoke should be functionally equivalent to generate
        # ainvoke provides a simpler interface that calls generate with defaults
        import inspect

        # Check that both methods exist and are callable
        assert callable(manager.ainvoke)
        assert callable(manager.generate)

        # Check that both methods are async
        assert inspect.iscoroutinefunction(manager.ainvoke)
        assert inspect.iscoroutinefunction(manager.generate)

        # ainvoke should have a simpler signature (only request parameter)
        # while generate has additional optional parameters
        ainvoke_sig = inspect.signature(manager.ainvoke)
        generate_sig = inspect.signature(manager.generate)

        # ainvoke should have exactly one parameter (request)
        assert len(ainvoke_sig.parameters) == 1
        assert "request" in ainvoke_sig.parameters

        # generate should have the request parameter plus optional parameters
        assert "request" in generate_sig.parameters
        assert len(generate_sig.parameters) > 1  # Has additional optional parameters

    @pytest.mark.asyncio
    async def test_ainvoke_functionality(
        self, mock_config_manager, mock_provider_manager
    ):
        """Test ainvoke method functionality."""
        from local_coding_assistant.providers.base import ProviderLLMResponse

        # Mock provider and router
        mock_provider = MagicMock()
        mock_provider.name = "test_provider"
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Test response",
                model="test-model",
                model_used="test-model",
                tokens_used=10,
            )
        )

        # Create the manager with mocks
        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )

        # Setup router mock
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "test-model")
        )
        manager.router = mock_router

        request = LLMRequest(prompt="Test")

        result = await manager.ainvoke(request)

        # Should call generate and return same result
        assert result.content == "Test response"
        assert (
            result.model_used == "test-model"
        )  # Should match the model we set in the mock


class TestLLMManagerBackwardCompatibility:
    """Test backward compatibility with existing code."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        mock = MagicMock()
        mock.global_config = {"providers": {}, "models": {}}
        return mock

    @pytest.fixture
    def mock_provider_manager(self):
        """Create a mock provider manager."""
        return MagicMock()

    def test_request_response_models_unchanged(self):
        """Test that LLMRequest and LLMResponse models are still compatible."""
        # Test LLMRequest creation (should work as before)
        request = LLMRequest(
            prompt="Test prompt",
            context={"user": "test"},
            system_prompt="System prompt",
            tools=[{"type": "function", "function": {"name": "test"}}],
            tool_outputs={"test": "output"},
        )

        assert request.prompt == "Test prompt"
        assert request.context == {"user": "test"}
        assert request.system_prompt == "System prompt"
        if request.tools is not None:
            assert len(request.tools) == 1
        assert request.tool_outputs == {"test": "output"}

        # Test LLMResponse creation (should work as before)
        response = LLMResponse(
            content="Test response",
            model_used="gpt-4",
            tokens_used=100,
            tool_calls=[{"id": "call_1", "type": "function"}],
        )

        assert response.content == "Test response"
        assert response.model_used == "gpt-4"
        assert response.tokens_used == 100
        if response.tool_calls is not None:
            assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_streaming_fallback_behavior(self):
        """Test that streaming falls back to regular generation when streaming not available."""
        from local_coding_assistant.providers.base import ProviderLLMResponse
        from local_coding_assistant.providers.exceptions import (
            ProviderConnectionError,
        )

        # Mock provider manager and router
        mock_provider_manager = MagicMock()
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()

        # Mock failing provider for streaming, successful for generation
        mock_provider = MagicMock()
        mock_provider.name = "openai"

        # Create an async generator that raises ProviderConnectionError immediately
        class FailingAsyncGenerator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ProviderConnectionError("Connection failed", provider="openai")

        # Mock stream_with_retry to return the failing async generator
        def mock_stream_with_retry(*args, **kwargs):
            return FailingAsyncGenerator()

        mock_provider.stream_with_retry = mock_stream_with_retry

        # Mock generate_with_retry to return a successful response
        mock_provider.generate_with_retry = AsyncMock(
            return_value=ProviderLLMResponse(
                content="Fallback response",
                model="gpt-3.5-turbo",
                tokens_used=50,
            )
        )

        # Mock router
        mock_router = MagicMock()
        mock_router.get_provider_for_request = AsyncMock(
            return_value=(mock_provider, "gpt-3.5-turbo")
        )
        mock_router._is_critical_error = MagicMock(return_value=True)

        manager = LLMManager(
            config_manager=mock_config_manager, provider_manager=mock_provider_manager
        )
        manager.router = mock_router

        request = LLMRequest(prompt="Test")

        # When streaming fails, should fall back to generate
        chunks = []
        async for chunk in manager.stream(request):
            chunks.append(chunk)

        # Should have yielded the complete response as a single chunk
        assert len(chunks) == 1
        assert chunks[0] == "Fallback response"

    def test_stream_method_documentation(self, mock_config_manager):
        """Test that stream method has proper documentation."""
        manager = LLMManager(config_manager=mock_config_manager)

        # Check that the method has a docstring
        docstring = manager.stream.__doc__
        assert docstring is not None

        # Check for key phrases in the docstring (case-insensitive)
        docstring_lower = docstring.lower()
        assert "streaming response" in docstring_lower
        assert "yield" in docstring_lower


class TestLLMManagerParameterConsistency:
    """Test parameter consistency between generate and stream methods."""

    @pytest.mark.asyncio
    async def test_parameter_passing_consistency(self):
        """Test that parameters are passed consistently between methods."""
        # This test is not applicable to v2 architecture which uses provider system
        # The v2 manager passes parameters through the provider system differently
        pass
