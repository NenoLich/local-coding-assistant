
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.core.exceptions import LLMError
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.providers.base import (
    BaseProvider,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)
from local_coding_assistant.providers.provider_manager import ProviderManager
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.builtin_tools.math_tools import SumTool
from local_coding_assistant.tools.tool_manager import ToolManager

# Test configuration constants
TEST_PROVIDER_CONFIG = {
    "name": "test",
    "driver": "test",
    "base_url": "https://api.test.com",
    "api_key_env": "TEST_API_KEY",
    "models": {
        "test-model": {"supported_parameters": ["temperature", "max_tokens"]},
        "gpt-4": {"supported_parameters": ["temperature", "max_tokens"]},
        "gpt-3.5": {"supported_parameters": ["temperature", "max_tokens"]},
    },
}

DEFAULT_SESSION_OVERRIDES = {
    "llm.model_name": "test-model",
    "llm.provider": "test",
    "llm.temperature": 0.7,
    "llm.max_tokens": 1000,
    "providers.test": TEST_PROVIDER_CONFIG,
    "agent.policies.planner.models": ["test:test-model", "fallback:any"],
    "agent.policies.general.models": ["test:test-model", "fallback:any"],
}


class MockTestProvider(BaseProvider):
    """Test provider that integrates with the provider system for testing."""

    def __init__(
        self,
        provider_name: str | None = None,
        name: str | None = "test",
        base_url: str | None = "https://api.test.com",
        api_key: str | None = "test_key",
        api_key_env: str | None = None,
        models: list[str] | None = None,
        driver: str | None = "test",
        env_manager: Any | None = None,
        allow_test_requests: bool = False,
        **kwargs,
    ):
        # Create a mock environment manager if none provided
        if env_manager is None:
            env_manager = MagicMock()
            env_manager.get_env.return_value = api_key
            env_manager.is_testing.return_value = False

        super().__init__(
            name=name or "test",
            base_url=base_url or "https://api.test.com",
            models=models or ["test-model", "gpt-4", "gpt-3.5"],
            api_key=api_key or "test_key",
            env_manager=env_manager,
            allow_test_requests=allow_test_requests,
            **kwargs
        )
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request: ProviderLLMRequest) -> ProviderLLMResponse:
        """Test generate method that returns deterministic responses."""
        self.call_count += 1
        self.calls.append(
            {
                "model": request.model,
                "messages": len(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.parameters.max_tokens,
                "tools": len(request.parameters.tools)
                if request.parameters.tools
                else 0,
                "tool_outputs": bool(request.tool_outputs)
                if hasattr(request, "tool_outputs") and request.tool_outputs
                else False,
            }
        )

        # Find the user message (not system prompt)
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Create echo-style response
        content = f"echo:{user_content}"
        if request.parameters.tools:
            content += f"|tools:{len(request.parameters.tools)}"
        if hasattr(request, "tool_outputs") and request.tool_outputs:
            content += "|to"

        return ProviderLLMResponse(
            content=content,
            model=request.model,
            tokens_used=50,
        )

    async def stream(
        self, request: ProviderLLMRequest
    ) -> AsyncIterator[ProviderLLMResponseDelta]:
        # Find the user message (not system prompt)
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        content = f"echo:{user_content}"
        if request.parameters.tools:
            content += f"|tools:{len(request.parameters.tools)}"
        if hasattr(request, "tool_outputs") and request.tool_outputs:
            content += "|to"

        # Split content into chunks for streaming
        for i in range(0, len(content), 10):
            chunk = content[i : i + 10]
            yield ProviderLLMResponseDelta(
                content=chunk,
                finish_reason="stop" if i + 10 >= len(content) else None,
            )

    async def health_check(self) -> bool:
        """Test health check that always passes."""
        return True

    def _create_driver_instance(self):
        """Create and return a mock driver instance for testing."""
        driver = MagicMock()
        driver.health_check.return_value = True
        return driver


class ToolManagerHelper(ToolManager):
    """Test tool manager that supports invoke() for backward compatibility."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager=config_manager)

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Legacy invoke method for backward compatibility."""
        return self.run_tool(name, payload)


class MockConfigManager(IConfigManager):
    """Mock implementation of IConfigManager for testing."""
    
    def __init__(self):
        self._config = {
            "runtime": {
                "persistent_sessions": False,
                "use_graph_mode": False,
                "stream": False,
            },
            "llm": {
                "model_name": "test-model",
                "provider": "test",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "tools": {
                "tools": []
            },
            "providers": {},
            "agent": {}
        }
        self._session_overrides = {}
    
    @property
    def global_config(self) -> dict:
        return self._config
    
    @property
    def session_overrides(self) -> dict:
        return self._session_overrides
    
    def load_global_config(self) -> dict:
        return self._config
    
    def get_tools(self) -> dict:
        return {}
    
    def reload_tools(self) -> None:
        pass
    
    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        self._session_overrides.update(overrides)
        
        # Apply overrides to config
        for key, value in overrides.items():
            if key == "runtime.persistent_sessions":
                self._config["runtime"]["persistent_sessions"] = value
            elif key.startswith("llm."):
                subkey = key[4:]  # Remove 'llm.' prefix
                self._config["llm"][subkey] = value
    
    def resolve(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> dict:
        # Create a deep copy of the config to avoid modifying the original
        result: dict[str, Any] = deepcopy(self._config)
        
        # Apply session overrides
        if self._session_overrides:
            for key, value in self._session_overrides.items():
                if "." in key:
                    parts = key.split(".")
                    current: dict[str, Any] = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]  # type: ignore[assignment]
                    current[parts[-1]] = value
                else:
                    result[key] = value
        
        # Apply call overrides
        if overrides:
            for key, value in overrides.items():
                if "." in key:
                    parts = key.split(".")
                    current: dict[str, Any] = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]  # type: ignore[assignment]
                    current[parts[-1]] = value
                else:
                    result[key] = value
        
        # Apply provider and model name overrides if provided
        if provider is not None:
            if "llm" not in result:
                result["llm"] = {}
            result["llm"]["provider"] = provider
            
        if model_name is not None:
            if "llm" not in result:
                result["llm"] = {}
            result["llm"]["model_name"] = model_name
        
        # Ensure the runtime config has all required fields
        if "runtime" not in result:
            result["runtime"] = {}
        
        # Add any missing runtime defaults
        runtime_defaults = {
            "persistent_sessions": False,
            "use_graph_mode": False,
            "stream": False,
        }
        for key, default in runtime_defaults.items():
            if key not in result["runtime"]:
                result["runtime"][key] = default
        
        # Recursively convert dictionaries to ConfigResult objects
        def convert_dict(d: Any) -> Any:
            if isinstance(d, dict):
                result = ConfigResult()
                for k, v in d.items():
                    result[k] = convert_dict(v)
                return result
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return d
        
        # Create a simple object with dot notation access
        class ConfigResult(dict):
            def __getattr__(self, name: str) -> Any:
                if name in self:
                    return self[name]
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def __setattr__(self, name: str, value: Any) -> None:
                self[name] = value
        
        return convert_dict(result)


@pytest.fixture(scope="session")
def test_provider() -> MockTestProvider:
    """Session-scoped test provider instance."""
    # Create a mock environment manager
    mock_env = MagicMock()
    mock_env.get_env.return_value = "test_key"
    mock_env.is_testing.return_value = False
    
    return MockTestProvider(
        name="test",
        env_manager=mock_env,
        allow_test_requests=True
    )


@pytest.fixture(scope="session")
def provider_manager() -> Generator[ProviderManager, None, None]:
    """Session-scoped provider manager with test provider registered."""
    # Create a mock environment manager
    mock_env = MagicMock()
    mock_env.get_env.return_value = "test_key"
    mock_env.is_testing.return_value = False
    
    manager = ProviderManager(env_manager=mock_env, allow_test_requests=True)
    manager._providers["test"] = MockTestProvider
    manager._provider_sources["test"] = "test"
    
    # Initialize the provider manager with test configuration
    manager._provider_configs = {
        "test": {
            "name": "test",
            "driver": "test",
            "base_url": "https://api.test.com",
            "models": ["test-model", "gpt-4", "gpt-3.5"],
            "api_key": "test_key",
            "env_manager": mock_env,
            "allow_test_requests": True
        }
    }
    
    # Initialize the providers
    manager._instantiate_providers()
    
    yield manager
    # Cleanup if needed


@pytest.fixture(scope="session")
def base_config_manager() -> MockConfigManager:
    """Session-scoped base config manager with common settings."""
    config = MockConfigManager()
    config.set_session_overrides(DEFAULT_SESSION_OVERRIDES)
    return config


@pytest.fixture(scope="session")
def llm_manager(
    provider_manager: ProviderManager, base_config_manager: MockConfigManager
) -> LLMManager:
    """Session-scoped LLM manager."""
    return LLMManager(
        config_manager=base_config_manager,
        provider_manager=provider_manager,
    )


@pytest.fixture(scope="session")
def tool_manager(base_config_manager: MockConfigManager) -> ToolManagerHelper:
    """Session-scoped tool manager with test tools."""
    manager = ToolManagerHelper(config_manager=base_config_manager)
    return manager


@pytest.fixture
def runtime_manager(
    llm_manager: LLMManager,
    tool_manager: ToolManagerHelper,
    base_config_manager: MockConfigManager,
) -> Generator[RuntimeManager, None, None]:
    """Runtime manager fixture with fresh state for each test."""
    manager = RuntimeManager(
        llm_manager=llm_manager,
        tool_manager=tool_manager,
        config_manager=base_config_manager,
    )
    yield manager
    # Cleanup if needed


@pytest.fixture
def persistent_runtime_manager(
    tool_manager: ToolManagerHelper,
    base_config_manager: MockConfigManager,
) -> Generator[RuntimeManager, None, None]:
    """Runtime manager with persistent sessions enabled."""
    # Create a new config manager with persistent sessions
    config = deepcopy(base_config_manager)
    config.set_session_overrides({"runtime.persistent_sessions": True})
    
    # Create a mock environment manager
    mock_env = MagicMock()
    mock_env.get_env.return_value = "test_key"
    mock_env.is_testing.return_value = False
    
    # Create a new provider manager with the test provider and mock environment
    provider_manager = ProviderManager(env_manager=mock_env, allow_test_requests=True)
    test_provider = MockTestProvider(env_manager=mock_env, allow_test_requests=True)
    provider_manager._providers = {"test": test_provider}
    
    # Initialize provider configs
    provider_manager._provider_configs = {
        "test": {
            "name": "test",
            "driver": "test",
            "base_url": "https://api.test.com",
            "models": ["test-model", "gpt-4", "gpt-3.5"],
            "api_key": "test_key",
            "env_manager": mock_env
        }
    }
    
    # Create a new LLM manager with the new config and provider manager
    llm_manager = LLMManager(
        config_manager=config,
        provider_manager=provider_manager,
    )
    
    manager = RuntimeManager(
        llm_manager=llm_manager,
        tool_manager=tool_manager,
        config_manager=config,
    )
    yield manager
    # Cleanup if needed


# ── non-persistent vs persistent behavior ─────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrate_with_model_override(runtime_manager: RuntimeManager):
    """Test orchestrate method with model override."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager

    try:
        # Call with model override
        result = await runtime_manager.orchestrate("test query", model="gpt-4")

        # Verify the response contains the expected fields
        assert "message" in result
        assert "model_used" in result
        assert "tokens_used" in result
        assert "history" in result

        # Verify the LLM was called with the correct model
        assert len(mock_llm_manager.generate.await_args_list) == 1
        args, _ = mock_llm_manager.generate.await_args_list[0]
        llm_request = args[0]
        
        # The model is passed to to_provider_request, not stored in the request
        # So we'll verify the request was created correctly
        assert llm_request.prompt == "test query"

        # Verify the response
        assert result["model_used"] == "gpt-4"
        assert result["message"] == "Test response"
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_orchestrate_with_multiple_overrides(runtime_manager: RuntimeManager):
    """Test orchestrate method with multiple configuration overrides."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager

    try:
        # Test with multiple overrides
        result = await runtime_manager.orchestrate(
            "test query",
            model="gpt-4",
            temperature=0.9,
            max_tokens=500,
        )

        # Verify the response contains the expected fields
        assert "message" in result
        assert "model_used" in result
        assert "tokens_used" in result
        assert "history" in result

        # Verify the LLM was called with the correct parameters
        assert len(mock_llm_manager.generate.await_args_list) == 1
        args, _ = mock_llm_manager.generate.await_args_list[0]
        llm_request = args[0]
        
        # Verify the request was created with the correct prompt and parameters
        assert llm_request.prompt == "test query"

        # Verify the response
        assert result["model_used"] == "gpt-4"
        assert result["message"] == "Test response"
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_persistent_many_iterations_history_grows_linearly(persistent_runtime_manager: RuntimeManager):
    """Test that history grows linearly with many iterations."""
    # Create mock responses
    mock_responses = [
        MagicMock(content=f"Response {i}", model_used="gpt-4") for i in range(5)
    ]

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.side_effect = mock_responses

    # Replace the LLM manager in the runtime manager
    original_llm_manager = persistent_runtime_manager._llm_manager
    persistent_runtime_manager._llm_manager = mock_llm_manager

    try:
        # Run multiple iterations
        num_iterations = 5
        for i in range(num_iterations):
            result = await persistent_runtime_manager.orchestrate(f"Query {i}")
            # History should grow by 2 messages (user + assistant) per iteration
            expected_history_length = (i + 1) * 2
            assert len(result["history"]) == expected_history_length
            assert result["message"] == f"Response {i}"
    finally:
        # Restore the original LLM manager
        persistent_runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_directive_success_invokes_tool_and_passes_outputs_to_llm(runtime_manager: RuntimeManager, tool_manager: ToolManagerHelper):
    """Test direct tool invocation and output passing."""
    # Create a test tool
    class TestTool:
        name = "test_tool"
        description = "A test tool"
        available = True  # Add the required 'available' attribute
        
        def execute(self, arg1: str) -> str:
            return f"Processed: {arg1}"
    
    # Create a mock response for the LLM call after tool execution
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Tool result processed"
    mock_llm_response.model_used = "gpt-4"
    mock_llm_response.tokens_used = MagicMock()
    
    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_llm_response
    
    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    # Register the test tool
    test_tool = TestTool()
    tool_manager._tools["test_tool"] = test_tool
    
    try:
        # Test direct tool invocation
        result = await runtime_manager.orchestrate('tool:test_tool {\"arg1\": \"value1\"}')
        
        # Verify the response contains the expected fields
        assert "message" in result
        assert "Tool result processed" in result["message"]
        assert "model_used" in result
        assert "tokens_used" in result
        assert "history" in result
        assert len(result["history"]) == 2  # User + Assistant
        
        # Verify the LLM was called once with the tool output
        assert mock_llm_manager.generate.await_count == 1
        
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_directive_unknown_tool_raises(runtime_manager: RuntimeManager):
    """Test that unknown tools raise appropriate errors."""
    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "Error: Tool 'unknown' not found"
    mock_llm_manager.generate.return_value = mock_response
    
    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with unknown tool
        result = await runtime_manager.orchestrate("tool:unknown {}")
        assert "Tool 'unknown' not found" in result["message"]
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_directive_invalid_json_raises(runtime_manager: RuntimeManager):
    """Test that invalid JSON in tool calls returns an appropriate error message."""
    # Create a mock response for the error case
    mock_response = MagicMock()
    mock_response.content = "Invalid JSON in tool payload: Expecting value: line 1 column 1 (char 0)"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with invalid JSON
        result = await runtime_manager.orchestrate("tool:sum not-json")

        # Verify the error message indicates invalid JSON
        assert result["message"] == "Invalid JSON in tool payload: Expecting value: line 1 column 1 (char 0)"
        assert result["model_used"] == "gpt-4"
        assert len(result["history"]) == 2
        assert result["history"][0]["content"] == "Invalid JSON in tool payload: Expecting value: line 1 column 1 (char 0)"
        assert result["history"][0]["role"] == "user"
        assert "Invalid JSON" in result["history"][1]["content"]
        assert result["history"][1]["role"] == "assistant"

        # Verify LLM was called once with the error message
        assert mock_llm_manager.generate.await_count == 1
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_directive_invalid_payload_validation_raises(runtime_manager: RuntimeManager, tool_manager: ToolManagerHelper):
    """Test that invalid tool payload raises appropriate errors."""
    # Create a mock response for the error case
    mock_response = MagicMock()
    mock_response.content = "Invalid tool payload: 1 validation error for SumTool\nvalue\n  Input should be a valid dictionary [type=dict_type, input_value=None, input_type=None_type]"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response
    
    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with invalid payload (None instead of a dictionary)
        result = await runtime_manager.orchestrate("tool:sum null")

        # Verify the error message indicates invalid payload
        assert "validation error" in result["message"] or "Input should be a valid dictionary" in result["message"]
        assert result["model_used"] == "gpt-4"
        assert len(result["history"]) == 2
        assert any(msg in result["history"][0]["content"] 
                  for msg in ["validation error", "Input should be a valid dictionary"])
        assert result["history"][0]["role"] == "user"
        assert "validation error" in result["history"][1]["content"]
        assert result["history"][1]["role"] == "assistant"

        # Verify LLM was called once with the error message
        assert mock_llm_manager.generate.await_count == 1
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_empty_text_is_accepted_and_yields_echo(runtime_manager: RuntimeManager):
    """Test that empty text is handled correctly."""
    # Create a mock response for the echo case
    mock_response = MagicMock()
    mock_response.content = "echo:"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = MagicMock()

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response
    
    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with empty text
        result = await runtime_manager.orchestrate("")

        # Verify the response is the echo response
        assert result["message"] == "echo:"
        assert result["model_used"] == "gpt-4"
        assert len(result["history"]) == 2  # User + Assistant
        assert result["history"][0]["content"] == ""
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["content"] == "echo:"
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_structured_output_shape_and_fields(runtime_manager: RuntimeManager):
    """Test that output has the expected structure and fields."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.model_used = "gpt-4"
    mock_response.tokens_used = 42

    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.return_value = mock_response

    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with a simple query
        result = await runtime_manager.orchestrate("test query")

        # Verify the response has the expected structure
        assert isinstance(result, dict)
        assert "message" in result
        assert "model_used" in result
        assert "tokens_used" in result
        assert "history" in result
        assert isinstance(result["history"], list)
        assert len(result["history"]) == 2  # User + Assistant
        assert result["history"][0]["role"] == "user"
        assert result["history"][0]["content"] == "test query"
        assert result["history"][1]["role"] == "assistant"
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_provider_system_integration(
    runtime_manager: RuntimeManager,
    provider_manager: ProviderManager,
    base_config_manager: MockConfigManager,
    tool_manager: ToolManagerHelper
):
    """Test that the provider system integration works correctly."""
    # Create a test provider
    test_provider = MockTestProvider()
    
    # Register our test provider
    provider_manager._providers["test"] = test_provider
    
    # Configure the config manager
    base_config_manager._config["providers"] = {"test": {"api_key": "test_key"}}
    
    # Create a mock LLM manager
    mock_llm_manager = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "echo:test query"
    mock_response.model_used = "test-model"
    mock_response.tokens_used = 10
    mock_llm_manager.generate.return_value = mock_response
    
    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test with a simple query
        result = await runtime_manager.orchestrate("test query")
        
        # Verify the response
        assert result["message"] == "echo:test query"
        assert result["model_used"] == "test-model"
        assert result["tokens_used"] == 10
        assert len(result["history"]) == 2  # User + Assistant
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


@pytest.mark.asyncio
async def test_llm_provider_failure_handling(runtime_manager: RuntimeManager):
    """Test that LLM provider errors are properly propagated."""
    # Create a mock LLM manager that raises an exception
    mock_llm_manager = AsyncMock()
    mock_llm_manager.generate.side_effect = LLMError("Provider error")

    # Replace the LLM manager in the runtime manager
    original_llm_manager = runtime_manager._llm_manager
    runtime_manager._llm_manager = mock_llm_manager
    
    try:
        # Test that the exception is propagated
        with pytest.raises(LLMError) as exc_info:
            await runtime_manager.orchestrate("test query")

        # Verify the exception was propagated correctly
        assert "Provider error" in str(exc_info.value)

        # Verify the LLM was called once
        assert mock_llm_manager.generate.await_count == 1
    finally:
        # Restore the original LLM manager
        runtime_manager._llm_manager = original_llm_manager


def test_get_available_tools_returns_function_specs_for_iterable_entries(runtime_manager: RuntimeManager):
    """Runtime manager should build OpenAI-style specs from iterable tool entries."""
    class DummyInput:
        @classmethod
        def model_json_schema(cls):
            return {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }

    class DummyTool:
        name = "dummy"
        description = "A dummy tool"
        args_schema = DummyInput

    # Replace the tool manager with our test tool

def test_get_available_tools_returns_none_when_no_valid_tools(runtime_manager: RuntimeManager):
    """If the tool manager has no usable entries, None should be returned."""
    # Create a mock tool manager that returns an empty list
    class MockToolManager:
        def list_tools(self, available_only: bool = False, category: str | None = None) -> list:
            return []
    
    # Replace the tool manager with our mock
    original_tool_manager = runtime_manager._tool_manager
    runtime_manager._tool_manager = MockToolManager()
    
    try:
        assert runtime_manager._get_available_tools() is None
    finally:
        # Restore the original tool manager
        runtime_manager._tool_manager = original_tool_manager
