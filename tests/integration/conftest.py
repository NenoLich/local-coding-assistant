import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from local_coding_assistant.agent.agent_loop import AgentLoop
from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest, LLMResponse
from local_coding_assistant.config.schemas import RuntimeConfig
from local_coding_assistant.core import AppContext
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.providers import (
    BaseProvider,
    ProviderError,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
    ProviderRouter,
)
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools import ToolManager
from local_coding_assistant.tools.builtin import SumTool


class MockStreamingLLMManager(LLMManager):
    """Mock LLM manager that supports streaming responses and tool calls."""

    def __init__(
        self,
        responses: list[dict[str, Any]],
        config_manager=None,
        provider_manager: Any = None,
    ):
        # Initialize the parent LLMManager first
        super().__init__(
            config_manager=config_manager, provider_manager=provider_manager
        )
        self.responses = responses
        self.call_count = 0
        self.streaming_enabled = True

    async def generate(
        self,
        request: LLMRequest,
        *,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response based on the request."""
        if self.call_count >= len(self.responses):
            # Return a default response if we've exhausted our predefined responses
            response_data = {"content": "I've completed my analysis.", "tool_calls": []}
        else:
            response_data = self.responses[self.call_count]

        self.call_count += 1

        # Create proper LLMResponse object instead of MagicMock
        tool_calls_data = response_data.get("tool_calls", [])
        # Ensure tool_calls is either a list of dicts or None
        if isinstance(tool_calls_data, list) and all(
            isinstance(tc, dict) for tc in tool_calls_data
        ):
            tool_calls = tool_calls_data
        else:
            tool_calls = None

        return LLMResponse(
            content=str(response_data.get("content", "")),
            model_used="mock-model",
            tokens_used=50,
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        request: LLMRequest,
        *,
        provider: str | None = None,
        model: str | None = None,
        policy: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if self.call_count >= len(self.responses):
            response_data = {"content": "I've completed my analysis.", "tool_calls": []}
        else:
            response_data = self.responses[self.call_count]

        self.call_count += 1
        # Simulate streaming by yielding partial content
        content = str(response_data.get("content", ""))
        if content and isinstance(content, str):
            # Yield content in chunks
            words = content.split()
            current_chunk = ""
            for word in words:
                current_chunk += word + " "
                yield current_chunk.strip()
                await asyncio.sleep(0.01)  # Simulate streaming delay

        # Yield tool calls if present
        tool_calls_data = response_data.get("tool_calls", [])
        # Ensure tool_calls is either a list of dicts or None
        if isinstance(tool_calls_data, list) and all(
            isinstance(tc, dict) for tc in tool_calls_data
        ):
            tool_calls = tool_calls_data
        else:
            tool_calls = None

        if tool_calls:
            yield f"\n\nTool calls: {json.dumps(tool_calls)}"

    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        """Async invoke (alias for generate)."""
        return await self.generate(request)


class MockCalculatorTool:
    """Mock calculator tool for testing."""

    def __init__(self):
        self.name = "calculator"
        self.description = "Perform basic arithmetic calculations"

    def run(self, expression: str) -> dict[str, Any]:
        """Evaluate a mathematical expression."""
        try:
            # Simple evaluation for testing
            result = eval(expression, {"__builtins__": {}})
            return {"result": result, "expression": expression, "success": True}
        except Exception as e:
            return {"error": str(e), "expression": expression, "success": False}


class MockWeatherTool:
    """Mock weather tool for testing."""

    def __init__(self):
        self.name = "weather"
        self.description = "Get weather information for a location"

    def run(self, location: str) -> dict[str, Any]:
        """Get weather for a location."""
        # Mock weather data
        weather_data = {
            "new york": {"temperature": 72, "condition": "sunny", "humidity": 65},
            "london": {"temperature": 59, "condition": "cloudy", "humidity": 80},
            "tokyo": {"temperature": 78, "condition": "rainy", "humidity": 90},
        }

        data = weather_data.get(
            location.lower(),
            {"temperature": 70, "condition": "unknown", "humidity": 50},
        )

        return {
            "location": location,
            "temperature": data["temperature"],
            "condition": data["condition"],
            "humidity": data["humidity"],
            "success": True,
        }


class MockFinalAnswerTool:
    """Mock final answer tool for testing."""

    def __init__(self):
        self.name = "final_answer"
        self.description = "Provide a final answer to the user"

    def run(self, answer: str) -> dict[str, Any]:
        """Return the final answer."""
        return {"answer": answer, "success": True}


class MockToolManager(ToolManager):
    """Mock tool manager with calculator, weather, and final answer tools."""

    def __init__(self):
        super().__init__()
        self.calculator = MockCalculatorTool()
        self.weather = MockWeatherTool()
        self.final_answer = MockFinalAnswerTool()
        self.tools = [self.calculator, self.weather, self.final_answer]

    def __iter__(self):
        return iter(self.tools)

    def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Run a tool by name."""
        if tool_name == "calculator":
            expression = args.get("expression", "")
            return self.calculator.run(expression)
        elif tool_name == "weather":
            location = args.get("location", "")
            return self.weather.run(location)
        elif tool_name == "final_answer":
            answer = args.get("answer", "")
            return self.final_answer.run(answer)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


# Integration test fixtures


@pytest.fixture
def mock_llm_with_tools():
    """Create LLM manager with predefined responses for tool calling."""
    responses = [
        # First response: Ask about weather
        {
            "content": "I need to check the weather in New York to help plan the trip.",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": '{"location": "New York"}',
                    }
                }
            ],
        },
        # Second response: Use calculator after getting weather
        {
            "content": "The weather is sunny with 72°F. I should calculate what to pack for this temperature.",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "72 + 10"}',
                    }
                }
            ],
        },
        # Third response: Provide final answer
        {
            "content": "Based on the weather and temperature calculation, I recommend packing light summer clothes.",
            "tool_calls": [
                {
                    "function": {
                        "name": "final_answer",
                        "arguments": '{"answer": "Pack light summer clothes for your trip to New York. The weather will be sunny with around 72°F.", "reasoning": "Weather data shows sunny conditions at 72°F, and temperature calculation confirms comfortable weather."}',
                    }
                }
            ],
        },
    ]
    return MockStreamingLLMManager(
        responses, config_manager=None, provider_manager=None
    )


@pytest.fixture
def tool_manager():
    """Create tool manager with calculator and weather tools."""
    return MockToolManager()


@pytest.fixture
def runtime_manager(mock_llm_with_tools, tool_manager):
    """Create runtime manager with mocked dependencies."""
    runtime = MagicMock(spec=RuntimeManager)
    runtime._llm_manager = mock_llm_with_tools
    runtime._tool_manager = tool_manager

    # Store the agent loop instance for session persistence
    agent_loop_instance: AgentLoop | None = None

    # Mock the _run_agent_mode method to use our real AgentLoop
    async def mock_run_agent_mode(
        text,
        model=None,
        temperature=None,
        max_tokens=None,
        streaming=False,
        max_iterations=5,
        agent_mode=False,
        **kwargs,
    ):
        nonlocal agent_loop_instance

        # Create new agent loop if this is the first call or if the previous one has finished
        if agent_loop_instance is None or (
            agent_loop_instance is not None
            and agent_loop_instance.final_answer is not None
        ):
            agent_loop_instance = AgentLoop(
                llm_manager=MockStreamingLLMManager(
                    mock_llm_with_tools.responses,
                    config_manager=None,
                    provider_manager=None,
                ),
                tool_manager=tool_manager,
                name="integration_test_agent",
                max_iterations=max_iterations,
                streaming=streaming,
            )

        final_answer = await agent_loop_instance.run()

        return {
            "final_answer": final_answer,
            "iterations": agent_loop_instance.current_iteration,
            "history": agent_loop_instance.get_history(),
            "session_id": agent_loop_instance.session_id,
            "streaming_enabled": streaming,
        }

    runtime._run_agent_mode = mock_run_agent_mode
    return runtime


@pytest.fixture
def runtime_manager_with_streaming(mock_llm_with_tools, tool_manager):
    """Create runtime manager with mocked dependencies and streaming enabled."""
    runtime = MagicMock(spec=RuntimeManager)
    runtime._llm_manager = mock_llm_with_tools
    runtime._tool_manager = tool_manager

    # Store the agent loop instance for session persistence
    agent_loop_instance: AgentLoop | None = None

    # Mock the _run_agent_mode method to use our real AgentLoop with streaming
    async def mock_run_agent_mode(
        text,
        model=None,
        temperature=None,
        max_tokens=None,
        streaming=False,
        max_iterations=5,
        agent_mode=False,
        **kwargs,
    ):
        nonlocal agent_loop_instance

        # Create new agent loop if this is the first call or if the previous one has finished
        if agent_loop_instance is None or (
            agent_loop_instance is not None
            and agent_loop_instance.final_answer is not None
        ):
            agent_loop_instance = AgentLoop(
                llm_manager=MockStreamingLLMManager(
                    mock_llm_with_tools.responses,
                    config_manager=None,
                    provider_manager=None,
                ),
                tool_manager=tool_manager,
                name="integration_test_agent_streaming",
                max_iterations=max_iterations,
                streaming=streaming,
            )

        final_answer = await agent_loop_instance.run()

        return {
            "final_answer": final_answer,
            "iterations": agent_loop_instance.current_iteration,
            "history": agent_loop_instance.get_history(),
            "session_id": agent_loop_instance.session_id,
            "streaming_enabled": streaming,
        }

    runtime._run_agent_mode = mock_run_agent_mode
    return runtime


@pytest.fixture
def complex_scenario_llm():
    """LLM with complex multistep reasoning scenario."""
    responses = [
        # Initial observation leads to planning
        {
            "content": "I need to solve this complex problem step by step. First, I should gather information about the components involved.",
            "tool_calls": [
                {
                    "function": {
                        "name": "weather",
                        "arguments": '{"location": "New York"}',
                    }
                }
            ],
        },
        # After weather info, do calculation
        {
            "content": "Now I have the weather information. I need to calculate something based on the temperature to determine the next steps.",
            "tool_calls": [
                {
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "75 * 2 + 10"}',
                    }
                }
            ],
        },
        # Final synthesis
        {
            "content": "Based on all the information gathered, I can now provide a comprehensive solution to the user's query.",
            "tool_calls": [
                {
                    "function": {
                        "name": "final_answer",
                        "arguments": '{"answer": "Based on the weather in New York (75°F) and my calculations (160), the optimal approach is to prepare for warm weather activities.", "reasoning": "Weather data and mathematical analysis support this conclusion."}',
                    }
                }
            ],
        },
    ]
    return MockStreamingLLMManager(
        responses, config_manager=None, provider_manager=None
    )


@pytest.fixture
def complex_tool_manager():
    """Tool manager for complex scenarios."""
    return MockToolManager()


@pytest.fixture
def streaming_llm_single_response():
    """LLM with single response for streaming tests."""
    responses = [
        {
            "content": "This is a streaming response that should be processed in chunks for testing purposes.",
            "tool_calls": [],
        }
    ]
    return MockStreamingLLMManager(
        responses, config_manager=None, provider_manager=None
    )


@pytest.fixture(scope="function")
def ctx():
    """Fresh context per test to avoid state bleed."""
    return bootstrap()


@pytest.fixture(scope="function")
def cli_runner():
    """Provide a CliRunner instance for invoking CLI commands."""
    return CliRunner()


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response for testing."""
    return LLMResponse(
        content="[LLMManager] Echo: Received request with tool outputs",
        model_used="gpt-5-mini",
        tokens_used=50,
        tool_calls=None,
    )


@pytest.fixture
def mock_llm_manager(mock_llm_response):
    """
    Create a mock LLM manager that doesn't make real API calls.
    This mock supports both generate() and update_config().
    """

    # Create a comprehensive mock without spec constraints
    mock_manager = MagicMock()

    # Create a mock config that matches the new LLMManager expectations
    mock_config = MagicMock()
    mock_config.temperature = 0.7
    mock_config.max_tokens = 1000
    mock_config.max_retries = 3
    mock_config.retry_delay = 1.0
    mock_config.providers = []

    mock_manager.config = mock_config

    # Create a proper config manager for the new system
    from local_coding_assistant.config.schemas import LLMConfig

    llm_config = LLMConfig(
        temperature=0.7, max_tokens=1000, max_retries=3, retry_delay=1.0, providers=[]
    )
    mock_config_manager = MagicMock()
    # Set up session overrides to return the model when accessed
    mock_session_overrides = MagicMock()
    mock_session_overrides.copy.return_value = {"llm.model_name": "gpt-4.1"}
    mock_config_manager.session_overrides = mock_session_overrides

    async def mock_generate(request: LLMRequest, *, overrides=None) -> LLMResponse:
        # Use the model from overrides if provided, otherwise use default
        model_used = "gpt-5-mini"

        # Handle both dict and MagicMock cases
        if overrides:
            try:
                # Try to get model from dict-like object
                if hasattr(overrides, "get"):
                    model_name = overrides.get("llm.model_name")
                    if model_name:
                        model_used = model_name
                elif "llm.model_name" in overrides:
                    model_used = overrides["llm.model_name"]
            except (KeyError, TypeError, AttributeError):
                # If we can't extract the model, use default
                pass

        # Check if tool outputs are present and modify response accordingly
        if request.tool_outputs:
            content = "[LLMManager] Echo: Received request with tool outputs"
        else:
            content = f"[LLMManager] Echo: Received request with model {model_used}"

        return LLMResponse(
            content=str(content),
            model_used=model_used,
            tokens_used=50,
            tool_calls=None,  # Mock responses don't need tool calls
        )

    mock_manager.generate = AsyncMock(side_effect=mock_generate)

    def mock_update_config(
        *,
        model_name: str | None = None,
        provider: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """Mock update_config with proper validation."""
        # For the new system, we'll just update the mock config attributes
        # The new LLMManager doesn't use these parameters directly

        # If validation passes, update the mock config
        if temperature is not None:
            mock_manager.config.temperature = temperature
        if max_tokens is not None:
            mock_manager.config.max_tokens = max_tokens
        # Note: model_name, provider, and api_key are not used in the new config system

    mock_manager.update_config = MagicMock(side_effect=mock_update_config)
    return mock_manager


@pytest.fixture
def ctx_with_mocked_llm(mock_llm_manager):
    """
    Create a context with mocked LLM manager to avoid API quota issues.
    """
    from local_coding_assistant.config.schemas import LLMConfig

    # Create proper config objects for the new system
    llm_config = LLMConfig(
        temperature=0.7, max_tokens=1000, max_retries=3, retry_delay=1.0, providers=[]
    )
    runtime_config = RuntimeConfig(
        persistent_sessions=False,
        max_session_history=100,
        enable_logging=True,
        log_level="INFO",
    )

    # Create config manager with the proper config
    config_manager = MagicMock()
    config_manager.global_config = MagicMock()
    config_manager.global_config.llm = llm_config
    config_manager.global_config.providers = {}
    config_manager.resolve.return_value = config_manager.global_config

    # Set up session overrides to return the model when accessed
    mock_session_overrides = MagicMock()
    mock_session_overrides.copy.return_value = {"llm.model_name": "gpt-4.1"}
    config_manager.session_overrides = mock_session_overrides

    def mock_set_session_overrides(overrides):
        """Mock set_session_overrides with proper validation."""
        if not overrides:
            return

        # Validate temperature
        if "llm.temperature" in overrides:
            temperature = overrides["llm.temperature"]
            if temperature is not None and (temperature < 0.0 or temperature > 2.0):
                from local_coding_assistant.core.exceptions import AgentError

                raise AgentError(
                    "Configuration update validation failed: temperature must be between 0.0 and 2.0"
                )

        # Validate max_tokens
        if "llm.max_tokens" in overrides:
            max_tokens = overrides["llm.max_tokens"]
            if max_tokens is not None and max_tokens <= 0:
                from local_coding_assistant.core.exceptions import AgentError

                raise AgentError(
                    "Configuration update validation failed: max_tokens must be greater than 0"
                )

        # If validation passes, just store the overrides
        mock_session_overrides.clear()
        mock_session_overrides.update(overrides)

    config_manager.set_session_overrides = MagicMock(
        side_effect=mock_set_session_overrides
    )

    runtime = RuntimeManager(
        llm_manager=mock_llm_manager,
        tool_manager=ToolManager(),
        config_manager=config_manager,
    )
    # Register the SumTool for testing
    if runtime._tool_manager is not None:
        runtime._tool_manager.register_tool(SumTool, category="test")
    ctx = AppContext()
    ctx.register("llm", mock_llm_manager)
    ctx.register("tools", runtime._tool_manager)
    ctx.register("runtime", runtime)
    return ctx


# Provider integration test fixtures


@pytest.fixture
async def mock_provider():
    """Create a mock provider for integration testing."""
    provider = AsyncMock(spec=BaseProvider)
    provider.name = "test_provider"
    provider.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]

    # Mock successful response generation
    provider.generate_with_retry = AsyncMock(
        return_value=ProviderLLMResponse(
            content="Mock provider response",
            model="gpt-4",
            tokens_used=50,
            tool_calls=None,
            finish_reason="stop",
        )
    )

    # Mock streaming response
    async def mock_stream_with_retry(*args, **kwargs):
        yield ProviderLLMResponseDelta(content="Stream", finish_reason=None)
        yield ProviderLLMResponseDelta(content="ing", finish_reason=None)
        yield ProviderLLMResponseDelta(content=" response", finish_reason="stop")

    provider.stream_with_retry = AsyncMock(side_effect=mock_stream_with_retry)

    return provider


class MockProviderManager:
    """Mock provider manager that simulates YAML loading behavior."""

    def __init__(self):
        self._loaded_providers = {}
        self._provider_configs = {}
        self._setup_default_providers()

    def _setup_default_providers(self):
        """Set up default provider instances."""
        from unittest.mock import AsyncMock, MagicMock

        # Create specific provider instances that tests can reference
        openai_provider = AsyncMock(spec=BaseProvider)
        openai_provider.name = "openai"
        openai_provider.get_available_models = MagicMock(
            return_value=["gpt-4", "gpt-3.5-turbo"]
        )
        openai_provider.health_check = AsyncMock(return_value=True)

        google_provider = AsyncMock(spec=BaseProvider)
        google_provider.name = "google"
        google_provider.get_available_models = MagicMock(return_value=["gemini-pro"])
        google_provider.health_check = AsyncMock(return_value=False)

        # Store provider instances for the default ones
        self._loaded_providers = {
            "openai": {"instance": openai_provider, "source": "builtin"},
            "google": {"instance": google_provider, "source": "global"},
            "test_provider": {"instance": None, "source": "local"},
        }

    def list_providers(self):
        return list(self._loaded_providers.keys())

    def get_provider_source(self, name):
        if name in self._loaded_providers:
            return self._loaded_providers[name].get("source", "unknown")
        return "unknown"

    def get_provider(self, name, **kwargs):
        """Return the same provider instances based on name."""
        if name in self._loaded_providers:
            return self._loaded_providers[name]["instance"]

        # Default provider for unknown names
        from unittest.mock import AsyncMock

        provider = AsyncMock(spec=BaseProvider)
        provider.name = name
        provider.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
        provider.health_check = AsyncMock(return_value=True)
        return provider

    def reload(self, config_manager=None):
        """Mock reload method that simulates loading from YAML."""
        # For now, just ensure basic providers are available
        if not self._loaded_providers:
            self._setup_default_providers()
        return None

    def load_providers_from_yaml(self, yaml_content):
        """Helper method for tests to simulate loading providers from YAML."""
        self._loaded_providers.clear()
        self._provider_configs.clear()

        for provider_name, config in yaml_content.get("providers", {}).items():
            # Create a mock provider instance
            from unittest.mock import AsyncMock, MagicMock

            provider = AsyncMock(spec=BaseProvider)
            provider.name = provider_name
            provider.get_available_models = MagicMock(
                return_value=list(config.get("models", {}).keys())
            )
            provider.health_check = AsyncMock(return_value=True)

            # Determine source based on config location
            source = "local" if "api_key" in config else "global"

            self._loaded_providers[provider_name] = {
                "instance": provider,
                "source": source,
            }
            self._provider_configs[provider_name] = config


@pytest.fixture
def mock_provider_manager():
    """Create a mock provider manager for integration testing."""
    return MockProviderManager()


@pytest.fixture
def provider_config_yaml():
    """Sample provider configuration in YAML format."""
    return {
        "openai": {
            "driver": "openai_chat",
            "api_key_env": "OPENAI_API_KEY",
            "models": {
                "gpt-4": {"temperature": 0.7},
                "gpt-3.5-turbo": {"temperature": 0.5},
            },
        },
        "google": {
            "driver": "google_gemini",
            "api_key_env": "GOOGLE_API_KEY",
            "models": {
                "gemini-pro": {},
            },
        },
        "local_provider": {
            "driver": "local",
            "models": {
                "local-model": {"temperature": 0.8},
            },
        },
    }


@pytest.fixture
def sample_provider_configs():
    """Sample provider configurations for different scenarios."""
    return {
        "minimal": {
            "test_provider": {
                "driver": "openai_chat",
                "models": {"gpt-4": {}},
            }
        },
        "with_api_key": {
            "test_provider": {
                "driver": "openai_chat",
                "api_key": "test-key-123",
                "models": {"gpt-4": {}, "gpt-3.5-turbo": {}},
            }
        },
        "with_base_url": {
            "test_provider": {
                "driver": "openai_chat",
                "base_url": "https://custom.api.com/v1",
                "api_key_env": "CUSTOM_API_KEY",
                "models": {"custom-model": {"max_tokens": 2000}},
            }
        },
        "invalid": {
            "bad_provider": {
                "models": {"gpt-4": {}},  # Missing required 'driver' field
            }
        },
    }


@pytest.fixture
async def failing_provider():
    """Create a provider that fails for error handling tests."""
    provider = AsyncMock(spec=BaseProvider)
    provider.name = "failing_provider"
    provider.get_available_models.return_value = []

    # Mock failure in generation
    provider.generate_with_retry = AsyncMock(
        side_effect=ProviderError("API rate limit exceeded")
    )

    # Mock failure in streaming
    async def failing_stream(*args, **kwargs):
        raise ProviderError("Streaming failed")

    provider.stream_with_retry = AsyncMock(side_effect=failing_stream)

    # Mock health check failure
    provider.health_check = AsyncMock(return_value=False)

    return provider


@pytest.fixture
def mock_router_with_fallback():
    """Create a mock router with fallback provider logic."""
    router = AsyncMock(spec=ProviderRouter)

    # Track call count for fallback testing
    call_count = 0

    async def get_provider_for_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Return failing provider on first call
            failing_provider = AsyncMock(spec=BaseProvider)
            failing_provider.name = "primary_provider"
            failing_provider.generate_with_retry = AsyncMock(
                side_effect=ProviderError("Primary provider failed")
            )
            return failing_provider, "gpt-4"
        else:
            # Return fallback provider on second call
            fallback_provider = AsyncMock(spec=BaseProvider)
            fallback_provider.name = "fallback_provider"
            fallback_provider.generate_with_retry = AsyncMock(
                return_value=ProviderLLMResponse(
                    content="Fallback response",
                    model="gpt-3.5-turbo",
                    tokens_used=75,
                    tool_calls=None,
                    finish_reason="stop",
                )
            )
            return fallback_provider, "gpt-3.5-turbo"

    router.get_provider_for_request = AsyncMock(side_effect=get_provider_for_request)
    router._is_critical_error = MagicMock(return_value=True)
    router.mark_provider_failure = MagicMock()
    router.mark_provider_success = MagicMock()

    return router


@pytest.fixture
def temp_provider_config_file():
    """Create a temporary provider configuration file for testing."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_content = {
            "test_provider": {
                "driver": "openai_chat",
                "api_key": "test-key",
                "models": {"gpt-4": {}, "gpt-3.5-turbo": {}},
            }
        }
        yaml.dump(config_content, f)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    config_path.unlink(missing_ok=True)


@pytest.fixture
def integration_llm_manager_with_providers(mock_provider_manager):
    """Create LLM manager with provider manager for integration tests."""
    with patch("local_coding_assistant.config.get_config_manager"):
        with patch(
            "local_coding_assistant.agent.llm_manager.ProviderManager"
        ) as mock_pm_class:
            mock_pm_class.return_value = mock_provider_manager

            llm_manager = LLMManager.__new__(LLMManager)
            llm_manager.provider_manager = mock_provider_manager
            llm_manager.config_manager = MagicMock()
            llm_manager.router = MagicMock(spec=ProviderRouter)
            llm_manager._provider_status_cache = {}
            llm_manager._last_health_check = 0
            llm_manager._cache_ttl = 30 * 60

            return llm_manager


@pytest.fixture
def mock_streaming_provider():
    """Create a mock provider for streaming tests."""
    provider = AsyncMock(spec=BaseProvider)
    provider.name = "streaming_provider"

    # Create realistic streaming deltas
    streaming_deltas = [
        ProviderLLMResponseDelta(content="Integration", finish_reason=None),
        ProviderLLMResponseDelta(content=" test", finish_reason=None),
        ProviderLLMResponseDelta(content=" streaming", finish_reason=None),
        ProviderLLMResponseDelta(content=" response", finish_reason="stop"),
    ]

    async def mock_stream_with_retry(*args, **kwargs):
        for delta in streaming_deltas:
            yield delta

    provider.stream_with_retry = AsyncMock(side_effect=mock_stream_with_retry)
    provider.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]

    return provider
