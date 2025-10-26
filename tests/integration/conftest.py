import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from local_coding_assistant.agent.agent_loop import AgentLoop
from local_coding_assistant.agent.llm_manager_v2 import LLMManager, LLMRequest, LLMResponse
from local_coding_assistant.config import get_config_manager, load_config
from local_coding_assistant.config.schemas import LLMConfig, RuntimeConfig
from local_coding_assistant.core import AppContext
from local_coding_assistant.core.bootstrap import bootstrap
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
        super().__init__(config_manager=config_manager, provider_manager=provider_manager)
        self.responses = responses
        self.call_count = 0
        self.streaming_enabled = True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response based on the request."""
        if self.call_count >= len(self.responses):
            # Return a default response if we've exhausted our predefined responses
            response_data = {"content": "I've completed my analysis.", "tool_calls": []}
        else:
            response_data = self.responses[self.call_count]

        self.call_count += 1

        # Create mock response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.content = response_data.get("content", "")
        mock_response.tool_calls = response_data.get("tool_calls", [])

        return mock_response

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Generate a streaming response."""
        if self.call_count >= len(self.responses):
            response_data = {"content": "I've completed my analysis.", "tool_calls": []}
        else:
            response_data = self.responses[self.call_count]

        self.call_count += 1

        # Simulate streaming by yielding partial content
        content = response_data.get("content", "")
        if content and isinstance(content, str):
            # Yield content in chunks
            words = content.split()
            current_chunk = ""
            for word in words:
                current_chunk += word + " "
                yield current_chunk.strip()
                await asyncio.sleep(0.01)  # Simulate streaming delay

        # Yield tool calls if present
        tool_calls = response_data.get("tool_calls", [])
        if tool_calls:
            yield f"\n\nTool calls: {json.dumps(tool_calls)}"


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
    return MockStreamingLLMManager(responses, config_manager=None, provider_manager=None)


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
    agent_loop_instance = None

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
        if agent_loop_instance is None or agent_loop_instance.final_answer is not None:
            agent_loop_instance = AgentLoop(
                llm_manager=MockStreamingLLMManager(mock_llm_with_tools.responses, config_manager=None, provider_manager=None),
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
    agent_loop_instance = None

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
        if agent_loop_instance is None or agent_loop_instance.final_answer is not None:
            agent_loop_instance = AgentLoop(
                llm_manager=MockStreamingLLMManager(mock_llm_with_tools.responses, config_manager=None, provider_manager=None),
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
    """LLM with complex multi-step reasoning scenario."""
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
    return MockStreamingLLMManager(responses, config_manager=None, provider_manager=None)


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
    return MockStreamingLLMManager(responses, config_manager=None, provider_manager=None)


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
    from local_coding_assistant.core.exceptions import AgentError

    # Create a comprehensive mock without spec constraints
    mock_manager = MagicMock()
    mock_manager.config = LLMConfig(model_name="gpt-5-mini", provider="openai")

    async def mock_generate(request: LLMRequest) -> LLMResponse:
        # Use the model from the config if available, otherwise use the mock default
        model_used = (
            mock_manager.config.model_name
            if hasattr(mock_manager, "config") and mock_manager.config.model_name
            else "gpt-5-mini"
        )

        # Check if tool outputs are present and modify response accordingly
        if request.tool_outputs:
            content = "[LLMManager] Echo: Received request with tool outputs"
        else:
            content = f"[LLMManager] Echo: Received request with model {model_used}"

        return LLMResponse(
            content=content,
            model_used=model_used,
            tokens_used=50,
            tool_calls=None,
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
        updates = {}
        if model_name is not None:
            updates["model_name"] = model_name
        if provider is not None:
            updates["provider"] = provider
        if temperature is not None:
            updates["temperature"] = temperature
        if max_tokens is not None:
            updates["max_tokens"] = max_tokens
        if api_key is not None:
            updates["api_key"] = api_key

        if not updates:
            return

        # Validate the new configuration (same logic as real LLM manager)
        try:
            old_config_dict = mock_manager.config.model_dump()
            new_config_dict = {**old_config_dict, **updates}
            # This will raise ValidationError if invalid
            LLMConfig(**new_config_dict)
        except Exception as e:
            raise AgentError(f"Configuration update validation failed: {e}") from e

        # If validation passes, update the mock config
        if model_name is not None:
            mock_manager.config.model_name = model_name
        if provider is not None:
            mock_manager.config.provider = provider
        if temperature is not None:
            mock_manager.config.temperature = temperature
        if max_tokens is not None:
            mock_manager.config.max_tokens = max_tokens
        if api_key is not None:
            mock_manager.config.api_key = api_key

    mock_manager.update_config = MagicMock(side_effect=mock_update_config)
    return mock_manager


@pytest.fixture
def ctx_with_mocked_llm(mock_llm_manager):
    """
    Create a context with mocked LLM manager to avoid API quota issues.
    """
    load_config()
    runtime_config = RuntimeConfig(
        persistent_sessions=False,
        max_session_history=100,
        enable_logging=True,
        log_level="INFO",
    )

    # Initialize config manager with global configuration
    config_manager = get_config_manager()
    config_manager.load_global_config()

    runtime = RuntimeManager(
        llm_manager=mock_llm_manager,
        tool_manager=ToolManager(),
        config_manager=config_manager,
    )
    # Register the SumTool for testing
    runtime._tool_manager.register_tool(SumTool)
    ctx = AppContext()
    ctx.register("llm", mock_llm_manager)
    ctx.register("tools", runtime._tool_manager)
    ctx.register("runtime", runtime)
    return ctx
