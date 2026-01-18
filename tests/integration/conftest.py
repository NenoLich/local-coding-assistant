import asyncio
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from local_coding_assistant.agent.agent_loop import AgentLoop
from local_coding_assistant.agent.llm_manager import (
    LLMManager,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from local_coding_assistant.cli.commands import sandbox as sandbox_cli
from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.config.schemas import AppConfig, SandboxConfig
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.providers import (
    BaseProvider,
    ProviderError,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
    ProviderRouter,
)
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.tools.types import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolInfo,
)


@pytest.fixture
def path_manager_integration(tmp_path: Path) -> PathManager:
    """PathManager configured for integration tests with temporary project root."""
    return PathManager(is_testing=True, project_root=tmp_path)


@pytest.fixture
def sandbox_config_manager(tmp_path: Path):
    """Provide a stub config manager wired with a PathManager and sandbox config."""

    project_root = tmp_path / "sandbox-project"
    project_root.mkdir(parents=True, exist_ok=True)

    path_manager = PathManager(is_testing=True, project_root=project_root)
    sandbox_config = SandboxConfig(
        enabled=True,
        image="integration-sandbox:latest",
        memory_limit="256m",
        cpu_limit=0.25,
        network_enabled=False,
        allowed_imports=["json"],
        blocked_patterns=[r"eval"],
        blocked_shell_commands=["rm"],
        session_timeout=123,
        max_sessions=3,
    )
    app_config = AppConfig(sandbox=sandbox_config)

    class StubConfigManager:
        def __init__(
            self, config: AppConfig, path_manager: PathManager, project_root: Path
        ):
            self._global_config = config
            self.path_manager = path_manager
            self.project_root = project_root

        @property
        def global_config(self) -> AppConfig:
            return self._global_config

    return StubConfigManager(app_config, path_manager, project_root)


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

        tool_calls = self._parse_tool_calls(response_data.get("tool_calls", []))

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

        tool_calls = self._parse_tool_calls(response_data.get("tool_calls", []))

        if tool_calls:
            serializable_calls = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in tool_calls
            ]
            yield f"\n\nTool calls: {json.dumps(serializable_calls)}"

    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        """Async invoke (alias for generate)."""
        return await self.generate(request)

    def _parse_tool_calls(
        self, tool_calls_data: list[dict[str, Any]] | None
    ) -> list[ToolCall] | None:
        """Normalize raw tool call payloads into ToolCall objects."""
        if not tool_calls_data:
            return None

        parsed_calls: list[ToolCall] = []
        for index, tool_call in enumerate(tool_calls_data):
            if isinstance(tool_call, ToolCall):
                parsed_calls.append(tool_call)
                continue

            function_payload = (
                tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            )
            name = (
                function_payload.get("name") or tool_call.get("name") or f"tool_{index}"
            )
            raw_arguments = function_payload.get("arguments") or tool_call.get(
                "arguments", {}
            )

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}

            parsed_calls.append(
                ToolCall(
                    id=tool_call.get("id") if isinstance(tool_call, dict) else None,
                    name=name,
                    arguments=arguments,
                )
            )

        return parsed_calls


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


class MockToolManager:
    """Mock tool manager with calculator, weather, and final answer tools."""

    def __init__(self, config_manager: IConfigManager | None = None):
        # Use TestConfigManager if no config_manager is provided
        if config_manager is None:
            config_manager = TestConfigManager()

        # Store config manager
        self.config_manager = config_manager

        # Create tool instances
        self.calculator = MockCalculatorTool()
        self.weather = MockWeatherTool()
        self.final_answer = MockFinalAnswerTool()

        # Store tools in a list for iteration
        self.tools = [self.calculator, self.weather, self.final_answer]

        # Create a mapping of tool names to tool instances
        self._tools_map = {tool.name: tool for tool in self.tools}
        self._tool_info_map = {
            tool.name: self._build_tool_info(tool) for tool in self.tools
        }

    def __iter__(self):
        """Iterate over all registered tools."""
        return iter(self.tools)

    def list_tools(self, available_only: bool = False, **kwargs):
        """Return ToolInfo metadata consistent with real ToolManager."""
        tools = list(self._tool_info_map.values())
        if available_only:
            tools = [tool for tool in tools if tool.available]
        return tools

    def get_tool(self, tool_name: str):
        """Get a tool by name."""
        return self._tools_map.get(tool_name)

    def register_tool(self, tool):
        """Register a new tool."""
        self.tools.append(tool)
        self._tools_map[tool.name] = tool
        self._tool_info_map[tool.name] = self._build_tool_info(tool)
        return tool

    def run_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Run a tool by name.

        Args:
            tool_name: Name of the tool to run
            parameters: Parameters to pass to the tool
            **kwargs: Additional keyword arguments (ignored in mock)
        """
        if tool_name == "calculator":
            expression = parameters.get("expression", "")
            return self.calculator.run(expression)
        elif tool_name == "weather":
            location = parameters.get("location", "")
            return self.weather.run(location)
        elif tool_name == "final_answer":
            answer = parameters.get("answer", "")
            return self.final_answer.run(answer)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Sync execution API used by ToolManager."""
        start = time.perf_counter()
        result = self.run_tool(request.tool_name, request.payload or {})
        return self._build_execution_response(request.tool_name, result, start)

    async def execute_async(
        self, request: ToolExecutionRequest
    ) -> ToolExecutionResponse:
        await asyncio.sleep(0)
        return self.execute(request)

    def _build_tool_info(self, tool: Any) -> ToolInfo:
        """Construct ToolInfo for mock tools."""
        parameter_templates = {
            "calculator": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            "weather": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to fetch weather for",
                    }
                },
                "required": ["location"],
            },
            "final_answer": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Final response to share",
                    }
                },
                "required": ["answer"],
            },
        }

        return ToolInfo(
            name=tool.name,
            description=tool.description,
            available=True,
            parameters=parameter_templates.get(
                tool.name, {"type": "object", "properties": {}, "required": []}
            ),
        )

    def _build_execution_response(
        self, tool_name: str, result: dict[str, Any], start_time: float
    ) -> ToolExecutionResponse:
        elapsed = (time.perf_counter() - start_time) * 1000
        success = result.get("success", True)

        if tool_name == "final_answer":
            return ToolExecutionResponse(
                tool_name=tool_name,
                success=success,
                result=result.get("answer"),
                execution_time_ms=elapsed,
                is_final=True,
            )

        return ToolExecutionResponse(
            tool_name=tool_name,
            success=success,
            result=result,
            execution_time_ms=elapsed,
        )


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

    # Create a test config manager
    config_manager = TestConfigManager()

    # Create a custom MockStreamingLLMManager that validates the model
    class ValidatingMockStreamingLLMManager(MockStreamingLLMManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.valid_models = {
                "gpt-4",
                "gpt-3.5-turbo",
            }  # Add other valid models as needed

        async def generate(self, *args, **kwargs):
            model = kwargs.get("model")
            if model == "invalid-model":
                raise ValueError("Model 'invalid-model' not found")
            if model not in self.valid_models:
                raise ValueError(
                    f"Model '{model}' not found. Available models: {', '.join(self.valid_models)}"
                )
            return await super().generate(*args, **kwargs)

        # Alias generate to chat_completion for backward compatibility
        chat_completion = generate

    # Return our validating mock LLM manager
    return ValidatingMockStreamingLLMManager(
        responses=responses, config_manager=config_manager, provider_manager=None
    )


@pytest.fixture
def tool_manager():
    """Create tool manager with calculator, weather, and final answer tools."""
    # Create a TestConfigManager instance
    config_manager = TestConfigManager()

    # Create a MockToolManager with the config manager
    tool_manager = MockToolManager(config_manager=config_manager)

    # Verify the tools are properly registered
    assert hasattr(tool_manager, "final_answer"), (
        "final_answer tool not registered in MockToolManager"
    )
    assert hasattr(tool_manager, "calculator"), (
        "calculator tool not registered in MockToolManager"
    )
    assert hasattr(tool_manager, "weather"), (
        "weather tool not registered in MockToolManager"
    )

    # Verify tools are registered in the tool manager
    registered_tool_names = {info.name for info in tool_manager.list_tools()}
    for tool in tool_manager:
        assert tool.name in registered_tool_names, (
            f"{tool.name} not registered in tool manager"
        )

    return tool_manager


class TestConfigManager:
    """Test implementation of IConfigManager for integration tests."""

    def __init__(self):
        from local_coding_assistant.config.env_manager import EnvManager
        from local_coding_assistant.config.path_manager import PathManager
        from local_coding_assistant.config.schemas import (
            AppConfig,
            LLMConfig,
            RuntimeConfig,
        )

        # Initialize environment and path managers
        self.env_manager = EnvManager()
        self.path_manager = PathManager()

        # Initialize with default config
        self._global_config = AppConfig(
            llm=LLMConfig(
                temperature=0.7,
                max_tokens=1000,
                max_retries=3,
                retry_delay=1.0,
                providers=[],
            ),
            runtime=RuntimeConfig(
                persistent_sessions=False,
                max_session_history=100,
                enable_logging=True,
                log_level="INFO",
            ),
        )
        self._session_overrides = {}

    @property
    def global_config(self) -> Any | None:
        return self._global_config

    @property
    def session_overrides(self) -> dict[str, Any]:
        return self._session_overrides

    def load_global_config(self) -> Any:
        return self._global_config

    def get_tools(self) -> dict[str, Any]:
        return {}

    def reload_tools(self) -> None:
        pass

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        self._session_overrides = overrides or {}

    def resolve(
        self,
        provider: str | None = None,
        model_name: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Any:
        # Create a copy of the global config
        config = self._global_config.model_copy(deep=True)

        # Apply session overrides
        if self._session_overrides:
            for key, value in self._session_overrides.items():
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        # Apply call overrides
        call_overrides = {}
        if provider is not None:
            call_overrides["llm.provider"] = provider
        if model_name is not None:
            call_overrides["llm.model_name"] = model_name
        if overrides:
            call_overrides.update(overrides)

        if call_overrides:
            for key, value in call_overrides.items():
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        return config


@pytest.fixture
def runtime_manager(mock_llm_with_tools, tool_manager):
    """Create runtime manager with mocked dependencies."""
    config_manager = TestConfigManager()
    config_manager.load_global_config()

    runtime = MagicMock(spec=RuntimeManager)
    runtime._llm_manager = mock_llm_with_tools
    runtime._tool_manager = tool_manager
    runtime._config_manager = config_manager

    # Store the agent loop instance and session ID for persistence
    agent_loop_instance: AgentLoop | None = None
    session_id: str | None = None

    # Store agent loop instances by session ID
    agent_loops = {}
    # Track the last used session ID to maintain it between calls
    last_session_id = None

    # Mock the _run_agent_mode method to use our real AgentLoop
    async def mock_run_agent_mode(
        text,
        model=None,
        temperature=None,
        max_tokens=None,
        streaming=False,
        max_iterations=5,
        agent_mode=False,
        session_id=None,  # Allow explicit session ID for testing
        **kwargs,
    ):
        nonlocal agent_loops, last_session_id

        # Use the provided session ID, the last used session ID, or generate a new one
        if session_id is None:
            if last_session_id is not None and last_session_id in agent_loops:
                session_id = last_session_id
            else:
                session_id = f"test_session_{len(agent_loops) + 1}"

        # Update the last used session ID
        last_session_id = session_id

        # Reuse existing agent loop for this session or create a new one
        if session_id not in agent_loops:
            # Create a new mock LLM manager with the same responses
            llm_manager = MockStreamingLLMManager(
                responses=mock_llm_with_tools.responses,
                config_manager=config_manager,
                provider_manager=None,
            )

            # Create a new agent loop for this session
            agent_loop = AgentLoop(
                llm_manager=llm_manager,
                tool_manager=tool_manager,
                name="integration_test_agent",
                max_iterations=max_iterations,
                streaming=streaming,
            )
            # Set the session ID after creation
            agent_loop._session_id = session_id
            agent_loops[session_id] = agent_loop
        else:
            agent_loop = agent_loops[session_id]
            # Update any runtime parameters if needed
            agent_loop.max_iterations = max_iterations
            agent_loop.streaming = streaming

        # Process the request
        result = await agent_loop.run()

        # Extract the final answer from the result
        final_answer = None
        if isinstance(result, dict) and "final_answer" in result:
            final_answer = result["final_answer"]
        elif isinstance(result, str):
            final_answer = result

        # If we still don't have a final answer, try to get it from the agent's state
        if final_answer is None and hasattr(agent_loop, "final_answer"):
            final_answer = agent_loop.final_answer

        # If we still don't have a final answer, use a default value
        if final_answer is None:
            final_answer = "Mock final answer for testing purposes"

        return {
            "final_answer": final_answer,
            "iterations": getattr(agent_loop, "current_iteration", 1),
            "history": getattr(agent_loop, "get_history", lambda: [])(),
            "session_id": session_id,
            "streaming_enabled": streaming,
        }

    runtime._run_agent_mode = mock_run_agent_mode
    return runtime


@pytest.fixture
def runtime_manager_with_streaming(mock_llm_with_tools, tool_manager):
    """Create runtime manager with mocked dependencies and streaming enabled."""
    config_manager = TestConfigManager()
    config_manager.load_global_config()

    runtime = MagicMock(spec=RuntimeManager)
    runtime._llm_manager = mock_llm_with_tools
    runtime._tool_manager = tool_manager
    runtime._config_manager = config_manager

    # Store the agent loop instance and session ID for persistence
    agent_loop_instance: AgentLoop | None = None
    session_id: str | None = None

    # Mock the _run_agent_mode method to use our real AgentLoop with streaming
    async def mock_run_agent_mode(
        text,
        model=None,
        temperature=None,
        max_tokens=None,
        streaming=True,  # Default to True for this fixture
        max_iterations=5,
        agent_mode=False,
        **kwargs,
    ):
        nonlocal agent_loop_instance, session_id

        # Create a new mock LLM manager with the same responses
        llm_manager = MockStreamingLLMManager(
            responses=mock_llm_with_tools.responses,
            config_manager=config_manager,
            provider_manager=None,
        )

        # Create a new agent loop for each request to ensure clean state
        agent_loop_instance = AgentLoop(
            llm_manager=llm_manager,
            tool_manager=tool_manager,
            name="integration_test_agent_streaming",
            max_iterations=max_iterations,
            streaming=streaming,
        )

        # If we have a previous session ID, set it on the new instance
        if session_id is not None:
            agent_loop_instance._session_id = session_id
        else:
            # Store the new session ID for subsequent requests
            session_id = agent_loop_instance.session_id

        # Process the request
        final_answer = await agent_loop_instance.run()

        # Ensure we maintain the same session ID for the next request
        session_id = agent_loop_instance.session_id

        return {
            "final_answer": final_answer,
            "iterations": agent_loop_instance.current_iteration,
            "history": agent_loop_instance.get_history(),
            "session_id": session_id,  # Use the maintained session ID
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

    from local_coding_assistant.core.protocols import IConfigManager

    # Create a properly typed config manager mock
    config_manager = MagicMock(spec=IConfigManager)
    config_manager.get_tool_config.return_value = {}

    return MockStreamingLLMManager(
        responses, config_manager=config_manager, provider_manager=None
    )


@pytest.fixture
def complex_tool_manager():
    """Tool manager for complex scenarios."""
    # Use TestConfigManager which properly implements IConfigManager
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


class FakeSandbox:
    """Simple sandbox double used for CLI integration tests."""

    def __init__(self) -> None:
        self.executed_requests: list[Any] = []
        self.stop_calls: list[str] = []
        self.response: SimpleNamespace = SimpleNamespace(
            success=True, stdout="", stderr="", error=None
        )

    async def execute(self, request):
        self.executed_requests.append(request)
        return self.response

    async def stop_session(self, session_id: str):
        self.stop_calls.append(session_id)


class FakeSandboxManager:
    """Provides access to the fake sandbox used by the CLI tests."""

    def __init__(self) -> None:
        self.sandbox = FakeSandbox()
        self.get_sandbox_call_count = 0
        self.bootstrap_levels: list[int | None] = []

    def get_sandbox(self):
        self.get_sandbox_call_count += 1
        return self.sandbox


@pytest.fixture
def sandbox_cli_test_env(monkeypatch):
    """Patch CLI bootstrap to use a deterministic fake sandbox manager."""

    manager = FakeSandboxManager()

    def fake_bootstrap(**kwargs):
        manager.bootstrap_levels.append(kwargs.get("log_level"))
        return {"sandbox": manager}

    monkeypatch.setattr(sandbox_cli, "bootstrap", fake_bootstrap)
    return manager


@pytest.fixture
def tmp_yaml_config(tmp_path):
    """Fixture to create a temporary YAML config file for testing."""

    def _create_config(config_data):
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    return _create_config


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

        # Check if we have a model in the request (from orchestrate)
        if hasattr(request, "model") and request.model:
            model_used = request.model

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
    from local_coding_assistant.core.app_context import AppContext

    # Create a TestConfigManager instance
    config_manager = TestConfigManager()

    # Set up session overrides
    config_manager.set_session_overrides({"llm.model_name": "gpt-4.1"})

    # Create a mock runtime manager
    runtime = MagicMock(spec=RuntimeManager)
    runtime.config_manager = config_manager
    runtime.llm_manager = mock_llm_manager
    runtime._llm_manager = (
        mock_llm_manager  # Add _llm_manager for backward compatibility
    )

    # Set up the orchestrate method to use the mock LLM manager
    async def mock_orchestrate(
        query, model=None, temperature=None, max_tokens=None, **kwargs
    ):
        from dataclasses import make_dataclass

        from local_coding_assistant.core.exceptions import AgentError

        # Validate temperature
        if temperature is not None and (temperature < 0.0 or temperature > 2.0):
            raise AgentError(
                "Configuration update validation failed: temperature must be between 0.0 and 2.0"
            )

        # Validate max_tokens
        if max_tokens is not None and max_tokens <= 0:
            raise AgentError(
                "Configuration update validation failed: max_tokens must be greater than 0"
            )

        # Use the model from the arguments or the default from the config
        model_used = model or "gpt-5-mini"

        # Create a simple request object with model and tool_outputs
        Request = make_dataclass("Request", ["model", "tool_outputs"])
        request = Request(model=model_used, tool_outputs=None)

        # If the LLM manager has a side effect set, let it raise the error
        if hasattr(mock_llm_manager.generate, "side_effect"):
            try:
                result = await mock_llm_manager.generate(request)
                # Convert LLMResponse to dict for the test
                return {
                    "message": str(result.content),
                    "model_used": result.model_used,
                    "tokens_used": result.tokens_used,
                }
            except AgentError as e:
                raise e

        return {
            "message": f"Processed query with model: {model_used}",
            "model_used": model_used,
            "tokens_used": 50,
        }

    runtime.orchestrate = mock_orchestrate

    # Create and return the context
    ctx = AppContext()
    ctx.register("runtime", runtime)
    ctx.register("config_manager", config_manager)

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
