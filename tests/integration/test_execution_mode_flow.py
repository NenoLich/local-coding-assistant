import asyncio
from typing import Any

import pytest
from jinja2 import TemplateError

from local_coding_assistant.agent.llm_manager import LLMResponse
from local_coding_assistant.config.schemas import AppConfig
from local_coding_assistant.prompt.composer import PromptComposer
from local_coding_assistant.runtime.runtime_manager import RuntimeManager
from local_coding_assistant.runtime.runtime_types import ExecutionMode, RenderedPrompt
from local_coding_assistant.tools.types import ToolExecutionMode, ToolInfo


class StubConfigManager:
    """Lightweight config manager that applies overrides to AppConfig."""

    def __init__(
        self, *, sandbox_enabled: bool = True, tool_call_mode: str = "classic"
    ):
        self.global_config = AppConfig()
        self.global_config.sandbox.enabled = sandbox_enabled
        self.global_config.runtime.tool_call_mode = tool_call_mode
        self.session_overrides: dict[str, Any] = {}

    def load_global_config(self) -> AppConfig:
        return self.global_config

    def set_session_overrides(self, overrides: dict[str, Any] | None) -> None:
        overrides = overrides or {}
        self.session_overrides.update(overrides)
        for dotted_key, value in overrides.items():
            self._apply_override(dotted_key, value)

    def _apply_override(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        target = self.global_config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)


class FakeToolManager:
    """Tool manager stub that exposes classic/PTC tool inventories."""

    def __init__(
        self,
        *,
        classic_tools: list[ToolInfo] | None = None,
        ptc_tools: list[ToolInfo] | None = None,
        has_runtime: bool = True,
    ) -> None:
        self._tools_by_mode = {
            ToolExecutionMode.CLASSIC: classic_tools or [],
            ToolExecutionMode.PTC: ptc_tools or [],
        }
        self._has_runtime = has_runtime

    def list_tools(
        self,
        available_only: bool = True,
        execution_mode: ToolExecutionMode | None = None,
        **_: Any,
    ) -> list[ToolInfo]:
        if execution_mode is None:
            tools = [tool for mode in self._tools_by_mode.values() for tool in mode]
        else:
            tools = list(self._tools_by_mode.get(execution_mode, []))
        if not available_only:
            return tools
        return [tool for tool in tools if getattr(tool, "available", True)]

    def has_runtime(self, runtime_name: str) -> bool:
        return self._has_runtime


class StubLLMManager:
    """Async LLM manager stub that records generate calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any] | None]] = []

    async def generate(
        self, request: Any, overrides: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append((request, overrides))
        return LLMResponse(
            content="stub-response", model_used="stub-model", tokens_used=5
        )


def _make_tool(name: str, mode: ToolExecutionMode) -> ToolInfo:
    return ToolInfo(
        name=name,
        description=f"{name} tool",
        available=True,
        execution_mode=mode,
        parameters={"type": "object", "properties": {}, "required": []},
    )


def _build_runtime_manager(
    *,
    sandbox_enabled: bool = True,
    default_mode: str = "classic",
    tool_manager: FakeToolManager | None = None,
) -> tuple[RuntimeManager, StubConfigManager, StubLLMManager]:
    config_manager = StubConfigManager(
        sandbox_enabled=sandbox_enabled, tool_call_mode=default_mode
    )
    llm_manager = StubLLMManager()
    runtime = RuntimeManager(
        config_manager=config_manager,
        llm_manager=llm_manager,
        tool_manager=tool_manager,
    )
    return runtime, config_manager, llm_manager


@pytest.fixture
def prompt_capture(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    def fake_render(self: PromptComposer, context):
        captured["context"] = context
        return RenderedPrompt(
            system_messages=["stub-system"],
            user_messages=[context.user_input],
            tool_schemas=list(context.tools),
            metadata=context.metadata,
        )

    monkeypatch.setattr(PromptComposer, "render", fake_render)
    return captured


@pytest.mark.asyncio
async def test_orchestrate_resolves_reasoning_mode(prompt_capture: dict[str, Any]):
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=False,
        tool_manager=FakeToolManager(),
    )

    result = await runtime.orchestrate(
        "Explain recursion", tool_call_mode="reasoning_only"
    )

    assert "message" in result
    ctx = prompt_capture["context"]
    assert ctx.execution_mode == ExecutionMode.REASONING_ONLY
    assert ctx.tool_call_mode == "reasoning_only"
    assert ctx.tools == []


@pytest.mark.asyncio
async def test_orchestrate_enables_sandbox_python_when_capabilities_exist(
    prompt_capture: dict[str, Any],
):
    ptc_tools = [_make_tool("execute_python_code", ToolExecutionMode.PTC)]
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=True,
        tool_manager=FakeToolManager(ptc_tools=ptc_tools, has_runtime=True),
    )

    await runtime.orchestrate("Run code", tool_call_mode="ptc")

    ctx = prompt_capture["context"]
    assert ctx.execution_mode == ExecutionMode.SANDBOX_PYTHON
    assert ctx.metadata["sandbox_enabled"] is True
    assert len(ctx.tools) == 1
    assert ctx.tools[0]["function"]["name"] == "execute_python_code"


@pytest.mark.asyncio
async def test_orchestrate_falls_back_to_classic_when_sandbox_disabled(prompt_capture):
    ptc_tools = [_make_tool("execute_python_code", ToolExecutionMode.PTC)]
    runtime, config_manager, _ = _build_runtime_manager(
        sandbox_enabled=False,
        tool_manager=FakeToolManager(ptc_tools=ptc_tools, has_runtime=True),
    )

    # Simulate user requesting PTC mode through config while sandbox remains disabled.
    config_manager.global_config.runtime.tool_call_mode = "ptc"

    await runtime.orchestrate("Need sandbox")

    ctx = prompt_capture["context"]
    assert ctx.execution_mode == ExecutionMode.CLASSIC_TOOLS
    assert ctx.metadata["sandbox_enabled"] is False


@pytest.mark.asyncio
async def test_orchestrate_falls_back_to_reasoning_when_tool_manager_missing(
    prompt_capture,
):
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=True,
        tool_manager=None,
    )

    await runtime.orchestrate("Need sandbox but unavailable", tool_call_mode="ptc")

    ctx = prompt_capture["context"]
    assert ctx.execution_mode == ExecutionMode.REASONING_ONLY
    assert ctx.tools == []


@pytest.mark.asyncio
async def test_orchestrate_falls_back_to_classic_when_runtime_missing(prompt_capture):
    ptc_tools = [_make_tool("execute_python_code", ToolExecutionMode.PTC)]
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=True,
        tool_manager=FakeToolManager(ptc_tools=ptc_tools, has_runtime=False),
    )

    await runtime.orchestrate("Need python runtime", tool_call_mode="ptc")

    ctx = prompt_capture["context"]
    assert ctx.execution_mode == ExecutionMode.CLASSIC_TOOLS


@pytest.mark.asyncio
async def test_orchestrate_surfaces_prompt_composer_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=True,
        tool_manager=FakeToolManager(),
    )

    def failing_render(self: PromptComposer, context):
        raise TemplateError("Undefined variable 'missing_block'")

    monkeypatch.setattr(PromptComposer, "render", failing_render)

    with pytest.raises(TemplateError):
        await runtime.orchestrate("Cause template failure")


@pytest.mark.asyncio
async def test_orchestrate_rejects_invalid_tool_call_mode(prompt_capture):
    runtime, _, _ = _build_runtime_manager(
        sandbox_enabled=True,
        tool_manager=FakeToolManager(),
    )

    with pytest.raises(ValueError, match="Invalid tool_call_mode"):
        await runtime.orchestrate("Invalid mode", tool_call_mode="unsupported-mode")
