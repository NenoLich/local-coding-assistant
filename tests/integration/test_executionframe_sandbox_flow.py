"""Integration test for sandbox tool call expansion in ExecutionFrame flow."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from local_coding_assistant.agent.frame_agent import FrameAgent
from local_coding_assistant.agent.llm import LLMResult, LLMStreamChunk, LLMToolCall
from local_coding_assistant.core.telemetry_types import (
    ExecutionEnvelope,
    ResourceMetric,
    ResourceType,
    ToolCallTrace,
)
from local_coding_assistant.runtime.context_manager import ContextManager
from local_coding_assistant.runtime.events import EventType
from local_coding_assistant.runtime.execution_types import ActionKind
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionMode,
    ToolExecutionResponse,
    ToolInfo,
    ToolSource,
)

from .conftest import TestConfigManager

TestConfigManager.__test__ = False


class SandboxToolManagerStub:
    """Minimal tool manager stub for sandbox tool-call integration."""

    def __init__(self):
        self._ptc_tool = ToolInfo(
            name="execute_python_code",
            description="Execute Python in sandbox",
            category=ToolCategory.PTC,
            available=True,
            enabled=True,
            parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        )
        self._sandbox_tool = ToolInfo(
            name="tool1",
            description="Sandbox tool",
            source=ToolSource.SANDBOX,
            available=True,
            enabled=True,
            parameters={"type": "object", "properties": {}},
        )

    def list_tools(self, available_only: bool = False, execution_mode=None, **kwargs):
        tools = [self._ptc_tool, self._sandbox_tool]
        if available_only:
            tools = [tool for tool in tools if tool.available]
        if execution_mode == ToolExecutionMode.PTC:
            return [
                tool for tool in tools if tool.execution_mode == ToolExecutionMode.PTC
            ]
        if execution_mode == ToolExecutionMode.SANDBOX:
            return [tool for tool in tools if tool.source == ToolSource.SANDBOX]
        return tools

    def get_sandbox_tools_prompt(self, tools=None):
        return ["sandbox tools prompt"]

    async def execute_async(self, request):
        tool_calls = [
            ToolCallTrace(
                call_id="call-1",
                tool_name="tool1",
                success=True,
                input={"args": [1]},
                output="ok",
                duration_ms=12.0,
            ),
            ToolCallTrace(
                call_id="call-2",
                tool_name="tool2",
                success=False,
                input={"kwargs": {"q": "x"}},
                error="failed",
                duration_ms=8.0,
            ),
        ]
        envelope = ExecutionEnvelope(
            tool_name=request.tool_name,
            session_id="sandbox-session",
            success=True,
            system_metrics=[
                ResourceMetric(
                    type=ResourceType.MEMORY,
                    name="memory_rss_mb",
                    value=10.0,
                    unit="mb",
                )
            ],
        )
        return ToolExecutionResponse(
            success=True,
            tool_name=request.tool_name,
            tool_args=request.payload,
            execution_time_ms=50.0,
            tool_calls=tool_calls,
            envelope=envelope,
        )


class SimpleLLMService:
    """LLM stub that always requests sandbox execution."""

    async def generate(self, task, *, options=None):
        return LLMResult(
            content="Running in sandbox",
            model="test-model",
            provider="test-provider",
            total_tokens=50,
            prompt_tokens=25,
            completion_tokens=25,
            tool_calls=[
                LLMToolCall(
                    name="execute_python_code",
                    arguments={"code": "print(1)"},
                    type="function",
                )
            ],
        )

    async def stream(self, task, *, options=None) -> AsyncIterator[LLMStreamChunk]:
        """Stream a response that includes tool calls."""
        content = "Running in sandbox"

        # Yield content in chunks
        words = content.split()
        current_chunk = ""
        for word in words:
            current_chunk += word + " "
            yield LLMStreamChunk(
                content=current_chunk.strip(),
                provider="test-provider",
                model="test-model",
                is_final=False,
            )
            await asyncio.sleep(0.01)  # Simulate streaming delay

        # Final chunk with tool calls and usage information
        yield LLMStreamChunk(
            content=content,
            provider="test-provider",
            model="test-model",
            is_final=True,
            tool_calls=[
                LLMToolCall(
                    name="execute_python_code",
                    arguments={"code": "print(1)"},
                    type="function",
                )
            ],
            usage={
                "prompt_tokens": 25,
                "completion_tokens": 25,
                "total_tokens": 50,
            },
        )


@pytest.mark.asyncio
async def test_frame_agent_expands_sandbox_tool_calls():
    config_manager = TestConfigManager()
    config_manager._global_config.runtime.tool_call_mode = "ptc"
    config_manager._global_config.sandbox.enabled = True
    config_manager._global_config.runtime.stream = False

    llm = SimpleLLMService()
    tool_manager = SandboxToolManagerStub()
    context_manager = ContextManager(config_manager, tool_manager=tool_manager)

    agent = FrameAgent(
        llm_service=llm,
        tool_manager=tool_manager,
        context_manager=context_manager,
        config_manager=config_manager,
        max_iterations=1,
    )

    session = SessionState(id="session-1")

    # Consume the async generator returned by agent.run()
    async for event in agent.run("hello", session):
        if event.type == EventType.TURN_COMPLETE:
            break

    frames = agent.get_frames()
    assert len(frames) == 1
    frame = frames[0]

    tool_actions = [
        action for action in frame.actions if action.kind == ActionKind.TOOL_CALL
    ]
    tool_names = {action.name for action in tool_actions}
    assert "execute_python_code" in tool_names
    assert "tool1" in tool_names
    assert "tool2" in tool_names
    assert frame.result is not None
    assert len(frame.get_tool_results()) == 3
