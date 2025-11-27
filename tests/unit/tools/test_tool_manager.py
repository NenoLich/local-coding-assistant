"""Unit tests for the refactored ToolManager implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from types import MethodType
from typing import Callable
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.base import Tool
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
    ToolInfo,
    ToolSource,
)


class SyncTestTool(Tool):
    class Input(BaseModel):
        value: int

    class Output(BaseModel):
        result: int

    def run(self, input_data: Input) -> Output:
        return self.Output(result=input_data.value * 2)


class AsyncTestTool(Tool):
    class Input(BaseModel):
        value: int

    class Output(BaseModel):
        result: int

    async def run(self, input_data: Input) -> Output:
        await asyncio.sleep(0)
        return self.Output(result=input_data.value * 3)


class StreamingTestTool(Tool):
    class Input(BaseModel):
        value: int

    class Output(BaseModel):
        result: int

    def run(self, input_data: Input) -> Output:  # pragma: no cover - compatibility
        return self.Output(result=input_data.value)

    async def stream(self, input_data: Input) -> AsyncIterator[Output]:
        for offset in range(3):
            await asyncio.sleep(0)
            yield self.Output(result=input_data.value + offset)


class FailingTestTool(Tool):
    class Input(BaseModel):
        value: int

    class Output(BaseModel):
        result: int

    def run(self, _: Input) -> Output:  # pragma: no cover - defensive
        raise ValueError("boom")


def make_tool_info(
    name: str,
    tool_class: type[Tool],
    *,
    description: str = "",
    category: ToolCategory = ToolCategory.UTILITY,
    is_async: bool = False,
    supports_streaming: bool = False,
    enabled: bool = True,
    available: bool = True,
) -> ToolInfo:
    return ToolInfo(
        name=name,
        tool_class=tool_class,
        description=description,
        category=category,
        source=ToolSource.BUILTIN,
        permissions=[],
        tags=[],
        is_async=is_async,
        supports_streaming=supports_streaming,
        enabled=enabled,
        available=available,
    )


class StubToolConfig:
    def __init__(
        self,
        tool_info: ToolInfo,
        *,
        enabled: bool = True,
        available: bool = True,
    ) -> None:
        self._base_info = tool_info
        self.enabled = enabled
        self.available = available

    def to_tool_info(self) -> ToolInfo:
        return replace(
            self._base_info,
            enabled=self.enabled,
            available=self.available and self.enabled,
        )


def build_manager(
    tool_configs: dict[str, StubToolConfig],
    *,
    get_tools_factory: Callable[[], dict[str, StubToolConfig]] | None = None,
):
    config_manager = MagicMock()
    if get_tools_factory is None:
        config_manager.get_tools.return_value = tool_configs
    else:
        config_manager.get_tools.side_effect = lambda: get_tools_factory()
    config_manager.reload_tools = MagicMock()

    manager = ToolManager(config_manager=config_manager, auto_load=True)

    def _patched_run_sync(self: ToolManager, awaitable):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(awaitable)
        finally:
            loop.close()

    manager._run_sync = MethodType(_patched_run_sync, manager) # type: ignore
    return manager, config_manager


class TestToolLoading:
    def test_enabled_tools_create_runtimes(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, config_manager = build_manager({
            "sync_tool": StubToolConfig(sync_info),
        })

        config_manager.get_tools.assert_called_once()
        assert manager.get_tool_info("sync_tool") is not None
        assert "sync_tool" in manager._runtimes

    def test_disabled_tool_skipped(self):
        disabled_info = make_tool_info("disabled_tool", SyncTestTool)
        manager, _ = build_manager({
            "disabled_tool": StubToolConfig(disabled_info, enabled=False, available=False),
        })

        info = manager.get_tool_info("disabled_tool")
        assert info is not None
        assert info.enabled is False
        assert "disabled_tool" not in manager._runtimes

    def test_reload_tools_uses_config_manager(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        async_info = make_tool_info(
            "async_tool",
            AsyncTestTool,
            is_async=True,
        )
        configs = [
            {"sync_tool": StubToolConfig(sync_info)},
            {"async_tool": StubToolConfig(async_info)},
        ]

        def get_tools_factory():
            return configs.pop(0)

        manager, config_manager = build_manager({}, get_tools_factory=get_tools_factory)

        assert "sync_tool" in manager._tools

        manager.reload_tools()

        config_manager.reload_tools.assert_called_once()
        assert "async_tool" in manager._tools
        assert "async_tool" in manager._runtimes
        assert "sync_tool" not in manager._runtimes


class TestToolExecution:
    def test_run_tool_success_updates_stats(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        result = manager.run_tool("sync_tool", {"value": 21})

        assert result == {"result": 42}
        stats = manager.get_execution_stats()["sync_tool"]
        assert stats["call_count"] == 1
        assert stats["error_count"] == 0
        assert stats["last_called"] is not None

    def test_run_tool_invalid_input_raises_tool_registry_error(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        with pytest.raises(ToolRegistryError) as exc:
            manager.run_tool("sync_tool", {"wrong": "value"})

        # Check that we got a validation error
        assert "Error executing tool 'sync_tool'" in str(exc.value)

    def test_run_unknown_tool_raises(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        with pytest.raises(ToolRegistryError) as exc:
            manager.run_tool("missing", {"value": 1})

        assert "not found" in str(exc.value)

    @pytest.mark.asyncio
    async def test_run_tool_async_handles_async_tools(self):
        async_info = make_tool_info(
            "async_tool",
            AsyncTestTool,
            is_async=True,
        )
        manager, _ = build_manager({"async_tool": StubToolConfig(async_info)})

        result = await manager.run_tool_async("async_tool", {"value": 5})

        assert result == {"result": 15}

    @pytest.mark.asyncio
    async def test_stream_tool_yields_chunks(self):
        streaming_info = make_tool_info(
            "streaming_tool",
            StreamingTestTool,
            is_async=True,
            supports_streaming=True,
        )
        manager, _ = build_manager({
            "streaming_tool": StubToolConfig(streaming_info),
        })

        chunks = []
        async for chunk in manager.stream_tool("streaming_tool", {"value": 10}):
            chunks.append(chunk["result"])

        assert chunks == [10, 11, 12]

    def test_run_tool_records_errors(self):
        failing_info = make_tool_info("failing_tool", FailingTestTool)
        manager, _ = build_manager({"failing_tool": StubToolConfig(failing_info)})

        with pytest.raises(ToolRegistryError):
            manager.run_tool("failing_tool", {"value": 1})

        stats = manager.get_execution_stats()["failing_tool"]
        assert stats["call_count"] == 1
        assert stats["error_count"] == 1
        assert stats["last_error"] is not None


class TestToolIntrospection:
    def test_get_tool_info_returns_metadata(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool, description="Sync tool")
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        info = manager.get_tool_info("sync_tool")
        assert info is not None
        assert info.name == "sync_tool"
        assert info.description == "Sync tool"
        assert info.category == ToolCategory.UTILITY

    def test_list_tools_filters_by_category(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        async_info = make_tool_info(
            "async_tool",
            AsyncTestTool,
            is_async=True,
        )
        manager, _ = build_manager(
            {
                "sync_tool": StubToolConfig(sync_info),
                "async_tool": StubToolConfig(async_info),
            }
        )

        all_tools = manager.list_tools()
        assert {tool.name for tool in all_tools} == {"sync_tool", "async_tool"}

        utility_tools = manager.list_tools(category=ToolCategory.UTILITY)
        assert {tool.name for tool in utility_tools} == {"sync_tool", "async_tool"}

        missing_category = manager.list_tools(category="nonexistent")
        assert missing_category == []

    def test_iteration_over_manager(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        items = list(manager)
        assert items == [("sync_tool", manager.get_tool_info("sync_tool"))]


class TestExecuteAPI:
    def test_execute_success_response(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        request = ToolExecutionRequest(tool_name="sync_tool", payload={"value": 4})
        response = manager.execute(request)

        assert response.success is True
        assert response.result == {"result": 8}
        assert response.execution_time_ms is not None

    def test_execute_unknown_tool(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        request = ToolExecutionRequest(tool_name="missing", payload={})
        response = manager.execute(request)

        assert response.success is False
        assert response.result is None
        assert "missing" in response.error_message

    def test_execute_validation_error(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, _ = build_manager({"sync_tool": StubToolConfig(sync_info)})

        request = ToolExecutionRequest(tool_name="sync_tool", payload={})
        response = manager.execute(request)

        assert response.success is False
        assert response.result is None
        assert "Error executing tool 'sync_tool'" in response.error_message

