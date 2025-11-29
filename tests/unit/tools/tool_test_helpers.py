"""Shared helpers for ToolManager-related unit tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import MethodType
from typing import Callable
from unittest.mock import MagicMock

from pydantic import BaseModel

from local_coding_assistant.tools.base import Tool
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.tools.types import (
    ToolCategory,
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

    async def stream(self, input_data: Input):
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
    source: ToolSource = ToolSource.BUILTIN,
) -> ToolInfo:
    return ToolInfo(
        name=name,
        tool_class=tool_class,
        description=description,
        category=category,
        source=source,
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

    manager._run_sync = MethodType(_patched_run_sync, manager)  # type: ignore[attr-defined]
    return manager, config_manager


__all__ = [
    "AsyncTestTool",
    "FailingTestTool",
    "StubToolConfig",
    "StreamingTestTool",
    "SyncTestTool",
    "build_manager",
    "make_tool_info",
]
