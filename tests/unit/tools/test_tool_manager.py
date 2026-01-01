"""Unit tests for the refactored ToolManager implementation."""

from __future__ import annotations

import pytest

from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
    ToolSource,
)

from .tool_test_helpers import (
    AsyncTestTool,
    FailingTestTool,
    StreamingTestTool,
    StubToolConfig,
    SyncTestTool,
    build_manager,
    make_tool_info,
)


class TestToolLoading:
    def test_enabled_tools_create_runtimes(self):
        sync_info = make_tool_info("sync_tool", SyncTestTool)
        manager, config_manager = build_manager(
            {
                "sync_tool": StubToolConfig(sync_info),
            }
        )

        config_manager.get_tools.assert_called_once()
        assert manager.get_tool_info("sync_tool") is not None
        assert "sync_tool" in manager._runtimes

    def test_disabled_tool_skipped(self):
        disabled_info = make_tool_info("disabled_tool", SyncTestTool)
        manager, _ = build_manager(
            {
                "disabled_tool": StubToolConfig(
                    disabled_info, enabled=False, available=False
                ),
            }
        )

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
        stats = manager.get_execution_stats("sync_tool")
        assert stats["total_executions"] == 1
        assert stats["error_count"] == 0
        assert stats["success_count"] == 1
        assert stats["last_execution"] is not None
        assert stats["success_rate"] == 100.0
        assert isinstance(stats["metrics_summary"], dict)

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
        manager, _ = build_manager(
            {
                "streaming_tool": StubToolConfig(streaming_info),
            }
        )

        chunks = []
        async for chunk in manager.stream_tool("streaming_tool", {"value": 10}):
            chunks.append(chunk["result"])

        assert chunks == [10, 11, 12]

    def test_run_tool_records_errors(self):
        failing_info = make_tool_info("failing_tool", FailingTestTool)
        manager, _ = build_manager({"failing_tool": StubToolConfig(failing_info)})

        with pytest.raises(ToolRegistryError):
            manager.run_tool("failing_tool", {"value": 1})

        stats = manager.get_execution_stats("failing_tool")
        assert stats["total_executions"] == 1
        assert stats["error_count"] == 1
        assert stats["success_count"] == 0
        assert stats["last_execution"] is not None
        # The error is recorded in the call metrics, not directly in the stats
        assert stats["success_rate"] == 0.0
        assert isinstance(stats["metrics_summary"], dict)


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


class TestRuntimeHelpers:
    def test_runtime_from_instance_sync_tool(self):
        manager, _ = build_manager({})
        tool_info = make_tool_info("sync_tool", SyncTestTool)
        runtime = manager._runtime_from_instance(tool_info, SyncTestTool())

        assert runtime.kind == "tool"
        assert runtime.run_is_async is False
        # SyncTestTool has a sync run method and an async stream method
        # so supports_streaming should be True
        assert runtime.supports_streaming is True
        assert runtime.has_input_validation is True
        assert runtime.has_output_validation is True

    def test_runtime_from_instance_async_streaming_tool(self):
        manager, _ = build_manager({})
        tool_info = make_tool_info(
            "async_tool",
            StreamingTestTool,
            is_async=False,  # The run method is not async
            supports_streaming=True,
        )
        runtime = manager._runtime_from_instance(tool_info, StreamingTestTool())

        # The run method in StreamingTestTool is not async, so run_is_async should be False
        assert runtime.run_is_async is False
        # But it does support streaming via the async stream method
        assert runtime.supports_streaming is True
        assert runtime.has_input_validation is True
        assert runtime.has_output_validation is True

    def test_runtime_from_instance_mcp_tool_sets_kind(self):
        manager, _ = build_manager({})
        tool_info = make_tool_info(
            "mcp_tool",
            SyncTestTool,
            source=ToolSource.MCP,
        )

        class DummyExecutor:
            Input = SyncTestTool.Input
            Output = SyncTestTool.Output

        runtime = manager._runtime_from_instance(tool_info, DummyExecutor())

        assert runtime.kind == "mcp"
