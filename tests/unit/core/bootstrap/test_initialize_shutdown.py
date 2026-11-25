"""Unit tests for bootstrap initialize/shutdown and dashboard lifecycle."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import (
    _auto_start_dashboard,
    _close_async_resource,
    _close_sync_resource,
    _is_dashboard_running,
    _terminate_process,
    initialize,
    shutdown,
)


class TestInitialize:
    """Test async resource initialization."""

    @pytest.fixture
    def ctx(self) -> AppContext:
        ctx = AppContext()
        store = Mock()
        store.init = AsyncMock()
        ctx.register("execution_store", store)
        return ctx

    async def test_initialize_opens_store_and_tool_gateway_identity(self, ctx):
        tool_gateway = Mock()
        tool_gateway.init_identity = AsyncMock()
        ctx.register("tool_gateway", tool_gateway)

        await initialize(ctx)

        store = ctx.get("execution_store")
        assert store is not None
        store.init.assert_awaited_once_with()
        tool_gateway.init_identity.assert_awaited_once_with(store=store)

    async def test_initialize_missing_store_raises(self):
        ctx = AppContext()
        with pytest.raises(RuntimeError, match="Execution store missing"):
            await initialize(ctx)

    async def test_initialize_tool_gateway_without_init_identity(self, ctx):
        ctx.register("tool_gateway", Mock(spec=[]))

        await initialize(ctx)

        ctx.get("execution_store").init.assert_awaited_once_with()

    async def test_initialize_store_init_failure(self, ctx):
        ctx.get("execution_store").init = AsyncMock(side_effect=Exception("db error"))
        with pytest.raises(RuntimeError, match="Failed to initialize application"):
            await initialize(ctx)

    async def test_initialize_identity_failure(self, ctx):
        tool_gateway = Mock()
        tool_gateway.init_identity = AsyncMock(side_effect=Exception("identity error"))
        ctx.register("tool_gateway", tool_gateway)
        with pytest.raises(RuntimeError, match="Failed to initialize application"):
            await initialize(ctx)

    async def test_initialize_idempotent(self, ctx):
        tool_gateway = Mock()
        tool_gateway.init_identity = AsyncMock()
        ctx.register("tool_gateway", tool_gateway)

        await initialize(ctx)
        await initialize(ctx)

        assert ctx.get("execution_store").init.await_count == 2


class TestShutdown:
    """Test async resource shutdown."""

    def make_ctx(self, **resources: object) -> AppContext:
        ctx = AppContext()
        for name, resource in resources.items():
            ctx.register(name, resource)
        return ctx

    async def test_shutdown_closes_all_resources(self):
        pipeline = Mock()
        pipeline.close = AsyncMock()
        store = Mock()
        store.close = AsyncMock()
        repo = Mock()
        repo.close = Mock()
        process = Mock()
        process.terminate = Mock()
        process.wait = Mock()

        ctx = self.make_ctx(
            event_pipeline=pipeline,
            execution_store=store,
            repository_context=repo,
            dashboard_process=process,
        )

        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ) as mock_telemetry:
            await shutdown(ctx)

        pipeline.close.assert_awaited_once_with()
        store.close.assert_awaited_once_with()
        repo.close.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        mock_telemetry.assert_called_once_with()

    async def test_shutdown_no_resources(self):
        ctx = AppContext()
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ) as mock_telemetry:
            await shutdown(ctx)
        mock_telemetry.assert_called_once_with()

    async def test_shutdown_missing_optional_resources(self):
        ctx = self.make_ctx(event_pipeline=Mock())
        ctx.get("event_pipeline").close = AsyncMock()
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ):
            await shutdown(ctx)

    async def test_shutdown_pipeline_close_failure_does_not_raise(self):
        pipeline = Mock()
        pipeline.close = AsyncMock(side_effect=Exception("boom"))
        ctx = self.make_ctx(event_pipeline=pipeline)
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ):
            await shutdown(ctx)

    async def test_shutdown_store_close_failure_does_not_raise(self):
        store = Mock()
        store.close = AsyncMock(side_effect=Exception("boom"))
        ctx = self.make_ctx(execution_store=store)
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ):
            await shutdown(ctx)

    async def test_shutdown_terminate_process_wait_timeout_kills(self):
        process = Mock()
        process.terminate = Mock()
        process.wait = Mock(side_effect=Exception("wait failed"))
        ctx = self.make_ctx(dashboard_process=process)
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ):
            await shutdown(ctx)
        process.kill.assert_called_once_with()

    async def test_shutdown_telemetry_called_via_thread(self):
        ctx = AppContext()
        with patch(
            "local_coding_assistant.core.bootstrap.telemetry_module.shutdown_telemetry"
        ) as mock_telemetry:
            await shutdown(ctx)
        mock_telemetry.assert_called_once_with()


class TestCloseResources:
    """Test resource close helpers."""

    async def test_close_async_resource_success(self):
        close = AsyncMock()
        await _close_async_resource(close, "test resource")
        close.assert_awaited_once_with()

    async def test_close_async_resource_cancelled(self):
        close = AsyncMock(side_effect=asyncio.CancelledError)
        await _close_async_resource(close, "test resource")

    async def test_close_async_resource_failure_logs(self):
        close = AsyncMock(side_effect=Exception("failed"))
        await _close_async_resource(close, "test resource")
        close.assert_awaited_once_with()

    def test_close_sync_resource_success(self):
        close = Mock()
        _close_sync_resource(close, "test resource")
        close.assert_called_once_with()

    def test_close_sync_resource_cancelled(self):
        close = Mock(side_effect=asyncio.CancelledError)
        _close_sync_resource(close, "test resource")


class TestTerminateProcess:
    """Test dashboard process termination."""

    def test_terminate_process_success(self):
        process = Mock()
        process.terminate = Mock()
        process.wait = Mock()

        _terminate_process(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_terminate_process_wait_failure_kills(self):
        process = Mock()
        process.terminate = Mock()
        process.wait = Mock(side_effect=Exception("wait failed"))
        process.kill = Mock()

        _terminate_process(process)

        process.kill.assert_called_once_with()

    def test_terminate_process_terminate_failure_kills(self):
        process = Mock()
        process.terminate = Mock(side_effect=Exception("terminate failed"))
        process.kill = Mock()

        _terminate_process(process)

        process.kill.assert_called_once_with()


class TestIsDashboardRunning:
    """Test dashboard running check."""

    def test_is_dashboard_running_true(self):
        with patch(
            "local_coding_assistant.core.bootstrap.socket.create_connection"
        ) as mock_conn:
            mock_conn.return_value.__enter__.return_value = Mock()
            assert _is_dashboard_running("127.0.0.1", 8080) is True

    def test_is_dashboard_running_false(self):
        with patch(
            "local_coding_assistant.core.bootstrap.socket.create_connection",
            side_effect=OSError("refused"),
        ):
            assert _is_dashboard_running("127.0.0.1", 8080) is False


class TestAutoStartDashboard:
    """Test dashboard auto-start behavior."""

    def config_manager(
        self,
        auto_start: bool = True,
        host: str = "127.0.0.1",
        port: int = 8080,
        store_enabled: bool = True,
    ) -> MagicMock:
        cm = MagicMock()
        cm.global_config.dashboard.auto_start = auto_start
        cm.global_config.dashboard.host = host
        cm.global_config.dashboard.port = port
        store = Mock()
        store.enabled = store_enabled
        cm.global_config.execution_store = store
        cm.path_manager.resolve_path.return_value = Path("data/execution_store.db")
        cm.path_manager.get_project_root.return_value = Path(tempfile.mkdtemp())
        return cm

    def test_auto_start_disabled(self):
        ctx = AppContext()
        cm = self.config_manager(auto_start=False)
        with patch(
            "local_coding_assistant.core.bootstrap.subprocess.Popen"
        ) as mock_popen:
            _auto_start_dashboard(cm, ctx)
        mock_popen.assert_not_called()
        assert ctx.get("dashboard_process") is None

    def test_dashboard_already_running_skips(self):
        ctx = AppContext()
        cm = self.config_manager()
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=True,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.subprocess.Popen"
            ) as mock_popen:
                _auto_start_dashboard(cm, ctx)
        mock_popen.assert_not_called()
        assert ctx.get("dashboard_process") is None

    def test_auto_start_success_registers_process(self):
        ctx = AppContext()
        cm = self.config_manager()
        process = Mock()
        process.poll.return_value = None
        process.pid = 1234
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=False,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.subprocess.Popen",
                return_value=process,
            ):
                with patch("local_coding_assistant.core.bootstrap.time.sleep"):
                    _auto_start_dashboard(cm, ctx)
        assert ctx.get("dashboard_process") is process

    def test_auto_start_process_exited_not_registered(self):
        ctx = AppContext()
        cm = self.config_manager()
        process = Mock()
        process.poll.return_value = 1
        process.returncode = 1
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=False,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.subprocess.Popen",
                return_value=process,
            ):
                with patch("local_coding_assistant.core.bootstrap.time.sleep"):
                    _auto_start_dashboard(cm, ctx)
        assert ctx.get("dashboard_process") is None

    def test_auto_start_popen_command(self):
        ctx = AppContext()
        cm = self.config_manager()
        process = Mock()
        process.poll.return_value = None
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=False,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.subprocess.Popen",
                return_value=process,
            ) as mock_popen:
                with patch("local_coding_assistant.core.bootstrap.time.sleep"):
                    _auto_start_dashboard(cm, ctx)
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == sys.executable
        assert "--host" in cmd
        assert str(cm.global_config.dashboard.port) in cmd
        assert "--store-db-path" in cmd
        assert kwargs.get("shell") is False

    def test_auto_start_exception_no_crash(self):
        ctx = AppContext()
        cm = self.config_manager()
        cm.path_manager.resolve_path.side_effect = Exception("boom")
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=False,
        ):
            _auto_start_dashboard(cm, ctx)
        assert ctx.get("dashboard_process") is None

    def test_auto_start_store_disabled_uses_default_db_path(self):
        ctx = AppContext()
        cm = self.config_manager(store_enabled=False)
        process = Mock()
        process.poll.return_value = None
        process.pid = 1
        cm.global_config.execution_store = None
        with patch(
            "local_coding_assistant.core.bootstrap._is_dashboard_running",
            return_value=False,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.subprocess.Popen",
                return_value=process,
            ):
                with patch("local_coding_assistant.core.bootstrap.time.sleep"):
                    _auto_start_dashboard(cm, ctx)
        assert ctx.get("dashboard_process") is process
