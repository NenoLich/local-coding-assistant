"""Unit tests for DockerSandbox path and state management."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from local_coding_assistant.config.schemas import SandboxConfig, SandboxLoggingConfig
from local_coding_assistant.sandbox.docker_sandbox import DockerSandbox
from local_coding_assistant.sandbox.exceptions import SandboxRuntimeError
from local_coding_assistant.sandbox.sandbox_types import (
    ResourceType,
    SandboxExecutionResponse,
)


class FakePathManager:
    """Minimal PathManager stub for isolating filesystem behavior."""

    def __init__(self, project_root: Path):
        self._project_root = project_root
        self.project_root_calls = 0
        self.resolve_calls: list[str] = []

    def get_project_root(self) -> Path:
        self.project_root_calls += 1
        return self._project_root

    def resolve_path(self, path: str) -> Path:
        self.resolve_calls.append(path)
        if path == "@tools/sandbox_tools":
            resolved = self._project_root / "tools" / "sandbox_tools"
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        msg = f"Unexpected resolve_path call: {path}"
        raise ValueError(msg)


@pytest.fixture()
def sandbox_project_root(tmp_path: Path) -> Path:
    """Provide a temporary project root with expected sub-structure."""
    (tmp_path / "src" / "local_coding_assistant" / "sandbox" / "guest").mkdir(
        parents=True, exist_ok=True
    )
    return tmp_path


class DummyContainer:
    """Minimal container stub for reuse tests."""

    def __init__(self, *, status: str = "running", ident: str = "dummy"):
        self.status = status
        self.id = ident
        self.reload_calls = 0

    def reload(self) -> None:
        self.reload_calls += 1


@pytest.mark.asyncio
async def test_ensure_directories_is_cached(sandbox_project_root: Path) -> None:
    path_manager = FakePathManager(sandbox_project_root)
    sandbox = DockerSandbox(path_manager=path_manager)

    first_paths = await sandbox.ensure_directories()
    second_paths = await sandbox.ensure_directories()

    assert first_paths is second_paths
    assert path_manager.project_root_calls == 1
    assert path_manager.resolve_calls == ["@tools/sandbox_tools"]
    assert first_paths.host_workspace.exists()
    assert first_paths.workspace_tools_dir.exists()


@pytest.mark.asyncio
async def test_logs_directory_sanitizes_configured_path(sandbox_project_root: Path) -> None:
    config = SandboxConfig(
        logging=SandboxLoggingConfig(directory="/workspace/custom/logs"),
    )
    path_manager = FakePathManager(sandbox_project_root)
    sandbox = DockerSandbox(path_manager=path_manager, config=config)

    paths = await sandbox.ensure_directories()

    expected_logs_dir = (
        sandbox_project_root / ".sandbox_workspace" / "workspace/custom/logs"
    )
    assert paths.logs_dir == expected_logs_dir
    assert expected_logs_dir.exists()


def test_paths_access_requires_initialization(sandbox_project_root: Path) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))

    with pytest.raises(SandboxRuntimeError):
        _ = sandbox._project_root


@pytest.mark.asyncio
async def test_build_volume_mounts_uses_expected_directories(
    sandbox_project_root: Path,
) -> None:
    path_manager = FakePathManager(sandbox_project_root)
    sandbox = DockerSandbox(path_manager=path_manager)
    paths = await sandbox.ensure_directories()

    volumes = sandbox._build_volume_mounts("session-42")

    guest_dir = (
        sandbox_project_root
        / "src"
        / "local_coding_assistant"
        / "sandbox"
        / "guest"
    )

    assert volumes[str(paths.host_workspace)] == {"bind": "/workspace", "mode": "rw"}
    assert volumes[str(paths.logs_dir)] == {"bind": "/workspace/logs", "mode": "rw"}
    assert volumes[str(guest_dir)] == {"bind": "/agent", "mode": "ro"}
    tools_src_dir = sandbox._tools_src_dir
    assert volumes[str(tools_src_dir)] == {"bind": "/tools/sandbox_tools", "mode": "ro"}


def test_determine_container_parameters_based_on_persistence(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))

    persistent_name, command = sandbox._determine_container_parameters("abc", True)
    assert persistent_name == "locca-sandbox-abc"
    assert command is not None
    assert "--ipc-dir" in command
    assert any("locca_ipc_abc" in segment for segment in command)

    ephemeral_name, ephemeral_command = sandbox._determine_container_parameters(
        "abc", False
    )
    assert ephemeral_name.startswith("locca-sandbox-ephemeral-")
    assert ephemeral_command is None


def test_build_environment_serializes_logging_config(sandbox_project_root: Path) -> None:
    config = SandboxConfig(
        logging=SandboxLoggingConfig(
            directory="/workspace/custom",
            file_name="custom_{session_id}.log",
            console=False,
            file=True,
        )
    )
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root), config=config)

    logging_config = sandbox._build_logging_config("sess-1")
    env = sandbox._build_environment(logging_config, "sess-1")

    assert env["PYTHONPATH"] == "/agent:/tools:/workspace"
    assert env["IPC_DIR"].endswith("locca_ipc_sess-1")

    serialized = json.loads(env["LOGGING_CONFIG"])
    assert serialized["file_name"] == "custom_sess-1.log"
    assert serialized["directory"] == "/workspace/custom"
    assert serialized["console"] is False
    assert serialized["file"] is True


def test_validate_persistent_capacity_enforces_limit(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(
        path_manager=FakePathManager(sandbox_project_root), max_sessions=1
    )
    sandbox._containers["active"] = DummyContainer()

    with pytest.raises(SandboxRuntimeError):
        sandbox._validate_persistent_capacity()


@pytest.mark.asyncio
async def test_get_ipc_paths_creates_directories_and_unique_files(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()

    req1, resp1 = sandbox._get_ipc_paths("alpha")
    req2, resp2 = sandbox._get_ipc_paths("alpha")

    assert req1.parent.name == "requests"
    assert resp1.parent.name == "responses"
    assert req1.parent.parent.name == "locca_ipc_alpha"
    assert req1 != req2
    assert resp1 != resp2
    assert req1.parent.exists()
    assert resp1.parent.exists()


def test_build_execution_response_applies_defaults() -> None:
    sandbox = DockerSandbox()

    response = sandbox._build_execution_response(
        {"success": True, "stdout": "done"}, duration=1.2, stderr_fallback="fallback"
    )

    assert response.success is True
    assert response.stderr == "fallback"
    assert response.return_code == 0
    assert response.duration == 1.2


def test_extract_resource_metrics_converts_units() -> None:
    sandbox = DockerSandbox()
    call_data = {
        "timestamp": 1_700_000_000,
        "end_stats": {
            "cpu_percent": 12.5,
            "memory_rss_mb": 64,
            "read_bytes_per_sec": 10,
            "write_bytes_per_sec": 20,
        },
        "delta_stats": {
            "cpu_delta": 3.0,
            "memory_delta_mb": 1.5,
        },
    }

    metrics = sandbox._extract_resource_metrics(call_data)

    names = {metric.name for metric in metrics}
    assert {"cpu_usage", "memory_usage", "read_bytes_per_sec", "write_bytes_per_sec", "cpu_delta", "memory_delta"} == names
    memory_metric = next(m for m in metrics if m.name == "memory_usage")
    assert memory_metric.value == 64 * 1024 * 1024
    assert memory_metric.type == ResourceType.MEMORY


def test_process_tool_call_metrics_appends_calls() -> None:
    sandbox = DockerSandbox()
    response = SandboxExecutionResponse(success=True)
    call_data = {
        "tool_name": "python",
        "call_id": "abc",
        "start_time": "2024-01-01T00:00:00+00:00",
        "end_time": "2024-01-01T00:00:01+00:00",
        "duration": 1.0,
        "success": True,
        "end_stats": {
            "cpu_percent": 1.0,
            "memory_rss_mb": 1,
        },
    }

    sandbox._process_tool_call_metrics(
        response,
        {"tool_calls": [call_data]},
    )

    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.tool_name == "python"
    assert tool_call.call_id == "abc"
    assert any(metric.name == "cpu_usage" for metric in tool_call.resource_metrics)


@pytest.mark.asyncio
async def test_reuse_tracked_container_rejects_default_session(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._containers["default"] = DummyContainer(ident="cont-1")
    sandbox._cleanup_container = AsyncMock()  # type: ignore[attr-defined]

    reused = await sandbox._reuse_tracked_container("default", persistence=True)

    assert reused is None
    sandbox._cleanup_container.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_reuse_tracked_container_drops_non_running_container(
    sandbox_project_root: Path,
) -> None:
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    sandbox._cleanup_container = AsyncMock()  # type: ignore[attr-defined]
    sandbox._containers["session"] = DummyContainer(status="exited")

    reused = await sandbox._reuse_tracked_container("session", persistence=True)

    assert reused is None
    sandbox._cleanup_container.assert_not_awaited()  # type: ignore[attr-defined]
    assert "session" not in sandbox._containers


def test_create_resource_metrics_emits_cpu_memory_and_throttling() -> None:
    sandbox = DockerSandbox()
    start_stats = {"dummy": True}
    end_stats = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 200},
            "system_cpu_usage": 1000,
            "online_cpus": 2,
            "throttling_data": {"throttled_periods": 1, "throttled_time": 10},
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 50},
            "system_cpu_usage": 500,
        },
        "memory_stats": {"usage": 1024, "limit": 4096},
    }

    metrics = sandbox._create_resource_metrics(start_stats, end_stats)

    metric_names = {metric.name for metric in metrics}
    assert {"cpu_usage", "memory_usage", "memory_limit", "throttled_periods", "throttled_time"} <= metric_names
    cpu_metric = next(metric for metric in metrics if metric.name == "cpu_usage")
    assert cpu_metric.type == ResourceType.CPU
    assert cpu_metric.value == pytest.approx(60.0)
