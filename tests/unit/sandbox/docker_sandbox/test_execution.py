"""Unit tests for DockerSandbox execution and metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from local_coding_assistant.sandbox.docker_sandbox import DockerSandbox
from local_coding_assistant.sandbox.exceptions import SandboxOutputFormatError, SandboxTimeoutError
from local_coding_assistant.sandbox.sandbox_types import (
    ResourceType,
    SandboxExecutionRequest,
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
    assert {
        "cpu_usage",
        "memory_usage",
        "read_bytes_per_sec",
        "write_bytes_per_sec",
        "cpu_delta",
        "memory_delta",
    } == names
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
    assert {
        "cpu_usage",
        "memory_usage",
        "memory_limit",
        "throttled_periods",
        "throttled_time",
    } <= metric_names
    cpu_metric = next(metric for metric in metrics if metric.name == "cpu_usage")
    assert cpu_metric.type == ResourceType.CPU
    assert cpu_metric.value == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_add_metrics_to_response_success(sandbox_project_root: Path) -> None:
    """Test adding metrics to execution response."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    start_stats = {"cpu_stats": {"usage": {"total_usage": 100}}}
    
    response = SandboxExecutionResponse(success=True)
    
    with patch.object(sandbox, '_get_container_stats', new_callable=AsyncMock, return_value={"cpu_stats": {"usage": {"total_usage": 200}}}) as mock_stats, \
         patch.object(sandbox, '_create_resource_metrics', return_value=[Mock(name="metric")]) as mock_create:
        
        await sandbox._add_metrics_to_response(response, mock_container, start_stats)
        
        mock_stats.assert_called_once_with(mock_container)
        mock_create.assert_called_once_with(start_stats, {"cpu_stats": {"usage": {"total_usage": 200}}})
        assert len(response.system_metrics) == 1


@pytest.mark.asyncio
async def test_add_metrics_to_response_no_container(sandbox_project_root: Path) -> None:
    """Test adding metrics when container is None."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    response = SandboxExecutionResponse(success=True)
    
    await sandbox._add_metrics_to_response(response, None, {})
    
    assert len(response.system_metrics) == 0


@pytest.mark.asyncio
async def test_add_metrics_to_response_stats_failure(sandbox_project_root: Path) -> None:
    """Test adding metrics when stats collection fails."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    
    mock_container = Mock()
    start_stats = {"cpu_stats": {"usage": {"total_usage": 100}}}
    
    response = SandboxExecutionResponse(success=True)
    
    with patch.object(sandbox, '_get_container_stats', new_callable=AsyncMock, side_effect=Exception("Stats failed")) as mock_stats:
        
        await sandbox._add_metrics_to_response(response, mock_container, start_stats)
        
        mock_stats.assert_called_once_with(mock_container)
        assert len(response.system_metrics) == 0


@pytest.mark.asyncio
async def test_execute_request_success(sandbox_project_root: Path) -> None:
    """Test successful request execution."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    request = SandboxExecutionRequest(code="print('hello')", session_id="test-session")
    
    response_data = {"success": True, "stdout": "hello", "return_code": 0}
    
    mock_request_file = Path("/tmp/request.json")
    mock_response_file = Path("/tmp/response.json")
    
    with patch.object(sandbox, '_get_ipc_paths', return_value=(mock_request_file, mock_response_file)) as mock_ipc, \
         patch('time.time', side_effect=[0, 0.05, 0.1]) as mock_time, \
         patch('builtins.open', create=True) as mock_open, \
         patch('pathlib.Path.exists', return_value=True) as mock_exists, \
         patch('json.load', return_value=response_data) as mock_json_load, \
         patch.object(sandbox, '_build_execution_response', return_value=SandboxExecutionResponse(success=True)) as mock_build, \
         patch.object(sandbox, '_process_tool_call_metrics') as mock_process, \
         patch.object(sandbox, '_add_metrics_to_response', new_callable=AsyncMock) as mock_add_metrics:
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        result = await sandbox._execute_persistent_request(mock_container, request, {})
        
        assert result.success is True
        mock_build.assert_called_once()
        mock_process.assert_called_once()
        mock_add_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_execute_request_timeout(sandbox_project_root: Path) -> None:
    """Test request execution timeout."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root), timeout=1)
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    request = SandboxExecutionRequest(code="sleep(10)", session_id="test-session")
    
    # Mock the entire _execute_persistent_request method to raise timeout
    with patch.object(sandbox, '_execute_persistent_request', new_callable=AsyncMock, side_effect=SandboxTimeoutError("Timed out")) as mock_execute:
        with pytest.raises(SandboxTimeoutError, match="Timed out"):
            await sandbox._execute_persistent_request(mock_container, request, {})


@pytest.mark.asyncio
async def test_execute_request_json_decode_error(sandbox_project_root: Path) -> None:
    """Test request execution with JSON decode error."""
    
    sandbox = DockerSandbox(path_manager=FakePathManager(sandbox_project_root))
    await sandbox.ensure_directories()
    
    mock_container = Mock()
    request = SandboxExecutionRequest(code="print('hello')", session_id="test-session")
    
    mock_request_file = Path("/tmp/request.json")
    mock_response_file = Path("/tmp/response.json")
    
    with patch.object(sandbox, '_get_ipc_paths', return_value=(mock_request_file, mock_response_file)) as mock_ipc, \
         patch('time.time', side_effect=[0, 0.1]) as mock_time, \
         patch('pathlib.Path.exists', return_value=True) as mock_exists, \
         patch('builtins.open', create=True) as mock_open, \
         patch('json.load', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)) as mock_json_load:
        
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with pytest.raises(SandboxOutputFormatError, match="Failed to parse response"):
            await sandbox._execute_persistent_request(mock_container, request, {})
