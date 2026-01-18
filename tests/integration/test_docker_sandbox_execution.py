"""Integration-style tests for DockerSandbox security and execution flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_coding_assistant.config.schemas import SandboxConfig
from local_coding_assistant.sandbox.docker_sandbox import DockerSandbox
from local_coding_assistant.sandbox.exceptions import (
    SandboxRuntimeError,
    SandboxTimeoutError,
)
from local_coding_assistant.sandbox.sandbox_types import (
    SandboxExecutionRequest,
    SandboxExecutionResponse,
)
from local_coding_assistant.sandbox.security import SecurityManager


class StubPathManager:
    """Minimal PathManager replacement for sandbox tests."""

    def __init__(self, project_root: Path):
        self._project_root = project_root

    def get_project_root(self) -> Path:
        return self._project_root

    def resolve_path(self, path: str) -> Path:
        if path == "@tools/sandbox_tools":
            target = self._project_root / "tools" / "sandbox_tools"
            target.mkdir(parents=True, exist_ok=True)
            return target
        if path.startswith("@project/"):
            return self._project_root / path.split("@project/", 1)[1]
        resolved = self._project_root / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


@dataclass
class SandboxTestHarness:
    sandbox: DockerSandbox
    start_container: AsyncMock
    execute_ephemeral: AsyncMock
    execute_persistent: AsyncMock
    cleanup: AsyncMock
    container: SimpleNamespace


@pytest.fixture
def sandbox_harness(tmp_path: Path) -> SandboxTestHarness:
    """Provision a DockerSandbox with filesystem isolation and mocked Docker ops."""

    project_root = tmp_path / "project"
    (project_root / "src" / "local_coding_assistant" / "sandbox" / "guest").mkdir(
        parents=True, exist_ok=True
    )
    (project_root / "tools" / "sandbox_tools").mkdir(parents=True, exist_ok=True)

    allowed_imports = ["json", "math", "tools_api", "datetime"]
    security_manager = SecurityManager(allowed_imports=allowed_imports)
    config = SandboxConfig(enabled=True, allowed_imports=allowed_imports)

    sandbox = DockerSandbox(
        security_manager=security_manager,
        path_manager=StubPathManager(project_root),
        config=config,
        auto_build=False,
    )

    container = SimpleNamespace(id="locca-container", status="running")
    start_container = AsyncMock(return_value=container)
    execute_ephemeral = AsyncMock(return_value=SandboxExecutionResponse(success=True))
    execute_persistent = AsyncMock(return_value=SandboxExecutionResponse(success=True))
    cleanup = AsyncMock()

    sandbox.start_container = start_container  # type: ignore[assignment]
    sandbox._execute_ephemeral_request = execute_ephemeral  # type: ignore[attr-defined]
    sandbox._execute_persistent_request = execute_persistent  # type: ignore[attr-defined]
    sandbox._cleanup_container = cleanup  # type: ignore[attr-defined]
    sandbox._get_container_stats = AsyncMock(return_value=None)  # type: ignore[attr-defined]

    return SandboxTestHarness(
        sandbox=sandbox,
        start_container=start_container,
        execute_ephemeral=execute_ephemeral,
        execute_persistent=execute_persistent,
        cleanup=cleanup,
        container=container,
    )


@pytest.mark.asyncio
async def test_ptc_import_enforcement_blocks_disallowed_import(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """os import should be blocked before any container work occurs."""

    request = SandboxExecutionRequest(
        code="import os\nprint('hi')",
        session_id="default",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert response.error and "Import not allowed" in response.error
    sandbox_harness.start_container.assert_not_called()
    sandbox_harness.execute_ephemeral.assert_not_called()


@pytest.mark.asyncio
async def test_api_usage_enforcement_allows_tools_api_calls(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """tools_api imports should pass security checks and reach execution."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=True,
        stdout="3\n",
        result=3,
    )

    request = SandboxExecutionRequest(
        code="from tools_api import sum_tool\nprint(sum_tool(1, 2))",
        session_id="ptc-happy-path",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is True
    assert "3" in response.stdout
    sandbox_harness.start_container.assert_awaited_once()
    sandbox_harness.execute_ephemeral.assert_awaited_once()
    sandbox_harness.cleanup.assert_awaited_once()
    assert sandbox_harness.cleanup.await_args.kwargs == {"remove": True}


@pytest.mark.asyncio
async def test_forbidden_os_access_blocked_by_patterns(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Direct filesystem access such as open() should be rejected."""

    request = SandboxExecutionRequest(
        code='from tools_api import sum_tool\nopen("/etc/passwd").read()',
        session_id="security",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert response.error and "blocked pattern" in response.error
    sandbox_harness.start_container.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_response_bubbles_up_from_container(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Container-level timeouts should be surfaced in the sandbox response."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=False,
        error="Execution timed out",
        stderr="Timed out waiting for response from sandbox",
        return_code=124,
    )

    request = SandboxExecutionRequest(
        code="while True:\n    pass",
        session_id="timeout",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert "timed out" in (response.error or "").lower()
    assert response.return_code == 124


@pytest.mark.asyncio
async def test_final_answer_detection_preserved(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Final answers returned from the sandbox should remain attached to the response."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=True,
        stdout="done\n",
        final_answer={"answer": "All done", "format": "text"},
    )

    request = SandboxExecutionRequest(
        code="from tools_api import final_answer\nfinal_answer('All done')",
        session_id="final-answer",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is True
    assert response.final_answer == {"answer": "All done", "format": "text"}


@pytest.mark.asyncio
async def test_persistent_execution_delegates_to_ipc_flow(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Persistent sessions should use the persistent IPC execution path."""

    persistent_resp = SandboxExecutionResponse(success=True, stdout="persisted")
    sandbox_harness.execute_persistent.return_value = persistent_resp

    request = SandboxExecutionRequest(
        code="print('persist me')",
        session_id="persist",
        persistence=True,
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is True
    sandbox_harness.start_container.assert_awaited_once()
    assert sandbox_harness.start_container.await_args.args[1] is True
    sandbox_harness.execute_persistent.assert_awaited_once()
    sandbox_harness.execute_ephemeral.assert_not_called()


@pytest.mark.asyncio
async def test_files_metadata_passthrough(sandbox_harness: SandboxTestHarness) -> None:
    """Files created/modified inside the sandbox should be exposed to callers."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=True,
        files_created=["/workspace/new.txt"],
        files_modified=["/workspace/existing.md"],
        stdout="done",
    )

    request = SandboxExecutionRequest(
        code="print('touch files')",
        session_id="files",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.files_created == ["/workspace/new.txt"]
    assert response.files_modified == ["/workspace/existing.md"]


@pytest.mark.asyncio
async def test_container_start_failure_reports_error(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Startup failures should be converted into structured sandbox errors."""

    sandbox_harness.start_container.side_effect = SandboxRuntimeError(
        "Docker is not running"
    )

    request = SandboxExecutionRequest(
        code="print('hello')",
        session_id="broken-docker",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert "Docker is not running" in (response.error or "")
    sandbox_harness.execute_ephemeral.assert_not_called()


@pytest.mark.asyncio
async def test_execute_shell_blocks_disallowed_commands(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Shell execution should reject blocked commands immediately."""

    response = await sandbox_harness.sandbox.execute_shell("rm -rf /tmp")

    assert response.success is False
    assert response.error and "not allowed" in response.error
    sandbox_harness.start_container.assert_not_called()


@pytest.mark.asyncio
async def test_infinite_loop_timeout_is_reported_and_cleaned_up(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """An infinite loop should result in a timeout error without leaking containers."""

    sandbox_harness.execute_ephemeral.side_effect = SandboxTimeoutError(
        "Execution timed out", return_code=124
    )

    request = SandboxExecutionRequest(
        code="while True:\n    pass",
        session_id="infinite-loop",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert "timed out" in (response.error or "").lower()
    sandbox_harness.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_large_memory_allocation_error_is_propagated(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Large memory allocations should surface the container OOM message."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=False,
        error="Memory limit exceeded (137)",
        stderr="Killed",
        return_code=137,
    )

    request = SandboxExecutionRequest(
        code="data = b'0' * (1024 ** 3)\nprint(len(data))",
        session_id="oom",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert "memory" in (response.error or "").lower()
    sandbox_harness.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_forbidden_imports_fail_fast_without_container(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Forbidden imports should be rejected before any Docker interaction occurs."""

    request = SandboxExecutionRequest(
        code="import pathlib\nprint(pathlib.Path('.'))",
        session_id="forbidden-import",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is False
    assert response.error and "import not allowed" in response.error.lower()
    sandbox_harness.start_container.assert_not_called()
    sandbox_harness.execute_ephemeral.assert_not_called()


@pytest.mark.asyncio
async def test_valid_tools_api_usage_executes_successfully(
    sandbox_harness: SandboxTestHarness,
) -> None:
    """Valid tool usage should pass through and produce stdout from the sandbox."""

    sandbox_harness.execute_ephemeral.return_value = SandboxExecutionResponse(
        success=True,
        stdout="12\n",
        result=12,
    )

    request = SandboxExecutionRequest(
        code=(
            "from tools_api import math\n"
            "result = math(operation='add', numbers=[7, 5])\n"
            "print(result)"
        ),
        session_id="valid-tool",
    )

    response = await sandbox_harness.sandbox.execute(request)

    assert response.success is True
    assert response.stdout.strip().endswith("12")
    sandbox_harness.start_container.assert_awaited_once()
    sandbox_harness.execute_ephemeral.assert_awaited_once()
    sandbox_harness.cleanup.assert_awaited_once()
