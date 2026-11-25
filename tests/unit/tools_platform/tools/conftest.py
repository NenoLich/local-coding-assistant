"""Shared fixtures for tools platform tool tests.

Fixtures here are available to every test module under
``tests/unit/tools_platform/tools/``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import (
    ProcessState,
    ToolOutputChunk,
)
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory used as the tool workspace root."""
    return tmp_path


@pytest.fixture
def path_policy(temp_workspace: Path) -> PathPolicy:
    """Permissive path policy allowing every path under the workspace."""
    return PathPolicy(
        workspace_root=temp_workspace,
        allowed_patterns=["**"],
        denied_patterns=[],
        allow_symlinks=False,
        max_depth=None,
    )


@pytest.fixture
def workspace_manager(
    temp_workspace: Path, path_policy: PathPolicy
) -> WorkspaceManager:
    """WorkspaceManager backed by the temporary workspace."""
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=temp_workspace,
        path_policy=path_policy,
        config_manager=None,
    )


@pytest.fixture
def config_manager() -> ConfigManager:
    """ConfigManager loaded from config defaults (config-driven test config)."""
    manager = ConfigManager()
    manager.load_global_config()
    return manager


@pytest.fixture
def backend(temp_workspace: Path, workspace_manager: WorkspaceManager) -> HostBackend:
    """HostBackend wired to the temporary workspace (no config manager)."""
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=None,
    )


@pytest.fixture
def backend_with_config(
    temp_workspace: Path,
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> HostBackend:
    """HostBackend wired to the temporary workspace with a real config manager.

    Honors the configured ignore_policy.files (``.gitignore`` and ``.aiignore``)
    rather than the ``.gitignore``-only default used when config_manager is None.
    """
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def command_backend(temp_workspace: Path):
    """Factory returning a mock HostBackend whose ``execute_command`` yields given chunks.

    Each chunk must be a ``ToolOutputChunk``. The factory is useful for deterministic
    tool-implementation tests that do not want to spawn real subprocesses.

    The returned mock records every ``execute_command`` call in ``_calls`` (a list of
    dicts with ``argv``, ``cwd``, ``env`` and ``timeout_seconds`` keys).
    """

    def factory(chunks: list[ToolOutputChunk]) -> MagicMock:
        backend = MagicMock(spec=HostBackend)
        backend._workspace_root = temp_workspace
        calls: list[dict[str, Any]] = []

        async def _execute_command(
            argv: list[str],
            cwd: Path | None = None,
            env: dict[str, str] | None = None,
            timeout_seconds: float | None = None,
        ) -> AsyncIterator[ToolOutputChunk]:
            calls.append(
                {
                    "argv": argv,
                    "cwd": cwd,
                    "env": env,
                    "timeout_seconds": timeout_seconds,
                }
            )
            for chunk in chunks:
                yield chunk

        backend.execute_command = _execute_command
        backend._calls = calls
        return backend

    return factory


def _default_process_status_data(handle_id: str = "handle-1") -> dict[str, Any]:
    """Build a default process status dict shaped like HostBackend.check_process_status."""
    return {
        "handle_id": handle_id,
        "state": ProcessState.RUNNING.value,
        "return_code": None,
        "exit_signal": None,
        "os_pid": 4242,
        "backend": "host",
        "start_time": "2026-10-22T20:00:00+00:00",
        "last_activity": None,
        "elapsed_seconds": 2.5,
        "idle_seconds": None,
        "resource_usage": {},
        "metadata": {},
    }


@pytest.fixture
def process_status_backend(temp_workspace: Path) -> MagicMock:
    """Factory returning a mock HostBackend whose ``check_process_status`` returns given status data.

    Each call returns a fresh mock backend. Pass ``status_data`` for a successful
    status response or ``error`` to simulate a backend failure (e.g. unknown handle).
    """

    def factory(
        status_data: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> MagicMock:
        backend = MagicMock(spec=HostBackend)
        backend._workspace_root = temp_workspace
        backend._status_data = status_data or _default_process_status_data()
        backend._error = error
        calls: list[str] = []

        async def _check_process_status(handle_id: str) -> dict[str, Any]:
            calls.append(handle_id)
            if backend._error is not None:
                raise backend._error
            return dict(backend._status_data)

        backend.check_process_status = _check_process_status
        backend._calls = calls
        return backend

    return factory


@pytest.fixture
def kill_process_backend(temp_workspace: Path) -> MagicMock:
    """Factory returning a mock HostBackend whose ``check_process_status`` and
    ``cancel_execution`` are controllable.

    Pass ``status_data`` for a successful status response, ``status_error`` to
    simulate a backend failure (e.g. unknown handle), and ``cancel_error`` to
    simulate a cancellation failure. The returned mock records every
    ``check_process_status`` handle and ``cancel_execution`` execution_id.
    """

    def factory(
        status_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        status_error: Exception | None = None,
        cancel_error: Exception | None = None,
    ) -> MagicMock:
        backend = MagicMock(spec=HostBackend)
        backend._workspace_root = temp_workspace
        backend._status_error = status_error
        backend._cancel_error = cancel_error

        # Normalize status_data into a sequence returned across calls.
        if status_data is None:
            status_sequence = [_default_process_status_data()]
        elif isinstance(status_data, list):
            status_sequence = list(status_data)
        else:
            status_sequence = [status_data]
        backend._status_sequence = status_sequence
        backend._status_index = 0

        status_calls: list[str] = []
        cancel_calls: list[str] = []

        async def _check_process_status(handle_id: str) -> dict[str, Any]:
            status_calls.append(handle_id)
            if backend._status_error is not None:
                raise backend._status_error
            entry = backend._status_sequence[
                min(backend._status_index, len(backend._status_sequence) - 1)
            ]
            backend._status_index += 1
            return dict(entry)

        async def _cancel_execution(execution_id: str) -> None:
            cancel_calls.append(execution_id)
            if backend._cancel_error is not None:
                raise backend._cancel_error

        backend.check_process_status = _check_process_status
        backend.cancel_execution = _cancel_execution
        backend._status_calls = status_calls
        backend._cancel_calls = cancel_calls
        return backend

    return factory
