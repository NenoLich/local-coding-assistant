"""Integration tests for the sandbox manager wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from local_coding_assistant.sandbox.manager import SandboxManager


def test_sandbox_manager_builds_docker_sandbox_from_config(sandbox_config_manager) -> None:
    """Ensure the manager instantiates DockerSandbox using config values."""

    manager = SandboxManager(sandbox_config_manager)
    sandbox = manager.get_sandbox()

    config = sandbox_config_manager.global_config.sandbox

    assert sandbox.image == config.image
    assert sandbox.mem_limit == config.memory_limit
    assert sandbox.cpu_quota == int(config.cpu_limit * 100000)
    assert sandbox.network_enabled is config.network_enabled
    assert sandbox.max_sessions == config.max_sessions

    assert manager.security_manager is not None
    assert manager.security_manager.allowed_imports == set(config.allowed_imports)
    assert manager.security_manager.blocked_patterns == config.blocked_patterns
    assert manager.security_manager.blocked_shell_commands == config.blocked_shell_commands


@pytest.mark.asyncio
async def test_sandbox_manager_start_and_stop_reuse_single_instance(sandbox_config_manager) -> None:
    """Verify start/stop delegates to the cached sandbox instance."""

    manager = SandboxManager(sandbox_config_manager)
    sandbox = manager.get_sandbox()
    assert sandbox is manager.get_sandbox()

    sandbox.start = AsyncMock()
    sandbox.stop = AsyncMock()

    await manager.start()
    sandbox.start.assert_awaited_once()

    await manager.stop()
    sandbox.stop.assert_awaited_once()


def test_sandbox_manager_workspace_directory_is_materialized(sandbox_config_manager) -> None:
    """Workspace directories should be created beneath the configured project root."""

    manager = SandboxManager(sandbox_config_manager)

    workspace_dir = manager.get_workspace_dir()

    assert workspace_dir.exists()
    assert workspace_dir.name == ".sandbox_workspace"
    assert workspace_dir.parent == sandbox_config_manager.project_root
