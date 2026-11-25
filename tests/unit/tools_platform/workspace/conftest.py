"""Shared fixtures for workspace tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory."""
    return tmp_path


def _make_config_manager(tmp_path: Path) -> ConfigManager:
    """Build a ConfigManager rooted at tmp_path (config-driven)."""
    cm = ConfigManager(config_paths=["config/tool_policies.yaml"])
    cfg = cm.load_global_config()
    cfg.tools_platform.workspace_root = str(tmp_path)
    # Store snapshots outside the workspace tree (hermetic, under pytest temp).
    cfg.tools_platform.snapshots.directory = str(
        tmp_path.parent / "locca_workspace_snapshots"
    )
    return cm


@pytest.fixture
def config_manager(temp_workspace: Path) -> ConfigManager:
    return _make_config_manager(temp_workspace)


@pytest.fixture
def workspace_manager(config_manager: ConfigManager) -> WorkspaceManager:
    return WorkspaceManager.from_config_manager(config_manager)


@pytest.fixture
def path_policy(temp_workspace: Path) -> PathPolicy:
    return PathPolicy(
        workspace_root=temp_workspace,
        allowed_patterns=["**"],
        denied_patterns=[],
        allow_symlinks=False,
        max_depth=None,
    )


@pytest.fixture
def workspace_manager_direct(
    temp_workspace: Path, path_policy: PathPolicy
) -> WorkspaceManager:
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=temp_workspace,
        path_policy=path_policy,
        config_manager=None,
    )
