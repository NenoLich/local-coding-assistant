"""Shared fixtures for tools platform tests.

Fixtures here are available to every test module directly under
``tests/unit/tools_platform/``. Subdirectories (``tools/``, ``workspace/``)
define their own closer-scoped fixtures in their conftest.py files, which take
precedence for tests living in those directories.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.repository.service import RepositoryContextService
from local_coding_assistant.tools_platform.backends.host import HostBackend
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
def config_manager_env_allowlist() -> ConfigManager:
    """ConfigManager with a restrictive env_allowlist (only PATH reaches commands)."""
    manager = ConfigManager()
    manager.load_global_config()
    manager.global_config.tools_platform.env_allowlist = ["PATH"]
    return manager


@pytest.fixture
def repo_service() -> MagicMock:
    """Mocked RepositoryContextService for symbol search delegation tests."""
    return MagicMock(spec=RepositoryContextService)


@pytest.fixture
def filtered_backend(
    temp_workspace: Path,
    workspace_manager: WorkspaceManager,
    config_manager_env_allowlist: ConfigManager,
) -> HostBackend:
    """HostBackend whose env is filtered through ``env_allowlist=[\"PATH\"]``."""
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=config_manager_env_allowlist,
    )


@pytest.fixture
def host_backend(
    temp_workspace: Path,
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> HostBackend:
    """HostBackend wired to the temporary workspace with a real config manager.

    Honors the configured timeout, output_limits, env_allowlist and
    ignore_policy from config/defaults.yaml.
    """
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )
