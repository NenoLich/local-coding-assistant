"""Shared fixtures for tools platform integration tests.

Fixtures here are available to every test module in
``tests/integration/tools_platform/``. These provide the real
WorkspaceManager, HostBackend, and config manager needed for integration
testing of conflict detection and other workspace operations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.approval import ApprovalService
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.scheduler import Scheduler
from local_coding_assistant.tools_platform.tools.register import register_all_tools
from local_coding_assistant.tools_platform.workspace import (
    PathPolicy,
    WorkspaceManager,
)


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace with sample files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.py").write_text("assert True\n")
    (tmp_path / "README.md").write_text("# Project\n")
    (tmp_path / ".env").write_text("SECRET=123\n")
    (tmp_path / ".gitignore").write_text("*.secret\n.env*\n")
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
    """WorkspaceManager with permissive path policy."""
    return WorkspaceManager(
        workspace_id="test-ws-integration",
        workspace_root=temp_workspace,
        path_policy=path_policy,
    )


@pytest.fixture
def config_manager(temp_workspace: Path) -> ConfigManager:
    """ConfigManager with test configuration."""
    cm = ConfigManager(config_paths=["config/tool_policies.yaml"])
    cfg = cm.load_global_config()
    cfg.tools_platform.workspace_root = str(temp_workspace)
    cfg.tools_platform.path_policy.allowed_patterns = ["**"]
    cfg.tools_platform.path_policy.denied_patterns = []
    cfg.tools_platform.path_policy.allow_symlinks = False
    cfg.tools_platform.path_policy.max_depth = None
    cfg.tools_platform.default_policy = "ask"
    cfg.tools_platform.concurrency.write_conflict_policy = "deny"
    cfg.tools_platform.snapshots.enabled = True
    cfg.tools_platform.snapshots.directory = str(
        temp_workspace.parent / "locca_workspace_snapshots"
    )
    return cm


@pytest.fixture
def host_backend(
    temp_workspace: Path,
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> HostBackend:
    """HostBackend wired to the temporary workspace with real config manager."""
    return HostBackend(
        workspace_root=temp_workspace,
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def registry() -> ToolRegistry:
    """ToolRegistry with all built-in tools registered."""
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry


@pytest.fixture
def approval_service(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> ApprovalService:
    """ApprovalService wired to the test workspace and real config rules."""
    return ApprovalService(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def scheduler(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> Scheduler:
    """Scheduler wired to the test workspace and real config."""
    return Scheduler(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def gateway(registry: ToolRegistry, config_manager: ConfigManager) -> ToolGateway:
    """ToolGateway with a real middleware pipeline and real components.

    Wires a real WorkspaceManager, HostBackend, ApprovalService and Scheduler
    from the shared config manager, per the acceptance criteria.
    """
    return ToolGateway(registry=registry, config_manager=config_manager)
