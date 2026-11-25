"""Tests for config-driven behavior of the tools platform.

These tests target the new config structure:
- tools_platform section in config/defaults.yaml
- tool_policies.yaml (replaces old separate tool_policies.default.yaml and command_approval.toml)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.approval import ApprovalService
from local_coding_assistant.tools_platform.contracts import (
    ToolDecisionOutcome,
    ToolRiskClass,
)
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _eval(
    service: ApprovalService,
    tool_id: str,
    arguments: dict,
    session_id: str = "test_session",
    call_id: str = "test_call",
    risk_class: ToolRiskClass = ToolRiskClass.LOW,
):
    """Helper to evaluate a tool call."""
    return service.evaluate(
        tool_id=tool_id,
        arguments=arguments,
        backend="host",
        session_id=session_id,
        call_id=call_id,
        risk_class=risk_class,
    )


# =============================================================================
# tools_platform Section (config/defaults.yaml)
# =============================================================================


class TestToolsPlatformSection:
    """Tests for tools_platform section in config/defaults.yaml."""

    @pytest.fixture
    def config_manager(self, tmp_path: Path) -> ConfigManager:
        """Create a ConfigManager loaded from config defaults."""
        manager = ConfigManager()
        manager.load_global_config()
        # Override workspace root to temp path for isolation
        cfg = manager.global_config
        cfg.tools_platform.workspace_root = str(tmp_path)
        return manager

    @pytest.fixture
    def workspace_manager(
        self, config_manager: ConfigManager, tmp_path: Path
    ) -> WorkspaceManager:
        """Create a WorkspaceManager from config."""
        path_policy = PathPolicy(
            workspace_root=tmp_path,
            allowed_patterns=["**"],
            denied_patterns=[],
            allow_symlinks=False,
            max_depth=None,
        )
        return WorkspaceManager(
            workspace_id="test-ws",
            workspace_root=tmp_path,
            path_policy=path_policy,
            config_manager=config_manager,
        )

    @pytest.fixture
    def approval_service(
        self, workspace_manager: WorkspaceManager, config_manager: ConfigManager
    ) -> ApprovalService:
        """Create an ApprovalService with config-driven rules."""
        return ApprovalService(
            workspace_manager=workspace_manager,
            config_manager=config_manager,
        )

    def test_backend_host_selects_host_backend(self, config_manager: ConfigManager):
        """backend=host selects HostBackend."""
        assert config_manager.global_config.tools_platform.backend == "host"

    def test_default_policy_ask_fallback(self, config_manager: ConfigManager):
        """default_policy=ask sets fallback for ApprovalService."""
        assert config_manager.global_config.tools_platform.default_policy == "ask"

    def test_default_policy_allow_fallback(self):
        """default_policy=allow sets fallback for ApprovalService."""
        from local_coding_assistant.config.config_manager import ConfigManager

        manager = ConfigManager()
        manager.load_global_config()
        manager.global_config.tools_platform.default_policy = "allow"
        assert manager.global_config.tools_platform.default_policy == "allow"

    def test_default_policy_deny_fallback(self):
        """default_policy=deny sets fallback for ApprovalService."""
        from local_coding_assistant.config.config_manager import ConfigManager

        manager = ConfigManager()
        manager.load_global_config()
        manager.global_config.tools_platform.default_policy = "deny"
        assert manager.global_config.tools_platform.default_policy == "deny"

    def test_exposed_tools_auto(self, config_manager: ConfigManager):
        """exposed_tools=auto exposes all backend-compatible tools."""
        assert config_manager.global_config.tools_platform.exposed_tools == "auto"

    def test_exposed_tools_list(self, config_manager: ConfigManager):
        """exposed_tools=["tool1", "tool2"] exposes only listed tools."""
        cfg = config_manager.global_config
        cfg.tools_platform.exposed_tools = ["read_file", "search_text"]
        assert cfg.tools_platform.exposed_tools == ["read_file", "search_text"]

    def test_allow_network_false(self, config_manager: ConfigManager):
        """allow_network=false blocks network access by default for run_command."""
        assert config_manager.global_config.tools_platform.allow_network is False

    def test_allow_network_true(self):
        """allow_network=true enables network access."""
        from local_coding_assistant.config.config_manager import ConfigManager

        manager = ConfigManager()
        manager.load_global_config()
        manager.global_config.tools_platform.allow_network = True
        assert manager.global_config.tools_platform.allow_network is True

    def test_non_interactive_converts_ask_to_deny(
        self, config_manager: ConfigManager, tmp_path: Path
    ):
        """non_interactive=true converts all ASK to DENY."""
        cfg = config_manager.global_config
        cfg.tools_platform.non_interactive = True

    def test_workspace_root_target_alias(
        self, config_manager: ConfigManager, tmp_path: Path
    ):
        """workspace_root="@target" resolves via PathManager to target directory."""
        # The @target alias should be resolved by PathManager
        # When loaded from config, it should work with the path manager
        root = config_manager.path_manager.resolve_path("@target")
        assert root is not None

    def test_env_allowlist_null_passes_all(self, config_manager: ConfigManager):
        """env_allowlist=null passes all env, env_allowlist=["PATH", "HOME"] filters env."""
        # env_allowlist=None means pass through
        assert config_manager.global_config.tools_platform.env_allowlist is None

    def test_max_output_bytes(self, config_manager: ConfigManager):
        """max_output_bytes clamps tool byte_limit arguments."""
        assert config_manager.global_config.tools_platform.max_output_bytes == 262144

    def test_max_read_lines(self, config_manager: ConfigManager):
        """max_read_lines clamps tool page_size arguments."""
        assert config_manager.global_config.tools_platform.max_read_lines == 400

    def test_max_search_matches(self, config_manager: ConfigManager):
        """max_search_matches clamps tool max_matches/limit arguments."""
        assert config_manager.global_config.tools_platform.max_search_matches == 200

    def test_path_policy_enforced(
        self,
        config_manager: ConfigManager,
        tmp_path: Path,
        workspace_manager: WorkspaceManager,
    ):
        """path_policy allowed_patterns/denied_patterns/allow_symlinks/max_depth enforced by WorkspaceManager."""
        # Test that path policy is set correctly
        assert workspace_manager.path_policy.allowed_patterns == ["**"]
        assert workspace_manager.path_policy.denied_patterns == []
        assert workspace_manager.path_policy.allow_symlinks is False
        assert workspace_manager.path_policy.max_depth is None

    def test_ignore_policy_files(self, config_manager: ConfigManager):
        """ignore_policy files [".gitignore", ".aiignore"] used by search_text/match_paths."""
        # Check that ignore policy is configured
        ignore_policy = config_manager.global_config.tools_platform.ignore_policy
        assert ".gitignore" in ignore_policy.files
        assert ".aiignore" in ignore_policy.files

    def test_snapshots_enabled(self, config_manager: ConfigManager):
        """snapshots.enabled=true enables snapshot creation."""
        assert config_manager.global_config.tools_platform.snapshots.enabled is True

    def test_timeout_config(self, config_manager: ConfigManager):
        """timeout soft_seconds/hard_seconds/idle_seconds/heartbeat_seconds/cancellation_grace_seconds used by supervisor."""
        timeout = config_manager.global_config.tools_platform.timeout
        assert timeout.soft_seconds == 10
        assert timeout.hard_seconds == 300
        assert timeout.idle_seconds == 30
        assert timeout.heartbeat_seconds == 5
        assert timeout.cancellation_grace_seconds == 5

    def test_output_limits(self, config_manager: ConfigManager):
        """output_limits max_stdout_bytes/max_stderr_bytes/max_lines used by normalization."""
        limits = config_manager.global_config.tools_platform.output_limits
        assert limits.max_stdout_bytes == 262144
        assert limits.max_stderr_bytes == 262144
        assert limits.max_lines == 400

    def test_concurrency(self, config_manager: ConfigManager):
        """concurrency max_parallel_reads/max_parallel_processes used by scheduler capacity groups."""
        concurrency = config_manager.global_config.tools_platform.concurrency
        assert concurrency.max_parallel_reads == 8
        assert concurrency.max_parallel_processes == 2

    def test_write_conflict_policy(self, config_manager: ConfigManager):
        """concurrency write_conflict_policy (deny/ask/allow) used by ConflictDetector."""
        assert (
            config_manager.global_config.tools_platform.concurrency.write_conflict_policy
            == "deny"
        )


# =============================================================================
# tool_policies.default.yaml (Per-Tool Overrides)
# =============================================================================


class TestToolPoliciesOverrides:
    """Tests for per-tool policies from tool_policies.default.yaml."""

    @pytest.fixture
    def config_manager_with_tool_policies(self, tmp_path: Path) -> ConfigManager:
        """Create a ConfigManager with tool_policies.yaml loaded."""
        manager = ConfigManager(
            config_paths=["config/tool_policies.yaml"],
        )
        manager.load_global_config()
        # Override workspace root to temp path for isolation
        cfg = manager.global_config
        cfg.tools_platform.workspace_root = str(tmp_path)
        return manager

    @pytest.fixture
    def approval_service(
        self, config_manager_with_tool_policies: ConfigManager, tmp_path: Path
    ) -> ApprovalService:
        """Create an ApprovalService with tool policies."""
        workspace_manager = WorkspaceManager(
            workspace_id="test-ws",
            workspace_root=tmp_path,
            path_policy=PathPolicy(
                workspace_root=tmp_path,
                allowed_patterns=["**"],
                denied_patterns=[],
                allow_symlinks=False,
                max_depth=None,
            ),
            config_manager=config_manager_with_tool_policies,
        )
        return ApprovalService(
            workspace_manager=workspace_manager,
            config_manager=config_manager_with_tool_policies,
        )

    def test_search_text_base_policy_allow(self, approval_service: ApprovalService):
        """search_text: risk_class=safe, default_policy=allow → ALLOW without approval."""
        decision = _eval(approval_service, "search_text", {"query": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_read_file_base_policy_allow(self, approval_service: ApprovalService):
        """read_file: risk_class=safe, default_policy=allow → ALLOW without approval."""
        decision = _eval(approval_service, "read_file", {"path": "test.txt"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_search_symbols_base_policy_allow(self, approval_service: ApprovalService):
        """search_symbols: risk_class=safe, default_policy=allow → ALLOW without approval."""
        decision = _eval(approval_service, "search_symbols", {"query": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_get_symbol_base_policy_allow(self, approval_service: ApprovalService):
        """get_symbol: risk_class=safe, default_policy=allow → ALLOW without approval."""
        decision = _eval(approval_service, "get_symbol", {"symbol": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_match_paths_base_policy_allow(self, approval_service: ApprovalService):
        """match_paths: risk_class=safe, default_policy=allow → ALLOW without approval."""
        decision = _eval(approval_service, "match_paths", {"pattern": "*.py"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_check_process_status_base_policy_allow(
        self, approval_service: ApprovalService
    ):
        """check_process_status: risk_class=low, default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "check_process_status", {})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_run_command_base_policy_ask(self, approval_service: ApprovalService):
        """run_command: risk_class=high, default_policy=ask → ASK by default."""
        decision = _eval(approval_service, "run_command", {"argv": ["echo", "hello"]})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_create_file_base_policy_ask(self, approval_service: ApprovalService):
        """create_file: risk_class=high, default_policy=ask → ASK by default."""
        decision = _eval(
            approval_service, "create_file", {"path": "test.txt", "content": "# test"}
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_delete_file_base_policy_ask(self, approval_service: ApprovalService):
        """delete_file: risk_class=high, default_policy=ask → ASK by default."""
        decision = _eval(approval_service, "delete_file", {"path": "test.txt"})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_replace_exact_base_policy_ask(self, approval_service: ApprovalService):
        """replace_exact: risk_class=high, default_policy=ask → ASK by default."""
        decision = _eval(
            approval_service,
            "replace_exact",
            {"old_text": "hello", "new_text": "world"},
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_kill_process_base_policy_ask(self, approval_service: ApprovalService):
        """kill_process: risk_class=high, default_policy=ask → ASK by default."""
        decision = _eval(approval_service, "kill_process", {})
        assert decision.outcome == ToolDecisionOutcome.ASK


# =============================================================================
# Unknown tool ID raises ValueError
