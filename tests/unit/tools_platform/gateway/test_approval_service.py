"""Tests for ApprovalService - comprehensive coverage of approval logic."""

import os
from pathlib import Path

import pytest
import yaml

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.approval import (
    ApprovalService,
    PolicySource,
)
from local_coding_assistant.tools_platform.contracts import (
    ApprovalResponse,
    ToolDecisionOutcome,
    ToolRiskClass,
)
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TOOL_POLICIES_PATH = PROJECT_ROOT / "config" / "tool_policies.yaml"


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    return tmp_path


@pytest.fixture
def workspace_manager(temp_workspace: Path) -> WorkspaceManager:
    """Create a workspace manager for testing."""
    path_policy = PathPolicy(
        workspace_root=temp_workspace,
        allowed_patterns=["**"],
        denied_patterns=[],
        max_depth=None,
        allow_symlinks=False,
    )
    return WorkspaceManager(
        workspace_id="test_workspace",
        workspace_root=temp_workspace,
        path_policy=path_policy,
    )


@pytest.fixture
def config_manager(temp_workspace: Path) -> ConfigManager:
    """Create a config manager with test tool policies."""
    import local_coding_assistant.config.env_manager as env_manager_module
    from local_coding_assistant.config.env_manager import EnvManager

    saved_module_instance = env_manager_module._instance
    saved_class_instance = EnvManager._instance
    saved_locca_env = os.environ.get("LOCCA_ENV")

    os.environ["LOCCA_ENV"] = "development"
    env_manager_module._instance = None
    EnvManager._instance = None
    env_manager = EnvManager()

    # Use the actual tool_policies.yaml for realistic testing, but redirect
    # project overrides to a non-existent temp path so tests exercise only the
    # default policies. tool_policies.local.yaml is volatile (user-specific)
    # and must not influence test outcomes.
    try:
        redirect_path = temp_workspace / "tool_policies.local.ignore.yaml"
        redirect_path.write_text(
            "tools_platform:\n  tool_policies_local_path: "
            + str(temp_workspace / "tool_policies.local.yaml")
            + "\n"
        )
        cm = ConfigManager(
            config_paths=[str(TOOL_POLICIES_PATH), str(redirect_path)],
            env_manager=env_manager,
        )
        cfg = cm.load_global_config()
        # Override workspace root to point to temp workspace
        cfg.tools_platform.workspace_root = str(temp_workspace)
        # Set tool policies paths to temp directory for test isolation
        cfg.tools_platform.tool_policies_local_path = str(
            temp_workspace / "tool_policies.local.yaml"
        )
        return cm
    finally:
        env_manager_module._instance = saved_module_instance
        EnvManager._instance = saved_class_instance
        if saved_locca_env is None:
            os.environ.pop("LOCCA_ENV", None)
        else:
            os.environ["LOCCA_ENV"] = saved_locca_env


@pytest.fixture
def approval_service(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> ApprovalService:
    """Create an approval service with real config."""
    return ApprovalService(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def non_interactive_approval_service(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> ApprovalService:
    """Create an approval service in non-interactive mode."""
    # Modify the config manager's global config
    config_manager.global_config.tools_platform.non_interactive = True
    return ApprovalService(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


@pytest.fixture
def auto_approve_approval_service(
    workspace_manager: WorkspaceManager,
    config_manager: ConfigManager,
) -> ApprovalService:
    """Create an approval service with auto_approve mode enabled."""
    config_manager.global_config.tools_platform.auto_approve = True
    return ApprovalService(
        workspace_manager=workspace_manager,
        config_manager=config_manager,
    )


# =============================================================================
# Helper Functions
# =============================================================================


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


def _confirm(
    service: ApprovalService,
    token: str,
    session_id: str = "test_session",
    response: ApprovalResponse = ApprovalResponse.YES_ONCE,
):
    """Helper to confirm a tool call."""
    return service.confirm(token, session_id, response)


def _local_file(config_manager: ConfigManager) -> Path:
    """Get the local tool policies file path using the config manager's path manager."""
    # Get the local path from config
    local_path_str = (
        config_manager.global_config.tools_platform.tool_policies_local_path
    )
    return Path(config_manager.path_manager.resolve_path(local_path_str))


def _write_local_yaml(config_manager: ConfigManager, data: dict):
    """Write data to local tool policies file."""
    local_file = _local_file(config_manager)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    with open(local_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _read_local_yaml(config_manager: ConfigManager) -> dict:
    """Read data from local tool policies file."""
    local_file = _local_file(config_manager)
    if not local_file.exists():
        return {}
    with open(local_file) as f:
        return yaml.safe_load(f) or {}


def _cleanup_local_yaml(config_manager: ConfigManager):
    """Clean up local tool policies file."""
    local_file = _local_file(config_manager)
    if local_file.exists():
        local_file.unlink()


# =============================================================================
# Base Policy Tests
# =============================================================================


class TestBasePolicies:
    """Tests for base per-tool default policies."""

    def test_read_file_base_policy_allow(self, approval_service: ApprovalService):
        """read_file has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "read_file", {"path": "test.txt"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_search_text_base_policy_allow(self, approval_service: ApprovalService):
        """search_text has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "search_text", {"query": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_search_symbols_base_policy_allow(self, approval_service: ApprovalService):
        """search_symbols has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "search_symbols", {"query": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_get_symbol_base_policy_allow(self, approval_service: ApprovalService):
        """get_symbol has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "get_symbol", {"symbol": "test"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_match_paths_base_policy_allow(self, approval_service: ApprovalService):
        """match_paths has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "match_paths", {"pattern": "*.py"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_check_process_status_base_policy_allow(
        self, approval_service: ApprovalService
    ):
        """check_process_status has default_policy=allow → ALLOW."""
        decision = _eval(approval_service, "check_process_status", {"handle_id": "123"})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_run_command_base_policy_ask(self, approval_service: ApprovalService):
        """run_command has default_policy=ask → ASK."""
        decision = _eval(approval_service, "run_command", {"argv": ["unknown_cmd"]})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_write_file_base_policy_ask(self, approval_service: ApprovalService):
        """write_file has default_policy=ask → ASK."""
        decision = _eval(
            approval_service, "write_file", {"path": "test.txt", "content": "x"}
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_create_file_base_policy_ask(self, approval_service: ApprovalService):
        """create_file has default_policy=ask → ASK."""
        decision = _eval(
            approval_service, "create_file", {"path": "test.txt", "content": "x"}
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_delete_file_base_policy_ask(self, approval_service: ApprovalService):
        """delete_file has default_policy=ask → ASK."""
        decision = _eval(approval_service, "delete_file", {"path": "test.txt"})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_replace_exact_base_policy_ask(self, approval_service: ApprovalService):
        """replace_exact has default_policy=ask → ASK."""
        decision = _eval(
            approval_service,
            "replace_exact",
            {"path": "test.txt", "old_text": "a", "new_text": "b"},
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_kill_process_base_policy_ask(self, approval_service: ApprovalService):
        """kill_process has default_policy=ask → ASK."""
        decision = _eval(approval_service, "kill_process", {"handle_id": "123"})
        assert decision.outcome == ToolDecisionOutcome.ASK


# =============================================================================
# Command Pattern Matching Tests
# =============================================================================


class TestCommandPatterns:
    """Tests for command pattern matching in run_command."""

    def test_allow_git_diff(self, approval_service: ApprovalService):
        """git diff* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["git", "diff"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW
        assert "Allow git diff" in decision.reason

    def test_allow_git_status(self, approval_service: ApprovalService):
        """git status matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["git", "status"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_git_log(self, approval_service: ApprovalService):
        """git log* matches allow pattern → ALLOW."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["git", "log", "--oneline"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_cargo_check(self, approval_service: ApprovalService):
        """cargo check matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["cargo", "check"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_cargo_test(self, approval_service: ApprovalService):
        """cargo test* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["cargo", "test"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_pytest(self, approval_service: ApprovalService):
        """pytest* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["pytest", "-v"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_ruff_check(self, approval_service: ApprovalService):
        """ruff check* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["ruff", "check"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_ls(self, approval_service: ApprovalService):
        """ls* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["ls", "-la"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_allow_cat(self, approval_service: ApprovalService):
        """cat* matches allow pattern → ALLOW."""
        decision = _eval(approval_service, "run_command", {"argv": ["cat", "file.txt"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_ask_git_commit(self, approval_service: ApprovalService):
        """git commit* matches ask pattern → ASK."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["git", "commit", "-m", "msg"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ASK
        assert "Ask for git commit" in decision.reason

    def test_ask_git_push(self, approval_service: ApprovalService):
        """git push* matches ask pattern → ASK."""
        decision = _eval(approval_service, "run_command", {"argv": ["git", "push"]})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_ask_rm(self, approval_service: ApprovalService):
        """rm * matches ask pattern → ASK."""
        decision = _eval(approval_service, "run_command", {"argv": ["rm", "file.txt"]})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_ask_python(self, approval_service: ApprovalService):
        """python* matches ask pattern → ASK."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["python", "script.py"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_deny_sudo(self, approval_service: ApprovalService):
        """sudo * matches deny pattern → DENY."""
        decision = _eval(approval_service, "run_command", {"argv": ["sudo", "reboot"]})
        assert decision.outcome == ToolDecisionOutcome.DENY
        assert "Deny sudo" in decision.reason

    def test_deny_apt(self, approval_service: ApprovalService):
        """apt* matches deny pattern → DENY."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["apt", "install", "pkg"]}
        )
        assert decision.outcome == ToolDecisionOutcome.DENY

    def test_deny_fork_bomb(self, approval_service: ApprovalService):
        """fork bomb matches deny pattern → DENY."""
        decision = _eval(approval_service, "run_command", {"argv": [":(){ :|:& };:"]})
        assert decision.outcome == ToolDecisionOutcome.DENY

    def test_deny_id_rsa(self, approval_service: ApprovalService):
        """*id_rsa* matches deny pattern → DENY."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["cat", "~/.ssh/id_rsa"]}
        )
        assert decision.outcome == ToolDecisionOutcome.DENY


# =============================================================================
# Workspace Boundary Guard Tests
# =============================================================================


class TestWorkspaceBoundaryGuard:
    """Tests for workspace boundary guard rules."""

    def test_read_file_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """read_file outside workspace → ASK with confirmation_token."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_read_outside"
        assert "outside" in decision.reason.lower()
        assert decision.confirmation_token is not None

    def test_read_file_outside_workspace_non_interactive_deny(
        self, non_interactive_approval_service: ApprovalService, temp_workspace: Path
    ):
        """read_file outside workspace in non-interactive → DENY."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(
            non_interactive_approval_service, "read_file", {"path": str(outside_file)}
        )

        assert decision.outcome == ToolDecisionOutcome.DENY
        assert decision.policy_rule == "boundary_read_outside"

    def test_search_text_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """search_text outside workspace → ASK."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(
            approval_service,
            "search_text",
            {"query": "test", "path": str(outside_file)},
        )

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_read_outside"

    def test_match_paths_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """match_paths outside workspace → ASK."""
        outside_dir = temp_workspace.parent / "outside_dir"

        decision = _eval(
            approval_service, "match_paths", {"pattern": str(outside_dir / "*.txt")}
        )

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_read_outside"

    def test_create_file_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """create_file outside workspace → ask."""
        outside_file = temp_workspace.parent / "new_file.txt"

        decision = _eval(
            approval_service, "create_file", {"path": str(outside_file), "content": "x"}
        )

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_write_outside"

    def test_replace_exact_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """replace_exact outside workspace → ask."""
        outside_file = temp_workspace.parent / "config.txt"
        outside_file.write_text("old")

        decision = _eval(
            approval_service,
            "replace_exact",
            {"path": str(outside_file), "old_text": "old", "new_text": "new"},
        )

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_write_outside"

    def test_delete_file_outside_workspace_ask(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """delete_file outside workspace → ask."""
        outside_file = temp_workspace.parent / "file.txt"
        outside_file.write_text("x")

        decision = _eval(approval_service, "delete_file", {"path": str(outside_file)})

        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_write_outside"

    def test_read_file_inside_workspace_allow(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """read_file inside workspace → ALLOW (base policy)."""
        test_file = temp_workspace / "src" / "app.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("print('hello')")

        decision = _eval(approval_service, "read_file", {"path": str(test_file)})

        assert decision.outcome == ToolDecisionOutcome.ALLOW
        assert decision.policy_rule != "boundary_read_outside"


# =============================================================================
# Heuristic Rules Tests
# =============================================================================


class TestHeuristicRules:
    """Tests for heuristic rules (shell_mode, network)."""

    def test_shell_mode_ask(self, approval_service: ApprovalService):
        """run_command with shell_mode=true → ASK."""
        decision = _eval(
            approval_service,
            "run_command",
            {"argv": ["bash", "-c", "echo hi"], "shell_mode": True},
        )
        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "heuristic_shell_mode"
        assert "shell mode" in decision.reason.lower()
        assert decision.confirmation_token is not None

    def test_shell_mode_non_interactive_deny(
        self, non_interactive_approval_service: ApprovalService
    ):
        """run_command with shell_mode=true in non-interactive → DENY."""
        decision = _eval(
            non_interactive_approval_service,
            "run_command",
            {"argv": ["bash", "-c", "echo hi"], "shell_mode": True},
        )
        assert decision.outcome == ToolDecisionOutcome.DENY
        assert decision.policy_rule == "heuristic_shell_mode"
        assert "non-interactive" in decision.reason.lower()

    def test_allow_network_ask(self, approval_service: ApprovalService):
        """run_command with allow_network=true → ASK."""
        decision = _eval(
            approval_service,
            "run_command",
            {"argv": ["curl", "example.com"], "allow_network": True},
        )
        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "heuristic_network"
        assert "network access" in decision.reason.lower()
        assert decision.confirmation_token is not None

    def test_allow_network_non_interactive_deny(
        self, non_interactive_approval_service: ApprovalService
    ):
        """run_command with allow_network=true in non-interactive → DENY."""
        decision = _eval(
            non_interactive_approval_service,
            "run_command",
            {"argv": ["curl", "example.com"], "allow_network": True},
        )
        assert decision.outcome == ToolDecisionOutcome.DENY
        assert decision.policy_rule == "heuristic_network"


# =============================================================================
# Default Fallback Tests
# =============================================================================


class TestDefaultFallback:
    """Tests for global fallback rule."""

    def test_unknown_tool_fallback_ask(self, approval_service: ApprovalService):
        """Unknown tool falls back to default_policy=ask → ASK."""
        decision = _eval(approval_service, "unknown_tool", {"arg": "value"})
        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "fallback"

    def test_non_interactive_fallback_deny(
        self, non_interactive_approval_service: ApprovalService
    ):
        """Unknown tool in non-interactive with default_policy=ask → DENY."""
        decision = _eval(
            non_interactive_approval_service, "unknown_tool", {"arg": "value"}
        )
        assert decision.outcome == ToolDecisionOutcome.DENY
        assert decision.policy_rule == "fallback"
        assert "non-interactive" in decision.reason.lower()

    def test_auto_approve_fallback_allow(
        self, auto_approve_approval_service: ApprovalService
    ):
        """Unknown tool with auto_approve=True → ALLOW via pre-guard."""
        decision = _eval(
            auto_approve_approval_service, "unknown_tool", {"arg": "value"}
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW
        assert decision.policy_rule == "auto_approve"

    def test_auto_approve_bypasses_specific_ask_rules(
        self, auto_approve_approval_service: ApprovalService
    ):
        """Auto-approve bypasses ALL rules including specific ASK rules (e.g., write_file)."""
        # create_file has a specific ASK rule with higher priority than fallback
        decision = _eval(
            auto_approve_approval_service,
            "create_file",
            {"path": "test.txt", "content": "x"},
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW
        assert decision.policy_rule == "auto_approve"


# =============================================================================
# Confirmation Token Tests
# =============================================================================


class TestConfirmationTokens:
    """Tests for confirmation token handling."""

    def test_confirm_valid_token_yes_once(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """confirm(valid_token, YES_ONCE) → True, marks token used."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token
        assert token is not None

        result = _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ONCE
        )
        assert result is True

        # Token should be marked used
        token_obj = approval_service._tokens[token]
        assert token_obj.used is True
        assert token_obj.approval_response == ApprovalResponse.YES_ONCE

    def test_confirm_expired_token_returns_false(
        self, approval_service: ApprovalService
    ):
        """confirm(expired_token) → False."""
        from datetime import UTC, datetime, timedelta

        token = "expired_token"
        # Create an expired token manually
        from local_coding_assistant.tools_platform.approval import ConfirmationToken

        approval_service._tokens[token] = ConfirmationToken(
            token=token,
            call_id="call_1",
            session_id="test_session",
            rule_id="test_rule",
            created_at=datetime.now(UTC) - timedelta(minutes=10),
            expires_at=datetime.now(UTC) - timedelta(minutes=5),
        )

        result = _confirm(approval_service, token, "test_session")
        assert result is False

    def test_confirm_wrong_session_returns_false(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """confirm(wrong_session) → False."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        result = _confirm(approval_service, token, "wrong_session")
        assert result is False

    def test_confirm_already_used_token_returns_false(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """confirm(already_used_token) → False."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        _confirm(approval_service, token, "test_session", ApprovalResponse.YES_ONCE)
        result = _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ONCE
        )
        assert result is False

    def test_confirm_yes_always_session_adds_override(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """confirm(YES_ALWAYS_SESSION) → modifies rule in-place to ALLOW."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        result = _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ALWAYS_SESSION
        )
        assert result is True

        # Rule should be modified in-place
        rule = next(
            r for r in approval_service._rules if r.id == "boundary_read_outside"
        )
        assert rule.outcome == ToolDecisionOutcome.ALLOW
        assert rule.source == PolicySource.USER_APPROVAL

        # Subsequent call should ALLOW
        decision2 = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision2.outcome == ToolDecisionOutcome.ALLOW

    def test_confirm_yes_always_project_persists_override(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """confirm(YES_ALWAYS_PROJECT) → modifies rule + persists to local.yaml."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        result = _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ALWAYS_PROJECT
        )
        assert result is True

        # Rule should be modified in-place
        rule = next(
            r for r in approval_service._rules if r.id == "boundary_read_outside"
        )
        assert rule.outcome == ToolDecisionOutcome.ALLOW
        assert rule.source == PolicySource.USER_APPROVAL

        # local.yaml should have the override
        local_file = _local_file(approval_service._config_manager)
        assert local_file.exists()
        with open(local_file) as f:
            data = yaml.safe_load(f)
        overrides = data.get("tools_platform", {}).get("project_overrides", [])
        assert len(overrides) == 1
        assert overrides[0]["base_rule_id"] == "boundary_read_outside"
        assert overrides[0]["outcome"] == "allow"

        _cleanup_local_yaml(approval_service._config_manager)


# =============================================================================
# Config Merge Tests (Base + Local Overrides)
# =============================================================================


def _create_temp_tool_policies(
    temp_workspace: Path, rules: list[dict], local_path: str
) -> Path:
    """Create a temporary tool_policies.yaml file with given rules and local path."""
    temp_file = temp_workspace / "tool_policies.yaml"
    data = {
        "tools_platform": {
            "tool_policies_local_path": local_path,  # Use absolute path for local overrides
            "rules": rules,
        }
    }
    with open(temp_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return temp_file


def _create_temp_local_overrides(temp_workspace: Path, overrides: list[dict]) -> Path:
    """Create a temporary tool_policies.local.yaml file with given overrides."""
    temp_file = temp_workspace / "tool_policies.local.yaml"
    data = {"tools_platform": {"project_overrides": overrides}}
    with open(temp_file, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return temp_file


class TestConfigMerges:
    """Tests for config loading with project overrides using temp files."""

    def test_local_overrides_applied_on_load(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ):
        """Project overrides in local.yaml are applied when loading config."""
        # Load base rules from actual tool_policies.yaml
        import yaml

        with open(TOOL_POLICIES_PATH) as f:
            base_data = yaml.safe_load(f)
        base_rules = base_data.get("tools_platform", {}).get("rules", [])

        # Create temp local.yaml with override FIRST
        local_file = temp_workspace / "tool_policies.local.yaml"
        with open(local_file, "w") as f:
            yaml.safe_dump(
                {
                    "tools_platform": {
                        "project_overrides": [
                            {
                                "base_rule_id": "cmd_ask_git_commit",
                                "outcome": "allow",
                                "description": "Allow git commit",
                            }
                        ]
                    }
                },
                f,
                sort_keys=False,
            )

        # Create temp tool_policies.yaml with base rules AND local path
        base_file = temp_workspace / "tool_policies.yaml"
        with open(base_file, "w") as f:
            yaml.safe_dump(
                {
                    "tools_platform": {
                        "tool_policies_path": str(
                            base_file
                        ),  # Absolute path to THIS file
                        "tool_policies_local_path": str(
                            local_file
                        ),  # Absolute path to local file
                        "rules": base_rules,
                    }
                },
                f,
                sort_keys=False,
            )

        # Create ConfigManager with temp files
        cm = ConfigManager(config_paths=[str(base_file)])
        cfg = cm.load_global_config()
        cfg.tools_platform.workspace_root = str(temp_workspace)

        try:
            asvc = ApprovalService(workspace_manager, cm)

            # git commit should now be ALLOW (overridden)
            decision = _eval(
                asvc, "run_command", {"argv": ["git", "commit", "-m", "test"]}
            )
            assert decision.outcome == ToolDecisionOutcome.ALLOW

            # Verify rule was modified
            rule = next(r for r in asvc._rules if r.id == "cmd_ask_git_commit")
            assert rule.outcome == ToolDecisionOutcome.ALLOW
            assert rule.source == PolicySource.USER_APPROVAL
        finally:
            # Cleanup
            if local_file.exists():
                local_file.unlink()

    def test_multiple_overrides_applied(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ):
        """Multiple project overrides are all applied."""
        # Load base rules from actual tool_policies.yaml
        import yaml

        with open(TOOL_POLICIES_PATH) as f:
            base_data = yaml.safe_load(f)
        base_rules = base_data.get("tools_platform", {}).get("rules", [])

        # Create temp local.yaml with multiple overrides FIRST
        local_file = temp_workspace / "tool_policies.local.yaml"
        with open(local_file, "w") as f:
            yaml.safe_dump(
                {
                    "tools_platform": {
                        "project_overrides": [
                            {"base_rule_id": "cmd_ask_git_commit", "outcome": "allow"},
                            {"base_rule_id": "cmd_ask_git_push", "outcome": "allow"},
                            {
                                "base_rule_id": "boundary_read_outside",
                                "outcome": "allow",
                            },
                        ]
                    }
                },
                f,
                sort_keys=False,
            )

        # Create temp tool_policies.yaml with base rules AND local path
        base_file = temp_workspace / "tool_policies.yaml"
        with open(base_file, "w") as f:
            yaml.safe_dump(
                {
                    "tools_platform": {
                        "tool_policies_path": str(
                            base_file
                        ),  # Absolute path to THIS file
                        "tool_policies_local_path": str(
                            local_file
                        ),  # Absolute path to local file
                        "rules": base_rules,
                    }
                },
                f,
                sort_keys=False,
            )

        # Create ConfigManager with temp files
        cm = ConfigManager(config_paths=[str(base_file)])
        cfg = cm.load_global_config()
        cfg.tools_platform.workspace_root = str(temp_workspace)

        try:
            asvc = ApprovalService(workspace_manager, cm)

            # All three should be ALLOW
            for rule_id in [
                "cmd_ask_git_commit",
                "cmd_ask_git_push",
                "boundary_read_outside",
            ]:
                rule = next(r for r in asvc._rules if r.id == rule_id)
                assert rule.outcome == ToolDecisionOutcome.ALLOW, (
                    f"Rule {rule_id} not overridden"
                )
        finally:
            # Cleanup
            if local_file.exists():
                local_file.unlink()


# =============================================================================
# Rule Priority / Evaluation Order Tests
# =============================================================================


class TestRulePriority:
    """Tests for rule evaluation order by priority."""

    def test_deny_patterns_higher_than_allow(self, approval_service: ApprovalService):
        """Deny patterns (P190) evaluated before allow (P170) and ask (P150)."""
        # sudo is denied even though it might match other patterns
        decision = _eval(
            approval_service, "run_command", {"argv": ["sudo", "git", "diff"]}
        )
        assert decision.outcome == ToolDecisionOutcome.DENY
        assert decision.policy_rule == "cmd_deny_sudo"

    def test_allow_patterns_higher_than_ask(self, approval_service: ApprovalService):
        """Allow patterns (P170) evaluated before ask patterns (P150)."""
        # git diff is in both allow and could match broader patterns
        decision = _eval(
            approval_service, "run_command", {"argv": ["git", "diff", "HEAD"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_boundary_guard_higher_than_base_policy(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Boundary guard (P210/P220) evaluated before base policy (P100)."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        # read_file base policy is ALLOW (P100) but boundary guard is ASK (P210)
        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision.outcome == ToolDecisionOutcome.ASK
        assert decision.policy_rule == "boundary_read_outside"


# =============================================================================
# Rule Source Tracking Tests
# =============================================================================


class TestRuleSourceTracking:
    """Tests for rule source tracking (config vs user_approval)."""

    def test_base_rules_have_config_source(self, approval_service: ApprovalService):
        """Base rules from config have source=CONFIG."""
        for rule in approval_service._rules:
            if rule.id.startswith(
                ("base_", "cmd_", "boundary_", "heuristic_", "fallback")
            ):
                assert rule.source == PolicySource.CONFIG, (
                    f"Rule {rule.id} should have CONFIG source"
                )

    def test_session_override_changes_source(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Session override changes rule source to USER_APPROVAL."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ALWAYS_SESSION
        )

        rule = next(
            r for r in approval_service._rules if r.id == "boundary_read_outside"
        )
        assert rule.source == PolicySource.USER_APPROVAL

    def test_project_override_changes_source(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Project override changes rule source to USER_APPROVAL."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ALWAYS_PROJECT
        )

        rule = next(
            r for r in approval_service._rules if r.id == "boundary_read_outside"
        )
        assert rule.source == PolicySource.USER_APPROVAL


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for project override persistence."""

    def test_persist_override_writes_to_local_yaml(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """persist_tool_policies_overrides writes to tool_policies.local.yaml."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        token = decision.confirmation_token

        _confirm(
            approval_service, token, "test_session", ApprovalResponse.YES_ALWAYS_PROJECT
        )

        # Verify local.yaml content
        local_file = _local_file(approval_service._config_manager)
        with open(local_file) as f:
            data = yaml.safe_load(f)
        overrides = data.get("tools_platform", {}).get("project_overrides", [])
        assert len(overrides) == 1
        assert overrides[0]["base_rule_id"] == "boundary_read_outside"
        assert overrides[0]["outcome"] == "allow"

        _cleanup_local_yaml(approval_service._config_manager)

    def test_persist_updates_existing_override(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Second override for same rule updates existing entry."""
        # First override
        outside_file1 = temp_workspace.parent / "secret1.txt"
        outside_file1.write_text("secret1")
        decision = _eval(approval_service, "read_file", {"path": str(outside_file1)})
        _confirm(
            approval_service,
            decision.confirmation_token,
            "test_session",
            ApprovalResponse.YES_ALWAYS_PROJECT,
        )

        # Second override for same rule (simulating user changing mind)
        rule = next(
            r for r in approval_service._rules if r.id == "boundary_read_outside"
        )
        approval_service._config_manager.persist_tool_policies_overrides(
            rule_id=rule.id,
            outcome="deny",  # Change to deny
            description="Changed to deny",
        )

        # Verify only one entry exists with updated outcome
        local_file = _local_file(approval_service._config_manager)
        with open(local_file) as f:
            data = yaml.safe_load(f)
        overrides = data.get("tools_platform", {}).get("project_overrides", [])
        assert len(overrides) == 1
        assert overrides[0]["outcome"] == "deny"

        _cleanup_local_yaml(approval_service._config_manager)


# =============================================================================
# Decision Metadata Tests
# =============================================================================


class TestDecisionMetadata:
    """Tests for decision metadata fields."""

    def test_decision_has_policy_rule_id(self, approval_service: ApprovalService):
        """Decision has policy_rule set to rule ID."""
        decision = _eval(approval_service, "read_file", {"path": "test.txt"})
        assert decision.policy_rule is not None
        assert decision.policy_rule.startswith("base_")

    def test_decision_has_reason(self, approval_service: ApprovalService):
        """Decision has human-readable reason."""
        decision = _eval(approval_service, "read_file", {"path": "test.txt"})
        assert decision.reason
        assert len(decision.reason) > 0

    def test_ask_decision_has_confirmation_token(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """ASK decision includes confirmation_token."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision.confirmation_token is not None

    def test_ask_decision_has_proposed_rule(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """ASK decision includes proposed_rule for override preview."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision.proposed_rule is not None
        assert decision.proposed_rule.outcome == ToolDecisionOutcome.ALLOW

    def test_deny_decision_no_token(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """DENY decision has no confirmation_token."""

        decision = _eval(
            approval_service, "run_command", {"argv": ["apt", "install", "pkg"]}
        )
        assert decision.confirmation_token is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arguments_handled(self, approval_service: ApprovalService):
        """Empty arguments dict doesn't crash."""
        decision = _eval(approval_service, "run_command", {})
        assert decision.outcome in {ToolDecisionOutcome.ASK, ToolDecisionOutcome.DENY}

    def test_arguments_with_none_values(self, approval_service: ApprovalService):
        """None values in arguments handled gracefully."""
        decision = _eval(
            approval_service,
            "run_command",
            {"argv": ["echo", "hi"], "shell_mode": None},
        )
        # shell_mode=None should not trigger heuristic
        assert decision.outcome == ToolDecisionOutcome.ASK  # falls to base policy

    def test_tool_id_case_sensitive(self, approval_service: ApprovalService):
        """Tool IDs are case-sensitive."""
        decision = _eval(approval_service, "READ_FILE", {"path": "test.txt"})
        assert (
            decision.outcome == ToolDecisionOutcome.ASK
        )  # Unknown tool falls to fallback

    def test_session_isolation(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Session overrides modify rule globally (current design)."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        # Session 1 approves
        decision = _eval(
            approval_service,
            "read_file",
            {"path": str(outside_file)},
            session_id="session1",
        )
        _confirm(
            approval_service,
            decision.confirmation_token,
            "session1",
            ApprovalResponse.YES_ALWAYS_SESSION,
        )

        # Session 2 now also ALLOWs (global override)
        decision2 = _eval(
            approval_service,
            "read_file",
            {"path": str(outside_file)},
            session_id="session2",
        )
        assert decision2.outcome == ToolDecisionOutcome.ALLOW


# =============================================================================
# Matcher Tests
# =============================================================================


class TestMatchers:
    """Tests for matcher implementations."""

    def test_command_pattern_exact_match(self, approval_service: ApprovalService):
        """CommandPattern matches exact command."""
        decision = _eval(approval_service, "run_command", {"argv": ["git", "status"]})
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_command_pattern_glob_match(self, approval_service: ApprovalService):
        """CommandPattern matches glob pattern."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["git", "diff", "HEAD~1"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_command_pattern_prefix_match(self, approval_service: ApprovalService):
        """CommandPattern matches prefix with *."""
        decision = _eval(
            approval_service, "run_command", {"argv": ["pytest", "-v", "--tb=short"]}
        )
        assert decision.outcome == ToolDecisionOutcome.ALLOW

    def test_path_outside_workspace_matcher(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """PathOutsideWorkspace matches paths outside workspace."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision.outcome == ToolDecisionOutcome.ASK

    def test_path_inside_workspace_no_match(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """PathOutsideWorkspace doesn't match paths inside workspace."""
        test_file = temp_workspace / "src" / "app.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("print('hello')")

        decision = _eval(approval_service, "read_file", {"path": str(test_file)})
        assert decision.outcome == ToolDecisionOutcome.ALLOW


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegrationScenarios:
    """Integration-style tests combining multiple rules."""

    def test_git_workflow_allow_ask_sequence(self, approval_service: ApprovalService):
        """Typical git workflow: diff→ALLOW, commit→ASK, push→ASK."""
        assert (
            _eval(approval_service, "run_command", {"argv": ["git", "diff"]}).outcome
            == ToolDecisionOutcome.ALLOW
        )
        assert (
            _eval(
                approval_service, "run_command", {"argv": ["git", "commit", "-m", "x"]}
            ).outcome
            == ToolDecisionOutcome.ASK
        )
        assert (
            _eval(approval_service, "run_command", {"argv": ["git", "push"]}).outcome
            == ToolDecisionOutcome.ASK
        )

    def test_read_then_write_outside(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Read outside → ASK, write outside → DENY."""
        outside_file = temp_workspace.parent / "config.txt"
        outside_file.write_text("config")

        read_decision = _eval(
            approval_service, "read_file", {"path": str(outside_file)}
        )
        write_decision = _eval(
            approval_service,
            "write_file",
            {"path": str(outside_file), "content": "new"},
        )

        assert read_decision.outcome == ToolDecisionOutcome.ASK
        assert write_decision.outcome == ToolDecisionOutcome.ASK

    def test_approve_then_use_session_override(
        self, approval_service: ApprovalService, temp_workspace: Path
    ):
        """Approve once with YES_ALWAYS_SESSION, subsequent calls allowed."""
        outside_file = temp_workspace.parent / "secret.txt"
        outside_file.write_text("secret")

        # First call - ASK
        decision = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision.outcome == ToolDecisionOutcome.ASK

        # Approve for session
        _confirm(
            approval_service,
            decision.confirmation_token,
            "test_session",
            ApprovalResponse.YES_ALWAYS_SESSION,
        )

        # Second call - should ALLOW
        decision2 = _eval(approval_service, "read_file", {"path": str(outside_file)})
        assert decision2.outcome == ToolDecisionOutcome.ALLOW


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
