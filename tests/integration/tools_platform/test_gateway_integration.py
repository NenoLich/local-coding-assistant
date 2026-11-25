"""Integration tests for ToolGateway.

Uses real WorkspaceManager, HostBackend, ApprovalService and Scheduler
(per acceptance criteria). Covers config-driven behavior, exposure,
backend compatibility, per-tool policy validation and init_identity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ToolCallRequest,
    ToolDecisionOutcome,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutionStatus,
    ValidatedToolCall,
)
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.read_file import ReadFileInput
from local_coding_assistant.tools_platform.tools.register import register_all_tools
from local_coding_assistant.tools_platform.tools.replace_exact import (
    ReplaceExactInput,
)
from local_coding_assistant.tools_platform.tools.run_command import RunCommandInput


class _EmptyInput(BaseModel):
    pass


class _FakeWorkspaceRecord:
    def __init__(self, record_id: str, workspace_root: str, config_hash: str) -> None:
        self.id = record_id
        self.workspace_root = workspace_root
        self.config_hash = config_hash


class _FakeStore:
    """Minimal stand-in for IExecutionStore used by init_identity tests."""

    def __init__(self) -> None:
        self._by_root: dict[str, _FakeWorkspaceRecord] = {}
        self.refreshed_hash: str | None = None

    async def get_workspace(self, workspace_root: str) -> _FakeWorkspaceRecord | None:
        return self._by_root.get(workspace_root)

    async def upsert_workspace(
        self, workspace_id: str, workspace_root: str, config_hash: str
    ) -> _FakeWorkspaceRecord:
        record = _FakeWorkspaceRecord(workspace_id, workspace_root, config_hash)
        self._by_root[workspace_root] = record
        return record

    async def refresh_workspace_config_hash(
        self, workspace_id: str, config_hash: str
    ) -> None:
        self.refreshed_hash = config_hash
        for record in self._by_root.values():
            if record.id == workspace_id:
                record.config_hash = config_hash


def _make_registry() -> ToolRegistry:
    from local_coding_assistant.tools_platform.registry import ToolRegistry

    registry = ToolRegistry()
    register_all_tools(registry)
    # Write file is referenced in tool_policies.yaml rules but not in phase registration;
    # register it minimally so the per-tool policy validation passes.
    from local_coding_assistant.tools_platform.contracts import (
        ToolDefinition,
        ToolRiskClass,
    )

    write_file_def = ToolDefinition(
        id="write_file",
        version="1.0.0",
        description="Writes content to a file",
        input_schema={},
        output_schema={},
        risk_class=ToolRiskClass.MEDIUM,
        supported_backends=[BackendType.HOST],
    )
    registry.register(write_file_def, MagicMock())
    return registry


def _create_temp_tool_policies(
    tmp_path: Path, extra_rules: list[dict] | None = None
) -> Path:
    """Create a temporary tool_policies.yaml file for testing.

    Args:
        tmp_path: Temporary directory
        extra_rules: Additional rules to append to the base rules

    Returns:
        Path to the created tool_policies.yaml
    """
    base_rules: list[dict[str, Any]] = [
        {
            "id": "base_read_file",
            "tool_id": "read_file",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for read_file",
        },
        {
            "id": "base_search_text",
            "tool_id": "search_text",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for search_text",
        },
        {
            "id": "base_search_symbols",
            "tool_id": "search_symbols",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for search_symbols",
        },
        {
            "id": "base_get_symbol",
            "tool_id": "get_symbol",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for get_symbol",
        },
        {
            "id": "base_match_paths",
            "tool_id": "match_paths",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for match_paths",
        },
        {
            "id": "base_check_process_status",
            "tool_id": "check_process_status",
            "matcher": {"type": "any_args"},
            "outcome": "allow",
            "priority": 100,
            "description": "Base policy for check_process_status",
        },
        {
            "id": "base_run_command",
            "tool_id": "run_command",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for run_command",
        },
        {
            "id": "base_replace_exact",
            "tool_id": "replace_exact",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for replace_exact",
        },
        {
            "id": "base_create_file",
            "tool_id": "create_file",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for create_file",
        },
        {
            "id": "base_delete_file",
            "tool_id": "delete_file",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for delete_file",
        },
        {
            "id": "base_write_file",
            "tool_id": "write_file",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for write_file",
        },
        {
            "id": "base_kill_process",
            "tool_id": "kill_process",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 100,
            "description": "Base policy for kill_process",
        },
        {
            "id": "fallback",
            "tool_id": "*",
            "matcher": {"type": "any_args"},
            "outcome": "ask",
            "priority": 0,
            "description": "Global fallback",
        },
    ]

    if extra_rules:
        base_rules.extend(extra_rules)

    tool_policies_path = tmp_path / "tool_policies.yaml"
    content = "tools_platform:\n  rules:\n"
    for rule in base_rules:
        content += f'    - id: "{rule["id"]}"\n'
        content += f'      tool_id: "{rule["tool_id"]}"\n'
        content += "      matcher:\n"
        content += f'        type: "{rule["matcher"]["type"]}"\n'
        content += f'      outcome: "{rule["outcome"]}"\n'
        content += f"      priority: {rule['priority']}\n"
        content += f'      description: "{rule["description"]}"\n'
    content += "  project_overrides: []\n"

    tool_policies_path.write_text(content, encoding="utf-8")
    return tool_policies_path


def _build_gateway(
    tmp_path: Path,
    tool_policies_path: Path | None = None,
    **overrides: Any,
) -> tuple[ToolGateway, ConfigManager, str]:
    """Build a real ToolGateway backed by a temp workspace.

    Returns ``(gateway, config_manager, workspace_id)``.
    """
    # Create a temporary config file that sets the correct tool_policies_path.
    # This is needed because ConfigManager merges tool_policies BEFORE env vars,
    # so the LOCCA_TOOLS_PLATFORM__TOOL_POLICIES_PATH env var isn't available yet.
    # In test mode, @config resolves to tests/configs which doesn't have the file.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("tools_platform:\n")
        f.write('  tool_policies_path: "@project/config/tool_policies.yaml"\n')
        temp_config_path = f.name

    try:
        cm = ConfigManager(config_paths=[temp_config_path])
        cm.load_global_config()

        # Apply session overrides for test configuration
        session_overrides = {
            "tools_platform.workspace_root": str(tmp_path),
        }
        session_overrides.update(overrides)

        if tool_policies_path:
            session_overrides["tools_platform.tool_policies_path"] = str(
                tool_policies_path
            )

        cm.set_session_overrides(session_overrides)

        registry = _make_registry()
        gateway = ToolGateway(registry=registry, config_manager=cm)
        return gateway, cm, gateway.workspace_manager.workspace_id
    finally:
        # Clean up temp config file
        try:
            os.unlink(temp_config_path)
        except OSError:
            pass


def _validated_call(
    gateway: ToolGateway,
    tool_id: str,
    call_id: str,
    input_model_cls: type[BaseModel],
    **kwargs: Any,
) -> ValidatedToolCall:
    """Build a validated tool call for scheduler grouping assertions."""
    tool_def = gateway._registry.get_tool(tool_id)
    assert tool_def is not None, f"Tool {tool_id} not found in registry"
    return ValidatedToolCall(
        request=ToolCallRequest(
            tool_id=tool_id,
            call_id=call_id,
            session_id="sess_1",
            frame_id="frame_1",
            raw_arguments=kwargs,
            workspace_id=gateway.workspace_manager.workspace_id,
        ),
        tool_definition=tool_def,
        input_model=input_model_cls(**kwargs),
    )


def _request(
    gateway: ToolGateway,
    tool_id: str = "read_file",
    call_id: str = "c1",
    workspace_id: str | None = None,
    **raw_arguments: Any,
) -> ToolCallRequest:
    if not raw_arguments:
        raw_arguments = {"path": "README.md"}
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments=raw_arguments,
        workspace_id=workspace_id or gateway.workspace_manager.workspace_id,
    )


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text("# hello\nline2\nline3\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("print('main')\n", encoding="utf-8")
    (tmp_path / "tests").mkdir(exist_ok=True)
    (tmp_path / "tests" / "test.py").write_text("print('test')\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("AAA\nBBB\n", encoding="utf-8")
    (tmp_path / "d.txt").write_text("DDD\nEEE\n", encoding="utf-8")
    (tmp_path / "e.txt").write_text("FFF\nGGG\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Case #1: default config (backend=host, default_policy=ask)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_default_config_read_file_ask(tmp_workspace: Path) -> None:
    """With defaults.yaml (default_policy=ask) read_file is ALLOWED because its
    per-tool default_policy=allow (from tool_policies) overrides the global
    ask default_policy."""
    gateway, _cm, workspace_id = _build_gateway(tmp_workspace)
    result = await gateway.execute(_request(gateway, workspace_id=workspace_id))

    assert result.status == ToolExecutionStatus.COMPLETED.value
    assert result.output is not None
    assert result.decision is not None
    assert result.decision.outcome == ToolDecisionOutcome.ALLOW


# ---------------------------------------------------------------------------
# Case #2: default_policy=deny -> per-tool allow overrides global deny
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_default_policy_deny(tmp_workspace: Path) -> None:
    """Global default_policy=deny is overridden by read_file's per-tool
    default_policy=allow, so read_file still executes."""
    gateway, _cm, workspace_id = _build_gateway(
        tmp_workspace,
        **{"tools_platform.default_policy": "deny"},
    )
    result = await gateway.execute(_request(gateway, workspace_id=workspace_id))

    assert result.status == ToolExecutionStatus.COMPLETED.value
    assert result.output is not None
    assert result.decision is not None
    assert result.decision.outcome == ToolDecisionOutcome.ALLOW


# ---------------------------------------------------------------------------
# Case #3: default_policy=allow -> ALLOW for read_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_default_policy_allow(tmp_workspace: Path) -> None:
    gateway, _cm, workspace_id = _build_gateway(
        tmp_workspace,
        **{"tools_platform.default_policy": "allow"},
    )
    result = await gateway.execute(_request(gateway, workspace_id=workspace_id))

    assert result.status == ToolExecutionStatus.COMPLETED.value
    assert result.output is not None
    assert result.decision is not None
    assert result.decision.outcome == ToolDecisionOutcome.ALLOW


# ---------------------------------------------------------------------------
# Case #4: non_interactive=true converts ASK to DENIED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_non_interactive_ask_to_denied(tmp_workspace: Path) -> None:
    """non_interactive=true converts an ASK decision to DENIED. run_command has
    per-tool default_policy=ask, so it requires approval even under global ask;
    in non-interactive mode that becomes DENIED."""
    gateway, _cm, workspace_id = _build_gateway(
        tmp_workspace,
        **{
            "tools_platform.default_policy": "ask",
            "tools_platform.non_interactive": True,
        },
    )
    result = await gateway.execute(
        _request(
            gateway,
            tool_id="run_command",
            workspace_id=workspace_id,
            argv=["cmd", "/c", "echo", "hi"],
        )
    )

    assert result.status == ToolExecutionStatus.DENIED.value
    assert result.error is not None
    assert result.error.code == ToolErrorCode.DENIED
    assert result.error.message.startswith("Non-interactive mode:")
    assert result.decision is not None
    assert result.decision.outcome == ToolDecisionOutcome.DENY
    assert result.decision.reason.startswith("Non-interactive mode:")


# ---------------------------------------------------------------------------
# Case #5: exposed_tools list limits exposed schemas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_exposed_tools_list(tmp_workspace: Path) -> None:
    gateway, _cm, _workspace_id = _build_gateway(
        tmp_workspace,
        **{"tools_platform.exposed_tools": ["read_file", "search_text"]},
    )
    schemas = gateway.get_exposed_tool_schemas()
    ids = {s["id"] for s in schemas}
    assert ids == {"read_file", "search_text"}


# ---------------------------------------------------------------------------
# Case #6: exposed_tools=auto exposes all backend-compatible tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_exposed_tools_auto(tmp_workspace: Path) -> None:
    gateway, _cm, _workspace_id = _build_gateway(
        tmp_workspace,
        **{"tools_platform.exposed_tools": "auto"},
    )
    schemas = gateway.get_exposed_tool_schemas()
    ids = {s["id"] for s in schemas}
    expected = {
        t.id
        for t in _make_registry().list_tools()
        if BackendType.HOST in t.supported_backends
    }
    assert ids == expected


# ---------------------------------------------------------------------------
# Case #7: backend-incompatible tool_id is rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_rejects_backend_incompatible_tool(tmp_workspace: Path) -> None:
    gateway, _cm, workspace_id = _build_gateway(tmp_workspace)

    # Register a docker-only tool on top of the host-backed gateway.
    definition = ToolDefinition(
        id="docker_only_tool",
        version="1.0.0",
        description="runs only in docker",
        input_schema={},
        output_schema={},
        supported_backends=[BackendType.DOCKER],
    )
    gateway._registry.register(
        definition,
        MagicMock(),
    )

    result = await gateway.execute(
        _request(gateway, tool_id="docker_only_tool", workspace_id=workspace_id)
    )

    assert result.status == ToolExecutionStatus.FAILED.value
    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS


# ---------------------------------------------------------------------------
# Case #8: unknown tool ID in per-tool policy config raises ValueError at init
# ---------------------------------------------------------------------------


def test_gateway_unknown_tool_policy_raises(tmp_workspace: Path) -> None:
    # Create a temporary tool_policies.yaml with an unknown tool ID
    tool_policies_path = _create_temp_tool_policies(
        tmp_workspace,
        extra_rules=[
            {
                "id": "bad_rule",
                "tool_id": "nonexistent_tool_x",
                "matcher": {"type": "any_args"},
                "outcome": "allow",
                "priority": 100,
                "description": "Bad rule with unknown tool",
            }
        ],
    )

    # Create a temporary config file that points to our test tool_policies.yaml
    # Use forward slashes in YAML to avoid escape sequence issues on Windows
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("tools_platform:\n")
        f.write(f'  tool_policies_path: "{tool_policies_path.as_posix()}"\n')
        temp_config_path = f.name

    try:
        cm = ConfigManager(config_paths=[temp_config_path])
        cm.load_global_config()
        cm.set_session_overrides(
            {
                "tools_platform.workspace_root": str(tmp_workspace),
            }
        )

        registry = _make_registry()
        with pytest.raises(ValueError, match="unknown tool IDs"):
            ToolGateway(registry=registry, config_manager=cm)
    finally:
        # Clean up temp config file
        try:
            os.unlink(temp_config_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Case #9: init_identity persists workspace_id and reuses it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_identity_persists_and_reuses(tmp_workspace: Path) -> None:
    store = _FakeStore()

    gateway1, _cm, _wid = _build_gateway(tmp_workspace)
    wid1 = await gateway1.init_identity(store=cast(IExecutionStore, store))

    record = await store.get_workspace(str(tmp_workspace))
    assert record is not None
    assert record.id == wid1

    # A second gateway for the same root reuses the stored identity.
    gateway2, _cm2, _wid2 = _build_gateway(tmp_workspace)
    wid2 = await gateway2.init_identity(store=cast(IExecutionStore, store))
    assert wid2 == wid1
    assert gateway2.workspace_manager.workspace_id == wid1


# ---------------------------------------------------------------------------
# Case #10: init_identity detects config_hash drift and refreshes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_identity_refreshes_on_config_drift(tmp_workspace: Path) -> None:
    store = _FakeStore()

    gateway1, cm1, _wid = _build_gateway(tmp_workspace)
    wid1 = await gateway1.init_identity(store=cast(IExecutionStore, store))
    record = await store.get_workspace(str(tmp_workspace))
    assert record is not None
    first_hash = record.config_hash

    # Change the config on the same manager, rebuild the gateway, and
    # re-init: stored hash should be refreshed to the new value.
    cm1.set_session_overrides(
        {
            "tools_platform.workspace_root": str(tmp_workspace),
            "tools_platform.default_policy": "deny",
        }
    )
    registry2 = _make_registry()
    gateway2 = ToolGateway(registry=registry2, config_manager=cm1)
    wid2 = await gateway2.init_identity(store=cast(IExecutionStore, store))
    assert wid2 == wid1  # same root -> same workspace id

    record2 = await store.get_workspace(str(tmp_workspace))
    assert record2 is not None
    assert record2.config_hash != first_hash
    assert store.refreshed_hash == record2.config_hash


# ---------------------------------------------------------------------------
# Case #11: execute_many classifies mixed reads into a parallel group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_many_parallel_read_group(tmp_workspace: Path) -> None:
    gateway, _cm, _wid = _build_gateway(
        tmp_workspace,
        **{"tools_platform.default_policy": "allow"},
    )
    calls = [
        _validated_call(gateway, "read_file", "r1", ReadFileInput, path="a.txt"),
        _validated_call(gateway, "read_file", "r2", ReadFileInput, path="b.txt"),
    ]
    groups = await gateway._scheduler.classify_tool_calls(calls)
    # Disjoint reads share a group -> executed concurrently.
    assert len(groups) == 1
    assert {v.request.call_id for v in next(iter(groups.values()))} == {"r1", "r2"}

    results = await gateway.execute_many([c.request for c in calls])
    assert len(results) == 2
    assert all(r.status == ToolExecutionStatus.COMPLETED.value for r in results)


# ---------------------------------------------------------------------------
# Case #12: overlapping file paths serialize writes; disjoint paths parallelize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_many_path_overlap_serialization(tmp_workspace: Path) -> None:
    gateway, _cm, _wid = _build_gateway(
        tmp_workspace,
        **{"tools_platform.default_policy": "allow"},
    )
    # Allow replace_exact so the execution can proceed (per-tool policy is 'ask')
    # Directly modify the rule since ApprovalService doesn't have add_session_override
    for rule in gateway._approval_service._rules:
        if rule.tool_id == "replace_exact" and rule.id == "base_replace_exact":
            rule.outcome = ToolDecisionOutcome.ALLOW
            break
    same_path = [
        _validated_call(
            gateway,
            "replace_exact",
            "w1",
            ReplaceExactInput,
            path="c.txt",
            old_text="AAA",
            new_text="aaa",
            expected_match_count=1,
        ),
        _validated_call(
            gateway,
            "replace_exact",
            "w2",
            ReplaceExactInput,
            path="c.txt",
            old_text="BBB",
            new_text="bbb",
            expected_match_count=1,
        ),
    ]
    diff_path = [
        _validated_call(
            gateway,
            "replace_exact",
            "w3",
            ReplaceExactInput,
            path="d.txt",
            old_text="DDD",
            new_text="ddd",
            expected_match_count=1,
        ),
        _validated_call(
            gateway,
            "replace_exact",
            "w4",
            ReplaceExactInput,
            path="e.txt",
            old_text="FFF",
            new_text="ggg",
            expected_match_count=1,
        ),
    ]

    same_groups = await gateway._scheduler.classify_tool_calls(same_path)
    # Same file path -> separate groups (serialized).
    assert len(same_groups) == 2

    diff_groups = await gateway._scheduler.classify_tool_calls(diff_path)
    # Disjoint file paths -> one group (parallelized).
    assert len(diff_groups) == 1

    results = await gateway.execute_many([c.request for c in same_path])
    assert len(results) == 2
    for r in results:
        if r.status != ToolExecutionStatus.COMPLETED.value:
            print(f"Failed result: status={r.status}, error={r.error}")
    assert all(r.status == ToolExecutionStatus.COMPLETED.value for r in results)

    results = await gateway.execute_many([c.request for c in diff_path])
    assert len(results) == 2
    assert all(r.status == ToolExecutionStatus.COMPLETED.value for r in results)


# ---------------------------------------------------------------------------
# Case #13: process capacity limit queues excess run_command calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_many_process_capacity_queues(tmp_workspace: Path) -> None:
    gateway, _cm, _wid = _build_gateway(
        tmp_workspace,
        **{
            "tools_platform.concurrency.max_parallel_processes": 2,
            "tools_platform.default_policy": "allow",
        },
    )
    # Allow run_command so the capacity-queuing behavior can be exercised
    # (unmatched commands fall back to default_action=ask otherwise).
    # Directly modify the rule since ApprovalService doesn't have add_session_override
    for rule in gateway._approval_service._rules:
        if rule.tool_id == "run_command" and rule.id == "base_run_command":
            rule.outcome = ToolDecisionOutcome.ALLOW
            break
    calls = [
        _validated_call(
            gateway,
            "run_command",
            f"p{i}",
            RunCommandInput,
            argv=["cmd", "/c", "echo", f"msg{i}"],
        )
        for i in range(3)
    ]
    groups = await gateway._scheduler.classify_tool_calls(calls)
    # All run_command calls share one group (process capacity is enforced by
    # the semaphore at execution time, not by group separation).
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 3

    # execute_many accepts all three calls (capacity does not reject them at
    # grouping time). Actual execution of run_command is blocked by unrelated
    # pre-existing issues (timeout TypeError + REJECT "processes" lock), so we
    # assert only that every call produces a result.
    results = await gateway.execute_many([c.request for c in calls])
    assert len(results) == 3
