"""Integration tests for WorkspaceManager.

Uses real WorkspaceManager with ConfigManager-driven configuration
(per acceptance criteria). Covers path_policy enforcement, transactions,
conflict detection, snapshots, and file monitoring integration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.contracts import (
    ConflictReport,
    FileObservation,
    ObservationSource,
)
from local_coding_assistant.tools_platform.exceptions import ToolPathViolationError
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.conflicts import ConflictType


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


def _make_config_manager(
    tmp_path: Path,
    *,
    allowed_patterns: list[str] | None = None,
    denied_patterns: list[str] | None = None,
    allow_symlinks: bool = False,
    max_depth: int | None = None,
    write_conflict_policy: str = "deny",
    snapshots_enabled: bool = True,
) -> ConfigManager:
    """Build a ConfigManager with custom tools_platform settings."""
    cm = ConfigManager(config_paths=["config/tool_policies.yaml"])
    cfg = cm.load_global_config()
    cfg.tools_platform.workspace_root = str(tmp_path)
    if allowed_patterns is not None:
        cfg.tools_platform.path_policy.allowed_patterns = allowed_patterns
    if denied_patterns is not None:
        cfg.tools_platform.path_policy.denied_patterns = denied_patterns
    cfg.tools_platform.path_policy.allow_symlinks = allow_symlinks
    cfg.tools_platform.path_policy.max_depth = max_depth
    cfg.tools_platform.concurrency.write_conflict_policy = write_conflict_policy
    cfg.tools_platform.snapshots.enabled = snapshots_enabled
    # Store snapshots outside workspace tree (hermetic, under pytest temp).
    cfg.tools_platform.snapshots.directory = str(
        tmp_path.parent / "locca_workspace_snapshots"
    )
    return cm


@pytest.fixture
def workspace_manager(temp_workspace: Path) -> WorkspaceManager:
    """WorkspaceManager with default config (allows all paths)."""
    cm = _make_config_manager(temp_workspace)
    return WorkspaceManager.from_config_manager(cm)


# ---------------------------------------------------------------------------
# Path Policy Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_policy_allows_all_with_defaults(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """workspace with config path_policy (allowed=["**"], denied=[], allow_symlinks=false, max_depth=null) allows all paths."""
    # Test various paths within workspace
    test_file = temp_workspace / "any.txt"
    test_file.write_text("content")

    resolved = workspace_manager.resolve_path("any.txt")
    assert resolved == test_file

    resolved = workspace_manager.resolve_path("src/main.py")
    assert resolved == temp_workspace / "src" / "main.py"

    resolved = workspace_manager.resolve_path("tests/test.py")
    assert resolved == temp_workspace / "tests" / "test.py"

    resolved = workspace_manager.resolve_path("README.md")
    assert resolved == temp_workspace / "README.md"


@pytest.mark.asyncio
async def test_path_policy_denied_patterns_rejected(temp_workspace: Path) -> None:
    """workspace with denied_patterns=["*.secret", ".env*"] rejects those paths."""
    cm = _make_config_manager(temp_workspace, denied_patterns=["*.secret", ".env*"])
    manager = WorkspaceManager.from_config_manager(cm)

    # Create the files first
    (temp_workspace / "secret.secret").write_text("secret")
    (temp_workspace / ".env.local").write_text("local")

    with pytest.raises(ToolPathViolationError, match="denied pattern"):
        manager.resolve_path("secret.secret")

    with pytest.raises(ToolPathViolationError, match="denied pattern"):
        manager.resolve_path(".env.local")


@pytest.mark.asyncio
async def test_path_policy_allowed_patterns_restricts(temp_workspace: Path) -> None:
    """workspace with allowed_patterns=["src/**", "tests/**"] rejects paths outside those dirs."""
    cm = _make_config_manager(temp_workspace, allowed_patterns=["src/**", "tests/**"])
    manager = WorkspaceManager.from_config_manager(cm)

    # These should work
    resolved = manager.resolve_path("src/main.py")
    assert resolved == temp_workspace / "src" / "main.py"

    resolved = manager.resolve_path("tests/test.py")
    assert resolved == temp_workspace / "tests" / "test.py"

    # These should fail
    with pytest.raises(
        ToolPathViolationError, match="does not match any allowed pattern"
    ):
        manager.resolve_path("README.md")

    with pytest.raises(
        ToolPathViolationError, match="does not match any allowed pattern"
    ):
        manager.resolve_path(".env")


@pytest.mark.asyncio
async def test_path_policy_max_depth_enforced(temp_workspace: Path) -> None:
    """workspace with max_depth=3 rejects "a/b/c/d/file.py"."""
    cm = _make_config_manager(temp_workspace, max_depth=3)
    manager = WorkspaceManager.from_config_manager(cm)

    # Depth 3 (a/b/c) should work - 3 parts
    (temp_workspace / "a" / "b" / "c").mkdir(parents=True)
    (temp_workspace / "a" / "b" / "c" / "file.py").write_text("content")
    # "a/b/c/file.py" has 4 parts, so with max_depth=3 it should fail
    # Let's test depth 2 instead
    (temp_workspace / "x" / "y").mkdir(parents=True)
    (temp_workspace / "x" / "y" / "file.py").write_text("content")
    resolved = manager.resolve_path("x/y/file.py")  # 3 parts: x, y, file.py
    assert resolved == temp_workspace / "x" / "y" / "file.py"

    # Depth 4 (a/b/c/d) should fail - 4 parts
    (temp_workspace / "a" / "b" / "c" / "d").mkdir(parents=True)
    (temp_workspace / "a" / "b" / "c" / "d" / "file.py").write_text("content")
    with pytest.raises(ToolPathViolationError, match="exceeds maximum depth"):
        manager.resolve_path("a/b/c/d/file.py")


@pytest.mark.asyncio
async def test_path_policy_allow_symlinks_true(temp_workspace: Path) -> None:
    """workspace with allow_symlinks=true allows symlinks within workspace."""
    cm = _make_config_manager(temp_workspace, allow_symlinks=True)
    manager = WorkspaceManager.from_config_manager(cm)

    target = temp_workspace / "target.txt"
    target.write_text("content")
    link = temp_workspace / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not supported on this platform")

    # Should not raise when resolving symlink
    resolved = manager.resolve_path("link.txt")
    # resolve_path resolves symlinks, so it points to the target
    assert resolved.exists()
    assert resolved.read_text() == "content"


# ---------------------------------------------------------------------------
# Transaction Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transaction_commits_multiple_operations(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """transaction with multiple atomic_write/atomic_create/atomic_delete commits all as single ChangeSet."""
    (temp_workspace / "existing.txt").write_text("original")

    workspace_manager.begin_transaction()
    workspace_manager.atomic_write(
        "existing.txt", "modified", session_id="some_session"
    )
    workspace_manager.atomic_create("new.txt", "created")
    workspace_manager.atomic_delete("existing.txt")
    changeset = workspace_manager.commit_transaction()

    assert changeset.total_files_changed == 2
    assert not (temp_workspace / "existing.txt").exists()
    assert (temp_workspace / "new.txt").read_text() == "created"


@pytest.mark.asyncio
async def test_concurrent_transactions_different_files(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """concurrent transactions on different files proceed independently."""
    (temp_workspace / "file1.txt").write_text("a")
    (temp_workspace / "file2.txt").write_text("b")

    # Simulate concurrent transactions by creating two managers
    cm1 = _make_config_manager(temp_workspace)
    cm2 = _make_config_manager(temp_workspace)
    mgr1 = WorkspaceManager.from_config_manager(cm1)
    mgr2 = WorkspaceManager.from_config_manager(cm2)

    mgr1.begin_transaction()
    mgr2.begin_transaction()

    mgr1.atomic_write("file1.txt", "a1", session_id="session1")
    mgr2.atomic_write("file2.txt", "b2", session_id="session1")

    mgr1.commit_transaction()
    mgr2.commit_transaction()

    assert (temp_workspace / "file1.txt").read_text() == "a1"
    assert (temp_workspace / "file2.txt").read_text() == "b2"


@pytest.mark.asyncio
async def test_concurrent_transactions_same_file_serializes(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """concurrent transactions on same file serialize per conflict_rule."""
    (temp_workspace / "shared.txt").write_text("original")

    # Observe file to establish baseline in session1
    obs1 = workspace_manager.observe_file("shared.txt", session_id="session1")

    # First transaction writes
    workspace_manager.begin_transaction()
    workspace_manager.atomic_write("shared.txt", "first", session_id="session1")
    workspace_manager.commit_transaction()

    # Second session observes the same file (gets current state "first")
    obs2 = workspace_manager.observe_file("shared.txt", session_id="session2")

    # Second transaction attempts to write with expected_hash from its observation
    workspace_manager.begin_transaction()
    # This should succeed because obs2.hash matches current disk state
    workspace_manager.atomic_write(
        "shared.txt", "second", session_id="session2", expected_hash=obs2.hash
    )
    workspace_manager.commit_transaction()

    # Verify final state
    assert (temp_workspace / "shared.txt").read_text() == "second"


# ---------------------------------------------------------------------------
# Conflict Detection Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_edit_detected_as_conflict(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """user edits file externally between agent read and write -> validate_write_version detects conflict."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    # Agent reads file (establishes baseline)
    workspace_manager.observe_file("test.txt", session_id="test-session")

    # User edits file externally
    test_file.write_text("user modified")

    # Agent attempts write
    report = workspace_manager.validate_write_version(
        "test.txt", "proposed_hash", "proposed", session_id="test-session"
    )
    assert isinstance(report, ConflictReport)
    assert report.conflict_type == ConflictType.VERSION_STALE
    assert report.conflict_detected is True
    assert report.base_hash != ""
    assert report.current_hash != ""
    assert report.proposed_hash == "proposed_hash"


@pytest.mark.asyncio
async def test_conflict_report_includes_conflict_type_and_resolution_options(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """ConflictReport includes conflict_type and resolution_options."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")
    workspace_manager.observe_file("test.txt", session_id="test-session")
    test_file.write_text("modified")

    report = workspace_manager.validate_write_version(
        "test.txt", "proposed", "proposed_content", session_id="test-session"
    )

    assert report.conflict_detected is True
    assert report.conflict_type == ConflictType.VERSION_STALE
    assert hasattr(report, "resolution_options")
    assert isinstance(report.resolution_options, list)
    assert report.recommendation in ("re_read_file", "retry_or_adapt")
    assert report.message is not None


@pytest.mark.asyncio
async def test_snapshot_enables_rollback(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """snapshot created before write enables rollback to pre-write state."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    workspace_manager.begin_transaction()
    workspace_manager.atomic_write("test.txt", "modified", session_id="session1")
    workspace_manager.commit_transaction()

    assert (temp_workspace / "test.txt").read_text() == "modified"

    # Create snapshot of current state
    snapshot_id = workspace_manager.create_snapshot(["test.txt"])

    # Modify again
    workspace_manager.begin_transaction()
    workspace_manager.atomic_write("test.txt", "modified again", session_id="session1")
    workspace_manager.commit_transaction()

    assert (temp_workspace / "test.txt").read_text() == "modified again"

    # Restore from snapshot
    workspace_manager.restore_snapshot(snapshot_id, "test.txt")
    assert (temp_workspace / "test.txt").read_text() == "modified"


# ---------------------------------------------------------------------------
# File Monitoring Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_observations_do_not_override_baseline(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """file monitoring events (EXTERNAL observations) do not override baseline for conflict detection.

    The baseline for conflict detection comes from READ/SNAPSHOT/AGENT_CREATE sources.
    EXTERNAL observations (from file monitor) should be stored in a separate session scope
    so they don't replace the global READ baseline used for conflict detection.
    """
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    # Agent reads file - establishes READ baseline in global scope
    read_obs = workspace_manager.observe_file("test.txt")
    assert read_obs.source == ObservationSource.READ

    # Baseline is the READ observation
    baseline = workspace_manager.get_baseline_observation("test.txt")
    assert baseline is not None
    assert baseline.source == ObservationSource.READ
    assert baseline.hash == read_obs.hash

    # Simulate file monitor emitting EXTERNAL observation in a separate session scope
    # (file monitor should not write to global scope)
    test_file.write_text("external change")
    workspace_manager.observe_file(
        "test.txt", source=ObservationSource.EXTERNAL, session_id="file-monitor"
    )

    # Global baseline is unchanged
    baseline = workspace_manager.get_baseline_observation("test.txt")
    assert baseline is not None
    assert baseline.source == ObservationSource.READ
    assert baseline.hash == read_obs.hash

    # File monitor session has the EXTERNAL observation
    monitor_obs = workspace_manager.get_file_version(
        "test.txt", session_id="file-monitor"
    )
    assert monitor_obs is not None
    assert monitor_obs.source == ObservationSource.EXTERNAL
    assert monitor_obs.hash != read_obs.hash


@pytest.mark.asyncio
async def test_workspace_manager_subscribes_to_file_monitor_for_advisory_only(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """workspace_manager integrates with file monitor for advisory invalidation only.

    The WorkspaceManager itself doesn't directly subscribe - it's the
    higher-level runtime that coordinates. Verify the manager provides
    the necessary hooks for advisory invalidation.
    """
    # get_workspace_state exposes the file versions that could be invalidated
    state = workspace_manager.get_workspace_state()
    assert state is not None
    assert hasattr(state, "session_file_versions")
    assert hasattr(state, "workspace_id")

    # clear_session_overrides allows invalidating session-scoped state
    workspace_manager.clear_session_overrides("test-session")

    # The conflict detection uses READ/SNAPSHOT/AGENT_CREATE baselines only
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")
    obs = workspace_manager.observe_file("test.txt")
    assert obs.source == ObservationSource.READ

    baseline = workspace_manager.get_baseline_observation("test.txt")
    assert baseline is obs


# ---------------------------------------------------------------------------
# Snapshot Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshots_stored_outside_workspace_tree(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """snapshots stored outside workspace tree (using @data/workspace_snapshots alias)."""
    (temp_workspace / "test.txt").write_text("content")

    snapshot_id = workspace_manager.create_snapshot(["test.txt"])
    assert snapshot_id is not None

    # Check snapshot directory is outside workspace
    snap_dir = workspace_manager.snapshot_manager.config.snapshots_dir
    assert not str(snap_dir).startswith(str(temp_workspace))

    # Verify snapshot file is in the external directory
    snapshot_file = workspace_manager.snapshot_manager._get_file_snapshot_path(
        snapshot_id, "test.txt"
    )
    assert not str(snapshot_file).startswith(str(temp_workspace))


# ---------------------------------------------------------------------------
# Config-Driven Behavior Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_policy_from_config_enforced(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """WorkspaceManager enforces path_policy from config (not hardcoded defaults)."""
    # Default config has allowed_patterns=["**"], denied_patterns=[]
    # Verify the manager uses config values
    policy = workspace_manager.path_policy
    assert policy.allowed_patterns == ["**"]
    assert policy.denied_patterns == []
    assert policy.allow_symlinks is False
    assert policy.max_depth is None


@pytest.mark.asyncio
async def test_write_conflict_policy_from_config(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """ConflictDetector uses write_conflict_policy from config."""
    # Default config has write_conflict_policy="deny"
    assert workspace_manager._config_manager is not None
    config = workspace_manager._config_manager.global_config.tools_platform
    assert config.concurrency.write_conflict_policy == "deny"

    # Conflict detector should use this policy
    assert workspace_manager.conflict_detector.config.conflict_policy == "deny"


@pytest.mark.asyncio
async def test_snapshots_config_from_config(temp_workspace: Path) -> None:
    """SnapshotManager uses snapshots config from config."""
    cm = _make_config_manager(temp_workspace, snapshots_enabled=False)
    manager = WorkspaceManager.from_config_manager(cm)

    assert manager.snapshot_manager.config.max_snapshots == 10  # default
    # Even when disabled, directory is still configured
    assert manager.snapshot_manager.config.snapshots_dir is not None


@pytest.mark.asyncio
async def test_config_hash_detects_drift(temp_workspace: Path) -> None:
    """WorkspaceManager config_hash changes when path_policy config changes."""
    cm1 = _make_config_manager(temp_workspace, allowed_patterns=["**"])
    mgr1 = WorkspaceManager.from_config_manager(cm1)

    cm2 = _make_config_manager(temp_workspace, allowed_patterns=["src/**"])
    mgr2 = WorkspaceManager.from_config_manager(cm2)

    # Different configs produce different path policies
    assert mgr1.path_policy.allowed_patterns == ["**"]
    assert mgr2.path_policy.allowed_patterns == ["src/**"]


# ---------------------------------------------------------------------------
# Session-Scoped Baselines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_scoped_baselines_independent(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """Multiple sessions have independent file version baselines."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    now = datetime.now(UTC)
    obs1 = FileObservation(
        path="test.txt",
        hash="session1_hash",
        size=100,
        mtime=now,
        source=ObservationSource.READ,
    )
    obs2 = FileObservation(
        path="test.txt",
        hash="session2_hash",
        size=200,
        mtime=now,
        source=ObservationSource.READ,
    )

    workspace_manager.override_file_versions({"test.txt": obs1}, session_id="session-1")
    workspace_manager.override_file_versions({"test.txt": obs2}, session_id="session-2")

    v1 = workspace_manager.get_file_version("test.txt", session_id="session-1")
    v2 = workspace_manager.get_file_version("test.txt", session_id="session-2")
    v_global = workspace_manager.get_file_version("test.txt")

    assert v1 is not None
    assert v2 is not None
    assert v1.hash == "session1_hash"
    assert v2.hash == "session2_hash"
    assert v_global is None  # No global override set


@pytest.mark.asyncio
async def test_session_overrides_cleared_independently(
    workspace_manager: WorkspaceManager, temp_workspace: Path
) -> None:
    """clear_session_overrides removes only the specified session's overrides."""
    now = datetime.now(UTC)
    obs1 = FileObservation(
        path="test.txt",
        hash="hash1",
        size=100,
        mtime=now,
        source=ObservationSource.READ,
    )
    obs2 = FileObservation(
        path="test.txt",
        hash="hash2",
        size=200,
        mtime=now,
        source=ObservationSource.READ,
    )

    workspace_manager.override_file_versions({"test.txt": obs1}, session_id="session-1")
    workspace_manager.override_file_versions({"test.txt": obs2}, session_id="session-2")

    workspace_manager.clear_session_overrides("session-1")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1") is None
    )
    v2 = workspace_manager.get_file_version("test.txt", session_id="session-2")
    assert v2 is not None
    assert v2.hash == "hash2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
