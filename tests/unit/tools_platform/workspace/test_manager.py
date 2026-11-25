"""Unit tests for WorkspaceManager.

Covers the test_workspace_manager cases from
plan_docs/2026_09/tools_platform_tests.md:

- Path Policy
- File Observation & Baselines
- Transactions
- Atomic Operations with Conflict Detection
- Snapshots
- Session-Scoped Overrides
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.contracts import (
    ChangeSet,
    ConflictReport,
    FileObservation,
    ObservationSource,
)
from local_coding_assistant.tools_platform.exceptions import ToolPathViolationError
from local_coding_assistant.tools_platform.workspace.line_normalization import (
    LineEnding,
)
from local_coding_assistant.tools_platform.workspace import (
    ConflictType,
    PathPolicy,
    WorkspaceManager,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_manager(temp_workspace: Path, **overrides: object) -> WorkspaceManager:
    """Build a WorkspaceManager directly with a custom path policy."""
    policy_kwargs: dict[str, object] = {
        "workspace_root": temp_workspace,
        "allowed_patterns": ["**"],
        "denied_patterns": [],
        "allow_symlinks": False,
        "max_depth": None,
    }
    policy_kwargs.update(overrides)
    policy = PathPolicy(**policy_kwargs)  # type: ignore[arg-type]
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=temp_workspace,
        path_policy=policy,
        config_manager=None,
    )


# ---------------------------------------------------------------------------
# Path Policy
# ---------------------------------------------------------------------------


def test_resolve_path_relative(temp_workspace: Path) -> None:
    """resolve_path converts relative path to absolute within workspace_root."""
    manager = _make_manager(temp_workspace)
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    resolved = manager.resolve_path("test.txt")
    assert resolved == test_file


def test_resolve_path_absolute_within_workspace(temp_workspace: Path) -> None:
    """resolve_path accepts absolute paths within workspace_root."""
    manager = _make_manager(temp_workspace)
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    resolved = manager.resolve_path(test_file)
    assert resolved == test_file


def test_resolve_path_outside_workspace(temp_workspace: Path) -> None:
    """resolve_path rejects absolute path outside workspace_root with ToolPathViolationError."""
    manager = _make_manager(temp_workspace)
    outside_file = temp_workspace.parent / "outside.txt"

    with pytest.raises(ToolPathViolationError, match="outside workspace"):
        manager.resolve_path(outside_file)


def test_resolve_path_denied_patterns(temp_workspace: Path) -> None:
    """resolve_path rejects path matching denied_patterns with ToolPathViolationError."""
    manager = _make_manager(temp_workspace, denied_patterns=["*.tmp", "temp/*"])

    with pytest.raises(ToolPathViolationError, match="denied pattern"):
        manager.resolve_path("test.tmp")


def test_resolve_path_allowed_patterns(temp_workspace: Path) -> None:
    """resolve_path rejects path not matching any allowed_patterns with ToolPathViolationError."""
    manager = _make_manager(temp_workspace, allowed_patterns=["*.py", "*.txt"])

    with pytest.raises(
        ToolPathViolationError, match="does not match any allowed pattern"
    ):
        manager.resolve_path("test.js")


def test_resolve_path_depth_limit(temp_workspace: Path) -> None:
    """resolve_path rejects path with depth > max_depth with ToolPathViolationError."""
    manager = _make_manager(temp_workspace, max_depth=2)

    with pytest.raises(ToolPathViolationError, match="exceeds maximum depth"):
        manager.resolve_path("a/b/c/test.txt")


def test_resolve_path_symlinks_disallowed(temp_workspace: Path) -> None:
    """resolve_path rejects symlink when allow_symlinks=false with ToolPathViolationError."""
    manager = _make_manager(temp_workspace)
    target_file = temp_workspace / "target.txt"
    target_file.write_text("content")
    symlink = temp_workspace / "link.txt"
    try:
        symlink.symlink_to(target_file)
    except OSError:
        pytest.skip("symlink creation not supported on this platform")

    with pytest.raises(ToolPathViolationError, match="Symlinks are not allowed"):
        manager.resolve_path("link.txt")


def test_is_path_inside_workspace(temp_workspace: Path) -> None:
    """is_path_inside_workspace returns True within workspace, False otherwise."""
    manager = _make_manager(temp_workspace)
    inside = temp_workspace / "sub" / "file.txt"
    outside = temp_workspace.parent / "file.txt"

    assert manager.is_path_inside_workspace(inside) is True
    assert manager.is_path_inside_workspace(outside) is False
    assert manager.is_path_inside_workspace("rel.txt") is True


# ---------------------------------------------------------------------------
# File Observation & Baselines
# ---------------------------------------------------------------------------


def test_observe_file(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """observe_file returns FileObservation with hash, size, mtime, inode, source=READ."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")

    observation = workspace_manager.observe_file("test.txt")

    assert observation.path == "test.txt"
    assert observation.hash != ""
    assert observation.size == len("content")
    assert observation.mtime is not None
    assert observation.inode is not None
    assert observation.observed_at is not None
    assert observation.source == ObservationSource.READ


def test_observe_nonexistent_file(workspace_manager: WorkspaceManager) -> None:
    """observe_file on nonexistent file returns size=0, hash="", source=READ."""
    observation = workspace_manager.observe_file("nonexistent.txt")

    assert observation.path == "nonexistent.txt"
    assert observation.size == 0
    assert observation.hash == ""
    assert observation.source == ObservationSource.READ


def test_get_file_version_session_scoped(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """get_file_version returns session-scoped observation set via override_file_versions."""
    obs1 = FileObservation(
        path="test.txt",
        hash="hash1",
        size=100,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    obs2 = FileObservation(
        path="test.txt",
        hash="hash2",
        size=200,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )

    workspace_manager.override_file_versions({"test.txt": obs1}, session_id="session-1")
    workspace_manager.override_file_versions({"test.txt": obs2}, session_id="session-2")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1").hash
        == "hash1"
    )
    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-2").hash
        == "hash2"
    )
    assert workspace_manager.get_file_version("test.txt") is None


def test_clear_session_overrides(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """clear_session_overrides removes session-scoped versions."""
    obs = FileObservation(
        path="test.txt",
        hash="hash1",
        size=100,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    workspace_manager.override_file_versions({"test.txt": obs}, session_id="session-1")
    workspace_manager.clear_session_overrides("session-1")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1") is None
    )


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def test_begin_transaction(workspace_manager: WorkspaceManager) -> None:
    """begin_transaction returns transaction_id, sets active transaction."""
    tx_id = workspace_manager.begin_transaction()
    assert tx_id is not None
    assert workspace_manager.get_active_transaction_id() == tx_id


def test_commit_transaction(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """commit_transaction returns ChangeSet, clears active transaction."""
    workspace_manager.begin_transaction()
    workspace_manager.atomic_create("new.txt", "content", session_id="test")
    changeset = workspace_manager.commit_transaction()

    assert isinstance(changeset, ChangeSet)
    assert changeset.total_files_changed == 1
    assert workspace_manager.get_active_transaction_id() is None
    assert (temp_workspace / "new.txt").read_text() == "content"


def test_rollback_transaction(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """rollback_transaction clears active transaction, discards pending changes."""
    workspace_manager.begin_transaction()
    workspace_manager.atomic_create("new.txt", "content", session_id="test")
    workspace_manager.rollback_transaction()

    assert workspace_manager.get_active_transaction_id() is None
    assert not (temp_workspace / "new.txt").exists()


def test_nested_transactions_not_supported(workspace_manager: WorkspaceManager) -> None:
    """second begin_transaction errors when one is already active."""
    workspace_manager.begin_transaction()
    with pytest.raises(RuntimeError, match="Transaction already active"):
        workspace_manager.begin_transaction()
    workspace_manager.rollback_transaction()


# ---------------------------------------------------------------------------
# Atomic Operations with Conflict Detection
# ---------------------------------------------------------------------------


def test_atomic_write_correct_hash(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write with correct expected_hash writes content, returns (before, after)."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")
    current_hash = _sha256("original")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    before_obs, after_obs = workspace_manager.atomic_write(
        "test.txt", "new content", expected_hash=current_hash, session_id="test"
    )

    assert before_obs is not None
    assert before_obs.hash == current_hash
    assert after_obs.hash != current_hash
    assert after_obs.hash == _sha256("new content")
    assert test_file.read_text() == "new content"
    workspace_manager.commit_transaction()


def test_atomic_write_wrong_hash(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write with wrong expected_hash raises ValueError (hash mismatch)."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    with pytest.raises(ValueError, match="Hash mismatch"):
        workspace_manager.atomic_write(
            "test.txt", "new content", expected_hash="wrong_hash", session_id="test"
        )
    workspace_manager.rollback_transaction()

    assert test_file.read_text() == "original"


def test_atomic_write_without_expected_hash(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write without expected_hash uses current on-disk hash as baseline."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    before_obs, after_obs = workspace_manager.atomic_write(
        "test.txt", "new content", session_id="test"
    )

    assert before_obs is not None
    assert before_obs.hash == _sha256("original")
    assert after_obs.hash == _sha256("new content")
    assert test_file.read_text() == "new content"
    workspace_manager.commit_transaction()


def test_atomic_write_content_matches_proposed(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write verifies written content matches proposed_hash after commit."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")
    proposed_hash = _sha256("new content")

    workspace_manager.begin_transaction()
    _, after_obs = workspace_manager.atomic_write(
        "test.txt", "new content", session_id="test"
    )
    workspace_manager.commit_transaction()

    assert test_file.read_text() == "new content"
    assert after_obs.hash == proposed_hash


def test_atomic_write_rollback_restores(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write on rollback restores file to pre-transaction state from snapshot."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    workspace_manager.atomic_write("test.txt", "new content", session_id="test")
    assert test_file.read_text() == "new content"

    workspace_manager.rollback_transaction()
    assert test_file.read_text() == "original"


def test_atomic_write_creates_snapshot(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_write creates a snapshot before writing an existing file."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("original")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    workspace_manager.atomic_write("test.txt", "new content", session_id="test")
    workspace_manager.commit_transaction()

    history = workspace_manager.transaction_manager.get_transaction_history()
    tx = list(history.values())[-1]
    write_entry = tx.entries[0]
    snapshot_id = write_entry.metadata.get("snapshot_id")
    assert snapshot_id is not None
    # A file snapshot was created on disk before the write.
    snapshot_file = workspace_manager.snapshot_manager._get_file_snapshot_path(
        snapshot_id, "test.txt"
    )
    assert snapshot_file.exists()


def test_atomic_create(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_create creates new file, returns after_obs, fails with FileExistsError if exists."""
    workspace_manager.begin_transaction()
    after_obs = workspace_manager.atomic_create("new.txt", "content", session_id="test")

    assert after_obs.path == "new.txt"
    assert after_obs.hash != ""
    assert (temp_workspace / "new.txt").read_text() == "content"
    workspace_manager.commit_transaction()


def test_atomic_create_exists(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_create fails with FileExistsError if file exists."""
    (temp_workspace / "test.txt").write_text("original")

    workspace_manager.begin_transaction()
    with pytest.raises(FileExistsError):
        workspace_manager.atomic_create("test.txt", "content", session_id="test")
    workspace_manager.rollback_transaction()


def test_atomic_delete(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """atomic_delete removes file, returns before_obs, fails with FileNotFoundError if missing."""
    (temp_workspace / "test.txt").write_text("content")

    workspace_manager.observe_file("test.txt", session_id="test")

    workspace_manager.begin_transaction()
    before_obs = workspace_manager.atomic_delete("test.txt", session_id="test")

    assert before_obs is not None
    assert before_obs.path == "test.txt"
    assert before_obs.hash != ""
    assert not (temp_workspace / "test.txt").exists()
    workspace_manager.commit_transaction()


def test_atomic_delete_nonexistent(workspace_manager: WorkspaceManager) -> None:
    """atomic_delete fails with FileNotFoundError if missing."""
    workspace_manager.begin_transaction()
    with pytest.raises(FileNotFoundError):
        workspace_manager.atomic_delete("nonexistent.txt", session_id="test")
    workspace_manager.rollback_transaction()


def test_validate_write_version_match(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """validate_write_version returns no conflict when disk state matches the read baseline."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")
    workspace_manager.observe_file("test.txt", session_id="test-session")

    # Disk unchanged since the read baseline → no conflict.
    report = workspace_manager.validate_write_version(
        "test.txt", _sha256("new"), "new", session_id="test-session"
    )
    assert report.conflict_detected is False


def test_validate_write_version_conflict(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """validate_write_version returns ConflictReport when disk changed since read baseline."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")
    obs = workspace_manager.observe_file("test.txt", session_id="test-session")

    # External edit after the read → baseline is stale.
    test_file.write_text("modified content")

    report = workspace_manager.validate_write_version(
        "test.txt", _sha256("new"), "new", session_id="test-session"
    )
    assert isinstance(report, ConflictReport)
    assert report.conflict_type == ConflictType.VERSION_STALE
    assert report.conflict_detected is True
    assert report.base_hash == obs.hash
    assert report.current_hash == _sha256("modified content")


def test_validate_write_version_uses_read_baseline(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """validate_write_version compares current disk state against the read baseline."""
    test_file = temp_workspace / "test.txt"
    test_file.write_text("content")
    obs = workspace_manager.observe_file("test.txt", session_id="test-session")

    # Disk unchanged since the read baseline → no conflict.
    report = workspace_manager.validate_write_version(
        "test.txt", _sha256("new"), "new", session_id="test-session"
    )
    assert report.conflict_detected is False

    # Disk modified since the read baseline → conflict.
    test_file.write_text("modified content")
    report = workspace_manager.validate_write_version(
        "test.txt", _sha256("new"), "new", session_id="test-session"
    )
    assert report.conflict_detected is True
    assert report.base_hash == obs.hash
    assert report.current_hash == _sha256("modified content")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_create_snapshot(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """create_snapshot for multiple files returns snapshot_id."""
    (temp_workspace / "a.txt").write_text("a")
    (temp_workspace / "b.txt").write_text("b")

    snapshot_id = workspace_manager.create_snapshot(["a.txt", "b.txt"])

    assert snapshot_id is not None


def test_restore_snapshot(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """restore_snapshot restores file content from snapshot_id."""
    (temp_workspace / "a.txt").write_text("original")
    snapshot_id = workspace_manager.create_snapshot(["a.txt"])

    (temp_workspace / "a.txt").write_text("modified")
    workspace_manager.restore_snapshot(snapshot_id, "a.txt")

    assert (temp_workspace / "a.txt").read_text() == "original"


def test_delete_snapshot(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """delete_snapshot removes snapshot."""
    (temp_workspace / "a.txt").write_text("a")
    snapshot_id = workspace_manager.create_snapshot(["a.txt"])

    workspace_manager.delete_snapshot(snapshot_id)

    snapshot_dir = workspace_manager.snapshot_manager._get_snapshot_path(snapshot_id)
    assert not snapshot_dir.exists()


def test_snapshots_stored_outside_workspace(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """snapshots stored outside workspace tree (configured directory)."""
    snap_dir = workspace_manager.snapshot_manager.config.snapshots_dir
    assert not str(snap_dir).startswith(str(temp_workspace))

    (temp_workspace / "a.txt").write_text("a")
    snapshot_id = workspace_manager.create_snapshot(["a.txt"])
    snapshot_file = workspace_manager.snapshot_manager._get_file_snapshot_path(
        snapshot_id, "a.txt"
    )
    assert not str(snapshot_file).startswith(str(temp_workspace))


# ---------------------------------------------------------------------------
# Session-Scoped Overrides
# ---------------------------------------------------------------------------


def test_override_file_versions_with_session(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """override_file_versions with session_id sets versions scoped to that session."""
    obs = FileObservation(
        path="test.txt",
        hash="hash1",
        size=100,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    workspace_manager.override_file_versions({"test.txt": obs}, session_id="session-1")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1").hash
        == "hash1"
    )


def test_multiple_sessions_have_independent_baselines(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """multiple sessions have independent file version baselines."""
    obs1 = FileObservation(
        path="test.txt",
        hash="h1",
        size=1,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    obs2 = FileObservation(
        path="test.txt",
        hash="h2",
        size=2,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    workspace_manager.override_file_versions({"test.txt": obs1}, session_id="session-1")
    workspace_manager.override_file_versions({"test.txt": obs2}, session_id="session-2")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1").hash
        == "h1"
    )
    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-2").hash
        == "h2"
    )


def test_get_file_version_returns_session_specific(
    temp_workspace: Path, workspace_manager: WorkspaceManager
) -> None:
    """get_file_version(session_id) returns session-specific version."""
    obs = FileObservation(
        path="test.txt",
        hash="session-hash",
        size=5,
        mtime=datetime.now(UTC),
        inode=None,
        source=ObservationSource.READ,
    )
    workspace_manager.override_file_versions({"test.txt": obs}, session_id="session-1")

    assert (
        workspace_manager.get_file_version("test.txt", session_id="session-1").hash
        == "session-hash"
    )


# ---------------------------------------------------------------------------
# Line-ending normalization (WorkspaceManager methods)
# ---------------------------------------------------------------------------


def _make_manager_with_normalization(
    temp_workspace: Path, enabled: bool
) -> WorkspaceManager:
    """Build a WorkspaceManager with normalization toggled via config."""
    from local_coding_assistant.config.config_manager import ConfigManager

    cm = ConfigManager(config_paths=["config/tool_policies.yaml"])
    cfg = cm.load_global_config()
    cfg.tools_platform.workspace_root = str(temp_workspace)
    cfg.tools_platform.snapshots.directory = str(
        temp_workspace.parent / "locca_workspace_snapshots"
    )
    cfg.tools_platform.normalize_to_lf_line_endings = enabled
    return WorkspaceManager.from_config_manager(cm)


def test_workspace_manager_read_file_raw_returns_raw_bytes(
    temp_workspace: Path, workspace_manager_direct: WorkspaceManager
) -> None:
    """read_file_raw returns file content unchanged (raw disk bytes)."""
    path = temp_workspace / "test.py"
    path.write_text("line1\r\nline2\r\n", newline="")

    content = workspace_manager_direct.read_file_raw(path)
    assert content == "line1\r\nline2\r\n"


def test_workspace_manager_read_file_normalized_disabled_passthrough(
    temp_workspace: Path, workspace_manager_direct: WorkspaceManager
) -> None:
    """With config_manager=None normalization is disabled -> passthrough."""
    path = temp_workspace / "test.py"
    path.write_text("line1\r\nline2\r\n", newline="")

    content = workspace_manager_direct.read_file_normalized(path)
    assert content == "line1\r\nline2\r\n"


def test_workspace_manager_read_file_normalized_enabled_converts_to_lf(
    temp_workspace: Path,
) -> None:
    """With normalization enabled, CRLF content is converted to LF."""
    manager = _make_manager_with_normalization(temp_workspace, enabled=True)
    path = temp_workspace / "test.py"
    path.write_text("line1\r\nline2\r\n", newline="")

    content = manager.read_file_normalized(path)
    assert content == "line1\nline2\n"


def test_workspace_manager_normalize_for_match_disabled_passthrough(
    temp_workspace: Path, workspace_manager_direct: WorkspaceManager
) -> None:
    """normalize_for_match is pass-through when disabled."""
    assert workspace_manager_direct.normalize_for_match("a\r\nb") == "a\r\nb"


def test_workspace_manager_normalize_for_match_enabled_converts_to_lf(
    temp_workspace: Path,
) -> None:
    """normalize_for_match converts to LF when enabled."""
    manager = _make_manager_with_normalization(temp_workspace, enabled=True)
    assert manager.normalize_for_match("a\r\nb\r\nc") == "a\nb\nc"


def test_workspace_manager_restore_line_endings_disabled_passthrough(
    temp_workspace: Path, workspace_manager_direct: WorkspaceManager
) -> None:
    """restore_line_endings is pass-through when disabled."""
    assert (
        workspace_manager_direct.restore_line_endings("a\nb\n", LineEnding.CRLF)
        == "a\nb\n"
    )


def test_workspace_manager_restore_line_endings_enabled_converts(
    temp_workspace: Path,
) -> None:
    """restore_line_endings restores LF content to CRLF when enabled."""
    from local_coding_assistant.tools_platform.workspace.line_normalization import (
        LineEnding,
    )

    manager = _make_manager_with_normalization(temp_workspace, enabled=True)
    assert manager.restore_line_endings("a\nb\n", LineEnding.CRLF) == "a\r\nb\r\n"
    assert manager.restore_line_endings("a\nb\n", LineEnding.LF) == "a\nb\n"


def test_workspace_manager_detect_line_ending(temp_workspace: Path) -> None:
    """detect_line_ending delegates to the module function."""
    from local_coding_assistant.tools_platform.workspace.line_normalization import (
        LineEnding,
    )

    manager = _make_manager(temp_workspace)
    assert manager.detect_line_ending("a\r\nb\r\n") == LineEnding.CRLF
    assert manager.detect_line_ending("a\nb\n") == LineEnding.LF
    assert manager.detect_line_ending("a\rc\r") == LineEnding.CR
    assert manager.detect_line_ending("no newlines") == LineEnding.NONE
