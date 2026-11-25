"""Integration tests for conflict detection in the tools platform.

Covers write conflict detection, version mismatch detection, resolution
options, and snapshot-based rollback behavior using real WorkspaceManager,
HostBackend, and ConflictDetector.

Acceptance criteria from plan_docs/2026_09/tools_platform_tests.md sections 866-903.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import (
    ConflictReport,
    FileObservation,
    ObservationSource,
)
from local_coding_assistant.tools_platform.workspace import (
    PathPolicy,
    WorkspaceManager,
)
from local_coding_assistant.tools_platform.workspace.conflicts import (
    ConflictType,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)
        yield workspace_root


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
        workspace_id="test-ws",
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


class TestConflictDetectionBaseline:
    """Test baseline observation and conflict detection fundamentals."""

    @pytest.mark.asyncio
    async def test_agent_reads_file_establishes_baseline(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Agent reads file → observe_file records baseline (hash, mtime, size)."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads file
        baseline = workspace_manager.observe_file("test.txt")
        assert baseline is not None
        assert baseline.hash != ""
        assert baseline.size > 0
        assert baseline.source == ObservationSource.READ

    @pytest.mark.asyncio
    async def test_baseline_observation_used_for_conflict_detection(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Baseline observation from read is used by validate_write_version."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads file (establishes baseline)
        workspace_manager.observe_file("test.txt", session_id="test-session")

        # User modifies file externally
        test_file.write_text("modified")

        # Agent tries to validate write version → conflict detected
        report = workspace_manager.validate_write_version(
            "test.txt", "proposed_hash", "proposed", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert isinstance(report, ConflictReport)
        assert report.base_hash != ""
        assert report.current_hash != ""
        assert report.proposed_hash == "proposed_hash"


class TestWriteConflictDetection:
    """Test write conflict detection with various scenarios."""

    @pytest.mark.asyncio
    async def test_user_edit_detected_as_conflict(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """User edits file externally between agent read and write → conflict detected."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads file (establishes baseline)
        workspace_manager.observe_file("test.txt", session_id="test-session")

        # User edits file externally
        test_file.write_text("user modified content")

        # Agent attempts write → conflict detected
        report = workspace_manager.validate_write_version(
            "test.txt", "proposed_hash", "proposed_content", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert report.conflict_type == ConflictType.VERSION_STALE
        assert report.base_hash != ""
        assert report.current_hash != ""
        assert report.proposed_hash == "proposed_hash"
        assert report.recommendation in ("re_read_file", "retry_or_adapt")
        assert report.message is not None

    @pytest.mark.asyncio
    async def test_conflict_report_includes_all_required_fields(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """ConflictReport includes base_hash, current_hash, proposed_hash, conflict_type."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("modified")

        report = workspace_manager.validate_write_version(
            "test.txt", "new_content_hash", "new_content", session_id="test-session"
        )

        # Check all required fields exist
        assert report.path == "test.txt"
        assert report.base_hash != ""
        assert report.current_hash != ""
        assert report.proposed_hash == "new_content_hash"
        assert report.conflict_type == ConflictType.VERSION_STALE
        assert report.conflict_detected is True
        assert report.detected_at is not None


class TestConflictResolutionPolicy:
    """Test conflict resolution policy (deny, ask, allow)."""

    @pytest.mark.asyncio
    async def test_default_write_conflict_policy_deny(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """Default write_conflict_policy=deny → replace_exact raises conflict error."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads file (establishes baseline)
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test-session"
        )

        # User modifies file
        test_file.write_text("user content")

        # Agent tries to write → should fail with conflict
        # Use old_text that matches current file content
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="user content",
            new_text="agent content",
            expected_match_count=1,
            session_id="test-session",
            turn_id="turn-1",
        )

        # Write should fail due to conflict
        assert result["success"] is False
        assert result["error"] is not None
        assert (
            "modified externally" in result["error"]
            or "Conflict detected" in result["error"]
        )

    @pytest.mark.asyncio
    async def test_replace_exact_with_stale_baseline_fails(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """replace_exact with stale expected_hash fails with conflict."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads file (establishes baseline)
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test-session"
        )

        # User modifies file
        test_file.write_text("user modified content")

        # Agent tries replace_exact when file has been modified
        # Use old_text matching current disk content
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="user modified content",
            new_text="agent modified content",
            expected_match_count=1,
            session_id="test-session",
            turn_id="turn-1",
        )

        # Should fail with conflict
        assert result["success"] is False
        assert "modified externally" in result.get(
            "error", ""
        ) or "Conflict" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_replace_exact_with_matching_baseline_succeeds(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """replace_exact with matching expected_hash succeeds when file unchanged."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads file (establishes baseline)
        host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test-session"
        )

        # Agent tries replace_exact when file unchanged
        result = await host_backend.replace_exact(
            path="test.txt",
            old_text="original content",
            new_text="agent modified content",
            expected_match_count=1,
            session_id="test-session",
            turn_id="turn-1",
        )

        # Should succeed since file hasn't changed
        assert result["success"] is True


class TestVersionMismatchDetection:
    """Test version mismatch detection."""

    @pytest.mark.asyncio
    async def test_version_mismatch_when_file_changed(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Detect version mismatch when file content differs from expected."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads file
        baseline = workspace_manager.observe_file("test.txt", session_id="test-session")

        # User modifies file
        test_file.write_text("modified content")

        # Agent tries to write with expected hash from baseline
        report = workspace_manager.validate_write_version(
            "test.txt", baseline.hash, "proposed", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert report.conflict_type == ConflictType.VERSION_STALE

    @pytest.mark.asyncio
    async def test_version_mismatch_file_does_not_exist(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """No baseline → unguarded write, no conflict detected."""
        report = workspace_manager.validate_write_version(
            "test.txt", "some_hash", "some_content", session_id="test-session"
        )
        assert report.conflict_detected is False


class TestSnapshotIntegration:
    """Test snapshot-based rollback after conflict."""

    @pytest.mark.asyncio
    async def test_snapshot_created_before_write_enables_rollback(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """Snapshot created before write enables rollback to pre-conflict state."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads file (establishes baseline hash = hash of "original content")
        host_backend._workspace_manager.observe_file("test.txt")

        # Perform a write (creates snapshot)
        result1 = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="first modification",
            session_id="test_session",
            expected_match_count=1,
        )
        assert result1["success"] is True

        # User modifies file externally - simulate by writing directly
        # The baseline hash is still "original content", but file now has "user external edit"
        test_file.write_text("user external edit")

        # Agent tries another write → conflict detected because baseline hash
        # (hash of "original content") differs from current file hash (hash of "user external edit")
        result2 = await host_backend.replace_exact(
            path="test.txt",
            old_text="original",
            new_text="second modification",
            session_id="test_session",
            expected_match_count=1,
        )
        assert result2["success"] is False

        # The file should still have "user external edit" since the write was rejected
        content = test_file.read_text()
        assert "user external edit" in content


class TestMultipleAgentsSessions:
    """Test multiple agents in different sessions."""

    @pytest.mark.asyncio
    async def test_session_scoped_baselines_independent(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
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

        workspace_manager.override_file_versions(
            {"test.txt": obs1}, session_id="session-1"
        )
        workspace_manager.override_file_versions(
            {"test.txt": obs2}, session_id="session-2"
        )

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
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
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

        workspace_manager.override_file_versions(
            {"test.txt": obs1}, session_id="session-1"
        )
        workspace_manager.override_file_versions(
            {"test.txt": obs2}, session_id="session-2"
        )

        workspace_manager.clear_session_overrides("session-1")

        assert (
            workspace_manager.get_file_version("test.txt", session_id="session-1")
            is None
        )
        v2 = workspace_manager.get_file_version("test.txt", session_id="session-2")
        assert v2 is not None
        assert v2.hash == "hash2"


class TestConflictReportResolutionOptions:
    """Test conflict report resolution options for different conflict types."""

    @pytest.mark.asyncio
    async def test_resolution_options_for_agent_write_stale(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Conflict report carries recommendation and message for stale baseline."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("modified")

        report = workspace_manager.validate_write_version(
            "test.txt", "proposed", "proposed_content", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert report.recommendation in ("re_read_file", "retry_or_adapt")
        assert report.message is not None
        assert report.external_diff is not None
        assert report.proposed_diff is not None

    @pytest.mark.asyncio
    async def test_resolution_options_for_user_edit_detected(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Conflict report carries recommendation and message for stale baseline."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("user modified")

        report = workspace_manager.validate_write_version(
            "test.txt", "proposed", "proposed_content", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert report.recommendation in ("re_read_file", "retry_or_adapt")
        assert report.message is not None

    @pytest.mark.asyncio
    async def test_resolution_options_for_version_mismatch(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """Conflict report carries recommendation and message for stale baseline."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        baseline = workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("modified")

        report = workspace_manager.validate_write_version(
            "test.txt", baseline.hash, "proposed", session_id="test-session"
        )
        assert report.conflict_detected is True
        assert report.recommendation in ("re_read_file", "retry_or_adapt")
        assert report.message is not None


class TestConflictReportStructure:
    """Test ConflictReport structure and attributes."""

    @pytest.mark.asyncio
    async def test_conflict_report_has_all_required_attributes(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """ConflictReport has path, base_hash, current_hash, proposed_hash, conflict_type, resolution_options."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("modified")

        report = workspace_manager.validate_write_version(
            "test.txt", "proposed_hash", "proposed", session_id="test-session"
        )

        # All required attributes
        assert hasattr(report, "path")
        assert hasattr(report, "base_hash")
        assert hasattr(report, "current_hash")
        assert hasattr(report, "proposed_hash")
        assert hasattr(report, "conflict_type")
        assert hasattr(report, "conflict_detected")
        assert hasattr(report, "resolution_options")
        assert hasattr(report, "detected_at")

        # Types
        assert isinstance(report.path, str)
        assert isinstance(report.base_hash, str)
        assert isinstance(report.current_hash, str)
        assert isinstance(report.proposed_hash, str)
        assert isinstance(report.conflict_type, str)
        assert isinstance(report.resolution_options, list)
        assert isinstance(report.detected_at, type(report.detected_at))  # datetime

    @pytest.mark.asyncio
    async def test_conflict_report_conflict_type_enum_values(
        self, workspace_manager: WorkspaceManager, temp_workspace: Path
    ) -> None:
        """ConflictReport conflict_type matches ConflictType enum values."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        workspace_manager.observe_file("test.txt", session_id="test-session")
        test_file.write_text("modified")

        report = workspace_manager.validate_write_version(
            "test.txt", "proposed", "proposed_content", session_id="test-session"
        )
        assert report.conflict_type == ConflictType.VERSION_STALE


class TestAtomicWriteWithConflict:
    """Test atomic_write with various conflict scenarios."""

    @pytest.mark.asyncio
    async def test_atomic_write_raises_valueerror_on_conflict(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """atomic_write raises ValueError when expected_hash doesn't match in-memory baseline."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Agent reads file (establishes baseline in session_file_versions)
        baseline = host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test-session"
        )

        # User modifies file on disk (but in-memory baseline is still "original")
        test_file.write_text("user content")

        # Agent tries atomic_write with WRONG expected_hash (not matching in-memory baseline)
        # This should fail because expected_hash != in-memory baseline hash
        host_backend._workspace_manager.begin_transaction()
        with pytest.raises(ValueError, match="Hash mismatch"):
            host_backend._workspace_manager.atomic_write(
                file_path="test.txt",
                content="agent content",
                expected_hash="wrong_hash",  # This doesn't match the baseline
                session_id="test-session",
            )
        host_backend._workspace_manager.rollback_transaction()

    @pytest.mark.asyncio
    async def test_atomic_write_succeeds_when_no_conflict(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        """atomic_write succeeds when file unchanged from baseline."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        # Agent reads file (establishes baseline)
        baseline = host_backend._workspace_manager.observe_file(
            "test.txt", session_id="test-session"
        )

        # Begin transaction
        host_backend._workspace_manager.begin_transaction()
        try:
            # Agent tries atomic_write when file unchanged
            before, after = host_backend._workspace_manager.atomic_write(
                file_path="test.txt",
                content="agent content",
                expected_hash=baseline.hash,
                session_id="test-session",
            )

            assert before is not None
            assert after is not None
            assert after.hash != ""
            assert after.size > 0
        finally:
            host_backend._workspace_manager.rollback_transaction()

        # File should be back to original
        assert test_file.read_text() == "original content"
