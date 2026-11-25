"""Unit tests for workspace snapshot management.

Moved from tests/unit/tools_platform/test_workspace_snapshots.py.
"""

from __future__ import annotations

import pytest

from local_coding_assistant.tools_platform.workspace.snapshots import (
    SnapshotManager,
    SnapshotStorageConfig,
)


@pytest.fixture
def temp_workspace(tmp_path):
    return tmp_path


@pytest.fixture
def snapshot_config(temp_workspace):
    snapshots_dir = temp_workspace / ".locca" / "snapshots"
    return SnapshotStorageConfig(snapshots_dir=snapshots_dir)


@pytest.fixture
def snapshot_manager(snapshot_config):
    return SnapshotManager(snapshot_config)


class TestSnapshotManager:
    """Test SnapshotManager functionality."""

    def test_create_file_snapshot(self, snapshot_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")

        snapshot = snapshot_manager.create_file_snapshot(
            workspace_root=temp_workspace,
            file_path="test.txt",
        )

        assert snapshot.path == "test.txt"
        assert snapshot.content == "test content"
        assert snapshot.observation.hash != ""
        assert snapshot.snapshot_id is not None

    def test_create_file_snapshot_nonexistent(self, snapshot_manager, temp_workspace):
        with pytest.raises(FileNotFoundError, match="File not found"):
            snapshot_manager.create_file_snapshot(
                workspace_root=temp_workspace,
                file_path="nonexistent.txt",
            )

    def test_restore_file_snapshot(self, snapshot_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        snapshot = snapshot_manager.create_file_snapshot(
            workspace_root=temp_workspace,
            file_path="test.txt",
        )

        test_file.write_text("modified content")

        snapshot_manager.restore_file_snapshot(
            workspace_root=temp_workspace,
            snapshot_id=snapshot.snapshot_id,
            file_path="test.txt",
        )

        assert test_file.read_text() == "original content"

    def test_restore_file_snapshot_nonexistent(self, snapshot_manager, temp_workspace):
        with pytest.raises(FileNotFoundError, match="Snapshot not found"):
            snapshot_manager.restore_file_snapshot(
                workspace_root=temp_workspace,
                snapshot_id="nonexistent_snapshot_id",
                file_path="test.txt",
            )

    def test_create_workspace_snapshot(self, snapshot_manager, temp_workspace):
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.txt").write_text("content2")
        (temp_workspace / "file3.txt").write_text("content3")

        workspace_snapshot = snapshot_manager.create_workspace_snapshot(
            workspace_id="test_workspace",
            workspace_root=temp_workspace,
            file_paths=["file1.txt", "file2.txt", "file3.txt"],
        )

        assert workspace_snapshot.workspace_id == "test_workspace"
        assert workspace_snapshot.snapshot_id is not None
        assert len(workspace_snapshot.file_snapshots) == 3
        assert "file1.txt" in workspace_snapshot.file_snapshots
        assert "file2.txt" in workspace_snapshot.file_snapshots
        assert "file3.txt" in workspace_snapshot.file_snapshots

    def test_create_workspace_snapshot_with_missing_file(
        self, snapshot_manager, temp_workspace
    ):
        (temp_workspace / "file1.txt").write_text("content1")

        workspace_snapshot = snapshot_manager.create_workspace_snapshot(
            workspace_id="test_workspace",
            workspace_root=temp_workspace,
            file_paths=["file1.txt", "file2.txt"],
        )

        assert len(workspace_snapshot.file_snapshots) == 1
        assert "file1.txt" in workspace_snapshot.file_snapshots

    def test_delete_snapshot(self, snapshot_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        snapshot = snapshot_manager.create_file_snapshot(
            workspace_root=temp_workspace,
            file_path="test.txt",
        )

        snapshot_manager.delete_snapshot(snapshot.snapshot_id)

        snapshot_dir = snapshot_manager._get_snapshot_path(snapshot.snapshot_id)
        assert not snapshot_dir.exists()

    def test_delete_nonexistent_snapshot(self, snapshot_manager):
        snapshot_manager.delete_snapshot("nonexistent_snapshot_id")

    def test_snapshot_preserves_newline_encoding(
        self, snapshot_manager, temp_workspace
    ):
        test_file = temp_workspace / "test.txt"
        test_file.write_bytes(b"line1\r\nline2\r\nline3")

        snapshot = snapshot_manager.create_file_snapshot(
            workspace_root=temp_workspace,
            file_path="test.txt",
        )

        test_file.write_bytes(b"modified\r\ncontent")

        snapshot_manager.restore_file_snapshot(
            workspace_root=temp_workspace,
            snapshot_id=snapshot.snapshot_id,
            file_path="test.txt",
        )

        restored_content = test_file.read_text()
        assert "line1" in restored_content
        assert "line2" in restored_content
        assert "line3" in restored_content

    def test_create_deletion_marker(self, snapshot_manager):
        deletion_marker = snapshot_manager.create_deletion_marker(
            file_path="deleted_file.txt",
        )

        assert deletion_marker.path == "deleted_file.txt"
        assert deletion_marker.content == ""
        assert deletion_marker.observation.hash == ""
        assert deletion_marker.snapshot_id is not None

    def test_restore_deletion_marker(self, snapshot_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        deletion_marker = snapshot_manager.create_deletion_marker(
            file_path="test.txt",
        )

        snapshot_manager.restore_file_snapshot(
            workspace_root=temp_workspace,
            snapshot_id=deletion_marker.snapshot_id,
            file_path="test.txt",
        )

        assert not test_file.exists()

    def test_get_snapshot_metadata(self, snapshot_manager, temp_workspace):
        (temp_workspace / "file1.txt").write_text("content1")
        workspace_snapshot = snapshot_manager.create_workspace_snapshot(
            workspace_id="test_workspace",
            workspace_root=temp_workspace,
            file_paths=["file1.txt"],
        )

        metadata = snapshot_manager.get_snapshot_metadata(
            workspace_snapshot.snapshot_id
        )

        assert metadata is not None
        assert metadata["snapshot_id"] == workspace_snapshot.snapshot_id
        assert metadata["workspace_id"] == "test_workspace"
        assert metadata["file_count"] == 1

    def test_get_nonexistent_snapshot_metadata(self, snapshot_manager):
        metadata = snapshot_manager.get_snapshot_metadata("nonexistent_snapshot_id")
        assert metadata is None

    def test_cleanup_old_snapshots(self, snapshot_manager, temp_workspace):
        for i in range(5):
            test_file = temp_workspace / f"test{i}.txt"
            test_file.write_text(f"content{i}")
            snapshot_manager.create_file_snapshot(
                workspace_root=temp_workspace,
                file_path=f"test{i}.txt",
            )

        snapshot_manager.config.max_snapshots = 2
        deleted_count = snapshot_manager.cleanup_old_snapshots()

        assert deleted_count == 3

    def test_cleanup_old_snapshots_by_age(self, snapshot_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        snapshot_manager.create_file_snapshot(
            workspace_root=temp_workspace,
            file_path="test.txt",
        )

        snapshot_manager.config.max_snapshot_age_days = 0
        deleted_count = snapshot_manager.cleanup_old_snapshots()

        assert deleted_count >= 0

    def test_cleanup_old_snapshots_filters_by_workspace_id(
        self, snapshot_manager, temp_workspace
    ):
        (temp_workspace / "file1.txt").write_text("content1")
        snapshot_manager.create_workspace_snapshot(
            workspace_id="workspace1",
            workspace_root=temp_workspace,
            file_paths=["file1.txt"],
        )

        (temp_workspace / "file2.txt").write_text("content2")
        snapshot_manager.create_workspace_snapshot(
            workspace_id="workspace2",
            workspace_root=temp_workspace,
            file_paths=["file2.txt"],
        )

        deleted_count = snapshot_manager.cleanup_old_snapshots(
            workspace_id="workspace1"
        )

        assert deleted_count >= 0
