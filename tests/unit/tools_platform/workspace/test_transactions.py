"""Unit tests for workspace transaction management.

Moved from tests/unit/tools_platform/test_workspace_transactions.py.
"""

from __future__ import annotations

import pytest

from local_coding_assistant.tools_platform.workspace.content_store import (
    BaselineStore,
    BaselineStoreConfig,
)
from local_coding_assistant.tools_platform.workspace.snapshots import (
    SnapshotManager,
    SnapshotStorageConfig,
)
from local_coding_assistant.tools_platform.workspace.transactions import (
    TransactionConfig,
    TransactionManager,
)


@pytest.fixture
def temp_workspace(tmp_path):
    return tmp_path


@pytest.fixture
def snapshot_config(temp_workspace):
    snapshots_dir = temp_workspace / ".locca" / "snapshots"
    return SnapshotStorageConfig(snapshots_dir=snapshots_dir)


@pytest.fixture
def baseline_store(temp_workspace):
    baselines_dir = temp_workspace / ".locca" / "baselines"
    config = BaselineStoreConfig(base_dir=baselines_dir)
    return BaselineStore(config)


@pytest.fixture
def transaction_manager(temp_workspace, snapshot_config, baseline_store):
    snapshot_manager = SnapshotManager(snapshot_config)
    return TransactionManager(
        workspace_id="test_workspace",
        workspace_root=temp_workspace,
        snapshot_manager=snapshot_manager,
        baseline_store=baseline_store,
        config=TransactionConfig(enable_snapshots=True),
    )


class TestTransactionManager:
    """Test TransactionManager functionality."""

    def test_begin_transaction(self, transaction_manager):
        transaction = transaction_manager.begin_transaction()

        assert transaction.transaction_id is not None
        assert transaction.workspace_id == "test_workspace"
        assert transaction.state.value == "active"
        assert transaction.started_at is not None

    def test_begin_transaction_fails_when_active(self, transaction_manager):
        transaction_manager.begin_transaction()

        with pytest.raises(RuntimeError, match="Transaction already active"):
            transaction_manager.begin_transaction()

    def test_commit_transaction(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        transaction_manager.begin_transaction()
        transaction_manager.atomic_write(test_file, "new content")
        changeset = transaction_manager.commit_transaction()

        assert changeset.total_files_changed == 1
        assert len(changeset.changes) == 1
        assert changeset.changes[0].path == "test.txt"
        assert changeset.changes[0].change_type == "modify"
        assert test_file.read_text() == "new content"

    def test_commit_transaction_fails_when_none_active(self, transaction_manager):
        with pytest.raises(RuntimeError, match="No active transaction to commit"):
            transaction_manager.commit_transaction()

    def test_rollback_transaction(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        transaction_manager.begin_transaction()
        transaction_manager.atomic_write(test_file, "new content")
        transaction_manager.rollback_transaction()

        assert test_file.read_text() == "original content"

    def test_rollback_transaction_fails_when_none_active(self, transaction_manager):
        with pytest.raises(RuntimeError, match="No active transaction to rollback"):
            transaction_manager.rollback_transaction()

    def test_atomic_write_creates_file(self, transaction_manager, temp_workspace):
        transaction_manager.begin_transaction()
        test_file = temp_workspace / "new_file.txt"

        before_obs, after_obs = transaction_manager.atomic_write(
            test_file, "new content"
        )

        assert before_obs is None
        assert after_obs.hash != ""
        assert after_obs.size == len("new content")
        assert (temp_workspace / "new_file.txt").exists()
        assert (temp_workspace / "new_file.txt").read_text() == "new content"

    def test_atomic_write_with_expected_hash(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")
        original_hash = transaction_manager.observe_file(test_file).hash

        transaction_manager.begin_transaction()
        before_obs, after_obs = transaction_manager.atomic_write(
            test_file, "new content", expected_hash=original_hash
        )

        assert before_obs is not None
        assert before_obs.hash == original_hash
        assert after_obs.hash != original_hash

    def test_atomic_write_with_wrong_hash(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        transaction_manager.begin_transaction()
        with pytest.raises(ValueError, match="Hash mismatch"):
            transaction_manager.atomic_write(
                test_file, "new content", expected_hash="wrong_hash"
            )

    def test_atomic_create(self, transaction_manager, temp_workspace):
        transaction_manager.begin_transaction()
        test_file = temp_workspace / "new_file.txt"
        after_obs = transaction_manager.atomic_create(test_file, "content")

        assert after_obs.hash != ""
        assert after_obs.size == len("content")
        assert (temp_workspace / "new_file.txt").exists()
        assert (temp_workspace / "new_file.txt").read_text() == "content"

    def test_atomic_create_fails_if_exists(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("existing content")

        transaction_manager.begin_transaction()
        with pytest.raises(FileExistsError, match="File already exists"):
            transaction_manager.atomic_create(test_file, "new content")

    def test_atomic_delete(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content to delete")

        transaction_manager.begin_transaction()
        before_obs = transaction_manager.atomic_delete(test_file)

        assert before_obs is not None
        assert before_obs.hash != ""
        assert before_obs.size == len("content to delete")
        assert not test_file.exists()

    def test_atomic_delete_fails_if_not_exists(
        self, transaction_manager, temp_workspace
    ):
        transaction_manager.begin_transaction()
        nonexistent_file = temp_workspace / "nonexistent.txt"
        with pytest.raises(FileNotFoundError, match="File not found"):
            transaction_manager.atomic_delete(nonexistent_file)

    def test_observe_file(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")

        observation = transaction_manager.observe_file(test_file)

        assert observation.path == "test.txt"
        assert observation.hash != ""
        assert observation.size == len("test content")
        assert observation.mtime is not None

    def test_observe_nonexistent_file(self, transaction_manager, temp_workspace):
        nonexistent_file = temp_workspace / "nonexistent.txt"
        observation = transaction_manager.observe_file(nonexistent_file)

        assert observation.path == "nonexistent.txt"
        assert observation.hash == ""
        assert observation.size == 0

    def test_get_workspace_state(self, transaction_manager, temp_workspace):
        file1_file = temp_workspace / "file1.txt"
        file2_file = temp_workspace / "file2.txt"
        file1_file.write_text("content1")
        file2_file.write_text("content2")

        transaction_manager.observe_file(file1_file)
        transaction_manager.observe_file(file2_file)

        state = transaction_manager.get_workspace_state()

        assert state.workspace_id == "test_workspace"
        assert "file1.txt" in state.session_file_versions.get(None, {}) or any(
            "file1.txt" in versions for versions in state.session_file_versions.values()
        )
        assert "file2.txt" in state.session_file_versions.get(None, {}) or any(
            "file2.txt" in versions for versions in state.session_file_versions.values()
        )

    def test_transaction_journal_entries(self, transaction_manager, temp_workspace):
        transaction_manager.begin_transaction()
        file1_file = temp_workspace / "file1.txt"

        transaction_manager.atomic_create(file1_file, "content1")
        transaction_manager.atomic_write(file1_file, "content2")

        transaction = transaction_manager.get_active_transaction()

        assert transaction is not None
        assert len(transaction.entries) == 2
        assert transaction.entries[0].operation == "create"
        assert transaction.entries[1].operation == "write"

    def test_transaction_history(self, transaction_manager, temp_workspace):
        test_file = temp_workspace / "test.txt"
        transaction_manager.begin_transaction()
        transaction_manager.atomic_create(test_file, "content")
        transaction_manager.commit_transaction()

        history = transaction_manager.get_transaction_history()

        assert len(history) == 1
        transaction_id = next(iter(history.keys()))
        assert history[transaction_id].state.value == "committed"
