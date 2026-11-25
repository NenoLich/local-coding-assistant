"""Workspace identity repository unit tests."""

from __future__ import annotations

import uuid

from local_coding_assistant.storage.repositories import WorkspaceRecord


class TestWorkspaceRepo:
    async def test_get_by_root_finds_existing_workspace(self, workspace_repo) -> None:
        """get_by_root(): finds by workspace_root, updates updated_at on hit."""
        workspace_id = str(uuid.uuid4())
        await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )

        record = await workspace_repo.get_by_root("/path/to/workspace")
        assert record is not None
        assert isinstance(record, WorkspaceRecord)
        assert record.id == workspace_id
        assert record.workspace_root == "/path/to/workspace"
        assert record.config_hash == "hash123"
        assert record.updated_at > 0  # Should be updated

    async def test_get_by_root_returns_none_for_missing_root(
        self, workspace_repo
    ) -> None:
        record = await workspace_repo.get_by_root("/nonexistent/path")
        assert record is None

    async def test_get_by_root_updates_updated_at(self, workspace_repo) -> None:
        """get_by_root(): updates updated_at on hit (side effect on read)."""

        workspace_id = str(uuid.uuid4())
        await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )

        # Get initial record
        record_1 = await workspace_repo.get_by_root("/path/to/workspace")
        initial_updated_at = record_1.updated_at

        # Wait a bit and get again
        import asyncio

        await asyncio.sleep(0.01)

        record_2 = await workspace_repo.get_by_root("/path/to/workspace")
        assert record_2.updated_at > initial_updated_at

    async def test_create_inserts_new_workspace(self, workspace_repo) -> None:
        """create(): INSERT OR IGNORE, returns WorkspaceRecord."""
        workspace_id = str(uuid.uuid4())
        record = await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )

        assert isinstance(record, WorkspaceRecord)
        assert record.id == workspace_id
        assert record.workspace_root == "/path/to/workspace"
        assert record.config_hash == "hash123"
        assert record.updated_at > 0

    async def test_create_auto_fills_updated_at(self, workspace_repo) -> None:
        """create(): auto-fills updated_at."""
        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        workspace_id = str(uuid.uuid4())
        record = await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        assert before <= record.updated_at <= after

    async def test_create_handles_duplicate_workspace_id(self, workspace_repo) -> None:
        """create(): INSERT OR IGNORE handles duplicate workspace_id."""
        workspace_id = str(uuid.uuid4())

        record_1 = await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )

        record_2 = await workspace_repo.create(
            workspace_id=workspace_id,  # Same ID
            workspace_root="/different/path",
            config_hash="hash456",
        )

        # Second create should be ignored (INSERT OR IGNORE)
        # The record should still have the original values
        record = await workspace_repo.get_by_root("/path/to/workspace")
        assert record is not None
        assert record.config_hash == "hash123"  # Original value

    async def test_update_config_hash_updates_hash_and_updated_at(
        self, workspace_repo
    ) -> None:
        """update_config_hash(): updates config_hash + updated_at."""
        workspace_id = str(uuid.uuid4())
        await workspace_repo.create(
            workspace_id=workspace_id,
            workspace_root="/path/to/workspace",
            config_hash="hash123",
        )

        # Get initial updated_at
        record_1 = await workspace_repo.get_by_root("/path/to/workspace")
        initial_updated_at = record_1.updated_at

        # Wait a bit and update
        import asyncio

        await asyncio.sleep(0.01)
        await workspace_repo.update_config_hash(workspace_id, "hash456")

        record_2 = await workspace_repo.get_by_root("/path/to/workspace")
        assert record_2 is not None
        assert record_2.config_hash == "hash456"
        assert record_2.updated_at > initial_updated_at
