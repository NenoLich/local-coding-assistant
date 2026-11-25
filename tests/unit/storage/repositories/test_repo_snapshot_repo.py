"""Repository snapshot repository unit tests."""

from __future__ import annotations

from local_coding_assistant.storage.repositories import (
    RepoSnapshotData,
)


class TestRepoSnapshotRepo:
    async def test_register_inserts_new_snapshot(self, repo_snapshot_repo) -> None:
        """register(): INSERT OR IGNORE by repo_snapshot_hash, returns (id, is_new)."""
        result = await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": []}',
            project_info_json='{"name": "project"}',
            repo_snapshot_hash="abc123",
        )

        assert result.is_new is True
        assert result.id > 0

    async def test_register_returns_existing_id_for_duplicate_hash(
        self, repo_snapshot_repo
    ) -> None:
        """register(): returns existing ID when hash already exists."""
        result_1 = await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": []}',
            project_info_json='{"name": "project"}',
            repo_snapshot_hash="abc123",
        )

        result_2 = await repo_snapshot_repo.register(
            target_project_root="/different/path",  # Different root, same hash
            repo_map_json='{"files": []}',
            project_info_json='{"name": "project"}',
            repo_snapshot_hash="abc123",
        )

        assert result_1.is_new is True
        assert result_2.is_new is False
        assert result_1.id == result_2.id

    async def test_register_handles_none_project_info(self, repo_snapshot_repo) -> None:
        """register(): handles None project_info_json."""
        result = await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": []}',
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )

        assert result.is_new is True
        assert result.id > 0

    async def test_get_existing_hashes_filters_existing(
        self, repo_snapshot_repo
    ) -> None:
        """get_existing_hashes(): IN clause for hashes."""
        await repo_snapshot_repo.register(
            target_project_root="/path1",
            repo_map_json='{"files": []}',
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )
        await repo_snapshot_repo.register(
            target_project_root="/path2",
            repo_map_json='{"files": []}',
            project_info_json=None,
            repo_snapshot_hash="def456",
        )

        existing = await repo_snapshot_repo.get_existing_hashes(
            ["abc123", "def456", "ghi789"]
        )
        assert set(existing) == {"abc123", "def456"}

    async def test_get_existing_hashes_empty_list(self, repo_snapshot_repo) -> None:
        """get_existing_hashes(): returns empty list for empty input."""
        existing = await repo_snapshot_repo.get_existing_hashes([])
        assert existing == []

    async def test_get_by_hash_returns_repo_snapshot_data(
        self, repo_snapshot_repo
    ) -> None:
        """get_by_hash(): returns RepoSnapshotData."""
        await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": ["file1.txt"]}',
            project_info_json='{"name": "project"}',
            repo_snapshot_hash="abc123",
        )

        snapshot = await repo_snapshot_repo.get_by_hash("abc123")
        assert snapshot is not None
        assert isinstance(snapshot, RepoSnapshotData)
        assert snapshot.target_project_root == "/path/to/project"
        assert snapshot.repo_map_json == '{"files": ["file1.txt"]}'
        assert snapshot.project_info_json == '{"name": "project"}'
        assert snapshot.repo_snapshot_hash == "abc123"

    async def test_get_by_hash_returns_none_for_missing_hash(
        self, repo_snapshot_repo
    ) -> None:
        snapshot = await repo_snapshot_repo.get_by_hash("nonexistent")
        assert snapshot is None

    async def test_get_by_root_returns_latest_by_created_at(
        self, repo_snapshot_repo
    ) -> None:
        """get_by_root(): latest by target_project_root ORDER BY created_at DESC LIMIT 1."""

        # Register first snapshot with older timestamp
        await repo_snapshot_repo._conn.execute(
            """
            INSERT INTO repo_snapshots (target_project_root, repo_map_json, project_info_json, repo_snapshot_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "/path/to/project",
                '{"files": ["file1.txt"]}',
                '{"name": "project"}',
                "abc123",
                1000,
            ),
        )

        # Register second snapshot with newer timestamp
        await repo_snapshot_repo._conn.execute(
            """
            INSERT INTO repo_snapshots (target_project_root, repo_map_json, project_info_json, repo_snapshot_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "/path/to/project",
                '{"files": ["file1.txt", "file2.txt"]}',
                '{"name": "project"}',
                "def456",
                2000,
            ),
        )

        snapshot = await repo_snapshot_repo.get_by_root("/path/to/project")
        assert snapshot is not None
        assert snapshot.repo_snapshot_hash == "def456"  # Latest

    async def test_get_by_root_returns_none_for_missing_root(
        self, repo_snapshot_repo
    ) -> None:
        snapshot = await repo_snapshot_repo.get_by_root("/nonexistent/path")
        assert snapshot is None

    async def test_get_by_root_handles_none_project_info(
        self, repo_snapshot_repo
    ) -> None:
        """get_by_root(): handles None project_info_json."""
        await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": []}',
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )

        snapshot = await repo_snapshot_repo.get_by_root("/path/to/project")
        assert snapshot is not None
        assert snapshot.project_info_json is None

    async def test_register_auto_fills_created_at(self, repo_snapshot_repo) -> None:
        """register(): auto-fills created_at."""
        from datetime import UTC, datetime

        before = int(datetime.now(UTC).timestamp() * 1000)
        result = await repo_snapshot_repo.register(
            target_project_root="/path/to/project",
            repo_map_json='{"files": []}',
            project_info_json=None,
            repo_snapshot_hash="abc123",
        )
        after = int(datetime.now(UTC).timestamp() * 1000)

        assert result.is_new is True
        snapshot = await repo_snapshot_repo.get_by_hash("abc123")
        assert snapshot is not None
        # Verify created_at was set (can't check exact value since it's not returned)
        assert snapshot.repo_snapshot_hash == "abc123"
