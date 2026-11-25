"""Repository snapshot repository for execution store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    pass

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.repo_snapshot_repo")


@dataclass(frozen=True, slots=True)
class RepoSnapshotData:
    """Repository snapshot data for storage and reconstruction."""

    target_project_root: str
    repo_map_json: str
    project_info_json: str | None
    repo_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class RepoSnapshotResult:
    """Result of repo snapshot registration."""

    id: int
    is_new: bool


class RepoSnapshotRepo:
    """Repository for repository snapshot operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def register(
        self,
        target_project_root: str,
        repo_map_json: str,
        project_info_json: str | None,
        repo_snapshot_hash: str,
    ) -> RepoSnapshotResult:
        """Register a repo snapshot (insert or ignore by hash).

        Args:
            target_project_root: Root path of the target project
            repo_map_json: Serialized RepoMapData as JSON
            project_info_json: Serialized ProjectInfo as JSON (optional)
            repo_snapshot_hash: Content hash for deduplication

        Returns:
            RepoSnapshotResult with the snapshot ID and whether it was newly created
        """
        from datetime import UTC, datetime

        created_at = int(datetime.now(UTC).timestamp() * 1000)

        # Try to insert, ignore if hash already exists
        cursor = await self._conn.execute(
            """
            INSERT OR IGNORE INTO repo_snapshots (target_project_root, repo_map_json, project_info_json, repo_snapshot_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                target_project_root,
                repo_map_json,
                project_info_json,
                repo_snapshot_hash,
                created_at,
            ),
        )

        if cursor.rowcount > 0:
            # New row inserted
            snapshot_id = cursor.lastrowid
            await self._conn.commit()
            return RepoSnapshotResult(id=snapshot_id or 0, is_new=True)

        # Hash exists, get existing ID
        cursor = await self._conn.execute(
            "SELECT id FROM repo_snapshots WHERE repo_snapshot_hash = ?",
            (repo_snapshot_hash,),
        )
        row = await cursor.fetchone()
        if row:
            return RepoSnapshotResult(id=row[0], is_new=False)

        # Should not happen, but handle gracefully
        raise RuntimeError(
            f"Failed to register or find repo snapshot with hash {repo_snapshot_hash}"
        )

    async def get_existing_hashes(self, repo_snapshot_hashes: list[str]) -> list[str]:
        """Return the subset of repo snapshot hashes that exist in the table."""
        if not repo_snapshot_hashes:
            return []
        placeholders = ",".join("?" for _ in repo_snapshot_hashes)
        query = (
            "SELECT repo_snapshot_hash FROM repo_snapshots WHERE repo_snapshot_hash IN ("  # noqa S608
            + placeholders
            + ")"
        )
        cursor = await self._conn.execute(query, repo_snapshot_hashes)

        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_by_hash(self, repo_snapshot_hash: str) -> RepoSnapshotData | None:
        """Get repo snapshot by content hash."""
        cursor = await self._conn.execute(
            """
            SELECT target_project_root, repo_map_json, project_info_json, repo_snapshot_hash
            FROM repo_snapshots WHERE repo_snapshot_hash = ?
            """,
            (repo_snapshot_hash,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return RepoSnapshotData(
            target_project_root=row[0],
            repo_map_json=row[1],
            project_info_json=row[2],
            repo_snapshot_hash=row[3],
        )

    async def get_by_root(self, target_project_root: str) -> RepoSnapshotData | None:
        """Get latest repo snapshot by project root (for fallback)."""
        cursor = await self._conn.execute(
            """
            SELECT target_project_root, repo_map_json, project_info_json, repo_snapshot_hash
            FROM repo_snapshots WHERE target_project_root = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (target_project_root,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return RepoSnapshotData(
            target_project_root=row[0],
            repo_map_json=row[1],
            project_info_json=row[2],
            repo_snapshot_hash=row[3],
        )
