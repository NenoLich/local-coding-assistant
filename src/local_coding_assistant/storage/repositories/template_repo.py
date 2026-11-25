"""Template repository for execution store."""

from __future__ import annotations

from typing import Any

import aiosqlite

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.template_repo")


class TemplateRepo:
    """Repository for global template operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def register(
        self,
        path: str,
        content: str,
        content_hash: str,
        version: str = "1.0.0",
        created_at_ms: int | None = None,
    ) -> int:
        """Register a global template. Returns template_id.

        Deduplicates by content_hash. If content_hash exists, returns existing ID.
        Otherwise inserts new version with new ID.
        """
        if created_at_ms is None:
            from datetime import UTC, datetime

            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)

        # Check if template with this content_hash already exists
        cursor = await self._conn.execute(
            "SELECT id FROM templates WHERE content_hash = ?", (content_hash,)
        )
        row = await cursor.fetchone()
        if row:
            return row[0]

        # Insert new version
        cursor = await self._conn.execute(
            """
            INSERT INTO templates (path, content, content_hash, version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (path, content, content_hash, version, created_at_ms),
        )

        return cursor.lastrowid if cursor.lastrowid is not None else -1

    async def get(self, template_id: int) -> dict[str, Any] | None:
        """Get template by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, path, content, content_hash, version, created_at
            FROM templates WHERE id = ?
            """,
            (template_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "path": row[1],
            "content": row[2],
            "content_hash": row[3],
            "version": row[4],
            "created_at": row[5],
        }

    async def get_existing_ids(self, template_ids: list[int]) -> list[int]:
        """Return the subset of template IDs that exist in the table."""
        if not template_ids:
            return []
        placeholders = ",".join("?" for _ in template_ids)
        query = "SELECT id FROM templates WHERE id IN (" + placeholders + ")"  # noqa: S608 - parameterized ? placeholders, IDs bound as query args
        cursor = await self._conn.execute(query, template_ids)

        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_by_path(self, path: str) -> dict[str, Any] | None:
        """Get template by path."""
        cursor = await self._conn.execute(
            """
            SELECT id, path, content, content_hash, version, created_at
            FROM templates WHERE path = ?
            """,
            (path,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "path": row[1],
            "content": row[2],
            "content_hash": row[3],
            "version": row[4],
            "created_at": row[5],
        }

    async def list_all(self) -> list[dict[str, Any]]:
        """List all registered templates."""
        cursor = await self._conn.execute(
            """
            SELECT id, path, content, content_hash, version, created_at
            FROM templates ORDER BY path
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "path": row[1],
                "content": row[2],
                "content_hash": row[3],
                "version": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    async def get_session_templates(self, session_id: str) -> list[dict[str, Any]]:
        """Get all templates associated with a session."""
        cursor = await self._conn.execute(
            """
            SELECT t.id, t.path, t.content, t.content_hash, t.version, t.created_at
            FROM templates t
            INNER JOIN session_templates st ON t.id = st.template_id
            WHERE st.session_id = ?
            ORDER BY t.path
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "path": row[1],
                "content": row[2],
                "content_hash": row[3],
                "version": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    async def get_many(self, template_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple templates by IDs in a single query."""
        if not template_ids:
            return []
        placeholders = ",".join("?" for _ in template_ids)
        query = (
            "SELECT id, path, content, content_hash, version, created_at FROM templates WHERE id IN ("  # noqa S608
            + placeholders
            + ")"
        )
        cursor = await self._conn.execute(query, template_ids)

        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "path": row[1],
                "content": row[2],
                "content_hash": row[3],
                "version": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]
