"""Tool repository for execution store."""

from __future__ import annotations

from typing import Any

import aiosqlite

from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.repositories.tool_repo")


class ToolRepo:
    """Repository for global tool operations."""

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def register(
        self,
        name: str,
        schema_json: str,
        risk_class: str,
        description: str | None = None,
        version: str = "1.0.0",
        created_at_ms: int | None = None,
    ) -> int:
        """Register a global tool. Returns tool_id.

        Deduplicates by content_hash. If content_hash exists, returns existing ID.
        Otherwise inserts new version with new ID.
        """
        import hashlib

        if created_at_ms is None:
            from datetime import UTC, datetime

            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)

        # Compute content hash from all relevant fields for deduplication
        hash_input = f"{name}|{schema_json}|{risk_class}|{description or ''}|{version}"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # Check if tool with this content_hash already exists
        cursor = await self._conn.execute(
            "SELECT id FROM tools WHERE content_hash = ?", (content_hash,)
        )
        row = await cursor.fetchone()
        if row:
            return row[0]

        # Insert new version
        cursor = await self._conn.execute(
            """
            INSERT INTO tools (name, schema_json, risk_class, description, version, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                schema_json,
                risk_class,
                description,
                version,
                content_hash,
                created_at_ms,
            ),
        )

        return cursor.lastrowid if cursor.lastrowid is not None else -1

    async def get(self, tool_id: int) -> dict[str, Any] | None:
        """Get tool by ID."""
        cursor = await self._conn.execute(
            """
            SELECT id, name, schema_json, risk_class, description, version, created_at
            FROM tools WHERE id = ?
            """,
            (tool_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "schema_json": row[2],
            "risk_class": row[3],
            "description": row[4],
            "version": row[5],
            "created_at": row[6],
        }

    async def get_existing_ids(self, tool_ids: list[int]) -> list[int]:
        """Return the subset of tool IDs that exist in the table."""
        if not tool_ids:
            return []
        placeholders = ",".join("?" for _ in tool_ids)
        query = "SELECT id FROM tools WHERE id IN (" + placeholders + ")"  # noqa: S608 - parameterized ? placeholders, IDs bound as query args
        cursor = await self._conn.execute(query, tool_ids)

        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Get tool by name."""
        cursor = await self._conn.execute(
            """
            SELECT id, name, schema_json, risk_class, description, version, created_at
            FROM tools WHERE name = ?
            """,
            (name,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        return {
            "id": row[0],
            "name": row[1],
            "schema_json": row[2],
            "risk_class": row[3],
            "description": row[4],
            "version": row[5],
            "created_at": row[6],
        }

    async def list_all(self) -> list[dict[str, Any]]:
        """List all registered tools."""
        cursor = await self._conn.execute(
            """
            SELECT id, name, schema_json, risk_class, description, version, created_at
            FROM tools ORDER BY name
            """
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "schema_json": row[2],
                "risk_class": row[3],
                "description": row[4],
                "version": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    async def get_session_tools(self, session_id: str) -> list[dict[str, Any]]:
        """Get all tools associated with a session."""
        cursor = await self._conn.execute(
            """
            SELECT t.id, t.name, t.schema_json, t.risk_class, t.description, t.version, t.created_at
            FROM tools t
            INNER JOIN session_tools st ON t.id = st.tool_id
            WHERE st.session_id = ?
            ORDER BY t.name
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "schema_json": row[2],
                "risk_class": row[3],
                "description": row[4],
                "version": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    async def get_many(self, tool_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple tools by IDs in a single query."""
        if not tool_ids:
            return []
        placeholders = ",".join("?" for _ in tool_ids)
        query = (
            "SELECT id, name, schema_json, risk_class, description, version, created_at FROM tools WHERE id IN ("  # noqa S608
            + placeholders
            + ")"
        )
        cursor = await self._conn.execute(query, tool_ids)

        rows = await cursor.fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "schema_json": row[2],
                "risk_class": row[3],
                "description": row[4],
                "version": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]
