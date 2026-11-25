"""Debug service backed by the read-only execution store."""

from __future__ import annotations

from typing import Any

from local_coding_assistant.dashboard.exceptions import (
    NotFoundError,
    StoreUnavailableError,
)
from local_coding_assistant.storage.contracts import MissingRefs
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.debug_service")


class DebugService:
    """Read-only debug queries: snapshot resolution, prompt reconstruction, missing refs."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def resolve_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Resolve a context snapshot for debug display."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        snapshot = await self._store.get_stored_context_snapshot(snapshot_id)
        if snapshot is None:
            raise NotFoundError(f"Context snapshot not found: {snapshot_id}")

        return {
            "snapshot_id": snapshot.id,
            "turn_id": snapshot.turn_id,
            "schema_version": snapshot.schema_version,
            "content_hash": snapshot.content_hash,
            "tool_set_ref": snapshot.tool_set_ref,
            "template_set_ref": snapshot.template_set_ref,
            "repo_snapshot_ref": snapshot.repo_snapshot_ref,
            "memory_snapshot_ref": snapshot.memory_snapshot_ref,
            "message_boundary_json": snapshot.message_boundary_json,
            "payload_json": snapshot.payload_json,
            "created_at_ms": snapshot.created_at_ms,
        }

    async def check_missing_refs(self, snapshot_id: str) -> MissingRefs:
        """Check which context refs of a snapshot still exist in the store."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        snapshot = await self._store.get_stored_context_snapshot(snapshot_id)
        if snapshot is None:
            raise NotFoundError(f"Context snapshot not found: {snapshot_id}")

        tool_ids = self._parse_json_list(snapshot.tool_set_ref, "tool_ids")
        template_ids = self._parse_json_list(snapshot.template_set_ref, "template_ids")
        repo_hash = self._parse_json_str(
            snapshot.repo_snapshot_ref, "repo_snapshot_hash"
        )
        repo_hashes = [repo_hash] if repo_hash else None
        memory_ids = self._parse_json_list(snapshot.memory_snapshot_ref, "memory_ids")

        return await self._store.check_context_refs(
            tool_ids=tool_ids,
            template_ids=template_ids,
            repo_hashes=repo_hashes,
            memory_ids=memory_ids,
        )

    @staticmethod
    def _parse_json_list(ref_json: str | None, key: str) -> list[Any] | None:
        """Parse a JSON ref string and extract a list by key."""
        if not ref_json:
            return None
        import json

        try:
            data = json.loads(ref_json)
            if isinstance(data, dict):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        except Exception as e:
            log.warning("Failed to parse JSON ref", key=key, error=str(e))
        return None

    @staticmethod
    def _parse_json_str(ref_json: str | None, key: str) -> str | None:
        """Parse a JSON ref string and extract a string by key."""
        if not ref_json:
            return None
        import json

        try:
            data = json.loads(ref_json)
            if isinstance(data, dict):
                value = data.get(key)
                if isinstance(value, str):
                    return value
        except Exception as e:
            log.warning("Failed to parse JSON ref", key=key, error=str(e))
        return None
