"""Historical message service backed by the read-only execution store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from local_coding_assistant.dashboard.exceptions import (
    NotFoundError,
    StoreUnavailableError,
)
from local_coding_assistant.dashboard.models.messages import MessageDetail, MessageEntry
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.message_service")


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


def _parse_iso(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        return None


def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    import json

    try:
        return json.loads(metadata_json)
    except Exception:
        return {}


class MessageService:
    """Read-only historical message queries."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def _build_turn_index_map(self, session_id: str) -> dict[str, int]:
        """Build a mapping of turn_id -> turn_index for a session."""
        turns = await self._store.list_turns_by_session(session_id)
        return {t.id: t.turn_index for t in turns}

    async def list_messages_by_session(
        self,
        session_id: str,
        turn_from: int | None = None,
        turn_to: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
    ) -> list[MessageEntry]:
        """List all messages for a session with optional filters."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        messages = await self._store.get_all_messages(session_id)
        turn_index_map = await self._build_turn_index_map(session_id)

        created_from_dt = _parse_iso(created_from)
        created_to_dt = _parse_iso(created_to)

        entries: list[MessageEntry] = []
        for m in messages:
            turn_index = turn_index_map.get(m["turn_id"])
            created_at_ms = m.get("created_at")

            if not self._message_matches_filters(
                m,
                turn_index,
                created_at_ms,
                turn_from,
                turn_to,
                created_from_dt,
                created_to_dt,
            ):
                continue

            entries.append(
                MessageEntry(
                    seq=m["seq"],
                    role=m["role"],
                    content=m["content"] or "",
                    tool_call_id=m.get("tool_call_id"),
                    tool_name=m.get("tool_name"),
                    turn_id=m["turn_id"],
                    frame_id=m.get("frame_id"),
                    created_at=_ms_to_iso(int(created_at_ms))
                    if created_at_ms
                    else datetime.now(UTC),
                    turn_index=turn_index,
                    metadata=_parse_metadata(m.get("metadata_json")),
                )
            )

        entries.sort(key=lambda e: e.seq)
        return entries

    @staticmethod
    def _message_matches_filters(
        m: dict[str, Any],
        turn_index: int | None,
        created_at_ms: Any,
        turn_from: int | None,
        turn_to: int | None,
        created_from_dt: datetime | None,
        created_to_dt: datetime | None,
    ) -> bool:
        """Check whether a message row matches all active filters."""
        if turn_from is not None and turn_index is not None and turn_index < turn_from:
            return False
        if turn_to is not None and turn_index is not None and turn_index > turn_to:
            return False

        if created_from_dt or created_to_dt:
            if created_at_ms is None:
                return False
            created_dt = datetime.fromtimestamp(int(created_at_ms) / 1000, UTC)
            if created_from_dt and created_dt < created_from_dt:
                return False
            if created_to_dt and created_dt > created_to_dt:
                return False

        return True

    async def get_message_detail(self, session_id: str, seq: int) -> MessageDetail:
        """Get full detail for a single message including raw payload."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        messages = await self._store.get_all_messages(session_id)
        for m in messages:
            if int(m["seq"]) == seq:
                entry = MessageEntry(
                    seq=m["seq"],
                    role=m["role"],
                    content=m["content"] or "",
                    tool_call_id=m.get("tool_call_id"),
                    tool_name=m.get("tool_name"),
                    turn_id=m["turn_id"],
                    frame_id=m.get("frame_id"),
                    created_at=_ms_to_iso(int(m["created_at"]))
                    if m.get("created_at")
                    else datetime.now(UTC),
                    metadata=_parse_metadata(m.get("metadata_json")),
                )
                import json

                try:
                    raw_payload = json.loads(m.get("metadata_json") or "{}")
                except Exception:
                    raw_payload = {}
                return MessageDetail(message=entry, raw_payload=raw_payload)

        raise NotFoundError(f"Message not found: session={session_id}, seq={seq}")
