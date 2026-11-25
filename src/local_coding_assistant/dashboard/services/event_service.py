"""Historical event service backed by the read-only execution store."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from local_coding_assistant.dashboard.exceptions import StoreUnavailableError
from local_coding_assistant.dashboard.models.historical import EventTimelineEntry
from local_coding_assistant.dashboard.services.event_preview import preview_for_event
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader
from local_coding_assistant.utils.logging import get_logger

log = get_logger("dashboard.services.event_service")


def _ms_to_iso(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC)


class EventService:
    """Read-only historical event queries."""

    def __init__(self, store: ExecutionStoreReader):
        self._store = store

    async def get_events_by_turn(
        self,
        turn_id: str,
        event_type: list[str] | None = None,
    ) -> list[EventTimelineEntry]:
        """Get all events for a turn, optionally filtered by type."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        events_raw = await self._store.get_events_by_turn(turn_id)
        events = []
        import json

        for ev in events_raw:
            if event_type and ev.get("type") not in event_type:
                continue

            try:
                data = json.loads(ev.get("data_json") or "{}")
            except Exception:
                data = {}
            events.append(
                EventTimelineEntry(
                    event_id=int(ev["id"]),
                    seq=int(ev.get("seq", 0)),
                    type=ev.get("type", "unknown"),
                    timestamp=_ms_to_iso(int(ev.get("timestamp", 0)))
                    if ev.get("timestamp")
                    else datetime.now(UTC),
                    frame_id=ev.get("frame_id"),
                    preview=preview_for_event(ev.get("type", ""), data),
                    payload=data,
                    has_payload=bool(data),
                )
            )
        events.sort(key=lambda e: e.seq)
        return events

    async def get_events_by_frame(
        self,
        frame_id: str,
        event_type: list[str] | None = None,
    ) -> list[EventTimelineEntry]:
        """Get all events for a frame, optionally filtered by type."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        events_raw = await self._store.get_frame_events(frame_id)
        events = []
        import json

        for ev in events_raw:
            if event_type and ev.get("type") not in event_type:
                continue

            try:
                data = json.loads(ev.get("data_json") or "{}")
            except Exception:
                data = {}
            events.append(
                EventTimelineEntry(
                    event_id=int(ev["id"]),
                    seq=int(ev.get("seq", 0)),
                    type=ev.get("type", "unknown"),
                    timestamp=_ms_to_iso(int(ev.get("timestamp", 0)))
                    if ev.get("timestamp")
                    else datetime.now(UTC),
                    frame_id=ev.get("frame_id"),
                    preview=preview_for_event(ev.get("type", ""), data),
                    payload=data,
                    has_payload=bool(data),
                )
            )
        events.sort(key=lambda e: e.seq)
        return events

    async def get_event_payload(
        self,
        turn_id: str,
        event_id: int,
    ) -> dict[str, Any] | None:
        """Get the full payload for a specific event by its unique ID."""
        if self._store is None:
            raise StoreUnavailableError("Execution store not available")

        events_raw = await self._store.get_events_by_turn(turn_id)
        import json

        for ev in events_raw:
            if int(ev["id"]) == event_id:
                try:
                    return json.loads(ev.get("data_json") or "{}")
                except Exception:
                    return {}
        return None
