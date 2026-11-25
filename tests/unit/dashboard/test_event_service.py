"""Tests for historical dashboard event queries."""

import json
from datetime import UTC, datetime

import pytest

from local_coding_assistant.dashboard.services.event_service import EventService


class EventStoreStub:
    def __init__(self) -> None:
        self.events = [
            {
                "id": 101,
                "seq": 1,
                "type": "turn_start",
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                "frame_id": None,
                "data_json": json.dumps({"user_input": "first event"}),
            },
            {
                "id": 202,
                "seq": 1,
                "type": "frame_start",
                "timestamp": int(datetime.now(UTC).timestamp() * 1000),
                "frame_id": "frame-1",
                "data_json": json.dumps({"iteration": 2}),
            },
        ]

    async def get_events_by_turn(self, turn_id: str) -> list[dict[str, object]]:
        return self.events


@pytest.mark.asyncio
async def test_event_payload_uses_unique_event_id_when_sequence_repeats() -> None:
    service = EventService(EventStoreStub())

    payload = await service.get_event_payload("turn-1", 202)

    assert payload == {"iteration": 2}


@pytest.mark.asyncio
async def test_timeline_entries_include_unique_event_id() -> None:
    service = EventService(EventStoreStub())

    events = await service.get_events_by_turn("turn-1")

    assert [event.event_id for event in events] == [101, 202]
