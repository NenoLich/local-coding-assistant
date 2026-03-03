"""
Event emitter utilities for streaming execution.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from datetime import UTC
from typing import Any

from local_coding_assistant.runtime.events import ExecutionEvent


class EventStream:
    """
    Async context manager for emitting execution events.

    Usage:
        async with EventStream() as stream:
            await stream.emit(event)
            async for event in stream:
                # consume events
    """

    def __init__(self):
        self._queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> EventStream:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Signal end of stream
        await self._queue.put(None)
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def emit(self, event: ExecutionEvent) -> None:
        """Emit an event to the stream."""
        await self._queue.put(event)

    def __aiter__(self) -> AsyncIterator[ExecutionEvent]:
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[ExecutionEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event


async def filter_events(
    events: AsyncIterator[ExecutionEvent],
    predicate: Callable[[ExecutionEvent], bool],
) -> AsyncGenerator[ExecutionEvent]:
    """Filter events based on a predicate function."""
    async for event in events:
        if predicate(event):
            yield event


async def map_events(
    events: AsyncIterator[ExecutionEvent],
    transform: Callable[[ExecutionEvent], ExecutionEvent],
) -> AsyncGenerator[ExecutionEvent]:
    """Transform events using a mapping function."""
    async for event in events:
        yield transform(event)


def event_to_dict(event: ExecutionEvent) -> dict[str, Any]:
    """Serialize an event to a dictionary for persistence."""
    return {
        "type": event.type.value,
        "session_id": event.session_id,
        "frame_id": event.frame_id,
        "data": event.data,
        "timestamp": event.timestamp.timestamp(),  # Convert datetime to Unix timestamp
    }


def event_from_dict(data: dict[str, Any]) -> ExecutionEvent:
    """Deserialize an event from a dictionary."""
    from datetime import datetime

    from local_coding_assistant.runtime.events import EventType

    timestamp_value = data.get("timestamp", 0.0)
    if isinstance(timestamp_value, (int, float)):
        timestamp = datetime.fromtimestamp(timestamp_value, tz=UTC)
    else:
        timestamp = timestamp_value

    return ExecutionEvent(
        type=EventType(data["type"]),
        session_id=data["session_id"],
        frame_id=data.get("frame_id"),
        data=data.get("data", {}),
        timestamp=timestamp,
    )


def serialize_events(events: list[ExecutionEvent]) -> str:
    """Serialize a list of events to JSON."""
    return json.dumps([event_to_dict(event) for event in events], ensure_ascii=False)


def deserialize_events(json_str: str) -> list[ExecutionEvent]:
    """Deserialize a list of events from JSON."""
    data = json.loads(json_str)
    return [event_from_dict(item) for item in data]
