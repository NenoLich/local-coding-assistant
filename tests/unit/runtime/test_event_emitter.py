"""
Tests for event emitter utilities.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from local_coding_assistant.runtime.event_emitter import (
    EventStream,
    deserialize_events,
    event_from_dict,
    event_to_dict,
    filter_events,
    map_events,
    serialize_events,
)
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class TestEventStream:
    """Test EventStream async context manager."""

    @pytest.mark.asyncio
    async def test_event_stream_context_manager(self):
        """Test basic context manager lifecycle."""
        async with EventStream() as stream:
            assert isinstance(stream, EventStream)

    @pytest.mark.asyncio
    async def test_event_stream_emit_and_iterate(self):
        """Test emitting events and iterating over them."""
        events_to_emit = [
            ExecutionEvent(type=EventType.LLM_START, session_id="test"),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id="test", data={"content": "hello"}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id="test"),
        ]

        collected_events = []

        async with EventStream() as stream:
            # Start collecting events
            async def collect():
                async for event in stream:
                    collected_events.append(event)

            collect_task = asyncio.create_task(collect())

            # Emit events
            for event in events_to_emit:
                await stream.emit(event)

            # Wait a bit for events to be processed
            await asyncio.sleep(0.01)

            # End the stream
            # This is done by exiting the context manager

        # Wait for collection to finish
        await collect_task

        assert len(collected_events) == 3
        assert collected_events[0].type == EventType.LLM_START
        assert collected_events[1].type == EventType.LLM_CHUNK
        assert collected_events[1].data["content"] == "hello"
        assert collected_events[2].type == EventType.LLM_COMPLETE

class TestEventFiltering:
    """Test event filtering and mapping utilities."""

    @pytest.mark.asyncio
    async def test_filter_events(self, mock_event_stream):
        """Test filtering events by predicate."""
        # Filter only LLM events
        llm_events = []
        async for event in filter_events(mock_event_stream, lambda e: e.type.value.startswith("llm")):
            llm_events.append(event)

        assert len(llm_events) == 4  # LLM_START, two LLM_CHUNK, LLM_COMPLETE
        assert all(e.type.value.startswith("llm") for e in llm_events)

    @pytest.mark.asyncio
    async def test_map_events(self, mock_event_stream):
        """Test transforming events."""
        # Add a prefix to all event types
        transformed_events = []
        async for event in map_events(
            mock_event_stream,
            lambda e: ExecutionEvent(
                type=e.type,
                session_id=f"mapped_{e.session_id}",
                frame_id=e.frame_id,
                data=e.data,
                timestamp=e.timestamp
            )
        ):
            transformed_events.append(event)

        assert len(transformed_events) == 8
        assert all(e.session_id.startswith("mapped_") for e in transformed_events)

    @pytest.mark.asyncio
    async def test_filter_and_map_chain(self, mock_event_stream):
        """Test chaining filter and map operations."""
        # Filter LLM events and add prefix to session_id
        processed_events = []
        async for event in map_events(
            filter_events(mock_event_stream, lambda e: e.type.value.startswith("llm")),
            lambda e: ExecutionEvent(
                type=e.type,
                session_id=f"processed_{e.session_id}",
                frame_id=e.frame_id,
                data=e.data,
                timestamp=e.timestamp
            )
        ):
            processed_events.append(event)

        assert len(processed_events) == 4
        assert all(e.type.value.startswith("llm") for e in processed_events)
        assert all(e.session_id.startswith("processed_") for e in processed_events)


class TestEventSerialization:
    """Test event serialization and deserialization."""

    def test_event_to_dict(self, sample_events):
        """Test converting event to dictionary."""
        event = sample_events[0]  # TURN_START event

        event_dict = event_to_dict(event)

        expected = {
            "type": "turn_start",
            "session_id": "test_session",
            "frame_id": None,
            "data": {"user_input": "test query"},
            "timestamp": event.timestamp.timestamp(),  # Convert to Unix timestamp
        }

        assert event_dict == expected

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        event_dict = {
            "type": "llm_chunk",
            "session_id": "test_session",
            "frame_id": "frame_1",
            "data": {"content": "test content"},
            "timestamp": 1234567890.0,
        }

        event = event_from_dict(event_dict)

        assert event.type == EventType.LLM_CHUNK
        assert event.session_id == "test_session"
        assert event.frame_id == "frame_1"
        assert event.data == {"content": "test content"}
        from datetime import datetime, timezone
        assert event.timestamp == datetime.fromtimestamp(1234567890.0, tz=timezone.utc)

    def test_event_from_dict_defaults(self):
        """Test event_from_dict with missing optional fields."""
        event_dict = {
            "type": "frame_start",
            "session_id": "test_session",
        }

        event = event_from_dict(event_dict)

        assert event.type == EventType.FRAME_START
        assert event.session_id == "test_session"
        assert event.frame_id is None
        assert event.data == {}
        from datetime import datetime, timezone
        assert event.timestamp == datetime.fromtimestamp(0.0, tz=timezone.utc)

    def test_serialize_deserialize_events(self, sample_events):
        """Test round-trip serialization of events."""
        # Serialize
        json_str = serialize_events(sample_events)

        # Deserialize
        deserialized_events = deserialize_events(json_str)

        assert len(deserialized_events) == len(sample_events)
        for original, deserialized in zip(sample_events, deserialized_events):
            assert original.type == deserialized.type
            assert original.session_id == deserialized.session_id
            assert original.frame_id == deserialized.frame_id
            assert original.data == deserialized.data
            assert original.timestamp == deserialized.timestamp

    def test_serialize_events_json_format(self, sample_events):
        """Test that serialized events are valid JSON."""
        json_str = serialize_events(sample_events)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == len(sample_events)

        # Each item should be a dict
        for item in parsed:
            assert isinstance(item, dict)
            assert "type" in item
            assert "session_id" in item
            assert "timestamp" in item

    def test_deserialize_events_invalid_json(self):
        """Test deserialization with invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            deserialize_events("invalid json")

    def test_event_from_dict_invalid_type(self):
        """Test event_from_dict with invalid event type."""
        event_dict = {
            "type": "invalid_type",
            "session_id": "test_session",
        }

        with pytest.raises(ValueError):
            event_from_dict(event_dict)
