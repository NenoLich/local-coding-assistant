"""
Tests for streaming event infrastructure.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_defined(self):
        """Test that all expected event types are defined."""
        expected_types = {
            "TURN_START",
            "FRAME_START",
            "LLM_START",
            "LLM_CHUNK",
            "LLM_COMPLETE",
            "TOOL_START",
            "TOOL_RESULT",
            "FRAME_COMPLETE",
            "TURN_COMPLETE",
            "SESSION_START",
            "SESSION_RESUME",
            "ERROR",
        }

        actual_types = {e.name for e in EventType}
        assert actual_types == expected_types

    def test_event_type_values(self):
        """Test that event type values match their names in lowercase."""
        for event_type in EventType:
            expected_value = event_type.name.lower().replace("_", "_")
            assert event_type.value == expected_value


class TestExecutionEvent:
    """Test ExecutionEvent dataclass."""

    def test_execution_event_creation_minimal(self):
        """Test creating an ExecutionEvent with minimal required fields."""
        event = ExecutionEvent(
            type=EventType.LLM_START,
            session_id="test_session",
        )

        assert event.type == EventType.LLM_START
        assert event.session_id == "test_session"
        assert event.frame_id is None
        assert event.data == {}
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp <= datetime.now(UTC)

    def test_execution_event_creation_full(self):
        """Test creating an ExecutionEvent with all fields."""
        test_timestamp = datetime.fromtimestamp(1234567890.0, tz=UTC)
        test_data = {"content": "test content", "reasoning": "test reasoning"}

        event = ExecutionEvent(
            type=EventType.LLM_CHUNK,
            session_id="test_session",
            frame_id="test_frame",
            data=test_data,
            timestamp=test_timestamp,
        )

        assert event.type == EventType.LLM_CHUNK
        assert event.session_id == "test_session"
        assert event.frame_id == "test_frame"
        assert event.data == test_data
        assert event.timestamp == test_timestamp

    def test_execution_event_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = datetime.now(UTC)
        event = ExecutionEvent(
            type=EventType.FRAME_START,
            session_id="test_session",
        )
        after = datetime.now(UTC)

        assert before <= event.timestamp <= after

    def test_execution_event_data_immutable(self):
        """Test that modifying event data doesn't affect the original dict."""
        original_data = {"key": "value"}
        event = ExecutionEvent(
            type=EventType.TOOL_RESULT,
            session_id="test_session",
            data=original_data,
        )

        # Modify the original dict
        original_data["key"] = "modified"

        # Event data should remain unchanged
        assert event.data["key"] == "value"

    def test_execution_event_serialization(self):
        """Test that ExecutionEvent can be serialized to dict."""
        event = ExecutionEvent(
            type=EventType.ERROR,
            session_id="test_session",
            frame_id="test_frame",
            data={"error": "test error"},
        )

        # Test that it has all expected attributes
        event_dict = {
            "type": event.type,
            "session_id": event.session_id,
            "frame_id": event.frame_id,
            "data": event.data,
            "timestamp": event.timestamp,
        }

        assert event_dict["type"] == EventType.ERROR
        assert event_dict["session_id"] == "test_session"
        assert event_dict["frame_id"] == "test_frame"
        assert event_dict["data"] == {"error": "test error"}
        assert isinstance(event_dict["timestamp"], datetime)


# Test utilities for streaming tests


async def collect_events(
    event_iterator: AsyncIterator[ExecutionEvent],
) -> list[ExecutionEvent]:
    """Collect all events from an async iterator into a list."""
    events = []
    async for event in event_iterator:
        events.append(event)
    return events


async def collect_events_with_timeout(
    event_iterator: AsyncIterator[ExecutionEvent], timeout: float = 5.0
) -> list[ExecutionEvent]:
    """Collect events with a timeout to prevent hanging tests."""
    events = []

    async def collect():
        async for event in event_iterator:
            events.append(event)

    try:
        await asyncio.wait_for(collect(), timeout=timeout)
    except TimeoutError:
        pass  # Timeout is expected for infinite iterators in tests

    return events


@pytest.fixture
def sample_events():
    """Fixture providing sample events for testing."""
    return [
        ExecutionEvent(
            type=EventType.TURN_START,
            session_id="test_session",
            data={"user_input": "test query"},
        ),
        ExecutionEvent(
            type=EventType.FRAME_START,
            session_id="test_session",
            frame_id="frame_1",
            data={"iteration": 1},
        ),
        ExecutionEvent(
            type=EventType.LLM_START,
            session_id="test_session",
            frame_id="frame_1",
            data={"model": "test-model"},
        ),
        ExecutionEvent(
            type=EventType.LLM_CHUNK,
            session_id="test_session",
            frame_id="frame_1",
            data={"content": "Hello"},
        ),
        ExecutionEvent(
            type=EventType.LLM_CHUNK,
            session_id="test_session",
            frame_id="frame_1",
            data={"content": " world"},
        ),
        ExecutionEvent(
            type=EventType.LLM_COMPLETE,
            session_id="test_session",
            frame_id="frame_1",
            data={"finish_reason": "stop"},
        ),
        ExecutionEvent(
            type=EventType.FRAME_COMPLETE,
            session_id="test_session",
            frame_id="frame_1",
            data={"status": "success"},
        ),
        ExecutionEvent(
            type=EventType.TURN_COMPLETE,
            session_id="test_session",
            data={"report": "test report"},
        ),
    ]


@pytest.fixture
def mock_event_stream(sample_events):
    """Fixture providing a mock async iterator of events."""

    async def event_generator():
        for event in sample_events:
            yield event
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming

    return event_generator()
