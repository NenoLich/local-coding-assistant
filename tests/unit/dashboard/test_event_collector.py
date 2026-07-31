"""
Unit tests for EventCollector class.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from local_coding_assistant.dashboard.event_collector import (
    EventCollector,
    get_event_collector,
)
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class TestEventCollector:
    """Test EventCollector functionality."""

    @pytest.fixture
    def event_collector(self):
        """Create a fresh EventCollector instance for each test."""
        return EventCollector(max_events=100, max_recent_activity=50)

    @pytest.fixture
    def sample_events(self):
        """Create sample ExecutionEvents for testing."""
        now = datetime.now(UTC)
        session_id = "test-session-123"
        run_id = "test-run-456"
        frame_id = "test-frame-789"

        events = [
            ExecutionEvent(
                type=EventType.SESSION_START,
                session_id=session_id,
                timestamp=now,
                data={"user_query": "Test query"},
            ),
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                timestamp=now + timedelta(seconds=1),
                data={"run_id": run_id, "mode": "auto"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=2),
                data={"run_id": run_id, "frame_id": frame_id, "frame_number": 1},
            ),
            ExecutionEvent(
                type=EventType.TOOL_START,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=3),
                data={
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "tool_name": "test_tool",
                    "tool_args": {"param": "value"},
                },
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=4),
                data={"run_id": run_id, "frame_id": frame_id, "status": "completed"},
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                timestamp=now + timedelta(seconds=5),
                data={
                    "run_id": run_id,
                    "status": "completed",
                    "final_answer": "Test answer",
                },
            ),
        ]

        return events

    async def test_event_collector_initialization(self, event_collector):
        """Test EventCollector initialization."""
        assert event_collector.max_events == 100
        assert event_collector.max_recent_activity == 50
        assert len(event_collector._events) == 0
        assert len(event_collector._sessions) == 0
        assert len(event_collector._runs) == 0

    async def test_collect_single_event(self, event_collector, sample_events):
        """Test collecting a single event."""
        event = sample_events[0]

        await event_collector.collect_event(event)

        assert len(event_collector._events) == 1
        assert event_collector._events[0] == event

    async def test_collect_multiple_events(self, event_collector, sample_events):
        """Test collecting multiple events at once."""
        await event_collector.collect_events(sample_events)

        assert len(event_collector._events) == len(sample_events)
        for i, event in enumerate(sample_events):
            assert event_collector._events[i] == event

    async def test_session_creation_on_first_event(
        self, event_collector, sample_events
    ):
        """Test that session is created on first event."""
        event = sample_events[0]

        await event_collector.collect_event(event)

        assert event.session_id in event_collector._sessions
        session = event_collector._sessions[event.session_id]
        assert session["session_id"] == event.session_id
        assert session["start_time"] == event.timestamp
        assert session["last_activity"] == event.timestamp
        assert session["status"] == "running"
        assert session["events_count"] == 1

    async def test_session_activity_update(self, event_collector, sample_events):
        """Test that session activity is updated with each event."""
        # Collect first event
        await event_collector.collect_event(sample_events[0])

        # Collect second event
        await event_collector.collect_event(sample_events[1])

        session = event_collector._sessions[sample_events[0].session_id]
        assert session["events_count"] == 2
        assert session["last_activity"] == sample_events[1].timestamp

    async def test_run_creation_on_run_start(self, event_collector, sample_events):
        """Test that run is created on RUN_START event."""
        run_start_event = sample_events[1]  # RUN_START event

        await event_collector.collect_event(run_start_event)

        run_id = run_start_event.data["run_id"]
        assert run_id in event_collector._runs
        run = event_collector._runs[run_id]
        assert run["run_id"] == run_id
        assert run["session_id"] == run_start_event.session_id
        assert run["start_time"] == run_start_event.timestamp
        assert run["status"] == "running"

    async def test_run_update_on_run_end(self, event_collector, sample_events):
        """Test that run is updated on RUN_END event."""
        # First collect RUN_START
        await event_collector.collect_event(sample_events[1])

        # Then collect RUN_END
        run_end_event = sample_events[5]
        await event_collector.collect_event(run_end_event)

        run_id = run_end_event.data["run_id"]
        run = event_collector._runs[run_id]
        assert run["status"] == "completed"
        assert run["end_time"] == run_end_event.timestamp
        assert run["final_answer"] == "Test answer"

    async def test_frame_handling(self, event_collector, sample_events):
        """Test frame creation and updates."""
        session_id = sample_events[0].session_id
        run_id = sample_events[1].data["run_id"]
        frame_id = sample_events[2].data["frame_id"]

        # Collect session and turn start events first
        await event_collector.collect_event(sample_events[0])  # SESSION_START
        await event_collector.collect_event(sample_events[1])  # TURN_START

        # Collect frame events
        await event_collector.collect_event(sample_events[2])  # FRAME_START
        await event_collector.collect_event(sample_events[3])  # TOOL_CALL
        await event_collector.collect_event(sample_events[4])  # FRAME_END

        session = event_collector._sessions[session_id]
        assert len(session["frames"]) == 1

        frame = session["frames"][0]
        assert frame["frame_id"] == frame_id
        assert frame["run_id"] == run_id
        assert frame["status"] == "completed"
        assert len(frame["events"]) == 3  # START, TOOL_CALL, END

    async def test_max_events_limit(self, event_collector):
        """Test that events are limited to max_events."""
        # Create more events than the limit
        events = []
        for i in range(150):  # More than max_events (100)
            events.append(
                ExecutionEvent(
                    type=EventType.TOOL_START,
                    session_id="test-session",
                    timestamp=datetime.now(UTC) + timedelta(seconds=i),
                    data={"tool_name": f"tool_{i}", "tool_args": {}},
                )
            )

        await event_collector.collect_events(events)

        # Should only keep max_events (100) most recent events
        assert len(event_collector._events) == 100
        # Should keep the most recent events
        assert event_collector._events[-1].data["tool_name"] == "tool_149"

    async def test_get_sessions(self, event_collector, sample_events):
        """Test getting sessions list."""
        await event_collector.collect_events(sample_events)

        sessions = await event_collector.get_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sample_events[0].session_id

    async def test_get_runs(self, event_collector, sample_events):
        """Test getting runs list."""
        await event_collector.collect_events(sample_events)

        runs = await event_collector.get_runs_simple()
        assert len(runs) == 1
        assert runs[0]["run_id"] == sample_events[1].data["run_id"]

    async def test_get_events_by_session(self, event_collector, sample_events):
        """Test getting events by session ID."""
        await event_collector.collect_events(sample_events)

        session_events = await event_collector.get_events_by_session(
            sample_events[0].session_id
        )
        assert len(session_events) == len(sample_events)

        # Test with non-existent session
        other_events = await event_collector.get_events_by_session("non-existent")
        assert len(other_events) == 0

    async def test_get_events_by_run(self, event_collector, sample_events):
        """Test getting events by run ID."""
        await event_collector.collect_events(sample_events)

        run_id = sample_events[1].data["run_id"]
        run_events = await event_collector.get_events_by_run(run_id)

        # Should include events from the run
        assert len(run_events) >= 1  # At least the TURN_START event

        # Test with non-existent run
        other_events = await event_collector.get_events_by_run("non-existent")
        assert len(other_events) == 0

    async def test_get_events_by_frame(self, event_collector, sample_events):
        """Test getting events by frame ID."""
        await event_collector.collect_events(sample_events)

        frame_id = sample_events[2].data["frame_id"]
        frame_events = await event_collector.get_events_by_frame(frame_id)

        # Should include FRAME_START, TOOL_CALL, FRAME_END
        assert len(frame_events) == 3

        # Test with non-existent frame
        other_events = await event_collector.get_events_by_frame("non-existent")
        assert len(other_events) == 0

    async def test_get_recent_activity(self, event_collector, sample_events):
        """Test getting recent activity."""
        await event_collector.collect_events(sample_events)

        activity = await event_collector.get_recent_activity_simple()
        assert len(activity) == 1  # One run completed
        assert activity[0]["run_id"] == sample_events[1].data["run_id"]
        assert activity[0]["status"] == "completed"

    async def test_get_dashboard_stats(self, event_collector, sample_events):
        """Test getting dashboard statistics."""
        await event_collector.collect_events(sample_events)

        stats = await event_collector.get_dashboard_stats()

        assert stats["total_runs"] == 1
        assert stats["completed_runs"] == 1
        assert stats["error_runs"] == 0
        assert stats["active_sessions"] == 1

    async def test_error_handling(self, event_collector):
        """Test error handling with malformed events."""
        # Test with event missing required data
        malformed_event = ExecutionEvent(
            type=EventType.TURN_START,
            session_id="test-session",
            timestamp=datetime.now(UTC),
            data={},  # Missing run_id
        )

        # Should not raise exception
        await event_collector.collect_event(malformed_event)

        # Event should still be stored
        assert len(event_collector._events) == 1

    async def test_concurrent_event_collection(self, event_collector, sample_events):
        """Test concurrent event collection."""
        # Create multiple tasks to collect events concurrently
        tasks = []
        for event in sample_events:
            task = asyncio.create_task(event_collector.collect_event(event))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # All events should be collected
        assert len(event_collector._events) == len(sample_events)

    @patch(
        "local_coding_assistant.dashboard.event_collector.EventCollector._broadcast_event"
    )
    async def test_broadcast_event_called(
        self, mock_broadcast, event_collector, sample_events
    ):
        """Test that _broadcast_event is called when collecting events."""
        # Make broadcast_event a proper async mock
        mock_broadcast.return_value = None

        await event_collector.collect_event(sample_events[0])

        # Should call broadcast_event once
        mock_broadcast.assert_called_once()

    async def test_get_paginated_runs(self, event_collector, sample_events):
        """Test getting paginated runs."""
        await event_collector.collect_events(sample_events)

        # Test first page
        page1 = await event_collector.get_paginated_runs(offset=0, limit=10)
        assert len(page1["items"]) == 1
        assert page1["total"] == 1
        assert page1["offset"] == 0
        assert page1["limit"] == 10
        assert page1["has_next"] is False
        assert page1["has_prev"] is False

    async def test_multiple_sessions_and_runs(self, event_collector):
        """Test handling multiple sessions and runs."""
        now = datetime.now(UTC)

        # Create events for multiple sessions and runs
        events = []
        for session_idx in range(2):
            session_id = f"session-{session_idx}"
            for run_idx in range(2):
                run_id = f"run-{session_idx}-{run_idx}"

                events.extend(
                    [
                        ExecutionEvent(
                            type=EventType.SESSION_START,
                            session_id=session_id,
                            timestamp=now
                            + timedelta(seconds=session_idx * 100 + run_idx * 10),
                            data={"user_query": f"Query {session_idx}-{run_idx}"},
                        ),
                        ExecutionEvent(
                            type=EventType.TURN_START,
                            session_id=session_id,
                            timestamp=now
                            + timedelta(seconds=session_idx * 100 + run_idx * 10 + 1),
                            data={"run_id": run_id, "mode": "auto"},
                        ),
                        ExecutionEvent(
                            type=EventType.TURN_COMPLETE,
                            session_id=session_id,
                            timestamp=now
                            + timedelta(seconds=session_idx * 100 + run_idx * 10 + 5),
                            data={"run_id": run_id, "status": "completed"},
                        ),
                    ]
                )

        await event_collector.collect_events(events)

        sessions = await event_collector.get_sessions()
        runs = await event_collector.get_runs_simple()

        assert len(sessions) == 2
        assert len(runs) == 4


class TestEventCollectorSingleton:
    """Test EventCollector singleton pattern."""

    @patch("local_coding_assistant.dashboard.event_collector._event_collector", None)
    def test_get_event_collector_creates_instance(self):
        """Test that get_event_collector creates a new instance when none exists."""
        collector = get_event_collector()
        assert isinstance(collector, EventCollector)

    @patch("local_coding_assistant.dashboard.event_collector._event_collector", None)
    def test_get_event_collector_returns_same_instance(self):
        """Test that get_event_collector returns the same instance on subsequent calls."""
        collector1 = get_event_collector()
        collector2 = get_event_collector()
        assert collector1 is collector2

    @patch("local_coding_assistant.dashboard.event_collector._event_collector", Mock())
    def test_get_event_collector_returns_existing(self):
        """Test that get_event_collector returns existing instance."""
        mock_collector = Mock()
        with patch(
            "local_coding_assistant.dashboard.event_collector._event_collector",
            mock_collector,
        ):
            collector = get_event_collector()
            assert collector is mock_collector


class TestEventCollectorIntegration:
    """Integration tests for EventCollector with realistic scenarios."""

    async def test_complete_workflow(self):
        """Test a complete workflow from session start to run end."""
        collector = EventCollector()
        now = datetime.now(UTC)
        session_id = "workflow-session"
        run_id = "workflow-run"
        frame_id = "workflow-frame"

        # Complete workflow events
        events = [
            ExecutionEvent(
                type=EventType.SESSION_START,
                session_id=session_id,
                timestamp=now,
                data={"user_query": "Complete test query"},
            ),
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                timestamp=now + timedelta(seconds=1),
                data={"run_id": run_id, "mode": "auto"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=2),
                data={"run_id": run_id, "frame_id": frame_id, "frame_number": 1},
            ),
            ExecutionEvent(
                type=EventType.LLM_START,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=3),
                data={
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "model": "test-model",
                    "prompt": "Test prompt",
                    "response": "Test response",
                    "tokens_used": 100,
                },
            ),
            ExecutionEvent(
                type=EventType.TOOL_START,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=4),
                data={
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "tool_name": "test_tool",
                    "tool_args": {"param": "value"},
                    "result": "success",
                },
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                timestamp=now + timedelta(seconds=5),
                data={"run_id": run_id, "frame_id": frame_id, "status": "completed"},
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                timestamp=now + timedelta(seconds=6),
                data={
                    "run_id": run_id,
                    "status": "completed",
                    "final_answer": "Final answer",
                },
            ),
        ]

        await collector.collect_events(events)

        # Verify session
        sessions = await collector.get_sessions()
        assert len(sessions) == 1
        session = sessions[0]
        assert session["session_id"] == session_id
        assert session["events_count"] == 7
        assert len(session["frames"]) == 1

        # Verify run
        runs = await collector.get_runs_simple()
        assert len(runs) == 1
        run = runs[0]
        assert run["run_id"] == run_id
        assert run["status"] == "completed"

        # Verify frame
        frame = session["frames"][0]
        assert frame["frame_id"] == frame_id
        assert frame["run_id"] == run_id
        assert frame["status"] == "completed"
        assert len(frame["events"]) == 4  # START, LLM_CALL, TOOL_CALL, END
        stats = await collector.get_dashboard_stats()
        assert stats["total_runs"] == 1
        assert stats["completed_runs"] == 1
        assert stats["active_sessions"] == 1
