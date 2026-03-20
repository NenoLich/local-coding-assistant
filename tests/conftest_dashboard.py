"""
Dashboard-specific test fixtures and helpers.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app
from local_coding_assistant.dashboard.event_collector import EventCollector
from local_coding_assistant.dashboard.models import (
    ActivityItem,
    DashboardStats,
    FrameDetail,
    FrameSummary,
    RunDetail,
    RunSummary,
)
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


@pytest.fixture
def dashboard_client():
    """Create a FastAPI test client for the dashboard."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_event_collector():
    """Create a mock EventCollector for testing."""
    collector = AsyncMock(spec=EventCollector)
    
    # Setup default return values
    collector.get_dashboard_stats.return_value = {
        "total_runs": 10,
        "success_rate": "85.0%",
        "avg_duration": "2m 30s",
        "active_sessions": 2,
        "completed_runs": 8,
        "error_runs": 2,
    }
    
    collector.get_recent_activity.return_value = {
        "activities": [
            {
                "run_id": "test-run-1",
                "status": "completed",
                "timestamp": "2024-01-01T12:00:00Z",
                "duration": 120.5,
            }
        ],
        "last_updated": "2024-01-01T12:05:00Z",
    }
    
    collector.get_runs.return_value = {
        "runs": [
            {
                "run_id": "test-run-1",
                "session_id": "test-session-1",
                "status": "completed",
                "start_time": datetime.now(UTC),
                "end_time": datetime.now(UTC) + timedelta(minutes=2),
                "duration": 120.0,
                "events_count": 10,
            }
        ]
    }
    
    collector.get_run_details.return_value = {
        "run_id": "test-run-1",
        "session_id": "test-session-1",
        "status": "completed",
        "start_time": datetime.now(UTC),
        "end_time": datetime.now(UTC) + timedelta(minutes=2),
        "duration": 120.0,
        "events_count": 10,
        "events": [
            {"type": "run_start", "timestamp": datetime.now(UTC)},
            {"type": "run_end", "timestamp": datetime.now(UTC)},
        ],
        "final_answer": "Test final answer",
    }
    
    collector.get_frame_details.return_value = {
        "frame_id": "test-frame-1",
        "run_id": "test-run-1",
        "status": "completed",
        "start_time": datetime.now(UTC),
        "end_time": datetime.now(UTC) + timedelta(seconds=30),
        "duration": 30.0,
        "events_count": 5,
        "events": [
            {"type": "frame_start", "timestamp": datetime.now(UTC)},
            {"type": "frame_end", "timestamp": datetime.now(UTC)},
        ],
        "prompt_context": "Test prompt context",
        "llm_response": "Test LLM response",
        "tool_calls": [
            {"name": "test_tool", "args": {"param": "value"}}
        ],
    }
    
    collector.get_events_by_session.return_value = [
        {
            "type": "session_start",
            "session_id": "test-session-1",
            "timestamp": datetime.now(UTC),
            "data": {"user_query": "Test query"},
        }
    ]
    
    collector.get_active_sessions.return_value = [
        {
            "session_id": "test-session-1",
            "status": "running",
            "start_time": datetime.now(UTC),
            "events_count": 5,
        }
    ]
    
    return collector


@pytest.fixture
def sample_execution_events():
    """Create sample ExecutionEvents for testing."""
    now = datetime.now(UTC)
    session_id = "test-session-123"
    run_id = "test-run-456"
    frame_id = "test-frame-789"
    
    return [
        ExecutionEvent(
            type=EventType.SESSION_START,
            session_id=session_id,
            timestamp=now,
            data={"user_query": "Test query for dashboard"},
        ),
        ExecutionEvent(
            type=EventType.RUN_START,
            session_id=session_id,
            timestamp=now + timedelta(seconds=1),
            data={"run_id": run_id, "mode": "auto"},
        ),
        ExecutionEvent(
            type=EventType.FRAME_START,
            session_id=session_id,
            timestamp=now + timedelta(seconds=2),
            data={"run_id": run_id, "frame_id": frame_id, "frame_number": 1},
        ),
        ExecutionEvent(
            type=EventType.LLM_CALL,
            session_id=session_id,
            timestamp=now + timedelta(seconds=3),
            data={
                "run_id": run_id,
                "frame_id": frame_id,
                "model": "test-model",
                "prompt": "Test prompt",
                "response": "Test response",
                "tokens_used": 150,
            },
        ),
        ExecutionEvent(
            type=EventType.TOOL_CALL,
            session_id=session_id,
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
            type=EventType.FRAME_END,
            session_id=session_id,
            timestamp=now + timedelta(seconds=5),
            data={"run_id": run_id, "frame_id": frame_id, "status": "completed"},
        ),
        ExecutionEvent(
            type=EventType.RUN_END,
            session_id=session_id,
            timestamp=now + timedelta(seconds=6),
            data={"run_id": run_id, "status": "completed", "final_answer": "Test answer"},
        ),
    ]


@pytest.fixture
def sample_run_summary():
    """Create a sample RunSummary for testing."""
    return RunSummary(
        run_id="test-run-123",
        session_id="test-session-456",
        status="completed",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(minutes=2),
        duration=120.0,
        events_count=10,
        tokens_used=500,
    )


@pytest.fixture
def sample_run_detail():
    """Create a sample RunDetail for testing."""
    return RunDetail(
        run_id="test-run-123",
        session_id="test-session-456",
        status="completed",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(minutes=2),
        duration=120.0,
        events_count=10,
        tokens_used=500,
        events=[
            {"type": "run_start", "timestamp": datetime.now(UTC)},
            {"type": "llm_call", "timestamp": datetime.now(UTC)},
            {"type": "run_end", "timestamp": datetime.now(UTC)},
        ],
        final_answer="Test final answer",
    )


@pytest.fixture
def sample_frame_summary():
    """Create a sample FrameSummary for testing."""
    return FrameSummary(
        frame_id="test-frame-123",
        run_id="test-run-456",
        status="completed",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(seconds=30),
        duration=30.0,
        action_count=5,
    )


@pytest.fixture
def sample_frame_detail():
    """Create a sample FrameDetail for testing."""
    return FrameDetail(
        frame_id="test-frame-123",
        run_id="test-run-456",
        status="completed",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC) + timedelta(seconds=30),
        duration=30.0,
        action_count=5,
        prompt_context="Test prompt context",
        llm_response="Test LLM response",
        tool_calls=[
            {"name": "test_tool", "args": {"param": "value"}}
        ],
        actions=[
            {"type": "tool_call", "timestamp": datetime.now(UTC)},
        ],
    )


@pytest.fixture
def sample_dashboard_stats():
    """Create sample DashboardStats for testing."""
    return DashboardStats(
        total_runs=100,
        success_rate="85.5%",
        avg_duration="2m 30s",
        active_sessions=3,
        completed_runs=85,
        error_runs=15,
    )


@pytest.fixture
def sample_activity_items():
    """Create sample ActivityItems for testing."""
    return [
        ActivityItem(
            run_id="test-run-1",
            status="completed",
            timestamp="2024-01-01T12:00:00Z",
            duration=120.5,
        ),
        ActivityItem(
            run_id="test-run-2",
            status="error",
            timestamp="2024-01-01T11:30:00Z",
            duration=45.2,
        ),
        ActivityItem(
            run_id="test-run-3",
            status="running",
            timestamp="2024-01-01T12:15:00Z",
            duration=30.0,
        ),
    ]


@pytest.fixture
def event_collector_instance():
    """Create a real EventCollector instance for testing."""
    return EventCollector(max_events=1000, max_recent_activity=100)


class DashboardTestHelpers:
    """Helper class for dashboard testing."""

    @staticmethod
    def create_event_data(
        event_type: str,
        session_id: str,
        timestamp: str | None = None,
        data: dict[str, Any] | None = None,
        frame_id: str | None = None,
    ) -> dict[str, Any]:
        """Create event data in the format expected by the API."""
        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()
        
        if data is None:
            data = {}
        
        event_data = {
            "type": event_type,
            "session_id": session_id,
            "timestamp": timestamp,
            "data": data,
        }
        
        if frame_id:
            event_data["frame_id"] = frame_id
        
        return event_data

    @staticmethod
    def create_complete_session_events(
        session_id: str,
        run_id: str,
        frame_id: str,
        base_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Create a complete set of events for a session."""
        if base_time is None:
            base_time = datetime.now(UTC)
        
        return [
            DashboardTestHelpers.create_event_data(
                "session_start",
                session_id,
                (base_time).isoformat(),
                {"user_query": f"Test query for {session_id}"},
            ),
            DashboardTestHelpers.create_event_data(
                "run_start",
                session_id,
                (base_time + timedelta(seconds=1)).isoformat(),
                {"run_id": run_id, "mode": "auto"},
            ),
            DashboardTestHelpers.create_event_data(
                "frame_start",
                session_id,
                (base_time + timedelta(seconds=2)).isoformat(),
                {"run_id": run_id, "frame_id": frame_id, "frame_number": 1},
                frame_id,
            ),
            DashboardTestHelpers.create_event_data(
                "llm_call",
                session_id,
                (base_time + timedelta(seconds=3)).isoformat(),
                {
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "model": "test-model",
                    "prompt": "Test prompt",
                    "response": "Test response",
                    "tokens_used": 150,
                },
                frame_id,
            ),
            DashboardTestHelpers.create_event_data(
                "tool_call",
                session_id,
                (base_time + timedelta(seconds=4)).isoformat(),
                {
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "tool_name": "test_tool",
                    "tool_args": {"param": "value"},
                    "result": "success",
                },
                frame_id,
            ),
            DashboardTestHelpers.create_event_data(
                "frame_end",
                session_id,
                (base_time + timedelta(seconds=5)).isoformat(),
                {"run_id": run_id, "frame_id": frame_id, "status": "completed"},
                frame_id,
            ),
            DashboardTestHelpers.create_event_data(
                "run_end",
                session_id,
                (base_time + timedelta(seconds=6)).isoformat(),
                {"run_id": run_id, "status": "completed", "final_answer": "Test answer"},
            ),
        ]

    @staticmethod
    def assert_event_collected(mock_collector: Mock, expected_event_type: str):
        """Assert that an event of the expected type was collected."""
        mock_collector.collect_event.assert_called()
        call_args = mock_collector.collect_event.call_args
        if call_args:
            event = call_args[0][0]
            assert event.type.value == expected_event_type

    @staticmethod
    def assert_api_response_success(response: Any, expected_status_code: int = 200):
        """Assert that API response is successful."""
        assert response.status_code == expected_status_code
        return response.json()

    @staticmethod
    def create_mock_websocket():
        """Create a mock WebSocket for testing."""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.receive_json = AsyncMock()
        return websocket

    @staticmethod
    def create_performance_test_data(num_sessions: int, events_per_session: int) -> list[dict[str, Any]]:
        """Create performance test data with multiple sessions."""
        all_events = []
        base_time = datetime.now(UTC)
        
        for session_idx in range(num_sessions):
            session_id = f"perf-session-{session_idx}"
            run_id = f"perf-run-{session_idx}"
            
            for event_idx in range(events_per_session):
                event_time = base_time + timedelta(
                    seconds=session_idx * 10 + event_idx
                )
                
                event_data = DashboardTestHelpers.create_event_data(
                    "tool_call",
                    session_id,
                    event_time.isoformat(),
                    {
                        "run_id": run_id,
                        "tool_name": f"tool_{event_idx}",
                        "tool_args": {"index": event_idx},
                    },
                )
                all_events.append(event_data)
        
        return all_events


@pytest.fixture
def dashboard_helpers():
    """Provide DashboardTestHelpers as a fixture."""
    return DashboardTestHelpers


# Patch fixtures for common mocking scenarios
@pytest.fixture
def patch_event_collector():
    """Patch the get_event_collector function."""
    with pytest.MonkeyPatch().context() as m:
        mock_collector = AsyncMock()
        m.setattr(
            "local_coding_assistant.dashboard.routes.api.get_event_collector",
            lambda: mock_collector,
        )
        m.setattr(
            "local_coding_assistant.dashboard.routes.websocket.get_event_collector",
            lambda: mock_collector,
        )
        yield mock_collector


@pytest.fixture
def patch_connection_manager():
    """Patch the WebSocket connection manager."""
    with pytest.MonkeyPatch().context() as m:
        mock_manager = Mock()
        mock_manager.connect = AsyncMock()
        mock_manager.disconnect = AsyncMock()
        mock_manager.broadcast_event = AsyncMock()
        mock_manager.broadcast_stats_update = AsyncMock()
        mock_manager.broadcast_sessions_update = AsyncMock()
        mock_manager.active_connections = []
        
        m.setattr(
            "local_coding_assistant.dashboard.routes.websocket.manager",
            mock_manager,
        )
        yield mock_manager


# Async test helpers
@pytest.fixture
def async_test_event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Data validation helpers
@pytest.fixture
def validate_dashboard_response():
    """Helper to validate dashboard API responses."""
    def _validate(response_data: dict[str, Any], expected_fields: list[str]):
        """Validate that response contains expected fields."""
        for field in expected_fields:
            assert field in response_data, f"Missing field: {field}"
        return True
    
    return _validate


# Error scenario helpers
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "invalid_timestamp": {
            "type": "session_start",
            "session_id": "test-session",
            "timestamp": "invalid-timestamp",
            "data": {},
        },
        "missing_session_id": {
            "type": "session_start",
            "timestamp": "2024-01-01T12:00:00",
            "data": {},
        },
        "invalid_event_type": {
            "type": "invalid_type",
            "session_id": "test-session",
            "timestamp": "2024-01-01T12:00:00",
            "data": {},
        },
        "missing_data": {
            "type": "tool_call",
            "session_id": "test-session",
            "timestamp": "2024-01-01T12:00:00",
        },
    }


# Performance measurement helpers
@pytest.fixture
def performance_timer():
    """Helper to measure performance in tests."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
        
        @property
        def duration(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
    
    return Timer()
