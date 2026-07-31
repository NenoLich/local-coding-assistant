"""
Unit tests for dashboard models.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from local_coding_assistant.dashboard.models import (
    ActivityItem,
    DashboardStats,
    FrameDetail,
    FrameSummary,
    PaginatedResponse,
    RecentActivityResponse,
    RunDetail,
    RunsListResponse,
    RunSummary,
)


class TestRunSummary:
    """Test RunSummary model."""

    def test_valid_run_summary_creation(self):
        """Test creating a valid RunSummary."""
        run_id = "test-run-123"
        session_id = "test-session-456"
        start_time = datetime.now()
        duration = 120.5
        events_count = 10

        run = RunSummary(
            run_id=run_id,
            session_id=session_id,
            status="completed",
            start_time=start_time,
            duration=duration,
            events_count=events_count,
        )

        assert run.run_id == run_id
        assert run.session_id == session_id
        assert run.status == "completed"
        assert run.start_time == start_time
        assert run.duration == duration
        assert run.events_count == events_count
        assert run.tokens_used == 0  # Default value
        assert run.end_time is None  # Optional field

    def test_run_summary_with_all_fields(self):
        """Test creating RunSummary with all fields."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=120)

        run = RunSummary(
            run_id="test-run",
            session_id="test-session",
            status="completed",
            start_time=start_time,
            end_time=end_time,
            duration=120.0,
            events_count=5,
            tokens_used=1000,
        )

        assert run.end_time == end_time
        assert run.tokens_used == 1000

    def test_invalid_status_values(self):
        """Test that invalid status values are still accepted (Pydantic doesn't enforce enum by default)."""
        start_time = datetime.now()

        # This should work since status is just a string field
        run = RunSummary(
            run_id="test-run",
            session_id="test-session",
            status="invalid_status",  # Not in the documented enum
            start_time=start_time,
            duration=10.0,
            events_count=1,
        )

        assert run.status == "invalid_status"

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        start_time = datetime.now()

        with pytest.raises(ValidationError) as exc_info:
            RunSummary(
                run_id="test-run",
                # Missing session_id
                status="completed",
                start_time=start_time,
                duration=10.0,
                events_count=1,
            )

        assert "session_id" in str(exc_info.value)

    def test_invalid_duration_type(self):
        """Test that invalid duration type raises ValidationError."""
        start_time = datetime.now()

        with pytest.raises(ValidationError) as exc_info:
            RunSummary(
                run_id="test-run",
                session_id="test-session",
                status="completed",
                start_time=start_time,
                duration="invalid",  # Should be float
                events_count=1,
            )

        assert "duration" in str(exc_info.value)


class TestRunDetail:
    """Test RunDetail model."""

    def test_run_detail_inheritance(self):
        """Test that RunDetail inherits from RunSummary."""
        start_time = datetime.now()
        events = [{"type": "test", "data": "value"}]

        run_detail = RunDetail(
            run_id="test-run",
            session_id="test-session",
            status="completed",
            start_time=start_time,
            duration=120.0,
            events_count=2,
            events=events,
            final_answer="Test answer",
        )

        # Should have all RunSummary fields
        assert run_detail.run_id == "test-run"
        assert run_detail.session_id == "test-session"

        # Should have additional RunDetail fields
        assert run_detail.events == events
        assert run_detail.final_answer == "Test answer"
        assert run_detail.error_message is None

    def test_run_detail_with_error(self):
        """Test RunDetail with error message."""
        start_time = datetime.now()

        run_detail = RunDetail(
            run_id="test-run",
            session_id="test-session",
            status="error",
            start_time=start_time,
            duration=60.0,
            events_count=1,
            events=[],
            error_message="Something went wrong",
        )

        assert run_detail.status == "error"
        assert run_detail.error_message == "Something went wrong"
        assert run_detail.final_answer is None


class TestFrameSummary:
    """Test FrameSummary model."""

    def test_valid_frame_summary(self):
        """Test creating a valid FrameSummary."""
        start_time = datetime.now()

        frame = FrameSummary(
            frame_id="frame-123",
            run_id="run-456",
            status="completed",
            start_time=start_time,
            duration=30.5,
            action_count=5,
        )

        assert frame.frame_id == "frame-123"
        assert frame.run_id == "run-456"
        assert frame.status == "completed"
        assert frame.action_count == 5

    def test_frame_summary_defaults(self):
        """Test FrameSummary with default values."""
        start_time = datetime.now()

        frame = FrameSummary(
            frame_id="frame-123",
            run_id="run-456",
            status="running",
            start_time=start_time,
            duration=0.0,
            # action_count should default to 0
        )

        assert frame.action_count == 0


class TestFrameDetail:
    """Test FrameDetail model."""

    def test_frame_detail_inheritance(self):
        """Test that FrameDetail inherits from FrameSummary."""
        start_time = datetime.now()
        tool_calls = [{"name": "test_tool", "args": {"param": "value"}}]
        actions = [{"type": "tool_call", "timestamp": start_time.isoformat()}]

        frame_detail = FrameDetail(
            frame_id="frame-123",
            run_id="run-456",
            status="completed",
            start_time=start_time,
            duration=45.0,
            action_count=3,
            prompt_context="Test context",
            llm_response="Test response",
            tool_calls=tool_calls,
            actions=actions,
        )

        # Should inherit all FrameSummary fields
        assert frame_detail.frame_id == "frame-123"
        assert frame_detail.action_count == 3

        # Should have additional FrameDetail fields
        assert frame_detail.prompt_context == "Test context"
        assert frame_detail.llm_response == "Test response"
        assert frame_detail.tool_calls == tool_calls
        assert frame_detail.actions == actions


class TestDashboardStats:
    """Test DashboardStats model."""

    def test_valid_dashboard_stats(self):
        """Test creating valid DashboardStats."""
        stats = DashboardStats(
            total_runs=100,
            success_rate="85.5%",
            avg_duration="2m 30s",
            active_sessions=3,
            completed_runs=85,
            error_runs=15,
        )

        assert stats.total_runs == 100
        assert stats.success_rate == "85.5%"
        assert stats.avg_duration == "2m 30s"
        assert stats.active_sessions == 3
        assert stats.completed_runs == 85
        assert stats.error_runs == 15

    def test_stats_relationships(self):
        """Test that stats relationships make sense."""
        stats = DashboardStats(
            total_runs=50,
            success_rate="80.0%",
            avg_duration="1m 15s",
            active_sessions=2,
            completed_runs=40,
            error_runs=10,
        )

        # Total should equal completed + error + (running, if any)
        assert stats.total_runs >= stats.completed_runs + stats.error_runs


class TestPaginatedResponse:
    """Test PaginatedResponse model."""

    def test_paginated_response_first_page(self):
        """Test paginated response for first page."""
        items = [{"id": 1}, {"id": 2}]

        response = PaginatedResponse(
            items=items,
            total=10,
            offset=0,
            limit=2,
            has_next=True,
            has_prev=False,
        )

        assert response.items == items
        assert response.total == 10
        assert response.offset == 0
        assert response.limit == 2
        assert response.has_next is True
        assert response.has_prev is False

    def test_paginated_response_middle_page(self):
        """Test paginated response for middle page."""
        items = [{"id": 3}, {"id": 4}]

        response = PaginatedResponse(
            items=items,
            total=10,
            offset=2,
            limit=2,
            has_next=True,
            has_prev=True,
        )

        assert response.has_next is True
        assert response.has_prev is True

    def test_paginated_response_last_page(self):
        """Test paginated response for last page."""
        items = [{"id": 9}]

        response = PaginatedResponse(
            items=items,
            total=10,
            offset=9,
            limit=2,
            has_next=False,
            has_prev=True,
        )

        assert response.has_next is False
        assert response.has_prev is True


class TestRunsListResponse:
    """Test RunsListResponse model."""

    def test_runs_list_response(self):
        """Test RunsListResponse with proper typing."""
        runs = [
            RunSummary(
                run_id="run-1",
                session_id="session-1",
                status="completed",
                start_time=datetime.now(),
                duration=60.0,
                events_count=5,
            ),
            RunSummary(
                run_id="run-2",
                session_id="session-2",
                status="running",
                start_time=datetime.now(),
                duration=30.0,
                events_count=3,
            ),
        ]

        response = RunsListResponse(
            items=runs,
            total=2,
            offset=0,
            limit=10,
            has_next=False,
            has_prev=False,
        )

        assert len(response.items) == 2
        assert isinstance(response.items[0], RunSummary)
        assert response.items[0].run_id == "run-1"
        assert response.items[1].run_id == "run-2"


class TestActivityItem:
    """Test ActivityItem model."""

    def test_activity_item_creation(self):
        """Test creating ActivityItem."""
        timestamp = "2024-01-01T12:00:00Z"

        activity = ActivityItem(
            run_id="run-123",
            status="completed",
            timestamp=timestamp,
            duration=120.5,
        )

        assert activity.run_id == "run-123"
        assert activity.status == "completed"
        assert activity.timestamp == timestamp
        assert activity.duration == 120.5


class TestRecentActivityResponse:
    """Test RecentActivityResponse model."""

    def test_recent_activity_response(self):
        """Test RecentActivityResponse."""
        activities = [
            ActivityItem(
                run_id="run-1",
                status="completed",
                timestamp="2024-01-01T12:00:00Z",
                duration=60.0,
            ),
            ActivityItem(
                run_id="run-2",
                status="error",
                timestamp="2024-01-01T11:30:00Z",
                duration=30.0,
            ),
        ]
        last_updated = "2024-01-01T12:05:00Z"

        response = RecentActivityResponse(
            activities=activities,
            last_updated=last_updated,
        )

        assert len(response.activities) == 2
        assert response.activities[0].run_id == "run-1"
        assert response.activities[1].status == "error"
        assert response.last_updated == last_updated


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_run_summary_serialization(self):
        """Test RunSummary serialization to dict."""
        start_time = datetime.now()

        run = RunSummary(
            run_id="test-run",
            session_id="test-session",
            status="completed",
            start_time=start_time,
            duration=120.0,
            events_count=5,
            tokens_used=1000,
        )

        data = run.model_dump()

        assert data["run_id"] == "test-run"
        assert data["status"] == "completed"
        assert data["duration"] == 120.0
        assert data["tokens_used"] == 1000
        # datetime should be serialized as datetime object by default
        assert isinstance(data["start_time"], datetime)
        assert data["start_time"] == start_time

    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        start_time = datetime.now()

        run = RunSummary(
            run_id="test-run",
            session_id="test-session",
            status="completed",
            start_time=start_time,
            duration=120.0,
            events_count=5,
        )

        json_str = run.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-run" in json_str
        assert "completed" in json_str

    def test_nested_model_serialization(self):
        """Test serialization of nested models."""
        runs = [
            RunSummary(
                run_id="run-1",
                session_id="session-1",
                status="completed",
                start_time=datetime.now(),
                duration=60.0,
                events_count=5,
            )
        ]

        response = RunsListResponse(
            items=runs,
            total=1,
            offset=0,
            limit=10,
            has_next=False,
            has_prev=False,
        )

        data = response.model_dump()

        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["run_id"] == "run-1"
