"""Pydantic models for dashboard data structures."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunSummary(BaseModel):
    """Summary model for a run in list views."""

    run_id: str
    session_id: str
    status: str = Field(description="Run status: running, completed, error")
    start_time: datetime
    end_time: datetime | None = None
    duration: float = Field(description="Duration in seconds")
    events_count: int = Field(description="Number of events in this run")
    tokens_used: int = Field(default=0, description="Total tokens used")


class RunDetail(RunSummary):
    """Detailed model for a single run."""

    events: list[dict[str, Any]] = Field(description="All events in the run")
    final_answer: str | None = Field(
        default=None, description="Final answer if available"
    )
    error_message: str | None = Field(
        default=None, description="Error message if failed"
    )


class FrameSummary(BaseModel):
    """Summary model for a frame in list views."""

    frame_id: str
    run_id: str
    status: str = Field(description="Frame status: running, completed, error")
    start_time: datetime
    end_time: datetime | None = None
    duration: float = Field(description="Duration in seconds")
    action_count: int = Field(default=0, description="Number of actions")


class FrameDetail(FrameSummary):
    """Detailed model for a single frame."""

    prompt_context: str | None = Field(default=None, description="Prompt context")
    llm_response: str | None = Field(default=None, description="LLM response")
    tool_calls: list[dict[str, Any]] = Field(default=[], description="Tool calls made")
    actions: list[dict[str, Any]] = Field(default=[], description="Action timeline")


class DashboardStats(BaseModel):
    """Dashboard statistics model."""

    total_runs: int
    success_rate: str
    avg_duration: str
    active_sessions: int
    completed_runs: int
    error_runs: int


class PaginatedResponse(BaseModel):
    """Generic paginated response model."""

    items: list[Any]
    total: int
    offset: int
    limit: int
    has_next: bool
    has_prev: bool


class RunsListResponse(PaginatedResponse):
    """Response model for runs list endpoint."""

    items: list[RunSummary]


class ActivityItem(BaseModel):
    """Recent activity item."""

    run_id: str
    status: str
    timestamp: str
    duration: float


class RecentActivityResponse(BaseModel):
    """Response model for recent activity endpoint."""

    activities: list[ActivityItem]
    last_updated: str
