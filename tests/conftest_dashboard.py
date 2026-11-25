"""
Dashboard-specific test fixtures and helpers for the refactored dashboard.

Provides fixtures for the live (LiveMonitor) and historical (store-backed)
planes of the dashboard, plus builders for typed RuntimeEvent contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app
from local_coding_assistant.dashboard.models.live import (
    LiveEvent,
    LiveFrame,
    LiveRunSummary,
    LiveSession,
    LiveStats,
    LiveTurn,
)
from local_coding_assistant.dashboard.services.live_monitor import LiveMonitor
from local_coding_assistant.runtime.events.contracts import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ContinuationDecisionEvent,
    ErrorEvent,
    EventType,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    PartialResponseEvent,
    RuntimeEvent,
    SessionBranchEvent,
    SessionInterruptEvent,
    SessionResumeEvent,
    SessionStartEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.storage.contracts import (
    StoredFrame,
    StoredSession,
    StoredTurn,
)
from local_coding_assistant.storage.execution_store_reader import ExecutionStoreReader


@pytest.fixture
def live_monitor() -> LiveMonitor:
    """Fresh in-memory live monitor."""
    return LiveMonitor()


@pytest.fixture
def mock_store() -> AsyncMock:
    """AsyncMock standing in for a read-only ExecutionStoreReader."""
    return AsyncMock(spec=ExecutionStoreReader)


@pytest.fixture
def dashboard_client() -> TestClient:
    """FastAPI test client for the dashboard app (lifespan runs)."""
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_with_store(
    mock_store: AsyncMock, dashboard_client: TestClient
) -> TestClient:
    """Test client whose app.state.execution_store is a mock store."""
    dashboard_client.app.state.execution_store = mock_store
    return dashboard_client


@pytest.fixture
def make_runtime_event():
    """Factory for typed RuntimeEvent contracts with sensible defaults."""

    def _factory(model_cls: type[RuntimeEvent], **kwargs: Any) -> RuntimeEvent:
        base = {
            "trace_id": "trace-1",
            "turn_id": "turn-1",
            "session_id": "session-1",
            "seq": 0,
        }
        base.update(kwargs)
        return model_cls(**base)

    return _factory


@pytest.fixture
def event_data():
    """Factory for /api/events/ingest payloads.

    The ingest handler builds the typed event from ``data`` plus top-level
    ``session_id``/``frame_id``/``timestamp``, so ``trace_id`` and ``turn_id``
    must live inside ``data``.
    """

    def _factory(
        event_type: str,
        session_id: str = "session-1",
        turn_id: str = "turn-1",
        frame_id: str | None = None,
        data: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        if timestamp is None:
            timestamp = datetime.now(UTC).isoformat()
        payload_data = dict(data or {})
        payload_data.setdefault("trace_id", "trace-1")
        payload_data.setdefault("turn_id", turn_id)
        payload_data.setdefault("session_id", session_id)
        payload: dict[str, Any] = {
            "type": event_type,
            "session_id": session_id,
            "turn_id": turn_id,
            "data": payload_data,
            "timestamp": timestamp,
        }
        if frame_id is not None:
            payload["frame_id"] = frame_id
        return payload

    return _factory


@pytest.fixture
def stored_session() -> StoredSession:
    """A sample stored session."""
    return StoredSession(
        id="session-1",
        user_input="hello",
        status="idle",
        started_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        updated_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        agent_mode="frame_agent",
        model_name="gpt-4",
        total_input_tokens=100,
        total_output_tokens=50,
        total_turns=2,
        total_frames=3,
    )


@pytest.fixture
def stored_turn() -> StoredTurn:
    """A sample stored turn."""
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return StoredTurn(
        id="turn-1",
        session_id="session-1",
        trace_id="trace-1",
        turn_index=0,
        user_input="hello",
        status="completed",
        started_at_ms=now_ms,
        finished_at_ms=now_ms + 5000,
        context_snapshot_id="snap-1",
        config_snapshot_json='{"context_config": {}}',
        total_input_tokens=100,
        total_output_tokens=50,
        total_frames=2,
        final_answer="done",
    )


@pytest.fixture
def stored_frame() -> StoredFrame:
    """A sample stored frame."""
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    return StoredFrame(
        id="frame-1",
        session_id="session-1",
        turn_id="turn-1",
        trace_id="trace-1",
        iteration=0,
        status="completed",
        started_at_ms=now_ms,
        finished_at_ms=now_ms + 2000,
        duration_ms=2000,
        model_name="gpt-4",
        finish_reason="stop",
        error_message=None,
    )


__all__ = [
    "ApprovalDeniedEvent",
    "ApprovalGrantedEvent",
    "ApprovalRequiredEvent",
    "ContinuationDecisionEvent",
    "ErrorEvent",
    "EventType",
    "FrameCompleteEvent",
    "FrameStartEvent",
    "LLMChunkEvent",
    "LLMCompleteEvent",
    "LLMStartEvent",
    "LiveEvent",
    "LiveFrame",
    "LiveRunSummary",
    "LiveSession",
    "LiveStats",
    "LiveTurn",
    "PartialResponseEvent",
    "RuntimeEvent",
    "SessionBranchEvent",
    "SessionInterruptEvent",
    "SessionResumeEvent",
    "SessionStartEvent",
    "StoredFrame",
    "StoredSession",
    "StoredTurn",
    "ToolResultEvent",
    "ToolStartEvent",
    "TurnCompleteEvent",
    "TurnStartEvent",
    "client_with_store",
    "dashboard_client",
    "event_data",
    "live_monitor",
    "make_runtime_event",
    "mock_store",
    "stored_frame",
    "stored_session",
    "stored_turn",
]
