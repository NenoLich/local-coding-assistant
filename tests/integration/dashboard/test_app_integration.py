"""
Integration tests for the dashboard app (live plane).

These exercise the full FastAPI app through TestClient: app creation, lifespan,
the ingest endpoint feeding the LiveMonitor, and the live API endpoints.
"""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app


def _ts() -> str:
    return datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC).isoformat()


def _ingest_payload(
    event_type: str, data: dict, session_id="session-1", turn_id="turn-1", frame_id=None
):
    """Build a valid /api/events/ingest payload.

    The ingest handler constructs the typed event from ``data`` plus top-level
    ``session_id``/``frame_id``/``timestamp``, so ``trace_id`` and ``turn_id``
    must live inside ``data``.
    """
    payload = {
        "type": event_type,
        "session_id": session_id,
        "turn_id": turn_id,
        "data": {"trace_id": "trace-1", "turn_id": turn_id, **data},
        "timestamp": _ts(),
    }
    if frame_id is not None:
        payload["frame_id"] = frame_id
    return payload


class TestDashboardAppIntegration:
    def test_app_creation_and_health(self):
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_live_monitor_available_after_startup(self):
        app = create_app()
        with TestClient(app) as client:
            monitor = client.app.state.live_monitor
            assert monitor is not None

    def test_event_ingest_feeds_live_runs(self, dashboard_client):
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "session_start",
                {
                    "user_input": "hello",
                    "agent_mode": "frame_agent",
                    "model_name": "gpt-4",
                },
            ),
        )
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "turn_start",
                {
                    "user_input": "hello",
                    "turn_index": 0,
                    "agent_mode": "frame_agent",
                    "model_name": "gpt-4",
                },
            ),
        )

        response = dashboard_client.get("/api/live/runs")
        assert response.status_code == 200
        runs = response.json()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "turn-1"
        assert runs[0]["session_id"] == "session-1"
        assert runs[0]["status"] == "running"

    def test_live_stats_endpoint(self, dashboard_client):
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "session_start",
                {"user_input": "hello", "agent_mode": "frame_agent"},
            ),
        )
        response = dashboard_client.get("/api/live/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["active_sessions"] == 1
        assert stats["total_events"] >= 1

    def test_live_sessions_endpoint(self, dashboard_client):
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "session_start",
                {"user_input": "hello", "agent_mode": "frame_agent"},
            ),
        )
        response = dashboard_client.get("/api/live/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session-1"

    def test_live_run_detail_endpoint(self, dashboard_client):
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "session_start",
                {"user_input": "hello", "agent_mode": "frame_agent"},
            ),
        )
        dashboard_client.post(
            "/api/events/ingest",
            json=_ingest_payload(
                "turn_start",
                {"user_input": "hello", "turn_index": 0},
            ),
        )

        response = dashboard_client.get("/api/live/runs/turn-1")
        assert response.status_code == 200
        body = response.json()
        assert body["turn"]["turn_id"] == "turn-1"
        assert body["session"]["session_id"] == "session-1"

    def test_live_run_not_found(self, dashboard_client):
        response = dashboard_client.get("/api/live/runs/does-not-exist")
        assert response.status_code == 404

    def test_ingest_unknown_event_type_returns_400(self, dashboard_client):
        response = dashboard_client.post(
            "/api/events/ingest",
            json={
                "type": "not_a_real_event",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "data": {},
                "timestamp": _ts(),
            },
        )
        assert response.status_code == 400

    def test_ingest_missing_type_400(self, dashboard_client):
        response = dashboard_client.post(
            "/api/events/ingest",
            json={"session_id": "session-1", "data": {}},
        )
        assert response.status_code == 400


class TestHistoricalEndpointsUnavailableWithoutStore:
    def test_sessions_503_without_store(self, dashboard_client):
        response = dashboard_client.get("/api/sessions")
        assert response.status_code == 503

    def test_debug_health_works_without_store(self, dashboard_client):
        response = dashboard_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["store_available"] is False
        assert response.json()["live_monitor_available"] is True
