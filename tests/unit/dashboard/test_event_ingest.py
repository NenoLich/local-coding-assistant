"""
Unit tests for the /api/events/ingest endpoint.

Seals the wire protocol that feeds runtime events into the LiveMonitor and
broadcasts them to WebSocket clients.
"""

from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app


def _post_ingest(client: TestClient, payload: dict):
    return client.post("/api/events/ingest", json=payload)


class TestIngestValidEvent:
    def test_ingest_session_start_returns_ok_and_feeds_monitor(
        self, dashboard_client, make_runtime_event, event_data
    ):
        monitor = dashboard_client.app.state.live_monitor
        assert monitor is not None

        payload = event_data(
            "session_start",
            session_id="session-1",
            data={
                "user_input": "hello",
                "agent_mode": "frame_agent",
                "model_name": "gpt-4",
            },
        )
        response = _post_ingest(dashboard_client, payload)

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        session = monitor.get_session("session-1")
        assert session is not None
        assert session.status == "running"
        assert session.user_input == "hello"

    def test_ingest_turn_start_then_run_summary_appears(
        self, dashboard_client, event_data
    ):
        monitor = dashboard_client.app.state.live_monitor

        _post_ingest(
            dashboard_client,
            event_data(
                "session_start",
                session_id="session-1",
                data={"user_input": "hello", "agent_mode": "frame_agent"},
            ),
        )
        _post_ingest(
            dashboard_client,
            event_data(
                "turn_start",
                session_id="session-1",
                turn_id="turn-1",
                data={
                    "user_input": "hello",
                    "turn_index": 0,
                    "agent_mode": "frame_agent",
                    "model_name": "gpt-4",
                },
            ),
        )

        runs = monitor.get_run_summaries()
        assert len(runs) == 1
        assert runs[0].run_id == "turn-1"
        assert runs[0].status == "running"

    def test_ingest_frame_and_llm_complete_updates_tokens(
        self, dashboard_client, event_data
    ):
        monitor = dashboard_client.app.state.live_monitor
        _post_ingest(
            dashboard_client,
            event_data(
                "session_start",
                session_id="session-1",
                data={"user_input": "hello", "agent_mode": "frame_agent"},
            ),
        )
        _post_ingest(
            dashboard_client,
            event_data(
                "turn_start",
                session_id="session-1",
                turn_id="turn-1",
                data={"user_input": "hello", "turn_index": 0},
            ),
        )
        _post_ingest(
            dashboard_client,
            event_data(
                "frame_start",
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                data={
                    "iteration": 0,
                    "context_snapshot_id": "snap-1",
                    "context_hash": "hash-1",
                },
            ),
        )
        _post_ingest(
            dashboard_client,
            event_data(
                "llm_complete",
                session_id="session-1",
                turn_id="turn-1",
                frame_id="frame-1",
                data={
                    "content": "done",
                    "model": "gpt-4",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cached_tokens": 5,
                },
            ),
        )

        frame = monitor.get_session("session-1").turns["turn-1"].frames["frame-1"]
        assert frame.input_tokens == 100
        assert frame.output_tokens == 50
        assert frame.cache_tokens == 5


class TestIngestErrors:
    def test_ingest_missing_type_returns_400(self, dashboard_client):
        response = _post_ingest(dashboard_client, {"session_id": "s", "data": {}})
        assert response.status_code == 400

    def test_ingest_unknown_event_type_returns_400(self, dashboard_client):
        response = _post_ingest(
            dashboard_client,
            {
                "type": "not_a_real_event",
                "session_id": "s",
                "data": {},
            },
        )
        assert response.status_code == 400

    def test_ingest_503_when_live_monitor_unavailable(self, mock_store, event_data):
        app = create_app()
        app.state.execution_store = mock_store
        with TestClient(app) as client:
            client.app.state.live_monitor = None
            response = _post_ingest(
                client,
                event_data(
                    "session_start",
                    session_id="s",
                    data={"user_input": "x", "agent_mode": "a"},
                ),
            )
            assert response.status_code == 503
