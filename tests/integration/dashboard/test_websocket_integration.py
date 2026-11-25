"""
Integration tests for the dashboard WebSocket endpoint.
"""

import json

from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app


class TestWebSocketIntegration:
    def test_connect_receives_initial_stats(self):
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws/ws/live") as websocket:
                data = json.loads(websocket.receive_text())
                assert data["type"] == "stats_update"
                assert "active_sessions" in data["data"]
                assert "running_turns" in data["data"]

    def test_ping_receives_pong(self):
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws/ws/live") as websocket:
                websocket.receive_text()  # consume initial stats
                websocket.send_text(json.dumps({"type": "ping"}))
                data = json.loads(websocket.receive_text())
                assert data["type"] == "pong"

    def test_unknown_message_is_ignored(self):
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/ws/ws/live") as websocket:
                websocket.receive_text()  # consume initial stats
                websocket.send_text("not json")
                # Should not crash; connection stays open
                websocket.send_text(json.dumps({"type": "ping"}))
                data = json.loads(websocket.receive_text())
                assert data["type"] == "pong"
