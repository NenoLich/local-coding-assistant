"""
Unit tests for the dashboard app, HTML routes, and WebSocket manager.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

from local_coding_assistant.dashboard.app import create_app
from local_coding_assistant.dashboard.routes.websocket import ConnectionManager


class TestApp:
    def test_create_app_includes_routers(self):
        app = create_app()
        paths = {route.path for route in app.routes}
        assert "/" in paths
        assert "/health" in paths
        assert "/ws/ws/live" in paths
        assert "/api/live/runs" in paths
        assert "/api/sessions" in paths
        assert "/api/health" in paths
        assert app.title == "LOCCA Dashboard"

    def test_health_endpoint(self, dashboard_client):
        response = dashboard_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "locca-dashboard"}

    def test_cors_middleware(self, dashboard_client):
        response = dashboard_client.options(
            "/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://example.com"

    def test_not_found_handler_returns_404(self, client_with_store, mock_store):
        mock_store.get_session = AsyncMock(return_value=None)
        response = client_with_store.get("/api/sessions/nope")
        assert response.status_code == 404


class TestWebRoutes:
    def test_pages_return_html(self, dashboard_client):
        for path in ["/", "/runs", "/sessions", "/debug"]:
            response = dashboard_client.get(path)
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "LOCCA Dashboard" in response.text

    def test_detail_pages_return_html(self, dashboard_client):
        for path in ["/sessions/abc", "/runs/abc", "/turns/abc", "/frames/abc"]:
            response = dashboard_client.get(path)
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


class TestConnectionManager:
    def test_connect_and_disconnect(self):
        manager = ConnectionManager()
        websocket = _mock_websocket()
        asyncio.run(manager.connect(websocket))
        assert websocket in manager.active_connections
        assert len(manager.active_connections) == 1

        asyncio.run(manager.disconnect(websocket))
        assert websocket not in manager.active_connections
        assert len(manager.active_connections) == 0

    def test_broadcast_sends_to_all_connections(self):
        manager = ConnectionManager()
        w1 = _mock_websocket()
        w2 = _mock_websocket()
        asyncio.run(manager.connect(w1))
        asyncio.run(manager.connect(w2))

        asyncio.run(manager.broadcast({"type": "event", "data": {}}))
        w1.send_text.assert_called()
        w2.send_text.assert_called()

    def test_broadcast_removes_disconnected(self):
        manager = ConnectionManager()
        dead = _mock_websocket()
        dead.send_text.side_effect = Exception("gone")
        asyncio.run(manager.connect(dead))

        asyncio.run(manager.broadcast({"type": "x"}))
        assert dead not in manager.active_connections


def _mock_websocket():
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    return websocket
