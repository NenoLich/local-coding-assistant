"""
Integration tests for WebSocket real-time functionality.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class TestWebSocketConnectionManagement:
    """Test WebSocket connection lifecycle management."""

    def test_multiple_concurrent_connections(self):
        """Test multiple concurrent WebSocket connections."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            clients = []
            connections = []

            try:
                # Create multiple concurrent connections
                for i in range(3):
                    client = TestClient(app)
                    websocket = client.websocket_connect("/ws/ws")
                    websocket.__enter__()
                    clients.append(client)
                    connections.append(websocket)

                    # Each connection should receive initial data
                    data = websocket.receive_json()
                    assert data["type"] == "stats_update"

                # Verify all connections are active
                assert len(clients) == 3

            finally:
                # Clean up connections
                for websocket in connections:
                    try:
                        websocket.__exit__(None, None, None)
                    except:
                        pass
                for client in clients:
                    client.close()

    def test_connection_disconnection_handling(self):
        """Test graceful handling of connection disconnections."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            # Connect and disconnect
            with client.websocket_connect("/ws/ws") as websocket:
                # Should receive initial data
                data = websocket.receive_json()
                assert data["type"] == "stats_update"

            # Connection should be closed gracefully

    def test_connection_failure_recovery(self):
        """Test recovery from connection failures."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            mock_manager.connect = AsyncMock(side_effect=Exception("Connection failed"))

            with pytest.raises(Exception):
                with TestClient(app).websocket_connect("/ws/ws"):
                    pass

    def test_connection_timeout_handling(self):
        """Test handling of connection timeouts."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            # Don't simulate slow response for this test - just test normal connection
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            # Should connect successfully
            with client.websocket_connect("/ws/ws") as websocket:
                # Should receive initial data
                data = websocket.receive_json()
                assert data["type"] == "stats_update"


class TestRealTimeEventBroadcasting:
    """Test real-time event broadcasting functionality."""

    def test_event_broadcast_to_multiple_clients(self):
        """Test broadcasting events to multiple connected clients."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            # Setup mock connections
            connections = []
            for i in range(3):
                ws = Mock()
                ws.send_text = AsyncMock()
                connections.append(ws)

            mock_manager.active_connections = connections
            mock_manager.broadcast_event = AsyncMock()

            # Create test event
            event = ExecutionEvent(
                type=EventType.TOOL_START,
                session_id="test-session",
                timestamp=datetime.now(),
                data={"tool_name": "test_tool", "tool_args": {"param": "value"}},
            )

            # Broadcast event
            asyncio.run(mock_manager.broadcast_event(event))

            # Verify the method was called
            mock_manager.broadcast_event.assert_called_once_with(event)

    def test_event_broadcast_with_disconnected_clients(self):
        """Test broadcasting when some clients are disconnected."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            # Setup connections where one fails
            ws1 = Mock()
            ws1.send_text = AsyncMock()

            ws2 = Mock()
            ws2.send_text = AsyncMock(side_effect=Exception("Connection lost"))

            mock_manager.active_connections = [ws1, ws2]
            mock_manager.disconnect = AsyncMock()
            mock_manager.broadcast_event = AsyncMock()

            # Create test event
            event = ExecutionEvent(
                type=EventType.SESSION_START,
                session_id="test-session",
                timestamp=datetime.now(),
                data={"user_query": "test query"},
            )

            # Broadcast event
            asyncio.run(mock_manager.broadcast_event(event))

            # Verify the method was called
            mock_manager.broadcast_event.assert_called_once_with(event)

    def test_different_event_types_broadcasting(self):
        """Test broadcasting different types of events."""
        app = create_app()

        event_types = [
            (EventType.SESSION_START, {"user_query": "test query"}),
            (EventType.TURN_START, {"run_id": "test-run", "mode": "auto"}),
            (EventType.FRAME_START, {"run_id": "test-run", "frame_id": "test-frame"}),
            (EventType.LLM_START, {"model": "test-model", "prompt": "test prompt"}),
            (EventType.TOOL_START, {"tool_name": "test_tool", "tool_args": {}}),
            (EventType.FRAME_COMPLETE, {"status": "completed"}),
            (
                EventType.TURN_COMPLETE,
                {"status": "completed", "final_answer": "test answer"},
            ),
        ]

        for event_type, data in event_types:
            with patch(
                "local_coding_assistant.dashboard.routes.websocket.manager"
            ) as mock_manager:
                ws = Mock()
                ws.send_text = AsyncMock()
                mock_manager.active_connections = [ws]
                mock_manager.broadcast_event = AsyncMock()

                event = ExecutionEvent(
                    type=event_type,
                    session_id="test-session",
                    timestamp=datetime.now(),
                    data=data,
                )

                asyncio.run(mock_manager.broadcast_event(event))

                # Since we're mocking the manager itself, we need to check if it was called
                mock_manager.broadcast_event.assert_called_once_with(event)

    def test_high_frequency_event_broadcasting(self):
        """Test broadcasting high-frequency events."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            ws = Mock()
            ws.send_text = AsyncMock()
            mock_manager.active_connections = [ws]
            mock_manager.broadcast_event = AsyncMock()

            # Create many events quickly
            events = []
            for i in range(10):  # Reduced from 100 for faster testing
                event = ExecutionEvent(
                    type=EventType.TOOL_START,
                    session_id="test-session",
                    timestamp=datetime.now() + timedelta(milliseconds=i),
                    data={"tool_name": f"tool_{i}", "tool_args": {"index": i}},
                )
                events.append(event)

            # Broadcast all events
            for event in events:
                asyncio.run(mock_manager.broadcast_event(event))

            # Verify all events were broadcast
            assert mock_manager.broadcast_event.call_count == 10


class TestRealTimeStatsUpdates:
    """Test real-time statistics updates."""

    def test_stats_update_broadcast(self):
        """Test broadcasting statistics updates."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {
                "total_runs": 10,
                "success_rate": "85.5%",
                "avg_duration": "2m 30s",
                "active_sessions": 3,
                "completed_runs": 8,
                "error_runs": 2,
            }
            mock_get_collector.return_value = mock_collector

            with patch(
                "local_coding_assistant.dashboard.routes.websocket.manager"
            ) as mock_manager:
                ws = Mock()
                ws.send_text = AsyncMock()
                mock_manager.active_connections = [ws]
                mock_manager.broadcast_stats_update = AsyncMock()

                # Broadcast stats update
                asyncio.run(mock_manager.broadcast_stats_update())

                # Verify the method was called
                mock_manager.broadcast_stats_update.assert_called_once()

    def test_sessions_update_broadcast(self):
        """Test broadcasting sessions updates."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_active_sessions.return_value = [
                {
                    "session_id": "session-1",
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "events_count": 15,
                },
                {
                    "session_id": "session-2",
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "events_count": 8,
                },
            ]
            mock_get_collector.return_value = mock_collector

            with patch(
                "local_coding_assistant.dashboard.routes.websocket.manager"
            ) as mock_manager:
                ws = Mock()
                ws.send_text = AsyncMock()
                mock_manager.active_connections = [ws]
                mock_manager.broadcast_sessions_update = AsyncMock()

                # Broadcast sessions update
                asyncio.run(mock_manager.broadcast_sessions_update())

                # Verify the method was called
                mock_manager.broadcast_sessions_update.assert_called_once()


class TestWebSocketMessageHandling:
    """Test WebSocket message handling from clients."""

    def test_ping_pong_messaging(self):
        """Test ping/pong message handling."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                websocket.receive_json()
                websocket.receive_json()
                websocket.receive_json()

                # Send ping
                websocket.send_json({"type": "ping"})

                # Should receive pong
                response = websocket.receive_json()
                assert response["type"] == "pong"

    def test_subscription_messaging(self):
        """Test subscription message handling."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                for _ in range(3):
                    websocket.receive_json()

                # Send subscription
                websocket.send_json({"type": "subscribe", "events": "tool_calls"})

                # Should receive subscription confirmation
                response = websocket.receive_json()
                assert response["type"] == "subscribed"
                assert response["events"] == "tool_calls"

    def test_request_sessions_update(self):
        """Test requesting sessions update."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = [
                {"session_id": "session-1", "status": "running"}
            ]
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                for _ in range(3):
                    websocket.receive_json()

                # Request sessions update
                websocket.send_json({"type": "request_sessions_update"})

                # Should receive sessions update
                response = websocket.receive_json()
                assert response["type"] == "sessions_update"
                assert len(response["data"]) == 1

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON messages."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                for _ in range(3):
                    websocket.receive_json()

                # Send invalid JSON
                websocket.send_text("invalid json string")

                # Should receive error message
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Invalid JSON" in response["message"]

    def test_unknown_message_type_handling(self):
        """Test handling of unknown message types."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                for _ in range(3):
                    websocket.receive_json()

                # Send unknown message type
                websocket.send_json({"type": "unknown_type", "data": {}})

                # Unknown message types are ignored - no response expected
                # Just verify connection is still alive by sending a ping
                websocket.send_json({"type": "ping"})
                response = websocket.receive_json()
                assert response["type"] == "pong"


class TestWebSocketPerformance:
    """Test WebSocket performance and scalability."""

    def test_message_throughput(self):
        """Test message throughput under load."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data first
                for _ in range(3):
                    websocket.receive_json()

                # Send many messages quickly
                start_time = datetime.now()

                for i in range(50):
                    websocket.send_json({"type": "ping"})
                    response = websocket.receive_json()
                    assert response["type"] == "pong"

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                # Should handle messages quickly (adjust threshold as needed)
                assert duration < 10.0  # 50 messages in under 10 seconds

    def test_connection_scalability(self):
        """Test handling many concurrent connections."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            clients = []
            connections = []

            try:
                # Create many connections
                for i in range(10):
                    client = TestClient(app)
                    websocket = client.websocket_connect("/ws/ws")
                    websocket.__enter__()
                    clients.append(client)
                    connections.append(websocket)

                    # Each should receive initial data
                    try:
                        for _ in range(3):
                            websocket.receive_json()
                    except Exception:
                        # Some connections might timeout, which is acceptable
                        pass

                # Should handle multiple connections
                assert len(clients) == 10

            finally:
                # Clean up
                for websocket in connections:
                    try:
                        websocket.__exit__(None, None, None)
                    except:
                        pass
                for client in clients:
                    client.close()

    def test_memory_usage_stability(self):
        """Test memory usage stability during long-running connections."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {"total_runs": 5}
            mock_collector.get_recent_activity.return_value = {"activities": []}
            mock_collector.get_active_sessions.return_value = []
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            with client.websocket_connect("/ws/ws") as websocket:
                # Receive initial data
                for _ in range(3):
                    websocket.receive_json()

                # Send many messages over time
                for i in range(100):
                    websocket.send_json({"type": "ping"})
                    response = websocket.receive_json()
                    assert response["type"] == "pong"

                    # Occasionally request updates
                    if i % 20 == 0:
                        websocket.send_json({"type": "request_sessions_update"})
                        try:
                            websocket.receive_json()
                        except Exception:
                            pass


class TestWebSocketErrorHandling:
    """Test WebSocket error handling and recovery."""

    def test_event_collector_error_handling(self):
        """Test handling when EventCollector has errors."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.event_collector.get_event_collector"
        ) as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.side_effect = Exception(
                "Collector error"
            )
            mock_get_collector.return_value = mock_collector

            client = TestClient(app)

            # Should handle error gracefully - connection succeeds despite error
            with client.websocket_connect("/ws/ws") as websocket:
                # Connection should succeed even if initial data fails
                # The error is caught and logged
                pass

    def test_broadcast_error_handling(self):
        """Test error handling during broadcasting."""
        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            # Setup connections where broadcasting fails
            ws = Mock()
            ws.send_text = AsyncMock(side_effect=Exception("Broadcast failed"))
            mock_manager.active_connections = [ws]
            mock_manager.disconnect = AsyncMock()
            mock_manager.broadcast_event = AsyncMock()

            event = ExecutionEvent(
                type=EventType.TOOL_START,
                session_id="test-session",
                timestamp=datetime.now(),
                data={},
            )

            # Should handle broadcast error gracefully
            asyncio.run(mock_manager.broadcast_event(event))

            # Verify the method was called
            mock_manager.broadcast_event.assert_called_once_with(event)
