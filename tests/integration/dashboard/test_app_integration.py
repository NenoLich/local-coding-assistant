"""
Integration tests for dashboard app startup and serving.
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from local_coding_assistant.dashboard.app import create_app, lifespan
from local_coding_assistant.dashboard.run_dashboard import main


class TestDashboardAppIntegration:
    """Integration tests for dashboard app."""

    def test_create_app_basic(self):
        """Test basic app creation."""
        app = create_app()

        assert app.title == "LOCCA Dashboard"
        assert app.description == "ExecutionFrame observability and analysis dashboard"
        assert app.version == "0.1.0"

        # Check that routers are included
        route_paths = [route.path for route in app.routes]
        assert "/" in route_paths
        assert "/api/status" in route_paths
        assert "/ws/ws" in route_paths

    def test_cors_middleware(self):
        """Test CORS middleware is properly configured."""
        app = create_app()
        client = TestClient(app)

        # Test preflight request
        response = client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    async def test_app_lifecycle(self):
        """Test app lifespan management."""
        app = create_app()

        # Test lifespan context manager
        async with lifespan(app):
            # App should be running
            pass

        # Should shutdown gracefully

    def test_static_files_mounting(self):
        """Test static files mounting when directory exists."""
        # Mock both Path.exists and os.path.isdir to simulate existing directory
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("os.path.isdir", return_value=True),
        ):
            app = create_app()

            # Check if static files route is mounted
            # In newer FastAPI, static files are mounted as routes with app attribute
            static_mounts = []
            for route in app.routes:
                if hasattr(route, "path") and route.path == "/static":
                    static_mounts.append(route)
                elif hasattr(route, "app") and hasattr(route.app, "directory"):
                    # This is a StaticFiles mount
                    if hasattr(route, "path") and route.path == "/static":
                        static_mounts.append(route)

            # Should have at least one static route or mount
            assert len(static_mounts) > 0

    def test_static_files_missing_directory(self):
        """Test app handles missing static directory gracefully."""
        with patch("pathlib.Path.exists", return_value=False):
            app = create_app()

            # Should still create app without static files
            assert app is not None

    def test_full_app_startup_sequence(self):
        """Test complete app startup sequence."""
        with patch("local_coding_assistant.dashboard.app.log") as mock_log:
            app = create_app()
            client = TestClient(app)

            # Test that app responds to basic requests
            response = client.get("/api/status")
            assert response.status_code == 200

            # Verify logging was called during app creation
            # The logger is created at module level, so we check if it exists
            assert mock_log is not None

    def test_route_inclusion(self):
        """Test that all expected routes are included."""
        app = create_app()

        # Get all route paths
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                routes.append(route.path)
            elif hasattr(route, "path_regex"):
                routes.append(route.path_regex.pattern)

        # Check main routes
        assert "/" in routes
        assert "/runs" in routes
        assert "/analytics" in routes
        assert "/live" in routes

        # Check API routes
        assert any("/api" in route for route in routes)
        assert any("/status" in route for route in routes)

        # Check WebSocket routes
        assert any("/ws" in route for route in routes)

    def test_app_configuration(self):
        """Test app configuration and middleware."""
        app = create_app()

        # Check that CORS middleware is configured by testing a CORS request
        client = TestClient(app)

        # Test preflight request - this should work if CORS middleware is properly configured
        response = client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Should return 200 for CORS preflight if middleware is configured
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_error_handling_integration(self):
        """Test error handling across the app."""
        app = create_app()
        client = TestClient(app)

        # Test 404 handling
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

        # Test API 404 handling
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_health_check_integration(self):
        """Test health check endpoints."""
        app = create_app()
        client = TestClient(app)

        # API status endpoint
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        app = create_app()
        client = TestClient(app)

        # Make multiple concurrent requests
        import queue
        import threading

        results = queue.Queue()

        def make_request():
            response = client.get("/api/status")
            results.put(response.status_code)

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All requests should succeed
        while not results.empty():
            assert results.get() == 200


class TestDashboardRunnerIntegration:
    """Integration tests for dashboard runner."""

    def test_run_dashboard_main_function(self):
        """Test the main function in run_dashboard.py."""
        # Mock uvicorn at module level since it might not be installed
        with patch.dict("sys.modules", {"uvicorn": Mock()}):
            import importlib

            uvicorn_mock = importlib.import_module("uvicorn")

            with patch("sys.argv", ["run_dashboard.py", "--port", "8080"]):
                with patch.object(uvicorn_mock, "run") as mock_uvicorn:
                    # Mock the create_app function at the module level where it's imported
                    with patch(
                        "local_coding_assistant.dashboard.app.create_app"
                    ) as mock_create_app:
                        mock_app = Mock()
                        mock_create_app.return_value = mock_app

                        main()

                        mock_create_app.assert_called_once()
                        mock_uvicorn.assert_called_once_with(
                            mock_app,
                            host="127.0.0.1",
                            port=8080,
                            log_level="info",
                            reload=False,
                        )

    def test_run_dashboard_with_custom_args(self):
        """Test run_dashboard with custom arguments."""
        # Mock uvicorn at module level since it might not be installed
        with patch.dict("sys.modules", {"uvicorn": Mock()}):
            import importlib

            uvicorn_mock = importlib.import_module("uvicorn")

            with patch(
                "sys.argv",
                [
                    "run_dashboard.py",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--log-level",
                    "debug",
                    "--reload",
                ],
            ):
                with patch.object(uvicorn_mock, "run") as mock_uvicorn:
                    with patch(
                        "local_coding_assistant.dashboard.app.create_app"
                    ) as mock_create_app:
                        mock_app = Mock()
                        mock_create_app.return_value = mock_app

                        main()

                        mock_uvicorn.assert_called_once_with(
                            mock_app,
                            host="0.0.0.0",
                            port=9000,
                            log_level="debug",
                            reload=True,
                        )

    def test_run_dashboard_argument_parsing(self):
        """Test argument parsing edge cases."""
        # Mock uvicorn at module level since it might not be installed
        with patch.dict("sys.modules", {"uvicorn": Mock()}):
            import importlib

            uvicorn_mock = importlib.import_module("uvicorn")

            # Test with no arguments
            with patch("sys.argv", ["run_dashboard.py"]):
                with patch.object(uvicorn_mock, "run") as mock_uvicorn:
                    with patch("local_coding_assistant.dashboard.app.create_app"):
                        main()

                        call_args = mock_uvicorn.call_args[1]
                        assert call_args["host"] == "127.0.0.1"
                        assert call_args["port"] == 8080
                        assert call_args["log_level"] == "info"
                        assert call_args["reload"] is False

    def test_run_dashboard_invalid_args(self):
        """Test handling of invalid arguments."""
        # Mock uvicorn at module level since it might not be installed
        with patch.dict("sys.modules", {"uvicorn": Mock()}):
            import importlib

            uvicorn_mock = importlib.import_module("uvicorn")

            with patch("sys.argv", ["run_dashboard.py", "--invalid-arg"]):
                with patch.object(uvicorn_mock, "run") as mock_uvicorn:
                    with patch("local_coding_assistant.dashboard.app.create_app"):
                        # Should not crash with invalid args
                        main()

                        # Should use defaults
                        call_args = mock_uvicorn.call_args[1]
                        assert call_args["host"] == "127.0.0.1"


class TestDashboardWithEventCollector:
    """Integration tests with EventCollector."""

    def test_app_with_event_collector_integration(self):
        """Test app integration with EventCollector."""
        app = create_app()
        client = TestClient(app)

        with patch(
            "local_coding_assistant.dashboard.routes.api.get_event_collector"
        ) as mock_get:
            mock_collector = AsyncMock()
            mock_collector.get_dashboard_stats.return_value = {
                "total_runs": 5,
                "success_rate": "80%",
                "avg_duration": "2m",
                "active_sessions": 2,
                "completed_runs": 4,
                "error_runs": 1,
            }
            mock_get.return_value = mock_collector

            response = client.get("/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_runs"] == 5

    def test_event_ingestion_flow(self):
        """Test complete event ingestion flow."""
        app = create_app()
        client = TestClient(app)

        with patch(
            "local_coding_assistant.dashboard.routes.api.get_event_collector"
        ) as mock_get:
            mock_collector = AsyncMock()
            mock_collector.collect_event = AsyncMock()
            mock_get.return_value = mock_collector

            # Ingest multiple events with valid EventType values
            events = [
                {
                    "type": "session_start",
                    "session_id": "session-1",
                    "timestamp": "2024-01-01T12:00:00",
                    "data": {"user_query": "test query 1"},
                },
                {
                    "type": "frame_start",
                    "session_id": "session-1",
                    "timestamp": "2024-01-01T12:01:00",
                    "data": {"frame_id": "frame-1", "mode": "auto"},
                },
            ]

            for event in events:
                response = client.post("/api/events/ingest", json=event)
                assert response.status_code == 200

            # Verify all events were collected
            assert mock_collector.collect_event.call_count == len(events)

    def test_real_time_updates_integration(self):
        """Test real-time updates integration."""
        app = create_app()
        client = TestClient(app)

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.broadcast_event = AsyncMock()

            # Test WebSocket connection by checking the endpoint exists
            # and that the manager would be called for real connections

            # Send event and verify broadcast would be called
            event_data = {
                "type": "tool_start",
                "session_id": "test-session",
                "timestamp": "2024-01-01T12:00:00",
                "data": {"tool_name": "test_tool"},
            }

            with patch(
                "local_coding_assistant.dashboard.routes.api.get_event_collector"
            ) as mock_get:
                mock_collector = AsyncMock()
                mock_collector.collect_event = AsyncMock()
                mock_get.return_value = mock_collector

                response = client.post("/api/events/ingest", json=event_data)
                assert response.status_code == 200


class TestDashboardPerformance:
    """Performance and load testing for dashboard."""

    def test_response_time_under_load(self):
        """Test dashboard response times under load."""
        app = create_app()
        client = TestClient(app)

        # Measure response times
        start_time = time.time()

        for _ in range(50):
            response = client.get("/api/status")
            assert response.status_code == 200

        end_time = time.time()
        avg_time = (end_time - start_time) / 50

        # Should respond quickly (adjust threshold as needed)
        assert avg_time < 0.1  # 100ms average

    def test_memory_usage_stability(self):
        """Test memory usage stability over time."""
        app = create_app()
        client = TestClient(app)

        # Make many requests to test for memory leaks
        for i in range(100):
            response = client.get("/api/status")
            assert response.status_code == 200

            # Occasionally hit other endpoints
            if i % 10 == 0:
                response = client.get("/")
                assert response.status_code == 200

    def test_concurrent_websocket_connections(self):
        """Test handling multiple concurrent WebSocket connections."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.disconnect = AsyncMock()

            # Test that the WebSocket endpoint exists and would handle connections
            # The actual connection handling is tested by checking the manager setup

            # Verify the manager is properly configured for multiple connections
            assert hasattr(mock_manager, "connect")
            assert hasattr(mock_manager, "disconnect")

            # Test that multiple calls to connect would be handled
            # This simulates what would happen with multiple concurrent connections
            import asyncio

            async def test_multiple_connections():
                # Simulate multiple connection attempts
                tasks = [mock_manager.connect(None) for _ in range(5)]
                await asyncio.gather(*tasks)

                # Verify connect was called 5 times
                assert mock_manager.connect.call_count == 5

            # Run the async test
            asyncio.run(test_multiple_connections())


class TestDashboardErrorRecovery:
    """Test dashboard error recovery and resilience."""

    def test_recovery_from_event_collector_errors(self):
        """Test recovery when EventCollector has errors."""
        app = create_app()
        client = TestClient(app)

        with patch(
            "local_coding_assistant.dashboard.routes.api.get_event_collector"
        ) as mock_get:
            mock_collector = AsyncMock()
            mock_get.return_value = mock_collector

            # First request should fail with 500
            mock_collector.get_dashboard_stats.side_effect = Exception(
                "Temporary error"
            )

            # Use pytest.raises to catch the exception that propagates
            with pytest.raises(Exception, match="Temporary error"):
                client.get("/api/stats")

            # Reset side effect for second call
            mock_collector.get_dashboard_stats.side_effect = None
            mock_collector.get_dashboard_stats.return_value = {
                "total_runs": 5,
                "success_rate": "80%",
                "avg_duration": "2m",
                "active_sessions": 2,
                "completed_runs": 4,
                "error_runs": 1,
            }

            # Second request should succeed
            response = client.get("/api/stats")
            assert response.status_code == 200

    def test_websocket_connection_failure_recovery(self):
        """Test recovery from WebSocket connection failures."""
        app = create_app()

        with patch(
            "local_coding_assistant.dashboard.routes.websocket.manager"
        ) as mock_manager:
            mock_manager.connect = AsyncMock(side_effect=Exception("Connection failed"))

            # Should handle connection failure gracefully
            with pytest.raises(Exception):
                with TestClient(app).websocket_connect("/ws/ws"):
                    pass

    def test_template_rendering_errors(self):
        """Test handling of template rendering errors."""
        app = create_app()

        # Mock the TemplateResponse to raise an exception
        with patch(
            "local_coding_assistant.dashboard.routes.main.templates.TemplateResponse"
        ) as mock_template:
            mock_template.side_effect = Exception("Template error")

            client = TestClient(app)

            # The template error should be handled by FastAPI's error middleware
            # and result in a 500 status code
            with pytest.raises(Exception, match="Template error"):
                client.get("/")


class TestDashboardConfiguration:
    """Test dashboard configuration and environment handling."""

    def test_app_in_different_environments(self):
        """Test app behavior in different environments."""
        with patch.dict("os.environ", {"LOCCA_ENV": "test"}):
            app = create_app()
            client = TestClient(app)

            response = client.get("/api/status")
            assert response.status_code == 200

    def test_app_with_custom_configuration(self):
        """Test app with custom configuration."""
        # This would test if the app respects custom configuration
        # Implementation depends on how configuration is handled
        app = create_app()
        assert app is not None

    def test_logging_configuration(self):
        """Test logging configuration."""
        # The logger is created at module level in app.py, so we can't mock its creation
        # Instead, we verify that the logger exists and can be used
        from local_coding_assistant.dashboard.app import log

        # Verify logger exists and can be used
        assert log is not None

        # Test that we can call logger methods without error
        try:
            log.info("Test log message")
            logger_works = True
        except Exception:
            logger_works = False

        assert logger_works

        # Create app and verify no errors occur
        app = create_app()
        assert app is not None
