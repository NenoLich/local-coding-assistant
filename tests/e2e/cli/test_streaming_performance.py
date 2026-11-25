"""
End-to-end performance tests for run-query output (formerly streaming).

These tests mock the bootstrap / execute-query boundary and verify that the
rendered report content reaches standard output, completing within reasonable
time bounds.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from local_coding_assistant.cli.main import app


class PerformanceMetrics:
    """Utility class for collecting performance metrics during tests."""

    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.cpu_usage = []

    def start_measurement(self):
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []

    def record_metrics(self):
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)
        self.cpu_usage.append(process.cpu_percent())

    def stop_measurement(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        return {
            "duration": duration,
            "avg_memory_mb": (
                sum(self.memory_usage) / len(self.memory_usage)
                if self.memory_usage
                else 0
            ),
            "peak_memory_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_percent": (
                sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            ),
        }


class _MockAsyncContextManager:
    def __init__(self, ctx):
        self.ctx = ctx

    async def __aenter__(self):
        return self.ctx

    async def __aexit__(self, *args):
        return False


def _invoke_run(cli_runner, query, final_answer):
    mock_runtime = MagicMock()
    mock_runtime.event_pipeline = MagicMock()
    mock_ctx = {"runtime": mock_runtime}

    mock_bootstrap = MagicMock()
    mock_bootstrap.return_value = _MockAsyncContextManager(mock_ctx)

    mock_report = MagicMock()
    mock_report.final_answer = final_answer
    mock_report.status = "success"

    mock_execute_query = AsyncMock(return_value=(mock_report, MagicMock(id="sess-1")))

    with (
        patch("local_coding_assistant.cli.commands.run.bootstrap", mock_bootstrap),
        patch(
            "local_coding_assistant.cli.commands.run.execute_query",
            mock_execute_query,
        ),
    ):
        return cli_runner.invoke(app, ["run", "query", query])


@pytest.fixture
def performance_metrics():
    return PerformanceMetrics()


class TestStreamingPerformanceE2E:
    """End-to-end performance tests for run-query output."""

    def test_run_query_completes_quickly(self, cli_runner, performance_metrics):
        """Run-query completes within a reasonable time."""

        performance_metrics.start_measurement()
        result = _invoke_run(
            cli_runner, "Test streaming latency", "Streaming latency test completed"
        )
        metrics = performance_metrics.stop_measurement()

        assert result.exit_code == 0, result.stderr
        assert "Streaming latency test completed" in result.stdout
        assert metrics["duration"] < 2.0, f"Took too long: {metrics['duration']}s"

    def test_run_query_processes_large_content(self, cli_runner, performance_metrics):
        """Large content responses are rendered to standard output."""

        large_content = ". ".join([f"chunk {i}" for i in range(50)])
        final_answer = f"Large content test completed {large_content}"

        performance_metrics.start_measurement()
        result = _invoke_run(
            cli_runner,
            "Test memory usage with large content",
            final_answer,
        )
        metrics = performance_metrics.stop_measurement()

        assert result.exit_code == 0, result.stderr
        assert "Large content test completed" in result.stdout
        assert metrics["duration"] < 3.0, f"Took too long: {metrics['duration']}s"
        assert metrics["peak_memory_mb"] < 500

    def test_run_query_responsiveness_under_load(self, cli_runner, performance_metrics):
        """Multiple run-query invocations complete quickly in sequence."""

        performance_metrics.start_measurement()
        results = []
        for i in range(3):
            result = _invoke_run(
                cli_runner, f"Load test query {i}", f"Load test {i} completed"
            )
            results.append(result)
            assert result.exit_code == 0, result.stderr

        metrics = performance_metrics.stop_measurement()

        assert len(results) == 3
        assert all(r.exit_code == 0 for r in results)
        assert metrics["duration"] / len(results) < 1.0
        assert metrics["peak_memory_mb"] < 400
        assert metrics["avg_cpu_percent"] < 50

    def test_run_query_streaming_latency_comparison(
        self, cli_runner, performance_metrics
    ):
        """Run-query output completes within expected bounds."""

        content_chunks = [
            "This is a comprehensive analysis ",
            "that would typically be returned ",
            "as a single large response. ",
        ]
        full_content = "".join(content_chunks)
        final_answer = f"Latency comparison completed {full_content}"

        performance_metrics.start_measurement()
        result = _invoke_run(
            cli_runner,
            "Compare streaming vs non-streaming latency",
            final_answer,
        )
        metrics = performance_metrics.stop_measurement()

        assert result.exit_code == 0, result.stderr
        assert full_content in result.stdout
        assert metrics["avg_memory_mb"] < 300
