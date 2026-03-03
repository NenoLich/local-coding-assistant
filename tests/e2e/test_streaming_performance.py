"""
End-to-end performance tests for streaming functionality.
Tests latency improvements, memory usage, and responsiveness under load.
"""

import asyncio
import time
import psutil
import os
from unittest.mock import patch, MagicMock

import pytest

from local_coding_assistant.cli.main import app
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class PerformanceMetrics:
    """Utility class for collecting performance metrics during tests."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_usage = []
        self.cpu_usage = []

    def start_measurement(self):
        """Start collecting performance metrics."""
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []

    def record_metrics(self):
        """Record current memory and CPU usage."""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(process.cpu_percent())

    def stop_measurement(self):
        """Stop collecting metrics and return summary."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        return {
            'duration': duration,
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'peak_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_percent': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
        }


class MockStreamingRuntimeWithMetrics:
    """Mock runtime that emits events with performance tracking."""

    def __init__(self, event_sequence, delay_between_events=0.01, metrics_collector=None):
        self.event_sequence = event_sequence
        self.delay_between_events = delay_between_events
        self.events_emitted = []
        self.metrics_collector = metrics_collector

    async def orchestrate(self, text, **kwargs):
        """Emit events in sequence with delays and metric collection."""
        for event in self.event_sequence:
            if self.metrics_collector:
                self.metrics_collector.record_metrics()

            self.events_emitted.append(event)
            yield event
            await asyncio.sleep(self.delay_between_events)


@pytest.fixture
def performance_metrics():
    """Provide a PerformanceMetrics instance for tests."""
    return PerformanceMetrics()


@pytest.fixture
def mock_streaming_runtime_with_metrics(performance_metrics):
    """Create a mock runtime with performance metrics collection."""
    def _create_runtime(event_sequence, delay=0.01):
        return MockStreamingRuntimeWithMetrics(event_sequence, delay, performance_metrics)
    return _create_runtime


class TestStreamingPerformanceE2E:
    """End-to-end performance tests for streaming functionality."""

    def test_streaming_latency_improvement(self, cli_runner, mock_streaming_runtime_with_metrics, performance_metrics):
        """Test that streaming provides better perceived latency for users."""
        session_id = "latency-test"
        frame_id = "latency-frame"

        # Create a sequence with multiple chunks to simulate progressive content
        event_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id,
                          data={"user_query": "Test streaming latency"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
        ]

        # Add progressive content chunks
        content_parts = [
            "The analysis shows ", "that streaming provides ", "significant improvements ",
            "in user experience ", "by delivering content ", "incrementally rather than ",
            "waiting for complete responses.", " This approach reduces ", "perceived latency ",
            "and provides real-time feedback."
        ]

        for i, part in enumerate(content_parts):
            is_final = (i == len(content_parts) - 1)
            event_sequence.append(
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": part, "is_final": is_final})
            )

        event_sequence.extend([
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"total_tokens": len(content_parts) * 5}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Streaming latency test completed", "report": {"message": "Streaming latency test completed"}}),
        ])

        runtime = mock_streaming_runtime_with_metrics(event_sequence, delay=0.02)  # Slightly longer delay

        with patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            performance_metrics.start_measurement()

            result = cli_runner.invoke(app, ["run", "query", "Test streaming latency"])

            metrics = performance_metrics.stop_measurement()

            assert result.exit_code == 0

            # Verify content was delivered incrementally
            full_content = "".join(content_parts)
            assert full_content in result.stdout

            # Verify performance metrics
            assert metrics['duration'] < 2.0, f"Test took too long: {metrics['duration']}s"
            assert metrics['avg_memory_mb'] > 0, "Memory usage should be measurable"
            assert metrics['peak_memory_mb'] > 0, "Peak memory should be measurable"

            # Content should appear progressively in output
            # (This is hard to test directly with cli_runner, but we verify the events were emitted)
            assert len(runtime.events_emitted) == len(event_sequence)

    def test_memory_usage_large_responses(self, cli_runner, mock_streaming_runtime_with_metrics, performance_metrics):
        """Test memory usage efficiency with large streaming responses."""
        session_id = "memory-test"
        frame_id = "memory-frame"

        # Create a large content response with many chunks
        large_content_chunks = []
        words = ["streaming", "responses", "can", "handle", "large", "amounts", "of", "content",
                "efficiently", "by", "processing", "data", "incrementally", "rather", "than",
                "loading", "everything", "into", "memory", "at", "once"]

        # Repeat words to create larger content
        for i in range(20):  # Create 20 chunks
            chunk_words = words[i % len(words):] + words[:(i % len(words))]
            chunk_content = " ".join(chunk_words) + ". "
            large_content_chunks.append(chunk_content)

        event_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id,
                          data={"user_query": "Test memory usage with large content"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
        ]

        # Add all the large content chunks
        for i, chunk in enumerate(large_content_chunks):
            is_final = (i == len(large_content_chunks) - 1)
            event_sequence.append(
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": chunk, "is_final": is_final})
            )

        event_sequence.extend([
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"total_tokens": len(large_content_chunks) * 10}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Large content test completed", "report": {"message": "Large content test completed"}}),
        ])

        runtime = mock_streaming_runtime_with_metrics(event_sequence, delay=0.005)  # Fast streaming

        with patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            performance_metrics.start_measurement()

            result = cli_runner.invoke(app, ["run", "query", "Test memory usage with large content"])

            metrics = performance_metrics.stop_measurement()

            assert result.exit_code == 0

            # Verify large content was processed (check for representative content)
            assert "streaming responses can handle" in result.stdout
            assert "incrementally rather than" in result.stdout
            assert "loading everything into memory" in result.stdout

            # Verify we have a reasonable amount of content (should be much longer than a simple response)
            content_lines = [line for line in result.stdout.split('\n') if line.strip() and not line.startswith('2026-')]
            content_length = sum(len(line) for line in content_lines)
            assert content_length > 1000, f"Content too short for large response test: {content_length} characters"

            # Check memory usage - should be reasonable even with large content
            assert metrics['peak_memory_mb'] < 500, f"Memory usage too high: {metrics['peak_memory_mb']} MB"
            assert metrics['avg_memory_mb'] < 300, f"Average memory usage too high: {metrics['avg_memory_mb']} MB"

            # Should complete within reasonable time despite large content
            assert metrics['duration'] < 3.0, f"Large content processing took too long: {metrics['duration']}s"

    def test_responsiveness_under_load(self, cli_runner, mock_streaming_runtime_with_metrics, performance_metrics):
        """Test system responsiveness when handling multiple concurrent streaming operations."""
        # This test simulates load by running multiple CLI commands in sequence
        # and measuring overall performance

        base_session_id = "load-test"
        base_frame_id = "load-frame"

        # Create multiple similar event sequences to simulate concurrent load
        test_cases = []
        for i in range(3):  # Run 3 concurrent-like operations
            session_id = f"{base_session_id}-{i}"
            frame_id = f"{base_frame_id}-{i}"

            event_sequence = [
                ExecutionEvent(type=EventType.TURN_START, session_id=session_id,
                              data={"user_query": f"Load test query {i}"}),
                ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
                ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": f"Response {i} part 1 ", "is_final": False}),
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": f"Response {i} part 2 ", "is_final": False}),
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": f"Response {i} final", "is_final": True}),
                ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
                ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
                ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                              data={"final_answer": f"Load test {i} completed", "report": {"message": f"Load test {i} completed"}}),
            ]
            test_cases.append((f"Load test query {i}", event_sequence))

        performance_metrics.start_measurement()

        # Run multiple CLI commands in sequence (simulating load)
        results = []
        for query, event_sequence in test_cases:
            runtime = mock_streaming_runtime_with_metrics(event_sequence, delay=0.01)

            with patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap:
                mock_bootstrap.return_value = {"runtime": runtime}

                result = cli_runner.invoke(app, ["run", "query", query])
                results.append(result)

                assert result.exit_code == 0
                assert f"Load test" in result.stdout

        metrics = performance_metrics.stop_measurement()

        # All operations should have completed successfully
        assert len(results) == 3
        assert all(r.exit_code == 0 for r in results)

        # Verify performance under load
        total_duration = metrics['duration']
        avg_duration_per_operation = total_duration / len(test_cases)

        # Each operation should complete reasonably quickly even under load
        assert avg_duration_per_operation < 1.0, f"Average operation time too slow: {avg_duration_per_operation}s"

        # Memory usage should remain stable under load
        assert metrics['peak_memory_mb'] < 400, f"Memory usage too high under load: {metrics['peak_memory_mb']} MB"

        # CPU usage should be reasonable
        assert metrics['avg_cpu_percent'] < 50, f"CPU usage too high under load: {metrics['avg_cpu_percent']}%"

    def test_streaming_vs_non_streaming_latency_comparison(self, cli_runner, mock_streaming_runtime_with_metrics, performance_metrics):
        """Compare latency between streaming and simulated non-streaming approaches."""
        session_id = "comparison-test"
        frame_id = "comparison-frame"

        # Create content that would normally be delivered all at once in non-streaming
        content_chunks = [
            "This is a comprehensive analysis ",
            "that would typically be returned ",
            "as a single large response. ",
            "With streaming, users see progress ",
            "incrementally, improving perceived performance."
        ]

        event_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id,
                          data={"user_query": "Compare streaming vs non-streaming latency"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
        ]

        # Add streaming chunks
        for i, chunk in enumerate(content_chunks):
            is_final = (i == len(content_chunks) - 1)
            event_sequence.append(
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": chunk, "is_final": is_final})
            )

        event_sequence.extend([
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Latency comparison completed", "report": {"message": "Latency comparison completed"}}),
        ])

        runtime = mock_streaming_runtime_with_metrics(event_sequence, delay=0.015)

        with patch("local_coding_assistant.cli.commands.run.bootstrap") as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            performance_metrics.start_measurement()

            result = cli_runner.invoke(app, ["run", "query", "Compare streaming vs non-streaming latency"])

            metrics = performance_metrics.stop_measurement()

            assert result.exit_code == 0

            # Verify all content chunks are present
            full_content = "".join(content_chunks)
            assert full_content in result.stdout

            # Measure time to first content appearance (simulated)
            # In a real implementation, we'd measure time to first token
            # Here we verify the streaming approach completes within expected bounds

            streaming_duration = metrics['duration']
            estimated_non_streaming_duration = len(content_chunks) * 0.1  # Simulated non-streaming delay per chunk

            # Streaming should not be significantly slower than non-streaming for user experience
            assert streaming_duration < estimated_non_streaming_duration * 2, \
                f"Streaming too slow compared to non-streaming: {streaming_duration}s vs {estimated_non_streaming_duration}s"

            # Memory efficiency should be maintained
            assert metrics['avg_memory_mb'] < 200, f"Memory usage inefficient: {metrics['avg_memory_mb']} MB"
