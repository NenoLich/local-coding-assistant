"""
End-to-end tests for CLI streaming functionality.
Tests real-time content display, tool execution feedback, and progress indicators.
"""

import asyncio
import time
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from local_coding_assistant.cli.commands import run as run_cli
from local_coding_assistant.cli.main import app
from local_coding_assistant.runtime.events import EventType, ExecutionEvent


class CapturingConsole(Console):
    """A Rich console that captures output for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_output = []

    def print(self, *args, **kwargs):
        """Capture print calls and also call the parent method."""
        # Capture the output
        output = StringIO()
        temp_console = Console(file=output, width=80)
        temp_console.print(*args, **kwargs)
        self.captured_output.append(output.getvalue())

        # Also call the real print method
        super().print(*args, **kwargs)


class MockStreamingRuntime:
    """Mock runtime that simulates streaming event emission."""

    def __init__(self, event_sequence, delay_between_events=0.01):
        self.event_sequence = event_sequence
        self.delay_between_events = delay_between_events
        self.events_emitted = []

    async def orchestrate(self, text, **kwargs):
        """Emit events in sequence with delays to simulate real streaming."""
        for event in self.event_sequence:
            self.events_emitted.append(event)
            yield event
            await asyncio.sleep(self.delay_between_events)


@pytest.fixture
def capturing_console():
    """Provide a console that captures output for verification."""
    return CapturingConsole()


@pytest.fixture
def mock_streaming_runtime():
    """Create a mock runtime that emits predefined streaming events."""

    def _create_runtime(event_sequence, delay=0.01):
        return MockStreamingRuntime(event_sequence, delay)

    return _create_runtime


class TestCLIStreamingE2E:
    """End-to-end tests for CLI streaming functionality."""

    def test_streaming_content_display_basic(
        self, cli_runner, mock_streaming_runtime, capturing_console
    ):
        """Test that LLM content is displayed incrementally in real-time."""
        # Create a simple streaming sequence
        session_id = "streaming-test-1"
        frame_id = "frame-1"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Hello world"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "Hello ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "world!", "is_final": True},
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Hello world!",
                    "report": {"message": "Hello world!"},
                },
            ),
        ]

        runtime = mock_streaming_runtime(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            start_time = time.time()
            result = cli_runner.invoke(app, ["run", "query", "Hello world"])
            end_time = time.time()

            # Should complete successfully
            assert result.exit_code == 0

            # Should complete within reasonable time (allowing for async delays)
            assert end_time - start_time < 1.0

            # Should contain the final response
            assert "Hello world!" in result.stdout

            # Verify streaming occurred (runtime emitted events)
            assert len(runtime.events_emitted) == len(event_sequence)

    def test_streaming_with_tool_execution_feedback(
        self, cli_runner, mock_streaming_runtime
    ):
        """Test streaming output includes tool execution feedback."""
        session_id = "tool-streaming-test"
        frame_id = "tool-frame"

        # Create mock tool call and response objects
        mock_tool_call = MagicMock()
        mock_tool_call.name = "calculator"
        mock_tool_call.arguments = {"expression": "5 + 3"}

        mock_tool_response = MagicMock()
        mock_tool_response.success = True
        mock_tool_response.result = {"result": 8}

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Calculate 5 + 3"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "I need to calculate 5 + 3. ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "Let me use the calculator.", "is_final": True},
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.TOOL_START,
                session_id=session_id,
                frame_id=frame_id,
                data={"tool_call": mock_tool_call},
            ),
            ExecutionEvent(
                type=EventType.TOOL_RESULT,
                session_id=session_id,
                frame_id=frame_id,
                data={"tool_call": mock_tool_call, "response": mock_tool_response},
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "The result is 8",
                    "report": {"message": "The result is 8"},
                },
            ),
        ]

        runtime = mock_streaming_runtime(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(app, ["run", "query", "Calculate 5 + 3"])

            assert result.exit_code == 0

            # Should contain the streaming content
            assert "I need to calculate 5 + 3." in result.stdout
            assert "Let me use the calculator." in result.stdout

            # Should contain tool execution feedback
            assert "Executing tool: calculator" in result.stdout
            assert "Tool calculator completed successfully" in result.stdout

            # Should contain the final answer
            assert "The result is 8" in result.stdout

    def test_streaming_with_multiple_iterations(
        self, cli_runner, mock_streaming_runtime
    ):
        """Test streaming output with multiple frames/iterations."""
        session_id = "multi-frame-test"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Complex task requiring multiple steps"},
            ),
            # First iteration
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id="frame-1"
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id="frame-1"
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id="frame-1",
                data={
                    "content": "First, I need to gather information.",
                    "is_final": True,
                },
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE, session_id=session_id, frame_id="frame-1"
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id="frame-1"
            ),
            # Second iteration
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id="frame-2"
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id="frame-2"
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id="frame-2",
                data={
                    "content": "Now I can provide the final answer.",
                    "is_final": True,
                },
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE, session_id=session_id, frame_id="frame-2"
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id="frame-2"
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Task completed in multiple steps",
                    "report": {"message": "Task completed in multiple steps"},
                },
            ),
        ]

        runtime = mock_streaming_runtime(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(
                app, ["run", "query", "Complex task requiring multiple steps"]
            )

            assert result.exit_code == 0

            # Should contain content from both iterations
            assert "First, I need to gather information." in result.stdout
            assert "Now I can provide the final answer." in result.stdout

            # Should show iteration progress indicators
            assert "Starting iteration frame-1" in result.stdout
            assert "Starting iteration frame-2" in result.stdout
            assert "Iteration frame-1 completed" in result.stdout
            assert "Iteration frame-2 completed" in result.stdout

    def test_streaming_error_handling_display(self, cli_runner, mock_streaming_runtime):
        """Test that streaming properly displays errors."""
        session_id = "error-streaming-test"
        frame_id = "error-frame"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "This will cause an error"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "Starting to process", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.ERROR,
                session_id=session_id,
                frame_id=frame_id,
                data={"error": "API rate limit exceeded"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                data={"result": "error"},
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Sorry, I encountered an error. Please try again.",
                    "report": {
                        "message": "Sorry, I encountered an error. Please try again."
                    },
                },
            ),
        ]

        runtime = mock_streaming_runtime(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(
                app, ["run", "query", "This will cause an error"]
            )

            assert result.exit_code == 0  # CLI should handle errors gracefully

            # Should contain partial content before error
            assert "Starting to process" in result.stdout

            # Should display the error
            assert "Error: API rate limit exceeded" in result.stdout

            # Should contain error recovery message
            assert "Sorry, I encountered an error" in result.stdout

    def test_streaming_performance_timing(self, cli_runner, mock_streaming_runtime):
        """Test that streaming provides responsive real-time feedback."""
        session_id = "performance-test"
        frame_id = "perf-frame"

        # Create a sequence with many small chunks to test responsiveness
        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Performance test"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id
            ),
            ExecutionEvent(
                type=EventType.LLM_START, session_id=session_id, frame_id=frame_id
            ),
        ]

        # Add many small content chunks
        for i in range(10):
            event_sequence.append(
                ExecutionEvent(
                    type=EventType.LLM_CHUNK,
                    session_id=session_id,
                    frame_id=frame_id,
                    data={"content": f"Chunk {i} ", "is_final": False},
                )
            )

        event_sequence.extend(
            [
                ExecutionEvent(
                    type=EventType.LLM_COMPLETE,
                    session_id=session_id,
                    frame_id=frame_id,
                ),
                ExecutionEvent(
                    type=EventType.FRAME_COMPLETE,
                    session_id=session_id,
                    frame_id=frame_id,
                ),
                ExecutionEvent(
                    type=EventType.TURN_COMPLETE,
                    session_id=session_id,
                    data={
                        "final_answer": "Performance test completed",
                        "report": {"message": "Performance test completed"},
                    },
                ),
            ]
        )

        # Use very small delay to test responsiveness
        runtime = mock_streaming_runtime(event_sequence, delay=0.001)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            start_time = time.time()
            result = cli_runner.invoke(app, ["run", "query", "Performance test"])
            end_time = time.time()

            assert result.exit_code == 0

            # Should complete within reasonable time
            duration = end_time - start_time
            assert duration < 2.0, f"Streaming took too long: {duration}s"

            # Should contain all chunks
            for i in range(10):
                assert f"Chunk {i}" in result.stdout

            # Verify all events were emitted
            assert len(runtime.events_emitted) == len(event_sequence)

    def test_streaming_progress_indicators(self, cli_runner, mock_streaming_runtime):
        """Test that progress indicators are shown during streaming."""
        session_id = "progress-test"
        frame_id = "progress-frame"

        # Create mock tool call and response objects
        mock_tool_call = MagicMock()
        mock_tool_call.name = "calculator"
        mock_tool_call.arguments = {"expression": "2*3"}

        mock_tool_response = MagicMock()
        mock_tool_response.success = True
        mock_tool_response.result = {"result": 6}

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test progress indicators"},
            ),
            # First iteration
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id="progress-frame",
                data={"frame_type": "reasoning", "step": 1, "total_steps": 3},
            ),
            ExecutionEvent(
                type=EventType.LLM_START,
                session_id=session_id,
                frame_id="progress-frame",
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id="progress-frame",
                data={"content": "Step 1: Analyzing the problem", "is_final": True},
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE,
                session_id=session_id,
                frame_id="progress-frame",
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id="progress-frame",
            ),
            # Second iteration
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id="progress-frame-2",
                data={"frame_type": "tool_execution", "step": 2, "total_steps": 3},
            ),
            ExecutionEvent(
                type=EventType.TOOL_START,
                session_id=session_id,
                frame_id="progress-frame-2",
                data={"tool_call": mock_tool_call},
            ),
            ExecutionEvent(
                type=EventType.TOOL_RESULT,
                session_id=session_id,
                frame_id="progress-frame-2",
                data={"tool_call": mock_tool_call, "response": mock_tool_response},
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id="progress-frame-2",
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Completed with progress tracking",
                    "report": {"message": "Completed with progress tracking"},
                },
            ),
        ]

        runtime = mock_streaming_runtime(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(
                app, ["run", "query", "Test progress indicators"]
            )

            assert result.exit_code == 0

            # Should show frame start indicators
            assert "Starting iteration progress-frame" in result.stdout
            assert "Starting iteration progress-frame-2" in result.stdout

            # Should show frame completion indicators
            assert "Iteration progress-frame completed" in result.stdout
            assert "Iteration progress-frame-2 completed" in result.stdout

            # Should show tool execution feedback
            assert "Executing tool: calculator" in result.stdout
            assert "Tool calculator completed successfully" in result.stdout
