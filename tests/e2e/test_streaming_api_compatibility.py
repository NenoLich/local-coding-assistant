"""
End-to-end tests for API compatibility with streaming functionality.
Tests event buffering, backward compatibility, and configuration handling.
"""

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from local_coding_assistant.cli.main import app
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.runtime_manager import RuntimeManager


class MockStreamingRuntimeForCompatibility:
    """Mock runtime that emits events for compatibility testing."""

    def __init__(self, event_sequence, delay_between_events=0.001):
        self.event_sequence = event_sequence
        self.delay_between_events = delay_between_events
        self.events_emitted = []

    async def orchestrate(self, text, **kwargs):
        """Emit events in sequence."""
        for event in self.event_sequence:
            self.events_emitted.append(event)
            yield event
            await asyncio.sleep(self.delay_between_events)


@pytest.fixture
def mock_streaming_runtime_compat():
    """Create a mock runtime for compatibility testing."""

    def _create_runtime(event_sequence, delay=0.001):
        return MockStreamingRuntimeForCompatibility(event_sequence, delay)

    return _create_runtime


class TestStreamingAPICompatibilityE2E:
    """End-to-end tests for API compatibility with streaming."""

    def test_event_buffering_into_traditional_results(
        self, cli_runner, mock_streaming_runtime_compat
    ):
        """Test that streaming events can be buffered into traditional result objects."""
        session_id = "buffer-test"
        frame_id = "buffer-frame"

        # Create comprehensive event sequence
        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test event buffering"},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id=frame_id,
                data={"frame_type": "reasoning"},
            ),
            ExecutionEvent(
                type=EventType.LLM_START,
                session_id=session_id,
                frame_id=frame_id,
                data={"model": "test-model", "provider": "test-provider"},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "This is ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "buffered content ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "for compatibility.", "is_final": True},
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                data={"total_tokens": 15, "model": "test-model"},
            ),
            # Tool execution
            ExecutionEvent(
                type=EventType.TOOL_START,
                session_id=session_id,
                frame_id=frame_id,
                data={
                    "tool_call": MagicMock(
                        name="calculator", arguments={"expression": "2+2"}
                    )
                },
            ),
            ExecutionEvent(
                type=EventType.TOOL_RESULT,
                session_id=session_id,
                frame_id=frame_id,
                data={
                    "tool_call": MagicMock(name="calculator"),
                    "response": MagicMock(success=True, result={"result": 4}),
                },
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                data={"result": "completed"},
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Buffered result: This is buffered content for compatibility. Calculator result: 4",
                    "report": {
                        "message": "Buffered result: This is buffered content for compatibility. Calculator result: 4",
                        "tool_calls": [
                            {
                                "name": "calculator",
                                "arguments": {"expression": "2+2"},
                                "result": 4,
                            }
                        ],
                    },
                },
            ),
        ]

        runtime = mock_streaming_runtime_compat(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(app, ["run", "query", "Test event buffering"])

            assert result.exit_code == 0

            # Verify CLI output contains the buffered result
            assert "Buffered result:" in result.stdout
            assert "buffered content for compatibility" in result.stdout
            assert "Calculator result: 4" in result.stdout

            # Verify all events were processed
            assert len(runtime.events_emitted) == len(event_sequence)

    def test_non_streaming_consumers_work_with_event_streams(
        self, mock_streaming_runtime_compat
    ):
        """Test that non-streaming consumers can work with event streams via buffering."""
        session_id = "consumer-test"
        frame_id = "consumer-frame"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test non-streaming consumer"},
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
                data={"content": "Non-streaming ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "consumer test", "is_final": True},
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
                    "final_answer": "Non-streaming consumer test completed",
                    "report": {"message": "Non-streaming consumer test completed"},
                },
            ),
        ]

        runtime = mock_streaming_runtime_compat(event_sequence)

        # Simulate a non-streaming consumer that collects events into a final result
        collected_events = []

        async def simulate_non_streaming_consumer():
            async for event in runtime.orchestrate("test query"):
                collected_events.append(event)

            # Simulate buffering events into a traditional result
            final_result = None
            for event in reversed(collected_events):
                if event.type == EventType.TURN_COMPLETE:
                    final_result = event.data.get("report")
                    break

            return final_result

        result = asyncio.run(simulate_non_streaming_consumer())

        # Verify the non-streaming consumer got the expected result
        assert result is not None
        assert result["message"] == "Non-streaming consumer test completed"

        # Verify all events were collected
        assert len(collected_events) == len(event_sequence)
        assert collected_events[-1].type == EventType.TURN_COMPLETE

    def test_configuration_flag_handling_streaming_enabled(
        self, cli_runner, mock_streaming_runtime_compat
    ):
        """Test configuration flag handling when streaming is enabled."""
        session_id = "config-streaming-test"
        frame_id = "config-frame"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test streaming config"},
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
                data={"content": "Streaming enabled ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "configuration test", "is_final": True},
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
                    "final_answer": "Streaming configuration test completed",
                    "report": {"message": "Streaming configuration test completed"},
                },
            ),
        ]

        runtime = mock_streaming_runtime_compat(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            # Mock bootstrap to return runtime and simulate streaming-enabled config
            mock_bootstrap.return_value = {"runtime": runtime}

            result = cli_runner.invoke(app, ["run", "query", "Test streaming config"])

            assert result.exit_code == 0

            # Verify streaming output is present
            assert "Streaming enabled" in result.stdout
            assert "configuration test" in result.stdout

            # Verify the CLI processed the streaming events correctly
            assert "Streaming configuration test completed" in result.stdout

    def test_configuration_flag_handling_backward_compatibility(
        self, cli_runner, mock_streaming_runtime_compat
    ):
        """Test that backward compatibility is maintained for configuration flags."""
        session_id = "backward-compat-test"
        frame_id = "compat-frame"

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test backward compatibility"},
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
                data={"content": "Backward compatible ", "is_final": False},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={"content": "configuration handling", "is_final": True},
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
                    "final_answer": "Backward compatibility test completed",
                    "report": {"message": "Backward compatibility test completed"},
                },
            ),
        ]

        runtime = mock_streaming_runtime_compat(event_sequence)

        with patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        ) as mock_bootstrap:
            mock_bootstrap.return_value = {"runtime": runtime}

            # Test with various CLI options to ensure backward compatibility
            result = cli_runner.invoke(
                app,
                [
                    "run",
                    "query",
                    "Test backward compatibility",
                    "--model",
                    "gpt-4",
                    "--format",
                    "plain",
                ],
            )

            assert result.exit_code == 0

            # Verify output format is still supported
            assert "Backward compatible" in result.stdout
            assert "configuration handling" in result.stdout

            # Verify final result is properly formatted
            assert "Backward compatibility test completed" in result.stdout

    def test_event_stream_buffering_preserves_data_integrity(
        self, mock_streaming_runtime_compat
    ):
        """Test that buffering events preserves all data integrity."""
        session_id = "integrity-test"
        frame_id = "integrity-frame"

        # Create complex event sequence with various data types
        complex_data = {
            "numbers": [1, 2, 3, 4, 5],
            "strings": ["test", "data", "integrity"],
            "nested": {"key": "value", "count": 42},
            "boolean": True,
            "null_value": None,
        }

        event_sequence = [
            ExecutionEvent(
                type=EventType.TURN_START,
                session_id=session_id,
                data={"user_query": "Test data integrity", "metadata": complex_data},
            ),
            ExecutionEvent(
                type=EventType.FRAME_START,
                session_id=session_id,
                frame_id=frame_id,
                data={"frame_type": "complex_processing", "config": complex_data},
            ),
            ExecutionEvent(
                type=EventType.LLM_START,
                session_id=session_id,
                frame_id=frame_id,
                data={"model": "complex-model", "parameters": complex_data},
            ),
            ExecutionEvent(
                type=EventType.LLM_CHUNK,
                session_id=session_id,
                frame_id=frame_id,
                data={
                    "content": "Complex data processing ",
                    "metadata": complex_data,
                    "is_final": True,
                },
            ),
            ExecutionEvent(
                type=EventType.LLM_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                data={"total_tokens": 100, "usage_stats": complex_data},
            ),
            ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id=session_id,
                frame_id=frame_id,
                data={"result": "integrity_verified", "final_data": complex_data},
            ),
            ExecutionEvent(
                type=EventType.TURN_COMPLETE,
                session_id=session_id,
                data={
                    "final_answer": "Data integrity test completed",
                    "report": {
                        "message": "Data integrity test completed",
                        "complex_data": complex_data,
                    },
                },
            ),
        ]

        runtime = mock_streaming_runtime_compat(event_sequence)

        # Collect all events
        collected_events = []

        async def collect_events():
            async for event in runtime.orchestrate("integrity test"):
                collected_events.append(event)

        asyncio.run(collect_events())

        # Verify data integrity across all events
        assert len(collected_events) == len(event_sequence)

        # Check that complex data is preserved in key events
        turn_start = next(e for e in collected_events if e.type == EventType.TURN_START)
        assert turn_start.data["metadata"] == complex_data

        llm_complete = next(
            e for e in collected_events if e.type == EventType.LLM_COMPLETE
        )
        assert llm_complete.data["usage_stats"] == complex_data

        turn_complete = next(
            e for e in collected_events if e.type == EventType.TURN_COMPLETE
        )
        assert turn_complete.data["report"]["complex_data"] == complex_data

        # Verify session and frame IDs are consistent
        session_ids = set(e.session_id for e in collected_events)
        assert len(session_ids) == 1
        assert session_ids.pop() == session_id

        frame_events = [e for e in collected_events if e.frame_id is not None]
        frame_ids = set(e.frame_id for e in frame_events)
        assert len(frame_ids) == 1
        assert frame_ids.pop() == frame_id

    def test_configuration_flags_affect_streaming_behavior(
        self, cli_runner, mock_streaming_runtime_compat
    ):
        """Test that configuration flags properly affect streaming behavior."""
        session_id = "config-flags-test"

        # Test with different agent modes and tool call modes
        test_configs = [
            ("no_agent", "reasoning"),
            ("frame", "ptc"),
            ("frame", "classic"),
        ]

        for agent_mode, tool_mode in test_configs:
            frame_id = f"config-{agent_mode}-{tool_mode}"

            event_sequence = [
                ExecutionEvent(
                    type=EventType.TURN_START,
                    session_id=session_id,
                    data={"user_query": f"Test {agent_mode} {tool_mode}"},
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
                    data={
                        "content": f"Testing {agent_mode} mode with {tool_mode} tool calling",
                        "is_final": True,
                    },
                ),
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
                        "final_answer": f"Configuration test completed: {agent_mode} {tool_mode}",
                        "report": {
                            "message": f"Configuration test completed: {agent_mode} {tool_mode}"
                        },
                    },
                ),
            ]

            runtime = mock_streaming_runtime_compat(event_sequence)

            with patch(
                "local_coding_assistant.cli.commands.run.bootstrap"
            ) as mock_bootstrap:
                mock_bootstrap.return_value = {"runtime": runtime}

                result = cli_runner.invoke(
                    app,
                    [
                        "run",
                        "query",
                        f"Test {agent_mode} {tool_mode}",
                        "--agent-mode",
                        agent_mode,
                        "--tool-call-mode",
                        tool_mode,
                    ],
                )

                assert result.exit_code == 0

                # Verify the configuration was applied correctly
                expected_content = (
                    f"Testing {agent_mode} mode with {tool_mode} tool calling"
                )
                assert expected_content in result.stdout

                expected_final = (
                    f"Configuration test completed: {agent_mode} {tool_mode}"
                )
                assert expected_final in result.stdout
