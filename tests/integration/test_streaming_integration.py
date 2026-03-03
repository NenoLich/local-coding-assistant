"""
Integration tests for end-to-end streaming functionality.
Tests the event-driven execution flow from runtime to CLI output.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.frame_agent import FrameAgent
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.executor import ExecutionFrame, RuntimeExecutor
from local_coding_assistant.runtime.runtime_manager import RuntimeManager


class EventCollector:
    """Utility to collect and analyze event streams during integration tests."""

    def __init__(self):
        self.events: list[ExecutionEvent] = []
        self.start_time = time.time()

    def collect_event(self, event: ExecutionEvent) -> None:
        """Collect an event for later analysis."""
        self.events.append(event)

    async def collect_events(self, event_stream: AsyncIterator[ExecutionEvent]) -> list[ExecutionEvent]:
        """Collect all events from an async event stream."""
        self.events = []
        async for event in event_stream:
            self.collect_event(event)
        return self.events

    def get_events_by_type(self, event_type: EventType) -> list[ExecutionEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]

    def assert_event_sequence(self, expected_sequence: list[EventType]) -> None:
        """Assert that events occurred in the expected order."""
        actual_sequence = [e.type for e in self.events]
        assert actual_sequence == expected_sequence, f"Expected {expected_sequence}, got {actual_sequence}"

    def assert_event_timing(self, event_type: EventType, min_count: int = 1, max_count: int | None = None) -> None:
        """Assert timing constraints on events."""
        events = self.get_events_by_type(event_type)
        assert len(events) >= min_count, f"Expected at least {min_count} {event_type.value} events, got {len(events)}"
        if max_count is not None:
            assert len(events) <= max_count, f"Expected at most {max_count} {event_type.value} events, got {len(events)}"

    def assert_event_data_consistency(self, event_type: EventType, key: str, expected_value: Any) -> None:
        """Assert that all events of a type have consistent data."""
        events = self.get_events_by_type(event_type)
        for event in events:
            assert key in event.data, f"Event {event.type.value} missing key '{key}'"
            assert event.data[key] == expected_value, f"Event {event.type.value} has inconsistent {key}: {event.data[key]} != {expected_value}"

    def get_session_id(self) -> str | None:
        """Get the session ID from the first event."""
        if self.events:
            return self.events[0].session_id
        return None

    def get_frame_id(self) -> str | None:
        """Get the frame ID from FRAME_START events."""
        frame_events = self.get_events_by_type(EventType.FRAME_START)
        if frame_events:
            return frame_events[0].frame_id
        return None


class MockStreamingProvider:
    """Mock provider that emits predefined event sequences for testing."""

    def __init__(self, event_sequence: list[ExecutionEvent]):
        self.event_sequence = event_sequence
        self.current_index = 0

    async def stream_events(self) -> AsyncIterator[ExecutionEvent]:
        """Stream the predefined event sequence."""
        for event in self.event_sequence:
            yield event
            await asyncio.sleep(0.01)  # Small delay to simulate real streaming


class MockEventEmitter:
    """Mock event emitter that produces configurable event streams."""

    def __init__(self, session_id: str = "test-session", frame_id: str = "test-frame"):
        self.session_id = session_id
        self.frame_id = frame_id

    def create_event_sequence(self, scenario: str = "basic") -> list[ExecutionEvent]:
        """Create predefined event sequences for different test scenarios."""
        if scenario == "basic":
            return [
                ExecutionEvent(
                    type=EventType.TURN_START,
                    session_id=self.session_id,
                    data={"user_query": "Test query"}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"frame_type": "reasoning"}
                ),
                ExecutionEvent(
                    type=EventType.LLM_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"model": "test-model", "provider": "test-provider"}
                ),
                ExecutionEvent(
                    type=EventType.LLM_CHUNK,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"content": "This is a test", "is_final": False}
                ),
                ExecutionEvent(
                    type=EventType.LLM_CHUNK,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"content": " response for streaming", "is_final": False}
                ),
                ExecutionEvent(
                    type=EventType.LLM_COMPLETE,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"total_tokens": 10, "model": "test-model", "provider": "test-provider"}
                ),
                ExecutionEvent(
                    type=EventType.TOOL_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"tool_name": "test_tool", "tool_args": {"param": "value"}}
                ),
                ExecutionEvent(
                    type=EventType.TOOL_RESULT,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"tool_name": "test_tool", "result": {"success": True}, "execution_time_ms": 100}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_COMPLETE,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"result": "completed"}
                ),
                ExecutionEvent(
                    type=EventType.TURN_COMPLETE,
                    session_id=self.session_id,
                    data={"final_answer": "Test completed"}
                ),
            ]
        elif scenario == "error":
            return [
                ExecutionEvent(
                    type=EventType.TURN_START,
                    session_id=self.session_id,
                    data={"user_query": "Test error query"}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"frame_type": "reasoning"}
                ),
                ExecutionEvent(
                    type=EventType.ERROR,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"error_type": "TestError", "message": "Test error occurred"}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_COMPLETE,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"result": "error"}
                ),
                ExecutionEvent(
                    type=EventType.TURN_COMPLETE,
                    session_id=self.session_id,
                    data={"final_answer": "Error handled"}
                ),
            ]
        elif scenario == "tool_only":
            return [
                ExecutionEvent(
                    type=EventType.TURN_START,
                    session_id=self.session_id,
                    data={"user_query": "Tool only test"}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"frame_type": "tool_execution"}
                ),
                ExecutionEvent(
                    type=EventType.TOOL_START,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"tool_name": "calculator", "tool_args": {"expression": "2+2"}}
                ),
                ExecutionEvent(
                    type=EventType.TOOL_RESULT,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"tool_name": "calculator", "result": {"result": 4}, "execution_time_ms": 50}
                ),
                ExecutionEvent(
                    type=EventType.FRAME_COMPLETE,
                    session_id=self.session_id,
                    frame_id=self.frame_id,
                    data={"result": 4}
                ),
                ExecutionEvent(
                    type=EventType.TURN_COMPLETE,
                    session_id=self.session_id,
                    data={"final_answer": "4"}
                ),
            ]

        return []


@pytest.fixture
def event_collector():
    """Provide an EventCollector instance for tests."""
    return EventCollector()


@pytest.fixture
def mock_event_emitter():
    """Provide a MockEventEmitter instance for tests."""
    return MockEventEmitter()


@pytest.fixture
def mock_streaming_provider(mock_event_emitter):
    """Provide a MockStreamingProvider with basic event sequence."""
    events = mock_event_emitter.create_event_sequence("basic")
    return MockStreamingProvider(events)


@pytest.fixture
def mock_executor():
    """Provide a mock executor that yields events."""
    async def mock_execute() -> AsyncIterator[ExecutionEvent]:
        emitter = MockEventEmitter()
        events = emitter.create_event_sequence("basic")
        for event in events:
            yield event
            await asyncio.sleep(0.01)

    executor = MagicMock(spec=RuntimeExecutor)
    executor.execute = mock_execute
    return executor


@pytest.fixture
def mock_frame_agent(mock_executor):
    """Provide a mock frame agent that forwards events from executor."""
    async def mock_run() -> AsyncIterator[ExecutionEvent]:
        async for event in mock_executor.execute():
            yield event

    agent = MagicMock(spec=FrameAgent)
    agent.run = mock_run
    return agent


@pytest.fixture
def mock_runtime_manager(mock_frame_agent):
    """Provide a mock runtime manager that forwards events from frame agent."""
    async def mock_orchestrate() -> AsyncIterator[ExecutionEvent]:
        async for event in mock_frame_agent.run():
            yield event

    runtime = MagicMock(spec=RuntimeManager)
    runtime.orchestrate = mock_orchestrate
    return runtime


class TestEventFlowIntegration:
    """Integration tests for complete event chains in streaming execution."""

    @pytest.mark.asyncio
    async def test_end_to_end_basic_execution_flow(self, mock_runtime_manager, event_collector):
        """Test complete execution flow: TURN_START → FRAME_START → LLM_START → LLM_CHUNK* → LLM_COMPLETE → TOOL_START → TOOL_RESULT → FRAME_COMPLETE → TURN_COMPLETE"""
        # Collect all events from the execution
        events = await event_collector.collect_events(mock_runtime_manager.orchestrate())

        # Assert the complete event sequence
        expected_sequence = [
            EventType.TURN_START,
            EventType.FRAME_START,
            EventType.LLM_START,
            EventType.LLM_CHUNK,
            EventType.LLM_CHUNK,
            EventType.LLM_COMPLETE,
            EventType.TOOL_START,
            EventType.TOOL_RESULT,
            EventType.FRAME_COMPLETE,
            EventType.TURN_COMPLETE,
        ]

        event_collector.assert_event_sequence(expected_sequence)

        # Assert event data consistency
        session_id = event_collector.get_session_id()
        assert session_id is not None
        event_collector.assert_event_data_consistency(EventType.FRAME_START, "frame_type", "reasoning")

        # Assert timing constraints
        event_collector.assert_event_timing(EventType.LLM_CHUNK, min_count=2)
        event_collector.assert_event_timing(EventType.TOOL_START, min_count=1, max_count=1)

    @pytest.mark.asyncio
    async def test_event_ordering_and_timing(self, mock_runtime_manager, event_collector):
        """Test that events occur in the correct order and within expected timeframes."""
        start_time = time.time()

        events = await event_collector.collect_events(mock_runtime_manager.orchestrate())

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (allowing for async delays)
        assert duration < 1.0, f"Execution took too long: {duration}s"

        # Verify event ordering
        event_types = [e.type for e in events]

        # TURN_START should be first
        assert event_types[0] == EventType.TURN_START

        # TURN_COMPLETE should be last
        assert event_types[-1] == EventType.TURN_COMPLETE

        # FRAME_START should come before FRAME_COMPLETE
        frame_start_idx = event_types.index(EventType.FRAME_START)
        frame_complete_idx = event_types.index(EventType.FRAME_COMPLETE)
        assert frame_start_idx < frame_complete_idx

        # LLM events should be properly ordered
        llm_start_idx = event_types.index(EventType.LLM_START)
        llm_complete_idx = event_types.index(EventType.LLM_COMPLETE)
        llm_chunk_indices = [i for i, t in enumerate(event_types) if t == EventType.LLM_CHUNK]

        assert llm_start_idx < min(llm_chunk_indices)
        assert max(llm_chunk_indices) < llm_complete_idx

    @pytest.mark.asyncio
    async def test_event_data_consistency_across_components(self, mock_runtime_manager, event_collector):
        """Test that event data is consistent across the execution pipeline."""
        events = await event_collector.collect_events(mock_runtime_manager.orchestrate())

        # Get session and frame IDs
        session_id = event_collector.get_session_id()
        frame_id = event_collector.get_frame_id()

        # All events should have the same session_id
        for event in events:
            assert event.session_id == session_id

        # Events within a frame should have the same frame_id
        frame_events = [e for e in events if e.frame_id is not None]
        for event in frame_events:
            assert event.frame_id == frame_id

        # FRAME_START should have frame metadata
        frame_start_events = event_collector.get_events_by_type(EventType.FRAME_START)
        assert len(frame_start_events) == 1
        assert "frame_type" in frame_start_events[0].data

        # LLM events should have model information
        llm_events = event_collector.get_events_by_type(EventType.LLM_START) + \
                    event_collector.get_events_by_type(EventType.LLM_COMPLETE)
        for event in llm_events:
            assert "model" in event.data
            assert "provider" in event.data

        # Tool events should have tool information
        tool_events = event_collector.get_events_by_type(EventType.TOOL_START) + \
                     event_collector.get_events_by_type(EventType.TOOL_RESULT)
        for event in tool_events:
            assert "tool_name" in event.data

    @pytest.mark.asyncio
    async def test_tool_only_execution_flow(self, event_collector):
        """Test execution flow for tool-only scenarios."""
        # Create tool-only event sequence
        emitter = MockEventEmitter()
        events_sequence = emitter.create_event_sequence("tool_only")
        provider = MockStreamingProvider(events_sequence)

        events = await event_collector.collect_events(provider.stream_events())

        expected_sequence = [
            EventType.TURN_START,
            EventType.FRAME_START,
            EventType.TOOL_START,
            EventType.TOOL_RESULT,
            EventType.FRAME_COMPLETE,
            EventType.TURN_COMPLETE,
        ]

        event_collector.assert_event_sequence(expected_sequence)

        # Verify tool-specific data
        tool_start_events = event_collector.get_events_by_type(EventType.TOOL_START)
        assert len(tool_start_events) == 1
        assert tool_start_events[0].data["tool_name"] == "calculator"
        assert "tool_args" in tool_start_events[0].data

        tool_result_events = event_collector.get_events_by_type(EventType.TOOL_RESULT)
        assert len(tool_result_events) == 1
        assert tool_result_events[0].data["tool_name"] == "calculator"
        assert "result" in tool_result_events[0].data
        assert "execution_time_ms" in tool_result_events[0].data

    @pytest.mark.asyncio
    async def test_error_handling_event_flow(self, event_collector):
        """Test event flow when errors occur during execution."""
        # Create error event sequence
        emitter = MockEventEmitter()
        events_sequence = emitter.create_event_sequence("error")
        provider = MockStreamingProvider(events_sequence)

        events = await event_collector.collect_events(provider.stream_events())

        expected_sequence = [
            EventType.TURN_START,
            EventType.FRAME_START,
            EventType.ERROR,
            EventType.FRAME_COMPLETE,
            EventType.TURN_COMPLETE,
        ]

        event_collector.assert_event_sequence(expected_sequence)

        # Verify error event data
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert "error_type" in error_events[0].data
        assert "message" in error_events[0].data
        assert error_events[0].data["error_type"] == "TestError"

    @pytest.mark.asyncio
    async def test_multiple_frames_execution_flow(self, event_collector):
        """Test execution flow with multiple frames in a single turn."""
        # Create a sequence with multiple frames
        session_id = "multi-frame-session"
        emitter1 = MockEventEmitter(session_id, "frame-1")
        emitter2 = MockEventEmitter(session_id, "frame-2")

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Multi-frame test"}),
            # First frame
            *emitter1.create_event_sequence("tool_only")[1:-1],  # Skip TURN_START and TURN_COMPLETE
            # Second frame
            *emitter2.create_event_sequence("basic")[1:-1],     # Skip TURN_START and TURN_COMPLETE
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id, data={"final_answer": "Multi-frame completed"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Should have events from both frames
        frame_start_events = event_collector.get_events_by_type(EventType.FRAME_START)
        assert len(frame_start_events) == 2

        # Each frame should have its own frame_id
        frame_ids = set(e.frame_id for e in frame_start_events)
        assert len(frame_ids) == 2
        assert "frame-1" in frame_ids
        assert "frame-2" in frame_ids

        # All events should share the same session_id
        for event in events:
            assert event.session_id == session_id


class TestStreamingResponseIntegration:
    """Integration tests for streaming LLM responses from provider to CLI output."""

    @pytest.mark.asyncio
    async def test_llm_streaming_from_provider_to_output(self, event_collector):
        """Test that LLM streaming works from provider through to event stream."""
        # Create a streaming provider with chunked content
        session_id = "streaming-test-session"
        frame_id = "streaming-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Stream test"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id,
                          data={"model": "test-model", "provider": "test-provider"}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "This is ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "a streaming ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "response test.", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"total_tokens": 8, "model": "test-model"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id, data={"final_answer": "Test completed"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify LLM chunk events
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        assert len(llm_chunks) == 3

        # Verify content accumulation
        full_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        assert full_content == "This is a streaming response test."

        # Verify final chunk is marked as final
        assert not llm_chunks[0].data["is_final"]
        assert not llm_chunks[1].data["is_final"]
        assert llm_chunks[2].data["is_final"]

    @pytest.mark.asyncio
    async def test_incremental_content_rendering(self, event_collector):
        """Test that content is rendered incrementally as chunks arrive."""
        # Simulate progressive content building
        session_id = "incremental-test"
        frame_id = "incremental-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Hello", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": " world", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "!", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Collect content progressively
        accumulated_content = []
        current_content = ""

        for event in events:
            if event.type == EventType.LLM_CHUNK:
                current_content += event.data["content"]
                accumulated_content.append(current_content)

        # Verify progressive accumulation
        expected_progression = ["Hello", "Hello world", "Hello world!"]
        assert accumulated_content == expected_progression

    @pytest.mark.asyncio
    async def test_reasoning_and_tool_call_streaming(self, event_collector):
        """Test streaming of reasoning content followed by tool calls."""
        session_id = "reasoning-test"
        frame_id = "reasoning-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Calculate 5 + 3"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            # Reasoning content
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "I need to calculate 5 + 3. ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Let me use the calculator tool.", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
            # Tool execution
            ExecutionEvent(type=EventType.TOOL_START, session_id=session_id, frame_id=frame_id,
                          data={"tool_name": "calculator", "tool_args": {"expression": "5 + 3"}}),
            ExecutionEvent(type=EventType.TOOL_RESULT, session_id=session_id, frame_id=frame_id,
                          data={"tool_name": "calculator", "result": {"result": 8}, "execution_time_ms": 25}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id, data={"final_answer": "8"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify reasoning content
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        reasoning_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        assert "calculate 5 + 3" in reasoning_content.lower()
        assert "calculator tool" in reasoning_content.lower()

        # Verify tool call follows reasoning
        event_types = [e.type for e in events]
        llm_complete_idx = event_types.index(EventType.LLM_COMPLETE)
        tool_start_idx = event_types.index(EventType.TOOL_START)
        assert tool_start_idx > llm_complete_idx

        # Verify tool execution data
        tool_start_events = event_collector.get_events_by_type(EventType.TOOL_START)
        assert len(tool_start_events) == 1
        assert tool_start_events[0].data["tool_name"] == "calculator"
        assert tool_start_events[0].data["tool_args"]["expression"] == "5 + 3"

        tool_result_events = event_collector.get_events_by_type(EventType.TOOL_RESULT)
        assert len(tool_result_events) == 1
        assert tool_result_events[0].data["result"]["result"] == 8

    @pytest.mark.asyncio
    async def test_streaming_with_empty_chunks(self, event_collector):
        """Test streaming behavior with empty or whitespace-only chunks."""
        session_id = "empty-chunks-test"
        frame_id = "empty-chunks-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Hello", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "", "is_final": False}),  # Empty chunk
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "   ", "is_final": False}),  # Whitespace chunk
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "world", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        assert len(llm_chunks) == 4

        # Content should handle empty chunks gracefully
        contents = [chunk.data["content"] for chunk in llm_chunks]
        assert contents[0] == "Hello"
        assert contents[1] == ""
        assert contents[2] == "   "
        assert contents[3] == "world"

        # Final content should be correct
        full_content = "".join(contents)
        assert full_content == "Hello   world"

    @pytest.mark.asyncio
    async def test_streaming_performance_constraints(self, event_collector):
        """Test that streaming events are emitted within performance constraints."""
        import time

        session_id = "performance-test"
        frame_id = "performance-frame"

        # Create many small chunks to test throughput
        events_sequence = [
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
        ]

        # Add many small chunks
        for i in range(10):
            events_sequence.append(
                ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                              data={"content": f"chunk{i} ", "is_final": False})
            )

        events_sequence.append(
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "final", "is_final": True})
        )
        events_sequence.append(
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id)
        )

        provider = MockStreamingProvider(events_sequence)

        start_time = time.time()
        events = await event_collector.collect_events(provider.stream_events())
        end_time = time.time()

        duration = end_time - start_time

        # Should complete within reasonable time (allowing for async delays)
        assert duration < 2.0, f"Streaming took too long: {duration}s"

        # All chunks should be present
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        assert len(llm_chunks) == 11  # 10 chunks + 1 final

        # Content should be complete
        full_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        expected_content = "".join(f"chunk{i} " for i in range(10)) + "final"
        assert full_content == expected_content


class TestErrorHandlingIntegration:
    """Integration tests for error handling in streaming execution."""

    @pytest.mark.asyncio
    async def test_error_event_propagation(self, event_collector):
        """Test that errors are properly emitted and propagated through the event stream."""
        session_id = "error-prop-test"
        frame_id = "error-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test error"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Starting response", "is_final": False}),
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "LLMError", "message": "Model API rate limit exceeded", "component": "llm_provider"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "error", "error_handled": True}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Sorry, I encountered an error. Please try again."}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify error event is present
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1

        error_event = error_events[0]
        assert error_event.data["error_type"] == "LLMError"
        assert "rate limit" in error_event.data["message"].lower()
        assert error_event.data["component"] == "llm_provider"

        # Verify error occurs after LLM start but before completion
        event_types = [e.type for e in events]
        llm_start_idx = event_types.index(EventType.LLM_START)
        error_idx = event_types.index(EventType.ERROR)
        llm_complete_exists = EventType.LLM_COMPLETE in event_types

        assert error_idx > llm_start_idx
        assert not llm_complete_exists  # No LLM_COMPLETE due to error

    @pytest.mark.asyncio
    async def test_error_recovery_and_cleanup(self, event_collector):
        """Test error recovery mechanisms and proper cleanup after errors."""
        session_id = "error-recovery-test"
        frame_id = "error-recovery-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test recovery"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "NetworkError", "message": "Connection timeout", "recoverable": True}),
            # Recovery attempt
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id,
                          data={"retry_attempt": 1}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Recovered response", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Recovered successfully"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Should have one error event
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data.get("recoverable", False)

        # Should have two LLM_START events (original + retry)
        llm_start_events = event_collector.get_events_by_type(EventType.LLM_START)
        assert len(llm_start_events) == 2

        # Should have successful completion
        llm_complete_events = event_collector.get_events_by_type(EventType.LLM_COMPLETE)
        assert len(llm_complete_events) == 1

        turn_complete_events = event_collector.get_events_by_type(EventType.TURN_COMPLETE)
        assert len(turn_complete_events) == 1
        assert "Recovered successfully" in turn_complete_events[0].data["final_answer"]

    @pytest.mark.asyncio
    async def test_partial_failure_scenarios(self, event_collector):
        """Test handling of partial failures during streaming execution."""
        session_id = "partial-failure-test"
        frame_id = "partial-failure-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test partial failure"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Partial content ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "before failure", "is_final": True}),
            # Tool execution starts successfully
            ExecutionEvent(type=EventType.TOOL_START, session_id=session_id, frame_id=frame_id,
                          data={"tool_name": "calculator", "tool_args": {"expression": "2 + 2"}}),
            # Tool fails
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "ToolExecutionError", "message": "Tool execution failed", "tool_name": "calculator"}),
            # Frame completes with partial results
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "partial", "partial_content": "Partial content before failure"}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Partial result: Partial content before failure"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify partial content was received
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        partial_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        assert partial_content == "Partial content before failure"

        # Verify tool failure
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "ToolExecutionError"
        assert error_events[0].data["tool_name"] == "calculator"

        # Verify no TOOL_RESULT event (tool failed)
        tool_result_events = event_collector.get_events_by_type(EventType.TOOL_RESULT)
        assert len(tool_result_events) == 0

        # Verify frame completed with partial results
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].data["result"] == "partial"
        assert "partial_content" in frame_complete_events[0].data

    @pytest.mark.asyncio
    async def test_multiple_error_types_handling(self, event_collector):
        """Test handling of different types of errors in the event stream."""
        session_id = "multi-error-test"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test multiple errors"}),
            # Frame 1: LLM Error
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id="frame-1"),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id="frame-1"),
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id="frame-1",
                          data={"error_type": "AuthenticationError", "message": "Invalid API key", "component": "llm_provider"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id="frame-1",
                          data={"result": "error"}),
            # Frame 2: Tool Error
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id="frame-2"),
            ExecutionEvent(type=EventType.TOOL_START, session_id=session_id, frame_id="frame-2",
                          data={"tool_name": "weather", "tool_args": {"location": "invalid"}}),
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id="frame-2",
                          data={"error_type": "ToolValidationError", "message": "Invalid location parameter", "tool_name": "weather"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id="frame-2",
                          data={"result": "error"}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Multiple errors encountered during execution"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Should have 2 error events
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 2

        # Verify different error types
        error_types = {e.data["error_type"] for e in error_events}
        assert error_types == {"AuthenticationError", "ToolValidationError"}

        # Verify errors are in different frames
        frame_ids = {e.frame_id for e in error_events}
        assert len(frame_ids) == 2

        # Verify 2 frames completed with errors
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 2
        assert all(e.data["result"] == "error" for e in frame_complete_events)

    @pytest.mark.asyncio
    async def test_error_event_data_consistency(self, event_collector):
        """Test that error events contain consistent and required data fields."""
        session_id = "error-data-test"
        frame_id = "error-data-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test error data"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "TestError", "message": "Test error message", "component": "test_component",
                                "timestamp": "2023-01-01T12:00:00Z", "stack_trace": "test stack trace"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id, data={"final_answer": "Error handled"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1

        error_event = error_events[0]

        # Required fields
        required_fields = ["error_type", "message"]
        for field in required_fields:
            assert field in error_event.data, f"Error event missing required field: {field}"

        # Optional but recommended fields
        recommended_fields = ["component", "timestamp"]
        for field in recommended_fields:
            assert field in error_event.data, f"Error event missing recommended field: {field}"

        # Verify data types
        assert isinstance(error_event.data["error_type"], str)
        assert isinstance(error_event.data["message"], str)
        assert error_event.data["error_type"] == "TestError"
        assert error_event.data["message"] == "Test error message"


class TestTimeoutCancellationIntegration:
    """Integration tests for timeout and cancellation handling in streaming execution."""

    @pytest.mark.asyncio
    async def test_execution_timeout_handling(self, event_collector):
        """Test handling of execution timeouts during streaming."""
        import asyncio

        session_id = "timeout-test"
        frame_id = "timeout-frame"

        # Simulate a timeout scenario
        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test timeout"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Starting slow response", "is_final": False}),
            # Simulate timeout error
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "TimeoutError", "message": "Execution timed out after 30 seconds",
                                "component": "runtime_manager", "timeout_duration": 30}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "timeout", "partial_content": "Starting slow response"}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Execution timed out. Partial result: Starting slow response"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify timeout error
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "TimeoutError"
        assert "timed out" in error_events[0].data["message"].lower()
        assert error_events[0].data["timeout_duration"] == 30

        # Verify partial content preservation
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].data["result"] == "timeout"
        assert "partial_content" in frame_complete_events[0].data

    @pytest.mark.asyncio
    async def test_cancellation_during_streaming(self, event_collector):
        """Test cancellation of streaming execution."""
        import asyncio

        session_id = "cancellation-test"
        frame_id = "cancellation-frame"

        # Simulate cancellation during LLM streaming
        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test cancellation"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Partial content ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "before cancellation", "is_final": False}),
            # Cancellation event
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "CancellationError", "message": "Execution cancelled by user",
                                "component": "runtime_manager"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "cancelled", "partial_content": "Partial content before cancellation"}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Execution cancelled. Partial result: Partial content before cancellation"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify cancellation error
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "CancellationError"
        assert "cancelled" in error_events[0].data["message"].lower()

        # Verify partial content was preserved
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        partial_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        assert partial_content == "Partial content before cancellation"

        # Verify frame marked as cancelled
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].data["result"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cleanup_on_interruption(self, event_collector):
        """Test proper cleanup when execution is interrupted."""
        session_id = "cleanup-test"
        frame_id = "cleanup-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test cleanup"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "Content before ", "is_final": False}),
            ExecutionEvent(type=EventType.TOOL_START, session_id=session_id, frame_id=frame_id,
                          data={"tool_name": "calculator", "tool_args": {"expression": "1 + 1"}}),
            # Interruption occurs
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "InterruptionError", "message": "Execution interrupted",
                                "component": "runtime_manager"}),
            # Cleanup events
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "interrupted", "cleanup_performed": True}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Execution interrupted during tool execution"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify interruption error
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "InterruptionError"

        # Verify cleanup was performed
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].data["result"] == "interrupted"
        assert frame_complete_events[0].data.get("cleanup_performed", False)

        # Verify no tool result (interrupted during tool execution)
        tool_result_events = event_collector.get_events_by_type(EventType.TOOL_RESULT)
        assert len(tool_result_events) == 0

        # But tool start should be present
        tool_start_events = event_collector.get_events_by_type(EventType.TOOL_START)
        assert len(tool_start_events) == 1

    @pytest.mark.asyncio
    async def test_graceful_timeout_with_partial_results(self, event_collector):
        """Test graceful handling of timeouts with partial results preservation."""
        session_id = "graceful-timeout-test"
        frame_id = "graceful-timeout-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test graceful timeout"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            # Multiple chunks of content
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "This is a ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "long running ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "response that ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "takes time to ", "is_final": False}),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "generate", "is_final": False}),
            # Timeout occurs
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "TimeoutError", "message": "Execution timed out after 60 seconds",
                                "component": "runtime_manager", "timeout_duration": 60}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "timeout", "partial_content": "This is a long running response that takes time to generate",
                                "chunks_received": 5, "progress_percentage": 60}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Timeout after 60s. Partial result: This is a long running response that takes time to generate"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify all partial content preserved
        llm_chunks = event_collector.get_events_by_type(EventType.LLM_CHUNK)
        assert len(llm_chunks) == 5

        partial_content = "".join(chunk.data["content"] for chunk in llm_chunks)
        expected_content = "This is a long running response that takes time to generate"
        assert partial_content == expected_content

        # Verify timeout metadata
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["timeout_duration"] == 60

        # Verify frame completion with progress information
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        frame_complete = frame_complete_events[0]

        assert frame_complete.data["result"] == "timeout"
        assert frame_complete.data["chunks_received"] == 5
        assert frame_complete.data["progress_percentage"] == 60
        assert frame_complete.data["partial_content"] == expected_content

    @pytest.mark.asyncio
    async def test_cancellation_during_tool_execution(self, event_collector):
        """Test cancellation specifically during tool execution phase."""
        session_id = "tool-cancel-test"
        frame_id = "tool-cancel-frame"

        events_sequence = [
            ExecutionEvent(type=EventType.TURN_START, session_id=session_id, data={"user_query": "Test tool cancellation"}),
            ExecutionEvent(type=EventType.FRAME_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_START, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.LLM_CHUNK, session_id=session_id, frame_id=frame_id,
                          data={"content": "I need to calculate something.", "is_final": True}),
            ExecutionEvent(type=EventType.LLM_COMPLETE, session_id=session_id, frame_id=frame_id),
            ExecutionEvent(type=EventType.TOOL_START, session_id=session_id, frame_id=frame_id,
                          data={"tool_name": "calculator", "tool_args": {"expression": "complex_calculation()"}}),
            # Cancellation during tool execution
            ExecutionEvent(type=EventType.ERROR, session_id=session_id, frame_id=frame_id,
                          data={"error_type": "CancellationError", "message": "User cancelled during tool execution",
                                "component": "tool_manager", "tool_name": "calculator"}),
            ExecutionEvent(type=EventType.FRAME_COMPLETE, session_id=session_id, frame_id=frame_id,
                          data={"result": "cancelled", "tool_cancelled": True, "cleanup_resources": True}),
            ExecutionEvent(type=EventType.TURN_COMPLETE, session_id=session_id,
                          data={"final_answer": "Tool execution cancelled by user"}),
        ]

        provider = MockStreamingProvider(events_sequence)
        events = await event_collector.collect_events(provider.stream_events())

        # Verify cancellation during tool execution
        error_events = event_collector.get_events_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0].data["error_type"] == "CancellationError"
        assert "tool execution" in error_events[0].data["message"]
        assert error_events[0].data["tool_name"] == "calculator"

        # Verify tool was started but not completed
        tool_start_events = event_collector.get_events_by_type(EventType.TOOL_START)
        assert len(tool_start_events) == 1

        tool_result_events = event_collector.get_events_by_type(EventType.TOOL_RESULT)
        assert len(tool_result_events) == 0  # No result due to cancellation

        # Verify cleanup metadata
        frame_complete_events = event_collector.get_events_by_type(EventType.FRAME_COMPLETE)
        assert len(frame_complete_events) == 1
        assert frame_complete_events[0].data["result"] == "cancelled"
        assert frame_complete_events[0].data.get("tool_cancelled", False)
        assert frame_complete_events[0].data.get("cleanup_resources", False)
