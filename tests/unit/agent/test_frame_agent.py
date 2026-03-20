"""
Tests for FrameAgent event emission and session updates.
"""

import pytest
from unittest.mock import MagicMock, patch

from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.execution_types import (
    ExecutionStatus,
    ExecutionResult,
)
from local_coding_assistant.runtime.runtime_types import (
    PromptContext,
    RenderedPrompt,
    ExecutionMode,
)
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.reporting import RunReport
from tests.unit.conftest import collect_events


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    return MagicMock()


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager."""
    return MagicMock()


@pytest.fixture
def mock_context_manager():
    """Mock context manager."""
    manager = MagicMock()
    # Mock build_context to return a PromptContext
    prompt_context = PromptContext(
        session_id="test_session",
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
        tool_call_mode="classic",
        user_input="Test input",
        tools=[],
        history=[],
    )
    manager.build_context.return_value = prompt_context
    return manager


@pytest.fixture
def mock_config_manager():
    """Mock config manager."""
    manager = MagicMock()
    manager.global_config.runtime.tool_call_mode = "classic"
    return manager


@pytest.fixture
def mock_composer():
    """Mock prompt composer."""
    composer = MagicMock()
    rendered_prompt = RenderedPrompt(
        user_messages=["Test input"],
        system_messages=[],
        tool_schemas=[],
    )
    composer.render.return_value = rendered_prompt
    return composer


@pytest.fixture
def mock_session():
    """Mock session state."""
    session = MagicMock(spec=SessionState)
    session.id = "test_session"
    session.metadata = {}
    return session


@pytest.fixture
def frame_agent(
    mock_llm_service,
    mock_tool_manager,
    mock_context_manager,
    mock_config_manager,
    mock_composer,
):
    """Create FrameAgent with mocked dependencies."""
    with patch(
        "local_coding_assistant.agent.frame_agent.PromptComposer",
        return_value=mock_composer,
    ):
        from local_coding_assistant.agent.frame_agent import FrameAgent

        agent = FrameAgent(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
            context_manager=mock_context_manager,
            config_manager=mock_config_manager,
        )
        return agent


@pytest.fixture
def mock_executor_events():
    """Predefined events that executor should emit."""
    frame_mock = MagicMock()
    frame_mock.result = ExecutionResult(
        status=ExecutionStatus.SUCCESS, final_answer="Final answer"
    )
    frame_mock.model_dump.return_value = {
        "id": "frame_1",
        "result": {"status": "success", "final_answer": "Final answer"},
    }
    frame_mock.get_llm_metrics.return_value = MagicMock(
        model="test-model", total_tokens=10
    )

    return [
        {
            "type": EventType.FRAME_START,
            "session_id": "test_session",
            "frame_id": "frame_1",
        },
        {
            "type": EventType.LLM_START,
            "session_id": "test_session",
            "frame_id": "frame_1",
        },
        {
            "type": EventType.LLM_CHUNK,
            "session_id": "test_session",
            "frame_id": "frame_1",
            "data": {"content": "Hello"},
        },
        {
            "type": EventType.LLM_COMPLETE,
            "session_id": "test_session",
            "frame_id": "frame_1",
            "data": {"result": MagicMock(content="Hello", finish_reason="stop")},
        },
        {
            "type": EventType.FRAME_COMPLETE,
            "session_id": "test_session",
            "frame_id": "frame_1",
            "data": {"frame": frame_mock},
        },
    ]


class TestFrameAgent:
    """Test FrameAgent event emission."""

    @pytest.mark.asyncio
    async def test_run_emits_turn_start_and_complete(
        self, frame_agent, mock_session, mock_executor_events
    ):
        """Test that run emits TURN_START and TURN_COMPLETE events."""

        # Mock executor to yield events
        async def mock_execute(frame):
            for event_data in mock_executor_events:
                yield ExecutionEvent(**event_data)

        frame_agent._executor.execute = mock_execute

        events = await collect_events(frame_agent.run("Test input", mock_session))

        assert len(events) >= 2  # At least TURN_START and TURN_COMPLETE
        assert events[0].type == EventType.TURN_START
        assert events[0].session_id == "test_session"

        # Find TURN_COMPLETE
        turn_complete_events = [e for e in events if e.type == EventType.TURN_COMPLETE]
        assert len(turn_complete_events) == 1
        assert turn_complete_events[0].session_id == "test_session"
        assert "report" in turn_complete_events[0].data
        report = turn_complete_events[0].data["report"]
        assert isinstance(report, RunReport)

    @pytest.mark.asyncio
    async def test_run_forwards_executor_events(
        self, frame_agent, mock_session, mock_executor_events
    ):
        """Test that run forwards events from executor."""

        async def mock_execute(frame):
            for event_data in mock_executor_events:
                yield ExecutionEvent(**event_data)

        frame_agent._executor.execute = mock_execute

        events = await collect_events(frame_agent.run("Test input", mock_session))

        # Check that executor events are forwarded (between TURN_START and TURN_COMPLETE)
        executor_event_types = [
            e.type for e in events[1:-1]
        ]  # Skip TURN_START and TURN_COMPLETE
        expected_types = [
            EventType.FRAME_START,
            EventType.LLM_START,
            EventType.LLM_CHUNK,
            EventType.LLM_COMPLETE,
            EventType.FRAME_COMPLETE,
        ]
        assert executor_event_types == expected_types

    @pytest.mark.asyncio
    async def test_run_updates_session(
        self, frame_agent, mock_session, mock_executor_events
    ):
        """Test that run updates session with frame results."""
        # Create a mock frame with actions
        mock_frame = MagicMock()
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS, final_answer="Answer"
        )
        mock_frame.rendered_prompt.get_user_prompt.return_value = "User prompt"
        mock_frame.model_response_raw = "Assistant response"
        mock_frame.actions = []
        mock_frame.model_dump.return_value = {
            "id": "frame_1",
            "result": {"status": "success"},
        }
        mock_frame.get_llm_metrics.return_value = None

        async def mock_execute(frame):
            for event_data in mock_executor_events[:-1]:  # All except FRAME_COMPLETE
                yield ExecutionEvent(**event_data)
            # Yield FRAME_COMPLETE with the mock frame
            yield ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id="test_session",
                frame_id="frame_1",
                data={"frame": mock_frame},
            )

        frame_agent._executor.execute = mock_execute

        await collect_events(frame_agent.run("Test input", mock_session))

        # Check that session methods were called
        mock_session.add_user_message.assert_called_with("User prompt")
        mock_session.add_assistant_message.assert_called_with(
            content="Assistant response"
        )

    @pytest.mark.asyncio
    async def test_run_handles_max_iterations(self, frame_agent, mock_session):
        """Test that run stops after max iterations."""
        frame_agent.max_iterations = 2

        # Mock executor to always return partial result
        async def mock_execute(frame):
            mock_frame = MagicMock(
                result=ExecutionResult(status=ExecutionStatus.SUCCESS)
            )
            mock_frame.get_llm_metrics.return_value = None
            mock_frame.model_dump.return_value = {
                "id": "frame_1",
                "result": {"status": "success"},
            }
            yield ExecutionEvent(EventType.FRAME_START, "test_session", "frame_1")
            yield ExecutionEvent(
                EventType.FRAME_COMPLETE,
                "test_session",
                "frame_1",
                {"frame": mock_frame},
            )

        frame_agent._executor.execute = mock_execute

        events = await collect_events(frame_agent.run("Test input", mock_session))

        # Should have 2 iterations + TURN_START + TURN_COMPLETE
        frame_start_events = [e for e in events if e.type == EventType.FRAME_START]
        assert len(frame_start_events) == 2

    @pytest.mark.asyncio
    async def test_run_stops_on_final_answer(
        self, frame_agent, mock_session, mock_executor_events
    ):
        """Test that run stops when final answer is reached."""
        mock_frame = MagicMock()
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS, final_answer="Final answer"
        )
        mock_frame.rendered_prompt.get_user_prompt.return_value = "User prompt"
        mock_frame.model_response_raw = "Assistant response"
        mock_frame.actions = []
        mock_frame.get_llm_metrics.return_value = None
        mock_frame.model_dump.return_value = {
            "id": "frame_1",
            "result": {"status": "success"},
        }

        async def mock_execute(frame):
            for event_data in mock_executor_events[:-1]:
                yield ExecutionEvent(**event_data)
            yield ExecutionEvent(
                type=EventType.FRAME_COMPLETE,
                session_id="test_session",
                frame_id="frame_1",
                data={"frame": mock_frame},
            )

        frame_agent._executor.execute = mock_execute

        events = await collect_events(frame_agent.run("Test input", mock_session))

        # Should have only 1 frame + TURN events
        frame_start_events = [e for e in events if e.type == EventType.FRAME_START]
        assert len(frame_start_events) == 1

        # Check TURN_COMPLETE has the final answer
        turn_complete_events = [e for e in events if e.type == EventType.TURN_COMPLETE]
        report = turn_complete_events[0].data["report"]
        assert report.final_answer == "Final answer"
