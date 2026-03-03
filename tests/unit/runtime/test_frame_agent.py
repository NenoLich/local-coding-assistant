"""Unit tests for FrameAgent with new ExecutionFrame workflow."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from local_coding_assistant.agent.frame_agent import FrameAgent
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.execution_types import (
    ExecutionFrame,
    ExecutionResult,
    ExecutionStatus,
)
from local_coding_assistant.runtime.runtime_types import (
    PromptContext,
    RenderedPrompt,
    AgentProfile,
    ExecutionMode,
    ToolSpec,
)

from .conftest import MockConfigManager, session_state


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    service = AsyncMock()
    return service


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager."""
    manager = AsyncMock()
    manager.execute_async = AsyncMock()
    return manager


@pytest.fixture
def mock_context_manager():
    """Mock context manager."""
    manager = MagicMock()
    manager.build_context = MagicMock()
    manager.config_manager = MockConfigManager()
    return manager


@pytest.fixture
def frame_agent(mock_llm_service, mock_tool_manager, mock_context_manager):
    """Frame agent fixture."""
    return FrameAgent(
        llm_service=mock_llm_service,
        tool_manager=mock_tool_manager,
        context_manager=mock_context_manager,
        config_manager=MockConfigManager(),
        name="test_agent",
        max_iterations=3,
    )


@pytest.fixture
def sample_prompt_context():
    """Sample prompt context for testing."""
    return PromptContext(
        session_id="test_session",
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
        tool_call_mode="classic",
        user_input="Test input",
        agent_profile=AgentProfile.default(),
        tools=[ToolSpec(name="test_tool", description="Test tool")],
        history=[],
    )


@pytest.fixture
def sample_rendered_prompt():
    """Sample rendered prompt for testing."""
    return RenderedPrompt(
        system_messages=["System message"],
        user_messages=["User message"],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        history=[],
    )


class TestFrameAgent:
    """Test cases for FrameAgent."""

    @pytest.mark.asyncio
    async def test_run_single_iteration_success(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test successful single iteration run."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Create mock frame
        mock_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            final_answer="Final answer",
        )
        
        # Mock executor to return FRAME_COMPLETE event with frame
        async def mock_execute_func(frame):
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": mock_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result
            assert final_answer == "Final answer"
            assert len(frame_agent.history) == 1
            assert frame_agent.history[0] == mock_frame

            # Verify context was built correctly
            mock_context_manager.build_context.assert_called_once_with(
                session=session_state,
                user_input="Test input",
                tool_call_mode="classic",
                agent_mode=True,
                handler_context=None,
            )

    @pytest.mark.asyncio
    async def test_run_multiple_iterations_with_tool_calls(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test multiple iterations with tool calls."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Mock first frame with tool calls (no final answer)
        first_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        first_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            final_answer=None,  # No final answer, continue iteration
        )
        # Add tool action to simulate tool call
        from local_coding_assistant.runtime.execution_types import ActionKind

        first_frame.add_action(ActionKind.TOOL_CALL, name="test_tool", _input={})

        # Mock second frame with final answer
        second_frame = ExecutionFrame(
            session_id="test_session",
            iteration=2,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        second_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            final_answer="Final answer after tools",
        )

        # Setup executor to return different frames
        frame_sequence = [first_frame, second_frame]
        frame_index = 0
        
        async def mock_execute_func(frame):
            nonlocal frame_index
            current_frame = frame_sequence[frame_index]
            frame_index += 1
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": current_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result and history
            assert final_answer == "Final answer after tools"
            assert len(frame_agent.history) == 2
            assert frame_agent.history[0] == first_frame
            assert frame_agent.history[1] == second_frame

    @pytest.mark.asyncio
    async def test_run_max_iterations_reached(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test behavior when max iterations is reached."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Mock frames without final answers
        mock_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            final_answer=None,
        )
        
        # Mock executor to always return the same frame
        async def mock_execute_func(frame):
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": mock_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent (should stop after max_iterations)
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result and history
            assert final_answer is None  # No final answer
            assert len(frame_agent.history) == 3  # max_iterations

    @pytest.mark.asyncio
    async def test_run_execution_failure(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test behavior when frame execution fails."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Mock failed frame
        mock_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.FAILED,
            error_message="Execution failed",
        )
        
        # Mock executor to return failed frame
        async def mock_execute_func(frame):
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": mock_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result and history
            assert final_answer is None  # No final answer due to failure
            assert len(frame_agent.history) == 1
            assert frame_agent.history[0] == mock_frame

    @pytest.mark.asyncio
    async def test_run_partial_success_continues(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test that partial success stops execution if handler cannot handle it."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Mock partial success frame
        partial_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        partial_frame.result = ExecutionResult(
            status=ExecutionStatus.PARTIAL,
            final_answer=None,
        )

        # Mock final success frame (but it won't be reached)
        final_frame = ExecutionFrame(
            session_id="test_session",
            iteration=2,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        final_frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            final_answer="Final answer",
        )

        # Mock executor to return frames in sequence
        frame_sequence = [partial_frame, final_frame]
        frame_index = 0
        
        async def mock_execute_func(frame):
            nonlocal frame_index
            current_frame = frame_sequence[frame_index]
            frame_index += 1
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": current_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result and history
            assert final_answer is None  # Handler stops execution
            assert len(frame_agent.history) == 1

    @pytest.mark.asyncio
    async def test_update_session_with_assistant_message(
        self, frame_agent, sample_prompt_context, sample_rendered_prompt, session_state
    ):
        """Test session update with assistant message."""
        # Create frame with assistant response
        frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        frame.model_response_raw = "Assistant response"
        frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
        )

        # Update session
        frame_agent._update_session(session_state, frame)

        # Verify session was updated
        assert len(session_state.history) == 2
        assert session_state.history[0].role == "user"
        assert session_state.history[0].content == "User message"
        assert session_state.history[-1].role == "assistant"
        assert session_state.history[-1].content == "Assistant response"

    @pytest.mark.asyncio
    async def test_update_session_with_tool_calls(
        self, frame_agent, sample_prompt_context, sample_rendered_prompt, session_state
    ):
        """Test session update with tool calls."""
        # Create frame with tool calls
        from local_coding_assistant.runtime.execution_types import ActionKind

        frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        frame.model_response_raw = "Assistant response"

        # Add tool actions
        tool_action1 = frame.add_action(
            ActionKind.TOOL_CALL, name="tool1", _input={"param": "value1"}
        )
        frame.complete_action(tool_action1.id, output={"result": "Result 1"})

        tool_action2 = frame.add_action(
            ActionKind.TOOL_CALL, name="tool2", _input={"param": "value2"}
        )
        frame.complete_action(tool_action2.id, output={"result": "Result 2"})

        frame.result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
        )

        # Update session
        frame_agent._update_session(session_state, frame)

        # Verify session was updated with assistant message and tool messages
        assert len(session_state.history) == 4
        assert session_state.history[0].role == "user"
        assert session_state.history[0].content == "User message"
        assert session_state.history[1].role == "assistant"
        assert session_state.history[1].content == "Assistant response"
        assert session_state.history[2].role == "tool"
        assert session_state.history[3].role == "tool"


    def test_frame_agent_initialization(
        self, mock_llm_service, mock_tool_manager, mock_context_manager
    ):
        """Test FrameAgent initialization."""
        # Test with all parameters
        agent = FrameAgent(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
            context_manager=mock_context_manager,
            config_manager=MockConfigManager(),
            name="test_agent",
            max_iterations=5,
        )

        assert agent.name == "test_agent"
        assert agent.max_iterations == 5
        assert agent._context_manager == mock_context_manager
        # The statistics manager is passed to the executor, not stored directly
        assert len(agent.history) == 0
        assert agent.session_id.startswith("agent_test_agent_")

        # Test with default parameters
        agent_default = FrameAgent(
            llm_service=mock_llm_service,
            tool_manager=mock_tool_manager,
            context_manager=mock_context_manager,
            config_manager=MockConfigManager(),
        )

        assert agent_default.name == "frame_agent"
        assert agent_default.max_iterations == 5
        # The statistics manager is passed to the executor, not stored directly
        assert len(agent_default.history) == 0

    def test_get_frames(
        self, frame_agent, sample_prompt_context, sample_rendered_prompt
    ):
        """Test get_frames method."""
        # Initially empty
        assert frame_agent.get_frames() == []

        # Add frames to history
        frame1 = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        frame2 = ExecutionFrame(
            session_id="test_session",
            iteration=2,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )

        frame_agent.history.extend([frame1, frame2])

        # Test get_frames
        frames = frame_agent.get_frames()
        assert len(frames) == 2
        assert frames[0] == frame1
        assert frames[1] == frame2

    @pytest.mark.asyncio
    async def test_run_with_blocked_status_stops(
        self,
        frame_agent,
        mock_context_manager,
        sample_prompt_context,
        sample_rendered_prompt,
        session_state,
    ):
        """Test that BLOCKED status stops execution."""
        # Setup mocks
        mock_context_manager.build_context.return_value = sample_prompt_context
        
        # Mock blocked frame
        mock_frame = ExecutionFrame(
            session_id="test_session",
            iteration=1,
            prompt_context=sample_prompt_context,
            rendered_prompt=sample_rendered_prompt,
        )
        mock_frame.result = ExecutionResult(
            status=ExecutionStatus.BLOCKED,
            error_message="Rate limited",
        )
        
        # Mock executor to return blocked frame
        async def mock_execute_func(frame):
            yield ExecutionEvent(EventType.FRAME_COMPLETE, frame.session_id, data={"frame": mock_frame})

        with patch.object(frame_agent._composer, 'render', return_value=sample_rendered_prompt), \
             patch.object(frame_agent._executor, 'execute', mock_execute_func):

            # Run agent
            final_answer = None
            async for event in frame_agent.run("Test input", session_state):
                if event.type == EventType.TURN_COMPLETE:
                    final_answer = event.data["report"].final_answer

            # Verify result and history
            assert final_answer is None  # No final answer due to blocked status
            assert len(frame_agent.history) == 1
            assert frame_agent.history[0].result.status == ExecutionStatus.BLOCKED
