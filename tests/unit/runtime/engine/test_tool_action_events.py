"""Unit tests for event-sourced tool actions and crash-resume skip logic.

Covers the runtime engine's point-1 behavior:
- TOOL_CALL actions are opened/closed exclusively from gateway tool events
- On resume, tool calls with terminal action statuses are skipped (not re-executed)
- In-flight (non-terminal) actions leave no transcript trace while open
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.engine import RuntimeEngine
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ExecutionStatus,
    RuntimeFrame,
)
from local_coding_assistant.runtime.engine.subscribers.transcript_subscriber import (
    TranscriptSubscriber,
)
from local_coding_assistant.runtime.events.contracts import (
    ToolResultEvent,
    ToolStartEvent,
)
from local_coding_assistant.runtime.events.pipeline import EventPipeline
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import RenderedPrompt
from local_coding_assistant.tools_platform.contracts import (
    ToolCallRequest,
    ToolErrorCategory,
    ToolErrorCode,
    ToolEvent,
    ToolEventType,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class EventEmittingGateway:
    """Mock gateway that emits realistic tool events via the async callback."""

    def __init__(
        self,
        outcomes: dict[str, ToolExecutionStatus] | None = None,
        fail_message: str = "tool broke",
    ):
        self.outcomes = outcomes or {}
        self.fail_message = fail_message
        self.events: list[str] = []
        self.executed: list[ToolCallRequest] = []
        self._callback: Any = None

    def register_event_callback(self, callback) -> None:
        self._callback = callback

    async def _emit(self, event: ToolEvent) -> None:
        self.events.append(event.type.value)
        if self._callback:
            await self._callback(event)

    async def execute_many(self, requests, interrupt_event=None):
        self.executed.extend(requests)
        results = []
        for request in requests:
            status = self.outcomes.get(request.tool_id, ToolExecutionStatus.COMPLETED)
            error = None
            if status == ToolExecutionStatus.FAILED:
                error = ToolExecutionError(
                    code=ToolErrorCode.BACKEND_FAILURE,
                    category=ToolErrorCategory.INFRASTRUCTURE,
                    message=self.fail_message,
                )
            result = ToolExecutionResult(
                status=status,
                output=f"out:{request.tool_id}"
                if status == ToolExecutionStatus.COMPLETED
                else None,
                error=error,
                timing={"elapsed_seconds": 0.01},
            )
            await self._emit(
                ToolEvent(
                    type=ToolEventType.TOOL_CALL_START,
                    call_id=request.call_id,
                    session_id=request.session_id,
                    frame_id=request.frame_id,
                    trace_id=str(uuid.uuid4()),
                    data={
                        "tool_id": request.tool_id,
                        "arguments": request.raw_arguments,
                        "backend": "classic",
                    },
                )
            )
            if status == ToolExecutionStatus.COMPLETED:
                await self._emit(
                    ToolEvent(
                        type=ToolEventType.TOOL_CALL_COMPLETE,
                        call_id=request.call_id,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        trace_id=str(uuid.uuid4()),
                        data={
                            "tool_id": request.tool_id,
                            "status": "completed",
                            "output": result.output,
                            "timing": result.timing,
                            "backend": "classic",
                            "arguments": request.raw_arguments,
                        },
                    )
                )
            else:
                await self._emit(
                    ToolEvent(
                        type=ToolEventType.TOOL_CALL_ERROR,
                        call_id=request.call_id,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        trace_id=str(uuid.uuid4()),
                        data={
                            "tool_id": request.tool_id,
                            "status": "failed",
                            "error": error.model_dump(),
                            "timing": result.timing,
                            "backend": "classic",
                            "arguments": request.raw_arguments,
                        },
                    )
                )
            results.append(result)
        return results


class MockEventPipeline:
    def __init__(self):
        self.events: list[Any] = []

    async def emit(self, event) -> None:
        self.events.append(event)


@pytest.fixture
def mock_llm_service():
    service = MagicMock()
    service.stream = MagicMock()
    return service


@pytest.fixture
def event_pipeline():
    return MockEventPipeline()


@pytest.fixture
def runtime_turn():
    return RuntimeTurn(
        turn_id="turn_1",
        session_id="sess_1",
        trace_id="trace_1",
        turn_index=1,
        user_input="test",
        context_snapshot=ContextSnapshot(
            id="ctx_1",
            turn_id="turn_1",
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_1",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
    )


@pytest.fixture
def resolved():
    from typing import ClassVar

    class MockResolvedContext:
        metadata: ClassVar[dict[str, Any]] = {
            "model": "test_model",
            "temperature": 0.7,
            "streaming": True,
        }
        streaming: ClassVar[bool] = True

    return MockResolvedContext()


@pytest.fixture
def rendered():
    return RenderedPrompt(
        system_messages=["sys"],
        user_messages=["test"],
        tool_schemas=[],
        history=[],
    )


def make_tools_frame(frame_id: str = "frame_tools") -> RuntimeFrame:
    return RuntimeFrame(
        id=frame_id,
        session_id="sess_1",
        turn_id="turn_1",
        trace_id="trace_1",
        iteration=1,
        status=ExecutionStatus.RUNNING_TOOLS,
    )


def attach_llm_action(frame: RuntimeFrame, tool_calls: list[dict[str, Any]]) -> None:
    llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
    frame.complete_action(
        llm_action.id,
        "I'll use tools",
        metadata={
            "tool_calls": tool_calls,
            "model": "test_model",
            "total_tokens": 100,
        },
    )


def attach_completed_tool_action(
    frame: RuntimeFrame, tool_id: str, call_id: str, output: str
) -> None:
    action = frame.add_action(ActionKind.TOOL_CALL, name=tool_id)
    action.tool_execution_data.update(
        {
            "tool_id": tool_id,
            "call_id": call_id,
            "input": {"p": 1},
            "status": "completed",
            "output": output,
        }
    )
    action.finished_at = datetime.now(UTC)


class TestEventSourcedToolActions:
    """TOOL_CALL actions are opened/closed from gateway events only."""

    async def test_actions_opened_and_closed_by_events(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame()
        attach_llm_action(
            frame, [{"id": "c1", "name": "tool_a", "arguments": {"p": 1}}]
        )

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        tool_actions = [
            a for a in result_frame.actions if a.kind == ActionKind.TOOL_CALL
        ]
        assert len(tool_actions) == 1
        action = tool_actions[0]
        data = action.tool_execution_data
        assert data["call_id"] == "c1"
        assert data["status"] == "completed"
        assert data["output"] == "out:tool_a"
        assert data["input"] == {"p": 1}
        assert action.finished_at is not None

        # Gateway saw the call and emitted both lifecycle events
        assert len(gateway.executed) == 1
        assert gateway.events == ["tool_call_start", "tool_call_complete"]

        # Runtime events carry the frame's turn/trace (not empty strings)
        start = next(e for e in event_pipeline.events if isinstance(e, ToolStartEvent))
        result = next(
            e for e in event_pipeline.events if isinstance(e, ToolResultEvent)
        )
        assert start.turn_id == "turn_1"
        assert start.arguments == {"p": 1}
        assert result.turn_id == "turn_1"
        assert result.success is True
        assert result.output == "out:tool_a"

        assert result_frame.status == ExecutionStatus.COMPLETED

    async def test_failed_tool_closes_action_failed_and_partial(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = EventEmittingGateway(
            outcomes={"tool_a": ToolExecutionStatus.FAILED},
            fail_message="disk on fire",
        )
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame()
        attach_llm_action(
            frame, [{"id": "c1", "name": "tool_a", "arguments": {"p": 1}}]
        )

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        tool_actions = [
            a for a in result_frame.actions if a.kind == ActionKind.TOOL_CALL
        ]
        assert len(tool_actions) == 1
        assert tool_actions[0].tool_execution_data["status"] == "failed"
        assert tool_actions[0].tool_execution_data["error"]["message"] == "disk on fire"
        assert result_frame.status == ExecutionStatus.PARTIAL
        assert "disk on fire" in (result_frame.result.error_message or "")

    async def test_idempotent_open_on_repeated_start(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        """A second TOOL_CALL_START for the same call_id must not duplicate."""
        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_idem")

        start_event = ToolEvent(
            type=ToolEventType.TOOL_CALL_START,
            call_id="c1",
            session_id="sess_1",
            frame_id="frame_idem",
            trace_id="trace_1",
            data={"tool_id": "tool_a", "arguments": {"p": 1}},
        )
        engine._current_frame = frame
        await engine._on_tool_event(start_event)
        await engine._on_tool_event(start_event)
        engine._current_frame = None

        assert len([a for a in frame.actions if a.kind == ActionKind.TOOL_CALL]) == 1


class TestCrashResumeSkipsTerminalCalls:
    """Resume must not re-execute calls whose actions are terminal."""

    async def test_completed_call_skipped_on_resume(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_resume")
        attach_llm_action(
            frame,
            [
                {"id": "c1", "name": "tool_a", "arguments": {"p": 1}},
                {"id": "c2", "name": "tool_a", "arguments": {"p": 2}},
            ],
        )
        # c1 completed before the crash (e.g. from reconstructed events)
        attach_completed_tool_action(frame, "tool_a", "c1", "out:tool_a")

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        # Only c2 is dispatched; c1 reuses its recorded outcome
        assert len(gateway.executed) == 1
        assert gateway.executed[0].call_id == "c2"
        assert result_frame.status == ExecutionStatus.COMPLETED
        assert "failed" not in (result_frame.result.final_answer or "")

        # No duplicate action for c1; both actions terminal
        tool_actions = [
            a for a in result_frame.actions if a.kind == ActionKind.TOOL_CALL
        ]
        assert len(tool_actions) == 2
        statuses = {
            a.tool_execution_data["call_id"]: a.tool_execution_data["status"]
            for a in tool_actions
        }
        assert statuses == {"c1": "completed", "c2": "completed"}

    async def test_inflight_call_is_redispatched(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_inflight")
        attach_llm_action(
            frame, [{"id": "c1", "name": "tool_a", "arguments": {"p": 1}}]
        )
        # c1 started but never finished (crashed mid-execution): status running
        action = frame.add_action(ActionKind.TOOL_CALL, name="tool_a")
        action.tool_execution_data.update(
            {
                "tool_id": "tool_a",
                "call_id": "c1",
                "input": {"p": 1},
                "status": "running",
            }
        )

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        assert len(gateway.executed) == 1
        tool_actions = [
            a for a in result_frame.actions if a.kind == ActionKind.TOOL_CALL
        ]
        assert len(tool_actions) == 1
        assert tool_actions[0].tool_execution_data["status"] == "completed"
        assert result_frame.status == ExecutionStatus.COMPLETED

    async def test_failed_recorded_call_makes_frame_partial(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_failed_recorded")
        attach_llm_action(
            frame, [{"id": "c1", "name": "tool_a", "arguments": {"p": 1}}]
        )
        action = frame.add_action(ActionKind.TOOL_CALL, name="tool_a")
        action.tool_execution_data.update(
            {
                "tool_id": "tool_a",
                "call_id": "c1",
                "input": {"p": 1},
                "status": "failed",
                "error": {"message": "old failure"},
            }
        )
        action.finished_at = datetime.now(UTC)

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        assert gateway.executed == []  # nothing re-executed
        assert result_frame.status == ExecutionStatus.PARTIAL
        assert "failed" in (result_frame.result.error_message or "")


class TestTranscriptProjectionFromEngineEvents:
    """Engine tool events project into the session transcript eagerly.

    A cancelled tool call leaves an interruption message (cancellation is
    data); a call still in flight leaves nothing until it finishes.
    """

    @staticmethod
    def _session() -> SessionState:
        return SessionState(
            id="sess_1",
            context_profile=ContextProfile(
                runtime_context_config=RuntimeContextConfig(),
                tool_set_ref=ToolSetRef(),
                template_set_ref=TemplateSetRef(),
            ),
        )

    @staticmethod
    def _pipeline(session: SessionState) -> EventPipeline:
        pipeline = EventPipeline()
        pipeline.subscribe(TranscriptSubscriber(lambda sid: session))
        return pipeline

    async def test_interrupted_call_leaves_cancellation_message(
        self, mock_llm_service, runtime_turn, resolved, rendered
    ):
        session = self._session()
        gateway = InterruptingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=self._pipeline(session),
        )

        frame = make_tools_frame("frame_int_transcript")
        attach_llm_action(
            frame,
            [
                {"id": "c1", "name": "tool_a", "arguments": {"p": 1}},
                {"id": "c2", "name": "tool_a", "arguments": {"p": 2}},
            ],
        )

        await engine.execute_frame(frame, runtime_turn, resolved, rendered)

        tool_messages = [m for m in session.history if m.role == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0].content == "out:tool_a"
        assert tool_messages[1].content.startswith("Tool call interrupted")
        assert tool_messages[1].tool_call_id == "c2"
        assert [m.seq for m in session.history] == [1, 2]

    async def test_in_flight_call_leaves_no_message(
        self, mock_llm_service, runtime_turn, resolved, rendered
    ):
        """A call that started but never finished produces no transcript message."""
        session = self._session()
        pipeline = self._pipeline(session)

        gateway = EventEmittingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=pipeline,
        )

        frame = make_tools_frame("frame_inflight_transcript")
        attach_llm_action(
            frame, [{"id": "c1", "name": "tool_a", "arguments": {"p": 1}}]
        )

        # Simulate an in-flight call: only a START event was emitted
        start_event = ToolEvent(
            type=ToolEventType.TOOL_CALL_START,
            call_id="c1",
            session_id="sess_1",
            frame_id="frame_inflight_transcript",
            trace_id="trace_1",
            data={"tool_id": "tool_a", "arguments": {"p": 1}},
        )
        engine._current_frame = frame
        await engine._on_tool_event(start_event)
        engine._current_frame = None

        assert session.history == []


class InterruptingGateway:
    """Gateway mock that interrupts after the first executed call.

    Mirrors the real gateway's interrupt contract: calls that have not
    started are reported as cancelled via TOOL_CALL_CANCELLED events, and the
    turn's interrupt event is honored between calls.
    """

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.events: list[str] = []
        self.seen_interrupt_event: Any = None
        self._callback: Any = None

    def register_event_callback(self, callback) -> None:
        self._callback = callback

    async def _emit(self, event: ToolEvent) -> None:
        self.events.append(event.type.value)
        if self._callback:
            await self._callback(event)

    async def execute_many(
        self,
        requests: list[ToolCallRequest],
        interrupt_event: Any = None,
    ) -> list[ToolExecutionResult]:
        self.seen_interrupt_event = interrupt_event
        self.executed = [r.call_id for r in requests]
        results = []

        for request in requests:
            if interrupt_event is not None and interrupt_event.is_set():
                error = ToolExecutionError(
                    code=ToolErrorCode.INTERRUPTED,
                    category=ToolErrorCategory.CONCURRENCY,
                    message="Interrupted before execution started",
                )
                await self._emit(
                    ToolEvent(
                        type=ToolEventType.TOOL_CALL_CANCELLED,
                        call_id=request.call_id,
                        session_id=request.session_id,
                        frame_id=request.frame_id,
                        trace_id=str(uuid.uuid4()),
                        data={
                            "tool_id": request.tool_id,
                            "status": "cancelled",
                            "error": error.model_dump(),
                            "reason": "Interrupted before execution started",
                            "timing": {},
                            "backend": "classic",
                            "arguments": request.raw_arguments,
                        },
                    )
                )
                results.append(
                    ToolExecutionResult(
                        status=ToolExecutionStatus.CANCELLED, error=error
                    )
                )
                continue

            await self._emit(
                ToolEvent(
                    type=ToolEventType.TOOL_CALL_START,
                    call_id=request.call_id,
                    session_id=request.session_id,
                    frame_id=request.frame_id,
                    trace_id=str(uuid.uuid4()),
                    data={
                        "tool_id": request.tool_id,
                        "arguments": request.raw_arguments,
                        "backend": "classic",
                    },
                )
            )
            result = ToolExecutionResult(
                status=ToolExecutionStatus.COMPLETED,
                output=f"out:{request.tool_id}",
                timing={"elapsed_seconds": 0.01},
            )
            await self._emit(
                ToolEvent(
                    type=ToolEventType.TOOL_CALL_COMPLETE,
                    call_id=request.call_id,
                    session_id=request.session_id,
                    frame_id=request.frame_id,
                    trace_id=str(uuid.uuid4()),
                    data={
                        "tool_id": request.tool_id,
                        "status": "completed",
                        "output": result.output,
                        "timing": result.timing,
                        "backend": "classic",
                    },
                )
            )
            results.append(result)
            if interrupt_event is not None:
                interrupt_event.set()

        return results


class TestToolPhaseInterrupt:
    async def test_interrupt_leaves_frame_unfinished_and_marks_cancelled(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = InterruptingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_interrupt")
        attach_llm_action(
            frame,
            [
                {"id": "c1", "name": "tool_a", "arguments": {"p": 1}},
                {"id": "c2", "name": "tool_a", "arguments": {"p": 2}},
            ],
        )

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        # The interrupt event reached the gateway and was honored
        assert gateway.seen_interrupt_event is runtime_turn.interrupt_event
        assert gateway.executed == ["c1", "c2"]

        # The frame is left unfinished - the engine loop pauses the turn
        assert result_frame.status == ExecutionStatus.RUNNING_TOOLS
        assert result_frame.finished_at is None

        # First call completed; second was reported cancelled (terminal)
        tool_actions = {
            a.tool_execution_data["call_id"]: a.tool_execution_data
            for a in result_frame.actions
            if a.kind == ActionKind.TOOL_CALL
        }
        assert tool_actions["c1"]["status"] == "completed"
        assert tool_actions["c2"]["status"] == "cancelled"

    async def test_resume_after_interrupt_skips_terminal_calls(
        self, mock_llm_service, event_pipeline, runtime_turn, resolved, rendered
    ):
        gateway = InterruptingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=event_pipeline,
        )

        frame = make_tools_frame("frame_interrupt_resume")
        attach_llm_action(
            frame,
            [
                {"id": "c1", "name": "tool_a", "arguments": {"p": 1}},
                {"id": "c2", "name": "tool_a", "arguments": {"p": 2}},
            ],
        )

        # First pass: interrupted after the first call
        await engine.execute_frame(frame, runtime_turn, resolved, rendered)
        assert frame.status == ExecutionStatus.RUNNING_TOOLS

        # Resume: clear the interrupt (as frame_agent.resume_turn does) and
        # re-execute the same frame - terminal calls are skipped
        runtime_turn.interrupt_event.clear()
        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        assert result_frame.status == ExecutionStatus.PARTIAL
        assert "recorded as cancelled" in (result_frame.result.error_message or "")

        tool_actions = {
            a.tool_execution_data["call_id"]: a.tool_execution_data
            for a in result_frame.actions
            if a.kind == ActionKind.TOOL_CALL
        }
        assert tool_actions["c1"]["status"] == "completed"
        assert tool_actions["c2"]["status"] == "cancelled"


class SilentFailingGateway:
    """Gateway mock returning a failed result with NO tool events.

    Mirrors a prepare-phase failure (e.g. validation error): the call never
    reached execution, so no TOOL_CALL_START/ERROR event was emitted.
    """

    def __init__(self) -> None:
        self.executed: list[str] = []
        self._callback: Any = None

    def register_event_callback(self, callback) -> None:
        self._callback = callback

    async def execute_many(self, requests, interrupt_event=None):
        self.executed = [r.call_id for r in requests]
        return [
            ToolExecutionResult(
                status=ToolExecutionStatus.FAILED,
                error=ToolExecutionError(
                    code=ToolErrorCode.INVALID_ARGUMENTS,
                    category=ToolErrorCategory.VALIDATION,
                    message="invalid arguments: missing path",
                ),
            )
            for _ in requests
        ]


class TestTerminalEventBackstop:
    async def test_eventless_failure_still_lands_in_transcript(
        self, mock_llm_service, runtime_turn, resolved, rendered
    ):
        session = SessionState(
            id="sess_1",
            context_profile=ContextProfile(
                runtime_context_config=RuntimeContextConfig(),
                tool_set_ref=ToolSetRef(),
                template_set_ref=TemplateSetRef(),
            ),
        )
        pipeline = EventPipeline()
        pipeline.subscribe(TranscriptSubscriber(lambda sid: session))

        gateway = SilentFailingGateway()
        engine = RuntimeEngine(
            llm_service=mock_llm_service,
            tool_gateway=gateway,
            event_pipeline=pipeline,
        )

        frame = make_tools_frame("frame_silent_fail")
        attach_llm_action(frame, [{"id": "c1", "name": "tool_a", "arguments": {}}])

        result_frame = await engine.execute_frame(
            frame, runtime_turn, resolved, rendered
        )

        # Action closed as failed even though no event came from the gateway
        tool_actions = [
            a for a in result_frame.actions if a.kind == ActionKind.TOOL_CALL
        ]
        assert len(tool_actions) == 1
        assert tool_actions[0].tool_execution_data["status"] == "failed"

        # And the failure is visible to the model in the transcript
        assert result_frame.status == ExecutionStatus.PARTIAL
        tool_messages = [m for m in session.history if m.role == "tool"]
        assert len(tool_messages) == 1
        assert "missing path" in tool_messages[0].content
