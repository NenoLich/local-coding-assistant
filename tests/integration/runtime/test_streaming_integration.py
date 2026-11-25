"""
Integration tests for runtime streaming events.

Exercises the real ``FrameAgent`` + ``RuntimeEngine`` + ``EventPipeline`` wiring
with a mocked LLM service and tool gateway, asserting on the typed event stream:

    TurnStart -> FrameStart -> LLMStart -> LLMChunk* -> LLMComplete
    -> ToolStart -> ToolResult -> FrameComplete -> TurnComplete

The previous version of this file imported removed modules
(``agent.frame_agent``, ``runtime.events.legacy``, ``runtime.executor``), asserted
against a flat ``ExecutionEvent`` model that no longer exists, and mocked every
component, so it tested no real runtime code. This rewrite drives the live engine
through a real event pipeline and asserts on the typed event contracts.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm.models import LLMStreamChunk, LLMToolCall
from local_coding_assistant.providers.exceptions import ProviderAuthError
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.engine import RuntimeEngine
from local_coding_assistant.runtime.engine.frame_agent import FrameAgent
from local_coding_assistant.runtime.events.contracts import (
    EventType,
    RuntimeEvent,
)
from local_coding_assistant.runtime.events.pipeline import EventPipeline
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import (
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
)
from local_coding_assistant.tools_platform.contracts import (
    ToolErrorCategory,
    ToolErrorCode,
    ToolEvent,
    ToolEventType,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class EventCollector:
    """Subscriber that records every runtime event it receives."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    async def on_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    async def close(self) -> None:
        return None

    def types(self) -> list[EventType]:
        return [e.type for e in self.events]

    def events_of(self, event_type: EventType) -> list[RuntimeEvent]:
        return [e for e in self.events if e.type == event_type]


class EventEmittingGateway:
    """Mock tool gateway that emits ToolEvents through a registered callback."""

    def __init__(self, outcomes: dict[str, ToolExecutionStatus] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.executed: list = []
        self._callback: object | None = None

    def register_event_callback(self, callback) -> None:
        self._callback = callback

    async def _emit(self, event: ToolEvent) -> None:
        if self._callback:
            await self._callback(event)

    async def execute_many(self, requests, interrupt_event=None):
        self.executed.extend(requests)
        results: list[ToolExecutionResult] = []
        for request in requests:
            status = self.outcomes.get(request.tool_id, ToolExecutionStatus.COMPLETED)
            error: ToolExecutionError | None = None
            output = f"result:{request.tool_id}"
            if status != ToolExecutionStatus.COMPLETED:
                output = None
                error = ToolExecutionError(
                    code=ToolErrorCode.BACKEND_FAILURE,
                    category=ToolErrorCategory.INFRASTRUCTURE,
                    message="tool failed",
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
                        "execution_mode": "classic",
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
                            "output": output,
                            "timing": {"elapsed_seconds": 0.01},
                            "execution_mode": "classic",
                            "backend": "classic",
                        },
                    )
                )
            results.append(
                ToolExecutionResult(
                    status=status,
                    output=output,
                    error=error,
                    timing={"elapsed_seconds": 0.01},
                )
            )
        return results

    @property
    def workspace_manager(self) -> SimpleNamespace:
        return SimpleNamespace(workspace_id="ws_test")


@pytest.fixture
def event_pipeline() -> EventPipeline:
    return EventPipeline()


@pytest.fixture
def event_collector(event_pipeline: EventPipeline) -> EventCollector:
    collector = EventCollector()
    event_pipeline.subscribe(collector)
    return collector


@pytest.fixture
def prompt_context() -> PromptContext:
    return PromptContext(
        session_id="session_1",
        execution_mode=ExecutionMode.CLASSIC_TOOLS,
        tool_call_mode="classic",
    )


@pytest.fixture
def rendered_prompt() -> RenderedPrompt:
    return RenderedPrompt(
        system_messages=["You are a helpful assistant."],
        user_messages=["What is 2 + 2?"],
        tool_schemas=[],
        history=[],
        metadata={"model": "test-model", "streaming": True},
    )


@pytest.fixture
def context_manager(
    prompt_context: PromptContext, rendered_prompt: RenderedPrompt
) -> MagicMock:
    cm = MagicMock()
    cm.resolve_context = AsyncMock(return_value=(prompt_context, rendered_prompt))
    cm.clear_turn_cache = MagicMock()
    return cm


@pytest.fixture
def session() -> SessionState:
    profile = ContextProfile(
        runtime_context_config=RuntimeContextConfig(),
        tool_set_ref=ToolSetRef(tool_ids=[], tool_names=[]),
        template_set_ref=TemplateSetRef(template_ids=[], template_paths=[]),
    )
    return SessionState(context_profile=profile)


@pytest.fixture
def make_turn(session: SessionState):
    def _make(user_input: str = "What is 2 + 2?") -> RuntimeTurn:
        return RuntimeTurn(
            turn_id="turn_1",
            session_id=session.id,
            trace_id="trace_1",
            turn_index=0,
            user_input=user_input,
            context_snapshot=ContextSnapshot(
                id="snap_1",
                turn_id="turn_1",
                tool_set_ref=ToolSetRef(),
                template_set_ref=TemplateSetRef(),
                repo_snapshot_ref=None,
                memory_snapshot_ref=None,
                message_boundary=None,
                content_hash="hash123",
                created_at_ms=1000,
            ),
            context_snapshot_id="snap_1",
            context_config=RuntimeContextConfig(llm_model="test-model", streaming=True),
            limits_config=RuntimeLimitsConfig(),
        )

    return _make


@pytest.fixture
def build_frame_agent(event_pipeline: EventPipeline, context_manager: MagicMock):
    def _build(llm_service, tool_gateway) -> FrameAgent:
        engine = RuntimeEngine(
            llm_service=llm_service,
            tool_gateway=tool_gateway,
            event_pipeline=event_pipeline,
        )
        return FrameAgent(
            engine=engine,
            context_manager=context_manager,
            event_pipeline=event_pipeline,
        )

    return _build


@pytest.fixture
def llm_service():
    def _make(call_responses: list[list[LLMStreamChunk]]):
        call_count = 0

        async def stream(llm_task, *, policy=None, options=None):
            nonlocal call_count
            index = min(call_count, len(call_responses) - 1)
            call_count += 1
            for chunk in call_responses[index]:
                yield chunk

        service = MagicMock()
        service.stream = stream
        return service

    return _make


@pytest.fixture
def tool_gateway():
    def _make(outcomes=None):
        return EventEmittingGateway(outcomes or {})

    return _make


class TestContentOnlyTurn:
    @pytest.mark.asyncio
    async def test_emits_full_happy_path_sequence(
        self,
        build_frame_agent,
        llm_service,
        tool_gateway,
        make_turn,
        session,
        event_collector,
    ):
        service = llm_service(
            [
                [
                    LLMStreamChunk(
                        content="The answer is ",
                        provider="test-provider",
                        model="test-model",
                        is_final=False,
                    ),
                    LLMStreamChunk(
                        content="4",
                        provider="test-provider",
                        model="test-model",
                        is_final=True,
                        finish_reason="stop",
                    ),
                ]
            ]
        )
        gateway = tool_gateway()
        agent = build_frame_agent(service, gateway)
        turn = make_turn()

        await agent.run_turn(turn, session)

        types = event_collector.types()
        assert types[0] == EventType.TURN_START
        assert types[-1] == EventType.TURN_COMPLETE

        turn_starts = event_collector.events_of(EventType.TURN_START)
        turn_completes = event_collector.events_of(EventType.TURN_COMPLETE)
        frame_starts = event_collector.events_of(EventType.FRAME_START)
        frame_completes = event_collector.events_of(EventType.FRAME_COMPLETE)
        llm_starts = event_collector.events_of(EventType.LLM_START)
        llm_chunks = event_collector.events_of(EventType.LLM_CHUNK)
        llm_completes = event_collector.events_of(EventType.LLM_COMPLETE)

        assert len(turn_starts) == 1
        assert len(turn_completes) == 1
        assert len(frame_starts) == 1
        assert len(frame_completes) == 1
        assert len(llm_starts) == 1
        assert len(llm_chunks) == 2
        assert len(llm_completes) == 1

        # Ordering within the single frame.
        chunk_positions = [i for i, t in enumerate(types) if t == EventType.LLM_CHUNK]
        assert types.index(EventType.TURN_START) < types.index(EventType.FRAME_START)
        assert types.index(EventType.FRAME_START) < types.index(EventType.LLM_START)
        assert types.index(EventType.LLM_START) < min(chunk_positions)
        assert max(chunk_positions) < types.index(EventType.LLM_COMPLETE)
        assert types.index(EventType.LLM_COMPLETE) < types.index(
            EventType.FRAME_COMPLETE
        )
        assert types.index(EventType.FRAME_COMPLETE) < types.index(
            EventType.TURN_COMPLETE
        )

        # Content accumulates across chunks.
        full_content = "".join(c.content for c in llm_chunks if c.content)
        assert full_content == "The answer is 4"
        assert llm_completes[0].content == "The answer is 4"

        # Turn start carries the user input and session.
        assert turn_starts[0].session_id == session.id
        assert turn_starts[0].user_input == "What is 2 + 2?"

        # Turn complete carries the final answer.
        assert turn_completes[0].session_id == session.id
        assert turn_completes[0].final_answer == "The answer is 4"
        assert turn_completes[0].status == "completed"

    @pytest.mark.asyncio
    async def test_chunk_content_and_final_markers(
        self,
        build_frame_agent,
        llm_service,
        tool_gateway,
        make_turn,
        session,
        event_collector,
    ):
        service = llm_service(
            [
                [
                    LLMStreamChunk(
                        content="Hello",
                        provider="test-provider",
                        model="test-model",
                        is_final=False,
                    ),
                    LLMStreamChunk(
                        content=" world",
                        provider="test-provider",
                        model="test-model",
                        is_final=False,
                    ),
                    LLMStreamChunk(
                        content="!",
                        provider="test-provider",
                        model="test-model",
                        is_final=True,
                        finish_reason="stop",
                    ),
                ]
            ]
        )
        gateway = tool_gateway()
        agent = build_frame_agent(service, gateway)
        turn = make_turn()

        await agent.run_turn(turn, session)

        chunks = event_collector.events_of(EventType.LLM_CHUNK)
        assert [c.content for c in chunks] == ["Hello", " world", "!"]
        assert event_collector.events_of(EventType.LLM_COMPLETE)[0].content == (
            "Hello world!"
        )


class TestToolTurn:
    @pytest.mark.asyncio
    async def test_tool_call_and_result_stream(
        self,
        build_frame_agent,
        llm_service,
        tool_gateway,
        make_turn,
        session,
        event_collector,
    ):
        # Frame 1: LLM asks for the calculator; frame 2: LLM gives the final answer.
        service = llm_service(
            [
                # Frame 1: LLM asks for the calculator.
                [
                    LLMStreamChunk(
                        content="Let me compute ",
                        provider="test-provider",
                        model="test-model",
                        is_final=False,
                    ),
                    LLMStreamChunk(
                        content="",
                        provider="test-provider",
                        model="test-model",
                        is_final=True,
                        finish_reason="stop",
                        tool_calls=[
                            LLMToolCall(
                                name="calculator",
                                arguments={"expression": "2 + 2"},
                                id="call_1",
                            )
                        ],
                    ),
                ],
                # Frame 2: LLM gives the final answer after the tool result.
                [
                    LLMStreamChunk(
                        content="The answer is 4",
                        provider="test-provider",
                        model="test-model",
                        is_final=True,
                        finish_reason="stop",
                    ),
                ],
            ]
        )
        gateway = tool_gateway()
        agent = build_frame_agent(service, gateway)
        turn = make_turn()

        await agent.run_turn(turn, session)

        types = event_collector.types()
        assert types[0] == EventType.TURN_START
        assert types[-1] == EventType.TURN_COMPLETE

        assert len(event_collector.events_of(EventType.TURN_START)) == 1
        assert len(event_collector.events_of(EventType.TURN_COMPLETE)) == 1
        assert len(event_collector.events_of(EventType.FRAME_START)) == 2
        assert len(event_collector.events_of(EventType.FRAME_COMPLETE)) == 2
        assert len(event_collector.events_of(EventType.LLM_START)) == 2
        assert len(event_collector.events_of(EventType.LLM_COMPLETE)) == 2
        assert len(event_collector.events_of(EventType.TOOL_START)) == 1
        assert len(event_collector.events_of(EventType.TOOL_RESULT)) == 1

        # Tool execution is wired through the gateway's event callback.
        assert len(gateway.executed) == 1
        assert gateway.executed[0].tool_id == "calculator"

        tool_start = event_collector.events_of(EventType.TOOL_START)[0]
        tool_result = event_collector.events_of(EventType.TOOL_RESULT)[0]
        assert tool_start.tool_name == "calculator"
        assert tool_start.arguments == {"expression": "2 + 2"}
        assert tool_result.tool_name == "calculator"
        assert tool_result.success is True
        assert tool_result.output == "result:calculator"

        # Tool start follows the first LLM complete; a second frame starts after
        # the tool result, and the turn completes with the final answer.
        assert types.index(EventType.TOOL_START) > types.index(EventType.LLM_COMPLETE)
        llm_start_positions = [
            i for i, t in enumerate(types) if t == EventType.LLM_START
        ]
        assert max(llm_start_positions) > types.index(EventType.TOOL_RESULT)
        assert types.index(EventType.FRAME_COMPLETE) < types.index(
            EventType.TURN_COMPLETE
        )

        turn_complete = event_collector.events_of(EventType.TURN_COMPLETE)[0]
        assert turn_complete.final_answer == "The answer is 4"
        assert turn_complete.status == "completed"


class TestErrorTurn:
    @pytest.mark.asyncio
    async def test_permanent_provider_error_fails_turn(
        self,
        build_frame_agent,
        llm_service,
        tool_gateway,
        make_turn,
        session,
        event_collector,
    ):
        async def failing_stream(llm_task, *, policy=None, options=None):
            if False:
                yield  # make this an async generator
            raise ProviderAuthError("Invalid API key")

        service = MagicMock()
        service.stream = failing_stream
        gateway = tool_gateway()
        agent = build_frame_agent(service, gateway)
        turn = make_turn()

        await agent.run_turn(turn, session)

        types = event_collector.types()
        assert EventType.TURN_START in types
        assert EventType.FRAME_START in types
        assert EventType.LLM_START in types
        assert EventType.LLM_COMPLETE not in types
        assert types[-1] == EventType.TURN_COMPLETE

        frame_completes = event_collector.events_of(EventType.FRAME_COMPLETE)
        assert len(frame_completes) == 1
        assert frame_completes[0].status == "failed"
        assert "Invalid API key" in frame_completes[0].error_message

        turn_completes = event_collector.events_of(EventType.TURN_COMPLETE)
        assert len(turn_completes) == 1
        assert turn_completes[0].status == "failed"
