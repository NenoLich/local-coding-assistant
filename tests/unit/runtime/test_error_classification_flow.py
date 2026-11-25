"""Tests for structured error classification in the runtime frame loop.

Covers the path that used to flatten every LLM failure into FAILED:
transient provider errors now keep the frame resumable (RUNNING_LLM),
map to FrameOutcome.PROVIDER_TRANSIENT_ERROR, and drive RETRY_GENERATION
with a bounded backoff; permanent errors still fail the frame.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.core.exceptions import LLMInfrastructureError
from local_coding_assistant.providers.error_classification import classify_llm_error
from local_coding_assistant.providers.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.engine import (
    ContinuationAction,
    ContinuationPolicy,
    RuntimeEngine,
    StopReason,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ExecutionStatus,
    FrameOutcome,
    FrameOutcomeDetail,
    RuntimeFrame,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.runtime.types import RenderedPrompt
from local_coding_assistant.tools_platform.exceptions import ToolPlatformError


@pytest.fixture
def mock_llm_service():
    service = MagicMock()
    service.stream = AsyncMock()
    return service


@pytest.fixture
def mock_tool_gateway():
    gateway = MagicMock()
    gateway.execute_many = AsyncMock(return_value=[])
    return gateway


@pytest.fixture
def mock_event_pipeline():
    pipeline = MagicMock()
    pipeline.emit = AsyncMock()
    return pipeline


@pytest.fixture
def runtime_engine(mock_llm_service, mock_tool_gateway, mock_event_pipeline):
    return RuntimeEngine(
        llm_service=mock_llm_service,
        tool_gateway=mock_tool_gateway,
        event_pipeline=mock_event_pipeline,
    )


@pytest.fixture
def sample_frame():
    return RuntimeFrame(
        id="frame_123",
        session_id="session_123",
        turn_id="turn_123",
        trace_id="trace_123",
        iteration=1,
        context_snapshot_id="ctx_123",
        status=ExecutionStatus.RUNNING_LLM,
    )


@pytest.fixture
def sample_resolved():
    return SimpleNamespace(
        metadata={"model": "test_model", "temperature": 0.7, "streaming": True},
        streaming=True,
    )


@pytest.fixture
def sample_rendered():
    return RenderedPrompt(
        messages=[{"role": "user", "content": "test input"}],
        system_messages=["You are a helpful assistant"],
        user_messages=["test input"],
        history=[],
        tool_schemas=[],
    )


@pytest.fixture
def sample_runtime_turn():
    turn = RuntimeTurn(
        turn_id="turn_123",
        session_id="session_123",
        trace_id="trace_123",
        turn_index=1,
        user_input="test input",
        context_snapshot=ContextSnapshot(
            id="ctx_123",
            turn_id="turn_123",
            tool_set_ref=ToolSetRef(),
            template_set_ref=TemplateSetRef(),
            repo_snapshot_ref=None,
            memory_snapshot_ref=None,
            message_boundary=None,
            content_hash="hash123",
            created_at_ms=1000,
        ),
        context_snapshot_id="ctx_123",
        context_config=RuntimeContextConfig(),
        limits_config=RuntimeLimitsConfig(),
    )
    return turn


def _llm_action(frame: RuntimeFrame):
    return next(a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE)


class TestLLMPhaseErrorClassification:
    """_execute_llm_phase must classify before flattening to FAILED."""

    @pytest.mark.asyncio
    async def test_transient_server_error_keeps_frame_resumable(
        self,
        runtime_engine,
        sample_frame,
        sample_runtime_turn,
        sample_resolved,
        sample_rendered,
    ) -> None:
        async def failing_stream(*args, **kwargs):
            raise ProviderConnectionError(
                "Server error (HTTP 503)", provider="p", status_code=503
            )
            yield  # pragma: no cover - makes this an async generator

        runtime_engine._llm_service.stream = failing_stream

        await runtime_engine.execute_frame(
            sample_frame, sample_runtime_turn, sample_resolved, sample_rendered
        )

        # Frame stays in a resumable state instead of FAILED
        assert sample_frame.status == ExecutionStatus.RUNNING_LLM
        llm_action = _llm_action(sample_frame)
        assert llm_action.metadata["transient_error"]
        assert llm_action.metadata["error_kind"] == "server"
        assert llm_action.metadata["status_code"] == 503
        assert sample_frame.result.metadata["transient_error"] is True

    @pytest.mark.asyncio
    async def test_transient_error_without_llm_action_records_metadata(
        self,
        runtime_engine,
        sample_frame,
        sample_runtime_turn,
        sample_resolved,
        sample_rendered,
    ) -> None:
        """Wrapped failover errors are still classified via the cause chain."""

        async def failing_stream(*args, **kwargs):
            try:
                raise ProviderRateLimitError(
                    "Rate limit exceeded (HTTP 429)",
                    provider="p",
                    retry_after=30,
                )
            except ProviderRateLimitError as e:
                raise LLMInfrastructureError(
                    "LLM streaming failed after exhausting failover attempts",
                    classification=classify_llm_error(e),
                ) from e
            yield  # pragma: no cover - makes this an async generator

        runtime_engine._llm_service.stream = failing_stream

        await runtime_engine.execute_frame(
            sample_frame, sample_runtime_turn, sample_resolved, sample_rendered
        )

        assert sample_frame.status == ExecutionStatus.RUNNING_LLM
        llm_action = _llm_action(sample_frame)
        assert llm_action.metadata["retry_after"] == 30.0
        assert llm_action.metadata["error_kind"] == "rate_limit"

    @pytest.mark.asyncio
    async def test_permanent_auth_error_fails_frame(
        self,
        runtime_engine,
        sample_frame,
        sample_runtime_turn,
        sample_resolved,
        sample_rendered,
    ) -> None:
        async def failing_stream(*args, **kwargs):
            raise ProviderAuthError("Invalid API key (HTTP 401)", provider="p")
            yield  # pragma: no cover - makes this an async generator

        runtime_engine._llm_service.stream = failing_stream

        await runtime_engine.execute_frame(
            sample_frame, sample_runtime_turn, sample_resolved, sample_rendered
        )

        assert sample_frame.status == ExecutionStatus.FAILED
        llm_action = _llm_action(sample_frame)
        assert "transient_error" not in llm_action.metadata
        assert "Invalid API key" in (sample_frame.result.error_message or "")

    @pytest.mark.asyncio
    async def test_str_classifier_delegates_to_structured_classifier(
        self, runtime_engine
    ) -> None:
        assert runtime_engine._is_transient_error(
            ProviderConnectionError("down", provider="p", status_code=500)
        )
        assert not runtime_engine._is_transient_error(
            ProviderAuthError("bad key", provider="p")
        )


class TestFrameOutcomeForProviderErrors:
    def test_transient_frame_maps_to_provider_transient_outcome(self) -> None:
        frame = RuntimeFrame(
            id="frame_1",
            session_id="s",
            turn_id="t",
            trace_id="tr",
            status=ExecutionStatus.RUNNING_LLM,
        )
        action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")
        action.metadata["transient_error"] = "Server error (HTTP 503)"
        action.metadata["error_kind"] = "server"
        action.metadata["status_code"] = 503
        action.metadata["retry_after"] = None
        from local_coding_assistant.runtime.engine.frame import ExecutionResult

        frame.result = ExecutionResult(
            error_message="Server error (HTTP 503)",
            metadata={"transient_error": True},
        )

        detail = FrameOutcomeDetail.from_frame(frame)

        assert detail.outcome is FrameOutcome.PROVIDER_TRANSIENT_ERROR
        assert detail.retryable is True
        assert detail.metadata["error_kind"] == "server"
        assert detail.metadata["status_code"] == 503
        assert detail.metadata["error_message"] == "Server error (HTTP 503)"

    def test_failed_frame_maps_to_non_transient_outcome(self) -> None:
        from local_coding_assistant.runtime.engine.frame import ExecutionResult

        frame = RuntimeFrame(
            id="frame_1",
            session_id="s",
            turn_id="t",
            trace_id="tr",
            status=ExecutionStatus.FAILED,
        )
        frame.result = ExecutionResult(error_message="Invalid API key (HTTP 401)")

        detail = FrameOutcomeDetail.from_frame(frame)

        assert detail.outcome is FrameOutcome.PROVIDER_NON_TRANSIENT_ERROR
        assert detail.metadata["error_message"] == "Invalid API key (HTTP 401)"


class TestContinuationPolicyForProviderErrors:
    def _policy(self, **overrides) -> ContinuationPolicy:
        from local_coding_assistant.config.schemas import ContinuationPolicyConfig

        config = ContinuationPolicyConfig(**overrides)
        return ContinuationPolicy(config=config)

    def _outcome(self, **metadata) -> FrameOutcomeDetail:
        return FrameOutcomeDetail(
            outcome=FrameOutcome.PROVIDER_TRANSIENT_ERROR,
            metadata=metadata,
        )

    def test_retry_generation_honors_retry_after(self, sample_runtime_turn) -> None:
        policy = self._policy(max_generation_retries=2)
        decision = policy.decide(
            self._outcome(error_kind="rate_limit", status_code=429, retry_after=30),
            sample_runtime_turn,
        )

        assert decision.action is ContinuationAction.RETRY_GENERATION
        assert decision.reason is StopReason.PROVIDER_TRANSIENT_ERROR
        assert decision.retry_count == 1
        # Dropped: max_tokens_adjustment was meaningless for network errors
        assert decision.max_tokens_adjustment is None
        assert decision.metadata["delay_before_retry"] == 30.0

    def test_exponential_backoff_without_retry_after(self, sample_runtime_turn) -> None:
        policy = self._policy(
            max_generation_retries=3,
            respect_retry_after=False,
            transient_retry_base_delay=2.0,
            transient_retry_max_delay=5.0,
        )
        sample_runtime_turn.metadata["generation_retry_count"] = 1
        decision = policy.decide(
            self._outcome(error_kind="server", status_code=503),
            sample_runtime_turn,
        )

        # 2.0 * 2**1 = 4.0, clamped at 5.0
        assert decision.metadata["delay_before_retry"] == 4.0

    def test_retry_after_clamped_to_max_delay(self, sample_runtime_turn) -> None:
        policy = self._policy(max_generation_retries=2, transient_retry_max_delay=10.0)
        decision = policy.decide(self._outcome(retry_after=600), sample_runtime_turn)

        assert decision.metadata["delay_before_retry"] == 10.0

    def test_exhausted_retries_stop(self, sample_runtime_turn) -> None:
        policy = self._policy(max_generation_retries=2)
        sample_runtime_turn.metadata["generation_retry_count"] = 2
        decision = policy.decide(
            self._outcome(error_kind="server", status_code=503),
            sample_runtime_turn,
        )

        assert decision.action is ContinuationAction.STOP
        assert decision.reason is StopReason.PROVIDER_TRANSIENT_ERROR
        assert decision.metadata["max_retries"] == 2

    def test_non_transient_stops_with_error_message(self, sample_runtime_turn) -> None:
        policy = self._policy()
        outcome = FrameOutcomeDetail(
            outcome=FrameOutcome.PROVIDER_NON_TRANSIENT_ERROR,
            metadata={"error_message": "Invalid API key (HTTP 401)"},
        )
        decision = policy.decide(outcome, sample_runtime_turn)

        assert decision.action is ContinuationAction.STOP
        assert decision.reason is StopReason.PROVIDER_NON_TRANSIENT_ERROR
        assert decision.metadata["error"] == "Invalid API key (HTTP 401)"


class TestContinuationPolicyFinalResponsePrecedence:
    """A final response completes the turn and must not be overridden by
    interrupts or limits, which are only meaningful for in-progress turns."""

    def _policy(self, **overrides) -> ContinuationPolicy:
        from local_coding_assistant.config.schemas import ContinuationPolicyConfig

        config = ContinuationPolicyConfig(**overrides)
        return ContinuationPolicy(config=config)

    def _final_response(self) -> FrameOutcomeDetail:
        return FrameOutcomeDetail(
            outcome=FrameOutcome.FINAL_RESPONSE,
            finish_reason="stop",
            final_content="The answer is 42.",
        )

    def test_final_response_overrides_interrupt(self, sample_runtime_turn) -> None:
        policy = self._policy()
        sample_runtime_turn.interrupt_event.set()

        decision = policy.decide(self._final_response(), sample_runtime_turn)

        assert decision.action is ContinuationAction.STOP
        assert decision.reason is StopReason.FINAL_ANSWER
        assert decision.metadata["content_length"] == len("The answer is 42.")

    def test_final_response_overrides_frame_cap(self, sample_runtime_turn) -> None:
        policy = self._policy(max_frames_per_turn=1)
        # Simulate the current frame already being tracked in the turn.
        sample_runtime_turn.add_frame_id("frame_1")

        decision = policy.decide(self._final_response(), sample_runtime_turn)

        assert decision.action is ContinuationAction.STOP
        assert decision.reason is StopReason.FINAL_ANSWER
        assert decision.metadata["content_length"] == len("The answer is 42.")
