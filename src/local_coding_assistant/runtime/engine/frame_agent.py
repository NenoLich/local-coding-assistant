"""FrameAgent - stateless continuation loop over IExecutionEngine.

Runs a bounded turn loop, executes frames through IExecutionEngine,
applies continuation decisions from ContinuationPolicy, and finalizes turns.

Transcript messages are never written here: they are projected from the
runtime event stream by TranscriptSubscriber (live session) and the store
adapter (durable), one message per event.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.schemas import ContinuationPolicyConfig

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context import ContextManager
from local_coding_assistant.runtime.engine import IExecutionEngine
from local_coding_assistant.runtime.engine.continuation import (
    FrameContinuation,
    ToolObservation,
)
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ExecutionStatus,
    FrameOutcomeDetail,
    RuntimeFrame,
)
from local_coding_assistant.runtime.engine.policy import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationPolicy,
    StopReason,
)
from local_coding_assistant.runtime.events.contracts import (
    ContinuationDecisionEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.pipeline import EventPipeline

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.session.turn import RuntimeTurn, TurnStatus
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.engine.frame_agent")


class FrameAgent:
    """Stateless FrameAgent running bounded turn loop over IExecutionEngine.

    Operates on RuntimeTurn state and explicit references.
    Does not retain cross-turn mutable state.
    """

    def __init__(
        self,
        engine: IExecutionEngine,
        context_manager: ContextManager,
        event_pipeline: EventPipeline,
    ) -> None:
        self._engine = engine
        self._ctx_mgr = context_manager
        self._event_pipeline = event_pipeline
        self._logger = get_logger("runtime.engine.frame_agent")

    async def run_turn(
        self,
        turn: RuntimeTurn,
        session: SessionState,
    ):
        """Execute a complete turn through the frame loop.

        Yields runtime events for streaming/observability.

        Args:
            turn: RuntimeTurn with user_input, limits, approvals, snapshot refs (MANDATORY)
            session: SessionState to update with new messages

        """
        self._logger.info(
            "Starting turn execution",
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            trace_id=turn.trace_id,
        )

        # Build continuation policy config from turn's limits_config
        policy_dict = turn.limits_config.continuation_policy
        policy_config = ContinuationPolicyConfig(**policy_dict)
        self._policy = ContinuationPolicy(config=policy_config)

        start_time = time.perf_counter()

        # Get snapshot from turn (MANDATORY - already created/resolved)
        snapshot = turn.context_snapshot
        if not snapshot:
            raise ValueError(f"Turn {turn.turn_id} has no context_snapshot")

        # Emit turn start event with context_snapshot (persistence happens via StoreSubscriber)
        config_data = {
            "context_config": turn.context_config.model_dump(),
            "limits_config": turn.limits_config.model_dump(),
        }
        turn_start_event = TurnStartEvent(
            trace_id=turn.trace_id,
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            user_input=turn.user_input,
            turn_index=turn.turn_index,
            config_snapshot_json=json.dumps(config_data),
            agent_mode="frame_agent",
            model_name=turn.context_config.llm_model if turn.context_config else None,
            context_snapshot_id=snapshot.id,
            context_snapshot_json=snapshot.model_dump_json(),
            context_profile_json=session.context_profile.model_dump_json()
            if session.context_profile
            else None,
        )
        await self._event_pipeline.emit(turn_start_event)

        final_answer: str | None = None
        turn_status = TurnStatus.RUNNING
        stop_reason: StopReason | None = None
        frame_count = 0
        completed_frame: RuntimeFrame | None = None

        # Initial continuation for first frame (empty directives)
        from local_coding_assistant.runtime.engine.continuation import FrameContinuation

        continuation = FrameContinuation()

        try:
            # Execute frame loop
            (
                turn_status,
                stop_reason,
                final_answer,
                frame_count,
                completed_frame,
            ) = await self._execute_frame_loop(
                turn=turn,
                session=session,
                snapshot=snapshot,
                continuation=continuation,
                frame_count=frame_count,
            )

        except Exception as e:
            self._logger.error(
                "Turn execution error",
                turn_id=turn.turn_id,
                error=str(e),
                exc_info=True,
            )

            # Check if the frame is in a resumable state (transient error)
            # The engine stores transient_error in frame.result.metadata
            if completed_frame is not None:
                frame_resumable = (
                    completed_frame.status
                    in (ExecutionStatus.RUNNING_LLM, ExecutionStatus.RUNNING_TOOLS)
                    and completed_frame.result
                    and completed_frame.result.metadata
                    and completed_frame.result.metadata.get("transient_error")
                )
            else:
                frame_resumable = False

            if frame_resumable:
                # Don't finalize turn - let continuation policy handle retry
                turn_status = TurnStatus.RUNNING
                stop_reason = None
                final_answer = (
                    completed_frame.result.error_message
                    if completed_frame is not None
                    and completed_frame.result is not None
                    else ""
                )
                # Re-raise to let the loop continue with policy decision
                raise
            else:
                turn_status = TurnStatus.FAILED
                stop_reason = StopReason.UNKNOWN
                final_answer = f"Turn execution error: {e}"

        finally:
            # Only finalize turn if it's actually complete (not paused)
            if turn_status != TurnStatus.RUNNING:
                await self._finalize_turn(
                    turn=turn,
                    turn_status=turn_status,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                    frame_count=frame_count,
                    start_time=start_time,
                )

    async def resume_turn(
        self,
        turn: RuntimeTurn,
        session: SessionState,
        frame_to_resume: RuntimeFrame | None = None,
    ) -> None:
        """Resume a turn from its current frame state.

        Continues execution from the last frame's status.
        Used when resuming an interrupted turn (session status='running').

        Args:
            turn: RuntimeTurn loaded from store (has context_snapshot, limits, approvals, frame_ids)
            session: SessionState with history loaded up to turn's message_boundary
            frame_to_resume: Optional specific frame to resume. If None, creates new frame.
        """
        self._logger.info(
            "Resuming turn execution",
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            trace_id=turn.trace_id,
            frame_id=frame_to_resume.id if frame_to_resume else None,
        )

        # Clear interrupt event so turn doesn't immediately re-pause
        turn.interrupt_event.clear()

        # Build continuation policy config from turn's limits_config
        policy_dict = turn.limits_config.continuation_policy
        policy_config = ContinuationPolicyConfig(**policy_dict)
        self._policy = ContinuationPolicy(config=policy_config)

        start_time = time.perf_counter()

        # Get snapshot from turn (MANDATORY - already created/resolved)
        snapshot = turn.context_snapshot
        if not snapshot:
            raise ValueError(f"Turn {turn.turn_id} has no context_snapshot")

        # NOTE: Do NOT emit TurnStartEvent - turn already exists in store
        # Do NOT add user message - already in history from resume_session

        final_answer: str | None = None
        turn_status = TurnStatus.RUNNING
        stop_reason: StopReason | None = None
        frame_count = len(turn.frame_ids)
        completed_frame: RuntimeFrame | None = None

        # Initial continuation for resume (empty directives - will be built from last frame)
        from local_coding_assistant.runtime.engine.continuation import FrameContinuation

        continuation = FrameContinuation()

        try:
            # Execute frame loop with optional frame to resume
            (
                turn_status,
                stop_reason,
                final_answer,
                frame_count,
                completed_frame,
            ) = await self._execute_frame_loop(
                turn=turn,
                session=session,
                snapshot=snapshot,
                continuation=continuation,
                frame_count=frame_count,
                frame_to_resume=frame_to_resume,
            )

        except Exception as e:
            self._logger.error(
                "Turn execution error",
                turn_id=turn.turn_id,
                error=str(e),
                exc_info=True,
            )

            # Check if the frame is in a resumable state (transient error)
            # The engine stores transient_error in frame.result.metadata
            if completed_frame is not None:
                frame_resumable = (
                    completed_frame.status
                    in (ExecutionStatus.RUNNING_LLM, ExecutionStatus.RUNNING_TOOLS)
                    and completed_frame.result
                    and completed_frame.result.metadata
                    and completed_frame.result.metadata.get("transient_error")
                )
            else:
                frame_resumable = False

            if frame_resumable:
                # Don't finalize turn - let continuation policy handle retry
                turn_status = TurnStatus.RUNNING
                stop_reason = None
                final_answer = (
                    completed_frame.result.error_message
                    if completed_frame is not None
                    and completed_frame.result is not None
                    else ""
                )
                # Re-raise to let the loop continue with policy decision
                raise
            else:
                turn_status = TurnStatus.FAILED
                stop_reason = StopReason.UNKNOWN
                final_answer = f"Turn execution error: {e}"

        finally:
            # Only finalize turn if it's actually complete (not paused)
            if turn_status != TurnStatus.RUNNING:
                await self._finalize_turn(
                    turn=turn,
                    turn_status=turn_status,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                    frame_count=frame_count,
                    start_time=start_time,
                )

    def _check_turn_limits(
        self,
        turn: RuntimeTurn,
        frame_count: int,
    ) -> tuple[bool, StopReason | None]:
        """Check turn limits and frame cap.

        Returns:
            Tuple of (should_continue, stop_reason).
            If should_continue is False, stop_reason contains the reason.
        """
        # Check turn limits
        limit_result = turn.check_limits()
        if not limit_result.within_limits:
            self._logger.warning(
                "Turn limit reached",
                turn_id=turn.turn_id,
                reason=limit_result.reason,
                detail=limit_result.detail,
            )
            return False, self._limit_reason_to_stop_reason(limit_result.reason)

        # Check frame cap
        max_frames = self._policy.config.max_frames_per_turn
        if frame_count >= max_frames:
            self._logger.warning(
                "Max frames per turn reached",
                turn_id=turn.turn_id,
                frame_count=frame_count,
                max_frames=max_frames,
            )
            return False, StopReason.FRAME_LIMIT_REACHED

        return True, None

    async def _execute_single_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        session: SessionState,
        snapshot: Any,
        continuation: FrameContinuation,
    ) -> tuple[RuntimeFrame, FrameOutcomeDetail]:
        """Execute a single frame through the engine.

        Returns:
            Tuple of (completed_frame, frame_outcome).
        """
        # Freeze the transcript view and sequence boundary for this frame.
        history_view = list(session.history)
        frame.history_through_seq = session.history_sequence
        resolved, rendered = await self._ctx_mgr.resolve_context(
            session=session,
            snapshot=snapshot,
            continuation=continuation,
            config_snapshot=turn.context_config,
            history=history_view,
            history_through_seq=frame.history_through_seq,
        )

        frame.prompt_hash = self._compute_prompt_hash(rendered)

        # Backoff before re-executing a frame retried after a
        # transient provider error (Retry-After / exponential).
        await self._await_retry_backoff(continuation, turn, frame)

        # Execute frame through engine
        completed_frame = await self._engine.execute_frame(
            frame, turn, resolved, rendered
        )

        # Preserve the prompt identity and transcript boundary for replay.
        completed_frame.prompt_hash = frame.prompt_hash
        completed_frame.history_through_seq = frame.history_through_seq

        # Derive semantic outcome for policy
        frame_outcome = FrameOutcomeDetail.from_frame(completed_frame)

        return completed_frame, frame_outcome

    async def _emit_continuation_decision(
        self,
        decision: ContinuationDecision,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
    ) -> None:
        """Emit continuation decision event and log."""
        decision_event = ContinuationDecisionEvent(
            trace_id=turn.trace_id,
            turn_id=turn.turn_id,
            frame_id=frame.id,
            session_id=turn.session_id,
            decision=decision.action.value,
            reason=decision.reason.value,
            adjusted_options={
                "max_tokens_adjustment": decision.max_tokens_adjustment,
                "temperature_adjustment": decision.temperature_adjustment,
                "compact_instruction": decision.compact_instruction,
            },
        )
        await self._event_pipeline.emit(decision_event)

        self._logger.info(
            "Continuation decision",
            turn_id=turn.turn_id,
            frame_id=frame.id,
            action=decision.action.value,
            reason=decision.reason.value,
        )

    def _apply_continuation_decision(
        self,
        decision: ContinuationDecision,
        frame_outcome: FrameOutcomeDetail,
        turn: RuntimeTurn,
    ) -> tuple[TurnStatus, StopReason | None, str | None]:
        """Apply continuation decision and determine turn state.

        Returns:
            Tuple of (turn_status, stop_reason, final_answer).
        """
        if decision.action == ContinuationAction.STOP:
            turn_status = self._map_reason_to_status(decision.reason)
            stop_reason = decision.reason
            final_answer = frame_outcome.final_content
            return turn_status, stop_reason, final_answer

        elif decision.action in (
            ContinuationAction.PAUSE_FOR_APPROVAL,
            ContinuationAction.PAUSE_FOR_INTERRUPT,
            ContinuationAction.PAUSE_FOR_DEADLINE,
            ContinuationAction.PAUSE_FOR_DENIED,
        ):
            # Frame paused in engine; yield control
            # Do NOT finalize turn - keep it "running" with pause metadata
            turn.status = TurnStatus.RUNNING
            turn.paused_at = __import__("datetime").datetime.now(
                __import__("datetime").UTC
            )
            turn.pause_reason = decision.reason.value
            # DO NOT emit TurnCompleteEvent - turn is not complete
            # DO NOT set finished_at
            turn_status = TurnStatus.RUNNING  # Keep as running for resume
            stop_reason = decision.reason
            final_answer = frame_outcome.final_content
            return turn_status, stop_reason, final_answer

        elif decision.action in (
            ContinuationAction.CONTINUE,
            ContinuationAction.COMPACT_AND_CONTINUE,
            ContinuationAction.RETRY_GENERATION,
        ):
            # Continue to next frame
            return TurnStatus.RUNNING, None, None

        else:
            self._logger.warning(
                "Unknown continuation action",
                action=decision.action.value,
            )
            return TurnStatus.FAILED, StopReason.UNKNOWN, None

    async def _execute_frame_loop(
        self,
        turn: RuntimeTurn,
        session: SessionState,
        snapshot: Any,
        continuation: FrameContinuation,
        frame_count: int,
        frame_to_resume: RuntimeFrame | None = None,
    ) -> tuple[TurnStatus, StopReason | None, str | None, int, RuntimeFrame | None]:
        """Execute the main frame loop.

        Returns:
            Tuple of (turn_status, stop_reason, final_answer, final_frame_count, completed_frame).
        """
        completed_frame: RuntimeFrame | None = None
        is_resumed = False

        while True:
            # Check turn limits before starting frame
            should_continue, stop_reason = self._check_turn_limits(turn, frame_count)
            if not should_continue:
                return (
                    TurnStatus.FAILED,
                    stop_reason,
                    None,
                    frame_count,
                    completed_frame,
                )

            # Find or create frame
            if frame_to_resume and not is_resumed:
                frame = frame_to_resume
                is_resumed = True
            else:
                frame = self._create_new_frame(turn, frame_count + 1)
                turn.add_frame_id(frame.id)
                frame_count += 1

            self._logger.info(
                "Executing frame",
                frame_id=frame.id,
                turn_id=turn.turn_id,
                iteration=frame.iteration,
            )

            # Execute frame
            completed_frame, frame_outcome = await self._execute_single_frame(
                frame, turn, session, snapshot, continuation
            )

            # Get continuation decision from policy
            decision = self._policy.decide(frame_outcome, turn)

            # Emit continuation decision event
            await self._emit_continuation_decision(decision, frame, turn)

            # Build continuation for NEXT frame based on decision
            continuation = self._build_continuation(decision, completed_frame, turn)

            # Apply decision
            turn_status, stop_reason, final_answer = self._apply_continuation_decision(
                decision, frame_outcome, turn
            )

            # Check if we should break the loop
            if turn_status != TurnStatus.RUNNING or stop_reason is not None:
                return (
                    turn_status,
                    stop_reason,
                    final_answer,
                    frame_count,
                    completed_frame,
                )

    async def _finalize_turn(
        self,
        turn: RuntimeTurn,
        turn_status: TurnStatus,
        stop_reason: StopReason | None,
        final_answer: str | None,
        frame_count: int,
        start_time: float,
    ) -> None:
        """Finalize turn completion.

        Only called when turn is actually complete (not paused).
        """
        # Finalize turn
        turn.finished_at = turn.finished_at or __import__("datetime").datetime.now(
            __import__("datetime").UTC
        )
        turn.status = turn_status
        turn.final_answer = final_answer

        # Build report
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        input_tokens_used = turn.limits.token_budget.input_tokens_used
        output_tokens_used = turn.limits.token_budget.output_tokens_used

        # Emit turn complete event
        turn_complete_event = TurnCompleteEvent(
            trace_id=turn.trace_id,
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            final_answer=final_answer,
            status=turn_status.value,
            total_input_tokens=input_tokens_used,
            total_output_tokens=output_tokens_used,
            total_frames=frame_count,
            finish_reason=stop_reason.value if stop_reason else None,
        )
        await self._event_pipeline.emit(turn_complete_event)

        self._logger.info(
            "Turn completed",
            turn_id=turn.turn_id,
            status=turn_status.value,
            stop_reason=stop_reason.value if stop_reason else None,
            frames=frame_count,
            total_input_tokens=input_tokens_used,
            total_output_tokens=output_tokens_used,
            latency_ms=total_latency_ms,
        )

        # Clear turn caches in context manager
        self._ctx_mgr.clear_turn_cache(turn.turn_id)

    async def _await_retry_backoff(
        self,
        continuation: FrameContinuation | None,
        turn: RuntimeTurn,
        frame: RuntimeFrame,
    ) -> None:
        """Wait out the retry backoff for a transient-error retry.

        The wait is clamped to the turn deadline so a long Retry-After can
        never push execution past it. Skipped entirely when no backoff is
        requested or the deadline is already (nearly) exhausted.
        """
        if continuation is None or not continuation.delay_before_retry:
            return

        delay = float(continuation.delay_before_retry)
        if delay <= 0:
            return

        remaining = turn.limits.deadline_tracker.seconds_remaining
        if remaining <= 0:
            return
        delay = min(delay, remaining)

        self._logger.info(
            "Backing off before generation retry",
            turn_id=turn.turn_id,
            frame_id=frame.id,
            delay_seconds=delay,
            generation_retry_count=continuation.generation_retry_count,
        )
        await asyncio.sleep(delay)

    def _create_new_frame(self, turn: RuntimeTurn, iteration: int) -> RuntimeFrame:
        """Create a new frame for the turn."""
        frame = RuntimeFrame(
            id=f"frame_{uuid.uuid4()}",
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            trace_id=turn.trace_id,
            iteration=iteration,
            context_snapshot_id=turn.context_snapshot.id
            if turn.context_snapshot
            else None,
            status=ExecutionStatus.RUNNING_LLM,
        )
        return frame

    def _build_continuation(
        self,
        decision: ContinuationDecision,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
    ) -> FrameContinuation:
        """Build continuation directives for next frame from policy decision.

        Directives only - transcript messages are projected from runtime
        events, not from frames or continuations.
        """
        from local_coding_assistant.runtime.engine.continuation import (
            FrameContinuation,
        )

        cont = FrameContinuation()

        if decision.action == ContinuationAction.CONTINUE:
            # Extract tool observations from completed frame
            observations = self._extract_tool_observations(frame)
            if observations:
                cont = cont.add_tool_observations(observations)

        elif decision.action == ContinuationAction.COMPACT_AND_CONTINUE:
            cont = cont.with_compaction(decision.compact_instruction or "")

        elif decision.action == ContinuationAction.RETRY_GENERATION:
            cont = cont.with_retry_params(
                max_tokens_adj=decision.max_tokens_adjustment,
                temp_adj=decision.temperature_adjustment,
                retry_count=decision.retry_count,
                delay_before_retry=(decision.metadata or {}).get("delay_before_retry"),
            )

        elif decision.action == ContinuationAction.PAUSE_FOR_APPROVAL:
            # Get approval info from turn
            for approval in turn.pending_approvals.values():
                cont = cont.with_approval(approval.token, approval.tool_name)
                break

        return cont

    def _extract_tool_observations(self, frame: RuntimeFrame) -> list[ToolObservation]:
        """Extract tool observations from frame actions."""
        observations = []

        # Find tool call actions
        for action in frame.actions:
            if action.kind == ActionKind.TOOL_CALL and action.tool_execution_data:
                data = action.tool_execution_data
                success = data.get("status") == "completed"

                # Get call_id from metadata or action
                call_id = (
                    data.get("call_id") or action.metadata.get("call_id") or action.id
                )

                obs = ToolObservation(
                    call_id=call_id,
                    tool_name=data.get("tool_id", action.name or "unknown"),
                    arguments=data.get("input", {}),
                    success=success,
                    output=data.get("output"),
                    error=data.get("error", {}).get("message")
                    if data.get("error")
                    else None,
                    side_effects_may_have_occurred=not success,
                    safe_to_retry=success,  # Only safe to retry if succeeded
                )
                observations.append(obs)

        return observations

    def _limit_reason_to_stop_reason(self, reason: Any) -> StopReason:
        """Convert LimitViolationReason to StopReason."""
        from local_coding_assistant.runtime.engine.config import LimitViolationReason

        reason_map = {
            LimitViolationReason.TOKEN_BUDGET_EXCEEDED: StopReason.TOKEN_BUDGET_DEPLETED,
            LimitViolationReason.DEADLINE_EXCEEDED: StopReason.DEADLINE_EXCEEDED,
            LimitViolationReason.LOOP_DETECTED: StopReason.LOOP_DETECTED,
            LimitViolationReason.FRAME_LIMIT_REACHED: StopReason.FRAME_LIMIT_REACHED,
            LimitViolationReason.GENERATION_RETRY_EXHAUSTED: StopReason.GENERATION_RETRY_EXHAUSTED,
            LimitViolationReason.COMPACTION_RETRY_EXHAUSTED: StopReason.TRUNCATION_RETRY_EXHAUSTED,
        }
        return reason_map.get(reason, StopReason.UNKNOWN)

    def _map_reason_to_status(self, reason: StopReason) -> TurnStatus:
        """Map stop reason to turn status."""
        success_reasons = {
            StopReason.FINAL_ANSWER,
            StopReason.TURN_COMPLETE,
        }
        partial_reasons = {
            StopReason.OUTPUT_TRUNCATED,
            StopReason.OUTPUT_TRUNCATED_WITH_TOOLS,
            StopReason.TRUNCATION_RETRY_EXHAUSTED,
        }
        blocked_reasons = {
            StopReason.CONTENT_BLOCKED,
            StopReason.APPROVAL_PENDING,
            StopReason.APPROVAL_DENIED,
            StopReason.APPROVAL_EXPIRED,
        }

        if reason in success_reasons:
            return TurnStatus.COMPLETED
        if reason in partial_reasons:
            return TurnStatus.PARTIAL
        if reason in blocked_reasons:
            return TurnStatus.BLOCKED
        return TurnStatus.FAILED

    def _compute_prompt_hash(self, rendered: Any) -> str:
        """Compute hash of rendered prompt for frame reconstruction."""
        import hashlib

        messages = rendered.to_full_prompt()
        prompt_text = "\n".join(msg.get("content") or "" for msg in messages)
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
