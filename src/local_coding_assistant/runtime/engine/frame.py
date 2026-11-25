from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from local_coding_assistant.agent.llm import LLMToolCall
from local_coding_assistant.tools_platform.contracts import (
    TERMINAL_TOOL_STATUSES,
    FileChange,
    ToolErrorCategory,
    ToolExecutionResult,
)
from local_coding_assistant.tools_platform.errors import ToolErrorCode

if TYPE_CHECKING:
    from local_coding_assistant.storage.contracts import StoredFrame


class ActionKind(str, Enum):
    LLM_MESSAGE = "llm_message"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


class ActionRecord(BaseModel):
    """A single action taken during an execution frame."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: ActionKind
    name: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    # LLM-specific
    llm_metrics: LLMMetrics | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    # Tool-specific - stores ToolExecutionResult data
    tool_execution_data: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMMetrics(BaseModel):
    """Metrics for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    model: str | None = None


class ExecutionStatus(str, Enum):
    # Engine state machine states (transient)
    RUNNING_LLM = "running_llm"
    RUNNING_TOOLS = "running_tools"

    # Terminal states
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    BLOCKED = "blocked"


class FrameOutcome(str, Enum):
    """Semantic outcome of a frame execution for continuation policy.

    This is separate from ExecutionStatus (engine lifecycle) and provides
    the semantic meaning needed for continuation decisions.
    """

    # Successful completion
    FINAL_RESPONSE = (
        "final_response"  # Model completed with user-facing answer, no tools
    )
    TOOL_ROUND_COMPLETED = (
        "tool_round_completed"  # Tools executed successfully, need next model turn
    )

    # Output truncation
    OUTPUT_TRUNCATED = "output_truncated"  # finish_reason="length" with ordinary text
    OUTPUT_TRUNCATED_WITH_TOOL_CALLS = (
        "output_truncated_with_tool_calls"  # Truncated mid-tool-call
    )

    # Context overflow
    CONTEXT_OVERFLOW = "context_overflow"  # Prompt too large for model input window

    # Tool failures
    TOOL_FAILURE_RECOVERABLE = (
        "tool_failure_recoverable"  # Tool failed but model can correct/retry
    )
    TOOL_FAILURE_NON_RECOVERABLE = (
        "tool_failure_non_recoverable"  # Tool failed, not safe to retry
    )
    INCOMPLETE_TOOL_CALL = (
        "incomplete_tool_call"  # Malformed/incomplete tool call from model
    )

    # Provider/request failures
    PROVIDER_TRANSIENT_ERROR = (
        "provider_transient_error"  # Network/timeout, safe to retry
    )
    PROVIDER_NON_TRANSIENT_ERROR = (
        "provider_non_transient_error"  # Auth/permission/invalid request
    )
    CONTENT_BLOCKED = "content_blocked"  # Safety filter blocked response

    # Approval
    APPROVAL_PENDING = "approval_pending"  # Tool requires approval, paused
    APPROVAL_DENIED = "approval_denied"  # Approval explicitly denied
    APPROVAL_EXPIRED = "approval_expired"  # Approval timed out

    # Limits
    FRAME_LIMIT_REACHED = "frame_limit_reached"  # Max frames per turn exceeded
    TOKEN_BUDGET_DEPLETED = (
        "token_budget_depleted"  # Turn token budget exceeded  # noqa: S105
    )
    DEADLINE_EXCEEDED = "deadline_exceeded"  # Turn deadline exceeded
    LOOP_DETECTED = "loop_detected"  # LoopGuard detected repetition
    GENERATION_RETRY_EXHAUSTED = (
        "generation_retry_exhausted"  # Max generation retries reached
    )

    # Unknown/error
    UNKNOWN = "unknown"  # Unknown or malformed state


class ToolCallInfo(BaseModel):
    """Information about a tool call from the frame."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    success: bool
    output: Any | None = None
    error_message: str | None = None
    error_category: ToolErrorCategory | None = None
    safe_to_retry: bool = False
    side_effects_may_have_occurred: bool = False


class FrameOutcomeDetail(BaseModel):
    """Detailed semantic outcome of a frame for continuation policy.

    This is what the ContinuationPolicy consumes to make decisions.
    It is derived from the frame's execution status, LLM finish_reason,
    tool results, and error classifications.
    """

    outcome: FrameOutcome
    finish_reason: str | None = (
        None  # Provider finish_reason (stop, length, tool_calls, content_filter, etc.)
    )
    final_content: str | None = None  # Final answer or partial content
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    any_tool_executed: bool = False
    side_effects_may_have_occurred: bool = False
    safe_to_retry: bool = False
    error_category: ToolErrorCategory | None = None
    retryable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_frame(cls, frame: RuntimeFrame) -> FrameOutcomeDetail:
        """Derive semantic outcome from a completed frame."""
        if not frame.result:
            return cls(
                outcome=FrameOutcome.UNKNOWN,
                final_content=None,
                metadata={"reason": "no_result"},
            )

        # Get LLM action and metrics
        llm_action = next(
            (a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE), None
        )
        finish_reason = frame.result.finish_reason
        final_content = frame.result.final_answer

        # Extract tool calls from LLM action
        tool_calls_info = []
        any_tool_executed = False
        side_effects_may_have_occurred = False
        safe_to_retry = False
        error_category = None

        if llm_action and llm_action.tool_calls:
            for tc in llm_action.tool_calls:
                # Find corresponding tool execution
                tool_action = next(
                    (
                        a
                        for a in frame.actions
                        if a.kind == ActionKind.TOOL_CALL
                        and a.tool_execution_data.get("call_id") == tc.id
                    ),
                    None,
                )
                if tool_action and tool_action.tool_execution_data:
                    data = tool_action.tool_execution_data
                    # Only terminal actions have a real outcome; in-flight
                    # ones (e.g. paused for approval) stay unknown
                    if data.get("status") not in TERMINAL_TOOL_STATUSES:
                        continue
                    success = data.get("status") == "completed"
                    any_tool_executed = True
                    if not success:
                        side_effects_may_have_occurred = True
                    tool_calls_info.append(
                        ToolCallInfo(
                            call_id=tc.id or "unknown",
                            name=tc.name,
                            arguments=tc.arguments,
                            success=success,
                            output=data.get("output"),
                            error_message=data.get("error", {}).get("message")
                            if data.get("error")
                            else None,
                        )
                    )

        # Determine semantic outcome based on status and details
        # Check for transient error in LLM action metadata
        transient_error = False
        error_kind = None
        status_code = None
        retry_after = None
        if llm_action and llm_action.metadata:
            transient_error = llm_action.metadata.get("transient_error") is not None
            error_kind = llm_action.metadata.get("error_kind")
            status_code = llm_action.metadata.get("status_code")
            retry_after = llm_action.metadata.get("retry_after")

        outcome = cls._determine_outcome(
            frame.status,
            finish_reason,
            tool_calls_info,
            any_tool_executed,
            frame.result.error_message,
            transient_error,
        )

        return cls(
            outcome=outcome,
            finish_reason=finish_reason,
            final_content=final_content,
            tool_calls=tool_calls_info,
            any_tool_executed=any_tool_executed,
            side_effects_may_have_occurred=side_effects_may_have_occurred,
            safe_to_retry=safe_to_retry,
            error_category=error_category,
            retryable=outcome
            in (
                FrameOutcome.OUTPUT_TRUNCATED,
                FrameOutcome.CONTEXT_OVERFLOW,
                FrameOutcome.TOOL_FAILURE_RECOVERABLE,
                FrameOutcome.PROVIDER_TRANSIENT_ERROR,
            ),
            metadata={
                "execution_status": frame.status.value,
                "input_tokens": frame.result.input_tokens if frame.result else 0,
                "output_tokens": frame.result.output_tokens if frame.result else 0,
                "error_message": frame.result.error_message,
                "error_kind": error_kind,
                "status_code": status_code,
                "retry_after": retry_after,
            },
        )

    @staticmethod
    def _determine_outcome(  # noqa: C901
        status: ExecutionStatus,
        finish_reason: str | None,
        tool_calls: list[ToolCallInfo],
        any_tool_executed: bool,
        error_message: str | None,
        transient_error: bool = False,
    ) -> FrameOutcome:
        """Determine semantic outcome from frame status and details."""
        # Transient error during running state -> retryable
        if transient_error and status in (
            ExecutionStatus.RUNNING_LLM,
            ExecutionStatus.RUNNING_TOOLS,
        ):
            return FrameOutcome.PROVIDER_TRANSIENT_ERROR

        # Terminal successful states
        if status == ExecutionStatus.COMPLETED:
            if tool_calls:
                return FrameOutcome.TOOL_ROUND_COMPLETED
            return FrameOutcome.FINAL_RESPONSE

        # Partial states - check finish_reason
        if status == ExecutionStatus.PARTIAL:
            if finish_reason == "length":
                if tool_calls:
                    return FrameOutcome.OUTPUT_TRUNCATED_WITH_TOOL_CALLS
                return FrameOutcome.OUTPUT_TRUNCATED
            # Other partial reasons
            if any_tool_executed:
                return FrameOutcome.TOOL_FAILURE_RECOVERABLE
            return FrameOutcome.OUTPUT_TRUNCATED

        # Failed states
        if status == ExecutionStatus.FAILED:
            if finish_reason == "length":
                return FrameOutcome.CONTEXT_OVERFLOW
            if error_message:
                # Could be more specific based on error analysis
                return FrameOutcome.PROVIDER_NON_TRANSIENT_ERROR
            return FrameOutcome.UNKNOWN

        # Blocked
        if status == ExecutionStatus.BLOCKED:
            if finish_reason in {"content_filter", "safety", "blocked"}:
                return FrameOutcome.CONTENT_BLOCKED
            return FrameOutcome.UNKNOWN

        # Running states shouldn't reach here for completed frames
        return FrameOutcome.UNKNOWN


class ExecutionResult(BaseModel):
    """The normalized result of an execution frame."""

    final_answer: str | None = None
    finish_reason: str | None = None
    total_latency_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None
    agent_file_changes: list[FileChange] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeFrame(BaseModel):
    """
    A single agent reasoning step.

    LIGHTWEIGHT FRAME MODEL:
    - No stored prompt_context (use turn_id + context_snapshot_id for reconstruction)
    - No rendered_prompt or full_prompt properties (ephemeral, not durable)
    - Carries trace_id + turn_id + context_snapshot_id + metadata only
    """

    # === IDENTIFICATION ===
    id: str = Field(default_factory=lambda: f"frame_{uuid.uuid4()}")
    session_id: str
    turn_id: str
    trace_id: str
    iteration: int = 1
    context_snapshot_id: str | None = None

    # === EXECUTION STATE ===
    status: ExecutionStatus
    model_response_raw: str | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    result: ExecutionResult | None = None

    # === WORKSPACE STATE ===
    transaction_id: str | None = None
    transaction_status: str | None = None
    conflicts: list[Any] = Field(default_factory=list)

    # === TIMING ===
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    # === PROMPT RECONSTRUCTION ===
    # Hash of the rendered prompt used for this frame (separate from snapshot's static context hash)
    prompt_hash: str | None = None
    # Last committed transcript sequence included in this frame's prompt.
    history_through_seq: int = 0

    # === CONVENIENCE PROPERTIES ===
    @property
    def duration_ms(self) -> float | None:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def is_successful(self) -> bool:
        return self.result is not None and self.status == ExecutionStatus.COMPLETED

    # === ON-DEMAND DATA ACCESS (from actions only) ===
    def get_tool_results(self) -> list[dict[str, Any]]:
        """Generate tool results from actions on-demand."""
        tool_results = []
        for action in self.actions:
            if action.kind == ActionKind.TOOL_CALL and action.tool_execution_data:
                data = action.tool_execution_data
                tool_result = {
                    "tool_name": data.get("tool_id", action.name or "unknown"),
                    "tool_args": data.get("input", {}),
                    "success": data.get("status") == "completed",
                    "result": data.get("output"),
                    "error_message": data.get("error", {}).get("message")
                    if data.get("error")
                    else None,
                    "execution_time_ms": data.get("timing", {}).get(
                        "elapsed_seconds", 0
                    )
                    * 1000,
                    "call_id": data.get("call_id"),
                    "start_time": None,
                    "end_time": None,
                    "parent_call_id": data.get("decision", {}).get("parent_call_id"),
                    "child_call_ids": [],
                    "execution_mode": data.get("backend"),
                    "source": "tools_platform",
                    "resource_metrics": [],
                    "metadata": data.get("metadata", {}),
                }
                tool_results.append(tool_result)
        return tool_results

    def get_tool_metrics(self) -> list[dict[str, Any]]:
        """Generate tool metrics from actions on-demand."""
        tool_metrics = []
        for action in self.actions:
            if action.kind == ActionKind.TOOL_CALL and action.tool_execution_data:
                data = action.tool_execution_data
                metrics = {
                    "tool_name": data.get("tool_id", action.name or "unknown"),
                    "call_id": data.get("call_id"),
                    "execution_time_ms": data.get("timing", {}).get(
                        "elapsed_seconds", 0
                    )
                    * 1000,
                    "start_time": None,
                    "end_time": None,
                    "success": data.get("status") == "completed",
                    "error_message": data.get("error", {}).get("message")
                    if data.get("error")
                    else None,
                    "parent_call_id": data.get("decision", {}).get("parent_call_id"),
                    "resource_metrics": [],
                    "metadata": data.get("metadata", {}),
                }
                tool_metrics.append(metrics)
        return tool_metrics

    def get_reasoning(self) -> str | None:
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.metadata:
                return action.metadata.get("reasoning")
        return None

    def get_reasoning_tokens(self) -> int:
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.llm_metrics:
                return getattr(action.llm_metrics, "reasoning_tokens", 0)
        return 0

    def get_llm_metrics(self) -> LLMMetrics | None:
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.llm_metrics:
                return action.llm_metrics
        return None

    # === ACTION MANAGEMENT ===
    def add_action(
        self, kind: ActionKind, name: str | None = None, _input: Any = None
    ) -> ActionRecord:
        action = ActionRecord(kind=kind, name=name)
        if kind == ActionKind.TOOL_CALL and _input is not None:
            action.tool_execution_data["input"] = _input
        elif _input is not None:
            action.metadata["input"] = _input

        self.actions.append(action)
        return action

    def _complete_llm_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        if metadata:
            # Convert tool_calls from dicts to LLMToolCall objects
            tool_calls_data = metadata.get("tool_calls", [])
            action.tool_calls = [
                LLMToolCall(**tc) if isinstance(tc, dict) else tc
                for tc in tool_calls_data
            ]
            llm_metrics: LLMMetrics
            if action.llm_metrics is not None:
                llm_metrics = action.llm_metrics
            else:
                llm_metrics = LLMMetrics()
                action.llm_metrics = llm_metrics
            llm_metrics.model = metadata.get("model")
            llm_metrics.prompt_tokens = metadata.get("prompt_tokens", 0)
            llm_metrics.completion_tokens = metadata.get("completion_tokens", 0)
            llm_metrics.reasoning_tokens = metadata.get("reasoning_tokens", 0)
            llm_metrics.cached_tokens = metadata.get("cached_tokens", 0)
            llm_metrics.cache_write_tokens = metadata.get("cache_write_tokens", 0)
            llm_metrics.latency_ms = metadata.get("latency_ms", 0.0)
            if "reasoning" in metadata:
                action.metadata["reasoning"] = metadata["reasoning"]
        action.finished_at = datetime.now(UTC)

    def _complete_tool_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        if isinstance(output, ToolExecutionResult):
            result = output
            action.tool_execution_data = {
                "tool_id": metadata.get("tool_id") if metadata else action.name,
                "status": result.status,
                "output": result.output,
                "error": result.error.model_dump() if result.error else None,
                "timing": result.timing,
                "output_chunks": [c.model_dump() for c in result.output_chunks]
                if result.output_chunks
                else [],
                "warnings": result.warnings,
                "changeset": result.changeset.model_dump()
                if result.changeset
                else None,
                "backend": result.backend.value if result.backend else None,
                "trace_id": result.trace_id,
                "call_id": metadata.get("call_id") if metadata else None,
                "decision": result.decision.model_dump() if result.decision else None,
                "input": action.tool_execution_data.get("input", {}),
            }
        elif metadata and "tool_execution_data" in metadata:
            action.tool_execution_data = metadata["tool_execution_data"]
        else:
            action.tool_execution_data = {
                "tool_id": metadata.get("tool_id", action.name)
                if metadata
                else action.name,
                "status": metadata.get("status", "completed")
                if metadata
                else "completed",
                "output": output,
                "error": metadata.get("error") if metadata else None,
                "timing": metadata.get("timing", {}) if metadata else {},
                "input": action.tool_execution_data.get("input", {}),
            }
        action.finished_at = datetime.now(UTC)

    def _complete_other_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        action.metadata["output"] = output
        if metadata:
            action.metadata.update(metadata)
        action.finished_at = datetime.now(UTC)

    def complete_action(
        self, action_id: str, output: Any, metadata: dict[str, Any] | None = None
    ):
        for action in self.actions:
            if action.id == action_id:
                if action.kind == ActionKind.LLM_MESSAGE:
                    self._complete_llm_action(action, output, metadata)
                elif action.kind == ActionKind.TOOL_CALL:
                    self._complete_tool_action(action, output, metadata)
                else:
                    self._complete_other_action(action, output, metadata)
                break

    def complete(
        self,
        final_answer: str | None = None,
        error_message: str | None = None,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        now = datetime.now(UTC)
        self.finished_at = now

        total_latency_ms = (now - self.started_at).total_seconds() * 1000

        input_tokens = 0
        output_tokens = 0
        agent_file_changes: set[FileChange] = set()
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.llm_metrics:
                input_tokens += action.llm_metrics.prompt_tokens
                output_tokens += (
                    action.llm_metrics.completion_tokens
                    + action.llm_metrics.reasoning_tokens
                )
            elif action.kind == ActionKind.TOOL_CALL:
                changeset = action.tool_execution_data.get("changeset") or {}
                for change in changeset.get("changes", []):
                    agent_file_changes.add(FileChange(**change))

        result = ExecutionResult(
            final_answer=final_answer,
            error_message=error_message,
            finish_reason=finish_reason,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            agent_file_changes=list(agent_file_changes),
            metadata=metadata or {},
        )
        self.result = result

        return result


# === Frame restoration from stored events ===


def build_frame(stored: StoredFrame, events: list[dict[str, Any]]) -> RuntimeFrame:
    """Reconstruct a frame from its stored row and its frame events.

    Used by the resume flow. Actions are rebuilt from the event log:
    LLM_COMPLETE restores the LLM_MESSAGE action (with its tool calls so a
    resumed frame knows what to reconcile); TOOL_START opens a tool action
    and TOOL_RESULT closes it with the recorded outcome.

    Args:
        stored: The frame's StoredFrame row
        events: Event rows for this frame (as returned by EventRepo), ordered by seq

    Returns:
        RuntimeFrame with restored actions and result
    """
    actions, model_response_raw = _rebuild_actions_from_events(events)

    frame = RuntimeFrame(
        id=stored.id,
        session_id=stored.session_id,
        turn_id=stored.turn_id or stored.session_id,
        trace_id=stored.trace_id,
        iteration=stored.iteration,
        context_snapshot_id=stored.context_snapshot_id,
        status=ExecutionStatus(stored.status),
        model_response_raw=model_response_raw,
        actions=[],
        started_at=datetime.fromtimestamp(stored.started_at_ms / 1000, UTC),
        finished_at=datetime.fromtimestamp(stored.finished_at_ms / 1000, UTC)
        if stored.finished_at_ms
        else None,
        history_through_seq=stored.history_through_seq,
        prompt_hash=stored.prompt_hash,
    )

    for action_data in actions:
        tool_calls = [
            LLMToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in (action_data.get("tool_calls") or [])
        ]
        llm_metrics_data = action_data.get("llm_metrics")
        action = ActionRecord(
            id=action_data["id"],
            kind=ActionKind(action_data["kind"]),
            name=action_data.get("name"),
            started_at=datetime.fromisoformat(
                action_data["started_at"].replace("Z", "+00:00")
            ),
            finished_at=datetime.fromisoformat(
                action_data["finished_at"].replace("Z", "+00:00")
            )
            if action_data.get("finished_at")
            else None,
            tool_calls=tool_calls,
            llm_metrics=LLMMetrics(**llm_metrics_data) if llm_metrics_data else None,
            metadata=action_data.get("metadata", {}),
            tool_execution_data=action_data.get("tool_execution_data", {}),
        )
        frame.actions.append(action)

    # Set result from FRAME_COMPLETE event
    for event_row in events:
        if event_row["type"] == "frame_complete":
            event_data = json.loads(event_row["data_json"])
            frame.result = ExecutionResult(
                final_answer=event_data.get("final_answer"),
                finish_reason=event_data.get("finish_reason"),
                total_latency_ms=event_data.get("duration_ms"),
                input_tokens=event_data.get("total_input_tokens") or 0,
                output_tokens=event_data.get("total_output_tokens") or 0,
                error_message=event_data.get("error_message"),
            )
            break

    return frame


def _rebuild_actions_from_events(
    event_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Rebuild action dicts from stored frame events.

    Returns:
        Tuple of action dicts and the raw model response (if any)
    """
    actions: list[dict[str, Any]] = []
    model_response_raw: str | None = None

    for event_row in event_rows:
        event_data = json.loads(event_row["data_json"])
        event_ts = datetime.fromtimestamp(event_row["timestamp"] / 1000, UTC)
        event_type = event_row["type"]

        if event_type == "llm_complete":
            model_response_raw = event_data.get("final_content") or event_data.get(
                "content"
            )
            actions.append(_llm_action_from_event(event_data, event_ts))
        elif event_type == "tool_start":
            actions.append(_tool_start_action(event_data, event_ts))
        elif event_type == "tool_result":
            _close_tool_action(actions, event_data, event_ts)

    return actions, model_response_raw


def _llm_action_from_event(
    event_data: dict[str, Any], event_ts: datetime
) -> dict[str, Any]:
    """Reconstruct an LLM_MESSAGE action dict from an LLM_COMPLETE event."""
    tool_calls = event_data.get("tool_calls") or []
    return {
        "id": str(uuid.uuid4()),
        "kind": "llm_message",
        "name": "llm_generation",
        "started_at": event_ts.isoformat(),
        "finished_at": event_ts.isoformat(),
        "tool_calls": tool_calls,
        "llm_metrics": {
            "model": event_data.get("model"),
            "prompt_tokens": event_data.get("prompt_tokens") or 0,
            "completion_tokens": event_data.get("completion_tokens") or 0,
            "reasoning_tokens": event_data.get("reasoning_tokens") or 0,
            "latency_ms": event_data.get("latency_ms") or 0.0,
        },
        "metadata": {
            "tool_calls": tool_calls,
            "reasoning": event_data.get("reasoning"),
        },
    }


def _tool_start_action(
    event_data: dict[str, Any], event_ts: datetime
) -> dict[str, Any]:
    """Open a tool action dict from a TOOL_START event."""
    return {
        "id": str(uuid.uuid4()),
        "kind": "tool_call",
        "name": event_data.get("tool_id"),
        "started_at": event_ts.isoformat(),
        "tool_execution_data": {
            "input": event_data.get("arguments", {}),
            "call_id": event_data.get("call_id"),
            "tool_id": event_data.get("tool_id"),
            "status": "running",
        },
    }


def _close_tool_action(
    actions: list[dict[str, Any]],
    event_data: dict[str, Any],
    event_ts: datetime,
) -> None:
    """Close the matching tool action from a TOOL_RESULT event.

    Falls back to creating the action defensively if no TOOL_START was
    recorded (e.g. events written before the start event was flushed).
    """
    call_id = event_data.get("call_id")
    output = event_data.get("output", event_data.get("result"))
    error = event_data.get("error")
    if event_data.get("success"):
        status = "completed"
    elif error and error.get("code") == ToolErrorCode.INTERRUPTED.value:
        status = "cancelled"
    else:
        status = "failed"
    closed_data = {
        "tool_id": event_data.get("tool_id"),
        "status": status,
        "output": output,
        "error": event_data.get("error"),
        "timing": event_data.get("timing", {}),
        "output_chunks": [],
        "warnings": event_data.get("warnings", []),
        "changeset": event_data.get("changeset"),
        "backend": event_data.get("backend"),
        "trace_id": event_data.get("trace_id"),
        "call_id": call_id,
        "decision": event_data.get("decision"),
        "input": event_data.get("arguments", {}),
    }

    for action in actions:
        existing = action.get("tool_execution_data", {})
        if existing and existing.get("call_id") == call_id:
            # Preserve the input recorded at start time
            closed_data["input"] = existing.get("input", closed_data["input"])
            action["tool_execution_data"] = closed_data
            action["finished_at"] = event_ts.isoformat()
            return

    actions.append(
        {
            "id": str(uuid.uuid4()),
            "kind": "tool_call",
            "name": event_data.get("tool_id"),
            "started_at": event_ts.isoformat(),
            "finished_at": event_ts.isoformat(),
            "tool_execution_data": closed_data,
        }
    )
