"""Turn models - StoredTurn (durable) and RuntimeTurn (ephemeral).

StoredTurn maps directly to the database table.
RuntimeTurn is the unified runtime object combining:
- Turn identity and snapshot refs
- Prompt resolution state (from TurnContext)
- Limits enforcement (from RuntimeLimits)
- Approval state (from TurnContext)
- Config-driven limits and continuation policy for resume/replay
- Pause/resume tracking for approval, interrupt, deadline, crash recovery
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_coding_assistant.storage import IExecutionStore

from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
)
from local_coding_assistant.runtime.engine.approval import (
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
)
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.limits import LimitCheckResult, RuntimeLimits
from local_coding_assistant.runtime.events.contracts import ApprovalResponseForm
from local_coding_assistant.storage.contracts import ResumeBatch, StoredTurn
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.turn")


class TurnStatus(str, Enum):
    """Session execution status for storage."""

    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    SUPERSEDED = "superseded"


@dataclass
class RuntimeTurn:
    """
    Unified runtime turn object for active execution.

    Combines what was previously:
    - TurnContext (prompt resolution, user_input, trace_id, pending_approvals)
    - RuntimeLimits (token_budget, deadline_tracker, loop_guard)
    - Turn metadata for replay/resume

    Does NOT hold session reference - pass session explicitly when needed.
    Does NOT hold full frame objects - only frame_ids for history tracking.
    """

    # Identity (from StoredTurn)
    turn_id: str
    session_id: str
    trace_id: str
    turn_index: int
    user_input: str

    # Snapshot (MANDATORY - not Optional)
    context_snapshot: ContextSnapshot
    context_snapshot_id: str  # convenience, same as context_snapshot.id

    # Config (from turn's config_snapshot_json - stored at creation)
    context_config: RuntimeContextConfig
    limits_config: RuntimeLimitsConfig

    # Limits enforcement (composite RuntimeLimits)
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)

    # Approvals
    # In-memory only: approval requests live for the duration of the process
    # (between the approval pause and the approval response). On resume the
    # approval flow simply re-triggers when the resumed frame re-executes the
    # tool call, so approvals are never restored from or persisted to store.
    pending_approvals: dict[str, ApprovalRequest] = field(default_factory=dict)

    # Replay/resume metadata
    is_resumed: bool = False
    resumed_from_turn_id: str | None = None

    # Frame history (lightweight refs, not full frames)
    frame_ids: list[str] = field(default_factory=list)

    # Metadata for continuation policy (compaction, retries, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: TurnStatus = TurnStatus.RUNNING
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    paused_at: datetime | None = None
    pause_reason: str | None = None
    resumed_at: datetime | None = None

    # Final answer
    final_answer: str | None = None

    # Interrupt handling
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    def check_limits(self, additional_tokens: int = 0) -> LimitCheckResult:
        """Check all limits (token budget, deadline, loop detection) - returns structured result."""
        return self.limits.check_limits(additional_tokens)

    def record_tool_call(
        self, call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Record a tool call for loop detection and limits."""
        self.limits.record_tool_call(call_id, tool_name, arguments)

    def record_tokens(
        self, input_tokens: int = 0, output_tokens: int = 0
    ) -> LimitCheckResult:
        """Record token usage."""
        return self.limits.record_tokens(input_tokens, output_tokens)

    def add_frame_id(self, frame_id: str) -> None:
        """Track frame ID for history."""
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)

    def get_limits_summary(self) -> dict[str, Any]:
        """Get summary of current limit state."""
        return self.limits.summary

    def get_final_answer(self) -> str | None:
        """Get the final answer"""
        return self.final_answer

    def reset_limits(self, timeout_seconds: int | None = None) -> None:
        """Reset all limits for new turn."""
        self.limits.reset(timeout_seconds)

    def get_limits_state_for_replay(self) -> dict[str, Any]:
        """Get limits state for serialization/resume."""
        return self.limits.get_state_for_replay()

    def restore_limits_state(self, state: dict[str, Any]) -> None:
        """Restore limits state from serialization."""
        self.limits.restore_state(state)

    def get_last_frame_id(self) -> str | None:
        """Get the last frame ID from the turn's frame history."""
        return self.frame_ids[-1] if self.frame_ids else None

    def apply_approval_response(
        self, response_form: ApprovalResponseForm
    ) -> ApprovalRequest:
        """Validate and apply approval response from form.

        Args:
            response_form: The approval response form containing
                approval_token, tool_call_id, and response

        Returns:
            The updated ApprovalRequest

        Raises:
            ValueError: If token not found, expired, or already responded
        """
        # Find approval by token (search in pending_approvals)
        approval = self.pending_approvals.get(response_form.tool_call_id, None)

        if not approval:
            raise ValueError(
                f"Pending approval for tool call id {response_form.tool_call_id} not found"
            )

        if approval.token != response_form.approval_token:
            raise ValueError(
                "Provided approval token does not match the expected token"
            )

        if approval.is_expired:
            approval.status = ApprovalStatus.EXPIRED
            raise ValueError(f"Approval token expired: {response_form.approval_token}")

        if approval.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Approval already responded: {response_form.approval_token} (status: {approval.status.value})"
            )

        # Convert response string to ApprovalResponse enum
        response = ApprovalResponse(response_form.response.lower())

        # Apply response
        approval.response = response
        approval.responded_at = datetime.now(UTC)
        approval.status = (
            ApprovalStatus.GRANTED
            if response != ApprovalResponse.NO
            else ApprovalStatus.DENIED
        )

        logger.info(
            "Approval response recorded",
            token=response_form.approval_token,
            tool=approval.tool_name,
            tool_call_id=approval.tool_call_id,
            response=response.value,
            status=approval.status.value,
        )

        return approval

    @classmethod
    async def from_store(
        cls, turn_id: str, store: IExecutionStore, stored_turn: StoredTurn | None = None
    ) -> RuntimeTurn:
        """Load turn AND snapshot from store for resume/replay.

        Uses STORED config (deterministic replay).
        """
        # 1. Load StoredTurn
        stored_turn = stored_turn or await store.get_turn(turn_id)
        if not stored_turn:
            raise ValueError(f"Turn not found: {turn_id}")

        # 2. Load ContextSnapshot (bound to turn)
        snapshot = await store.get_context_snapshot_by_turn(turn_id)
        if not snapshot:
            raise ValueError(f"No snapshot for turn {turn_id}")

        # 3. Parse stored config (deterministic replay)
        if not stored_turn.config_snapshot_json:
            raise ValueError(f"No config snapshot for turn {turn_id}")
        config_data = json.loads(stored_turn.config_snapshot_json)
        context_config = RuntimeContextConfig(**config_data.get("context_config", {}))
        limits_config = RuntimeLimitsConfig(**config_data["limits_config"])

        # 4. Restore limits state
        limits_state = None
        if stored_turn.metadata_json:
            metadata = json.loads(stored_turn.metadata_json)
            limits_state = metadata.get("limits_state")

        limits = RuntimeLimits.from_restored(limits_config, limits_state)

        # 5. Load frame IDs for this turn
        stored_frame_ids = await store.get_frame_ids_by_turn(turn_id)

        # 6. Build turn with MANDATORY snapshot
        turn = cls(
            turn_id=stored_turn.id,
            session_id=stored_turn.session_id,
            trace_id=stored_turn.trace_id,
            turn_index=stored_turn.turn_index,
            user_input=stored_turn.user_input,
            context_snapshot=snapshot,
            context_snapshot_id=snapshot.id,
            context_config=context_config,
            limits=limits,
            limits_config=limits_config,
            status=TurnStatus(stored_turn.status),
            is_resumed=True,
            paused_at=datetime.fromtimestamp(stored_turn.paused_at_ms / 1000, UTC)
            if stored_turn.paused_at_ms
            else None,
            pause_reason=stored_turn.pause_reason,
            resumed_at=datetime.fromtimestamp(stored_turn.resumed_at_ms / 1000, UTC)
            if stored_turn.resumed_at_ms
            else None,
            frame_ids=stored_frame_ids,
        )

        return turn

    @classmethod
    def from_batch(cls, batch: ResumeBatch) -> RuntimeTurn:
        """Build the resumed turn from a ResumeBatch loaded by the store.

        Uses the STORED config (deterministic replay): frame ids come from
        the batch's frame rows, pending approvals are intentionally empty
        (cache-only concept - the approval flow re-triggers on resume).
        """
        stored_turn = batch.turn
        snapshot = batch.snapshot
        if snapshot is None:
            raise ValueError(f"No snapshot for turn {stored_turn.id}")

        if not stored_turn.config_snapshot_json:
            raise ValueError(f"No config snapshot for turn {stored_turn.id}")
        config_data = json.loads(stored_turn.config_snapshot_json)
        context_config = RuntimeContextConfig(**config_data.get("context_config", {}))
        limits_config = RuntimeLimitsConfig(**config_data["limits_config"])

        limits_state = None
        if stored_turn.metadata_json:
            limits_state = json.loads(stored_turn.metadata_json).get("limits_state")

        limits = RuntimeLimits.from_restored(limits_config, limits_state)

        return cls(
            turn_id=stored_turn.id,
            session_id=stored_turn.session_id,
            trace_id=stored_turn.trace_id,
            turn_index=stored_turn.turn_index,
            user_input=stored_turn.user_input,
            context_snapshot=snapshot,
            context_snapshot_id=snapshot.id,
            context_config=context_config,
            limits=limits,
            limits_config=limits_config,
            status=TurnStatus(stored_turn.status),
            is_resumed=True,
            paused_at=datetime.fromtimestamp(stored_turn.paused_at_ms / 1000, UTC)
            if stored_turn.paused_at_ms
            else None,
            pause_reason=stored_turn.pause_reason,
            resumed_at=datetime.fromtimestamp(stored_turn.resumed_at_ms / 1000, UTC)
            if stored_turn.resumed_at_ms
            else None,
            final_answer=stored_turn.final_answer,
            frame_ids=[f.id for f in batch.frames],
        )

    def to_stored(self) -> StoredTurn:
        """Convert to StoredTurn for persistence."""
        config_data = {
            "context_config": self.context_config.model_dump(),
            "limits_config": self.limits_config.model_dump(),
        }

        metadata = {
            "limits_state": self.get_limits_state_for_replay(),
        }
        if self.metadata:
            metadata.update(self.metadata)

        return StoredTurn(
            id=self.turn_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            turn_index=self.turn_index,
            user_input=self.user_input,
            status=str(self.status),
            started_at_ms=int(self.started_at.timestamp() * 1000),
            finished_at_ms=int(self.finished_at.timestamp() * 1000)
            if self.finished_at
            else None,
            paused_at_ms=int(self.paused_at.timestamp() * 1000)
            if self.paused_at
            else None,
            pause_reason=self.pause_reason,
            resumed_at_ms=int(self.resumed_at.timestamp() * 1000)
            if self.resumed_at
            else None,
            final_answer=self.final_answer,
            context_snapshot_id=self.context_snapshot_id,
            config_snapshot_json=json.dumps(config_data),
            total_input_tokens=self.limits.token_budget.input_tokens_used,
            total_output_tokens=self.limits.token_budget.output_tokens_used,
            total_frames=len(self.frame_ids),
            metadata_json=json.dumps(metadata) if metadata else None,
        )


# Re-export StoredTurn for convenience
__all__ = ["RuntimeTurn", "StoredTurn"]
