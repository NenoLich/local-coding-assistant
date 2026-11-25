"""StoreEventAdapter - converts runtime events to stored events with in-memory rebase.

This adapter sits between the runtime event pipeline and the execution store.
It accumulates entity mutations in memory and prepares combined upsert operations
to avoid redundant INSERT+UPDATE cycles.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from local_coding_assistant.runtime.events.contracts import (
    ApprovalDeniedEvent,
    ApprovalGrantedEvent,
    ApprovalRequiredEvent,
    ErrorEvent,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMCompleteEvent,
    RuntimeEvent,
    SessionResumeEvent,
    SessionStartEvent,
    ToolResultEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)
from local_coding_assistant.runtime.events.projector import MessageService
from local_coding_assistant.storage.contracts import StoredEvent
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.store_event_adapter")


@dataclass
class PendingEntity:
    """Accumulated mutations for a single entity before flush."""

    session_id: str | None = None
    turn_id: str | None = None
    frame_id: str | None = None

    # Session
    session_create: dict[str, Any] | None = None
    session_update: dict[str, Any] | None = None
    session_branch: dict[str, Any] | None = None

    # Turn
    turn_create: dict[str, Any] | None = None
    turn_update: dict[str, Any] | None = None

    # Context Snapshot
    snapshot_create: dict[str, Any] | None = None
    snapshot_update: dict[str, Any] | None = None

    # Frame
    frame_create: dict[str, Any] | None = None
    frame_update: dict[str, Any] | None = None

    # Message
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Event (always queued individually)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Session-tool/template associations
    associate_tools: list[dict[str, Any]] = field(default_factory=list)
    associate_templates: list[dict[str, Any]] = field(default_factory=list)

    # File baseline upserts for conflict detection
    baseline_upserts: list[dict[str, Any]] = field(default_factory=list)

    # Counter deltas are combined into one atomic update per entity and batch.
    frame_input_tokens_delta: int = 0
    frame_output_tokens_delta: int = 0
    turn_input_tokens_delta: int = 0
    turn_output_tokens_delta: int = 0
    turn_frames_delta: int = 0
    session_input_tokens_delta: int = 0
    session_output_tokens_delta: int = 0
    session_turns_delta: int = 0
    session_frames_delta: int = 0


class StoreEventAdapter:
    """Converts runtime events to stored events with in-memory rebase.

    Also projects transcript messages from the events that carry them
    (TurnStartEvent -> user, LLMCompleteEvent -> assistant,
    ToolResultEvent -> tool). Message ops land in the same pending batch as
    the source event, so event and transcript commit in one transaction.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingEntity] = {}
        self._seq_counters: dict[str, int] = {}
        self._current_turn_id: str | None = None
        self._current_frame_id: str | None = None
        self._message_service = MessageService()

    def _get_pending(self, turn_id: str, frame_id: str | None = None) -> PendingEntity:
        """Get or create pending entity bucket for a turn/frame."""
        key = f"{turn_id}:{frame_id or 'turn'}"
        if key not in self._pending:
            self._pending[key] = PendingEntity()
        return self._pending[key]

    def _entity_key(self, event: RuntimeEvent) -> str:
        """Generate key for pending entity bucket."""
        if isinstance(
            event,
            (TurnStartEvent, TurnCompleteEvent, SessionStartEvent, SessionResumeEvent),
        ):
            return f"{event.turn_id}:turn"
        return f"{event.turn_id}:{event.frame_id or 'frame'}"

    @staticmethod
    def _json_ref_or_none(value: Any) -> str | None:
        """Serialize a context ref to JSON, or NULL when absent.

        ``json.dumps(None)`` would store the literal string ``"null"``, which
        then fails ``model_validate_json`` on read; absent refs must be SQL NULL.
        """
        if value is None:
            return None
        return json.dumps(value)

    def _project_message(
        self, event: RuntimeEvent, turn_pending: PendingEntity
    ) -> None:
        """Project a transcript message from an event into the pending batch.

        The message op joins the same batch (and therefore the same
        transaction) as the event itself, so a crash can never leave an
        event without its transcript message or vice versa.

        Args:
            event: Runtime event carrying transcript data
            turn_pending: Turn-level pending bucket to append the op to
        """
        built = self._message_service.build_message(event)
        if built is None:
            return

        message, dedupe_key = built
        turn_pending.messages.append(
            {
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "frame_id": event.frame_id or "turn_start",
                # seq=None: computed in SQL as MAX(seq)+1 per session
                "seq": None,
                "role": message.role,
                "content": message.content or "",
                "tool_call_id": message.tool_call_id,
                "tool_name": (
                    event.tool_name
                    if isinstance(event, ToolResultEvent)
                    else (message.tool_calls[0]["name"] if message.tool_calls else None)
                ),
                "metadata_json": json.dumps({"tool_calls": message.tool_calls})
                if message.tool_calls
                else None,
                "dedupe_key": dedupe_key,
                "created_at_ms": int(event.timestamp.timestamp() * 1000),
            }
        )

    def _next_seq(self, session_id: str) -> int:
        """Get next sequence number for session.

        Seq is globally unique and monotonic per session across all frames,
        so ordering by seq reproduces execution order. Keyed by session_id
        (not turn/frame) so the counter never resets mid-session, matching
        the message repo's per-session seq pattern.
        """
        self._seq_counters[session_id] = self._seq_counters.get(session_id, 0) + 1
        return self._seq_counters[session_id]

    def has_seq_counter(self, session_id: str) -> bool:
        """Return whether this adapter has initialized the session counter."""
        return session_id in self._seq_counters

    def initialize_seq_counter(self, session_id: str, last_seq: int) -> None:
        """Seed the counter from persisted events without resetting live state."""
        self._seq_counters.setdefault(session_id, last_seq)

    def process(self, event: RuntimeEvent) -> None:
        """Process a runtime event: assign seq, accumulate mutations, queue stored event."""
        # Assign sequence number. The EventPipeline assigns seq before fan-out;
        # this is a fallback for direct store.process_event() calls where seq
        # is still the default 0.
        if event.seq == 0:
            event.seq = self._next_seq(event.session_id)

        # Ensure trace_id
        if not event.trace_id:
            event.trace_id = self._generate_trace_id()

        # Track current turn/frame for context
        if isinstance(event, TurnStartEvent):
            self._current_turn_id = event.turn_id
        if isinstance(event, FrameStartEvent):
            self._current_frame_id = event.frame_id
            self._current_turn_id = event.turn_id

        # Accumulate entity mutations
        self._accumulate(event)

        # Convert to stored event and queue
        stored = event.to_stored_event()
        self._queue_stored_event(stored)

    def _accumulate(self, event: RuntimeEvent) -> None:
        """Accumulate entity mutations in memory for batched upsert."""
        key = self._entity_key(event)
        pending = self._pending.setdefault(key, PendingEntity())
        turn_pending = self._get_pending(event.turn_id, None)
        pending.session_id = event.session_id
        pending.turn_id = event.turn_id
        pending.frame_id = event.frame_id
        turn_pending.session_id = event.session_id
        turn_pending.turn_id = event.turn_id

        handler = self._EVENT_HANDLERS.get(type(event))
        if handler is not None:
            handler(self, event, pending, turn_pending)

    def _accumulate_session_start(
        self,
        event: SessionStartEvent,
        pending: PendingEntity,
        _turn_pending: PendingEntity,
    ) -> None:
        pending.session_create = {
            "op": "upsert_session",
            "data": {
                "session_id": event.session_id,
                "user_input": event.user_input,
                "status": "idle",
                "started_at_ms": int(event.timestamp.timestamp() * 1000),
                "agent_mode": event.agent_mode,
                "model_name": event.model_name,
                "metadata_json": event.metadata_json,
            },
        }

    def _accumulate_turn_start(
        self, event: TurnStartEvent, pending: PendingEntity, turn_pending: PendingEntity
    ) -> None:
        # Parse context snapshot JSON up front so we can back-reference
        # the snapshot on the turn row when we create it. This avoids
        # an extra lookup loop during branch_session cascade copy.
        snapshot_data: dict[str, Any] | None = None
        if event.context_snapshot_json:
            try:
                snapshot_data = json.loads(event.context_snapshot_json)
            except Exception as e:
                logger.warning(
                    "Failed to parse context_snapshot_json from TurnStartEvent",
                    error=str(e),
                )
        snapshot_id = (
            snapshot_data.get("id", event.context_snapshot_id)
            if snapshot_data
            else event.context_snapshot_id
        )

        pending.turn_create = {
            "op": "upsert_turn",
            "data": {
                "turn_id": event.turn_id,
                "session_id": event.session_id,
                "trace_id": event.trace_id,
                "turn_index": event.turn_index,
                "user_input": event.user_input,
                "status": "running",
                "started_at_ms": int(event.timestamp.timestamp() * 1000),
                "config_snapshot_json": event.config_snapshot_json,
                "context_snapshot_id": snapshot_id,
            },
        }
        turn_pending.session_turns_delta += 1
        # Create full context snapshot from TurnStartEvent (contains complete snapshot JSON)
        if snapshot_data:
            pending.snapshot_create = {
                "op": "upsert_context_snapshot",
                "data": {
                    "snapshot_id": snapshot_id,
                    "turn_id": event.turn_id,
                    "schema_version": snapshot_data.get("schema_version", 1),
                    "tool_set_ref": self._json_ref_or_none(
                        snapshot_data.get("tool_set_ref")
                    ),
                    "template_set_ref": self._json_ref_or_none(
                        snapshot_data.get("template_set_ref")
                    ),
                    "repo_snapshot_ref": self._json_ref_or_none(
                        snapshot_data.get("repo_snapshot_ref")
                    ),
                    "memory_snapshot_ref": self._json_ref_or_none(
                        snapshot_data.get("memory_snapshot_ref")
                    ),
                    "message_boundary_json": self._json_ref_or_none(
                        snapshot_data.get("message_boundary")
                    ),
                    "content_hash": snapshot_data.get("content_hash", ""),
                    "payload_json": snapshot_data.get("payload_json"),
                    "created_at_ms": snapshot_data.get("created_at_ms"),
                },
            }

        # Update session status to running on turn start and refresh
        # the stored context profile (the bundle cache may rebuild it
        # every turn).
        pending.session_update = {
            "op": "upsert_session",
            "data": {
                "session_id": event.session_id,
                "status": "running",
                "updated_at_ms": int(event.timestamp.timestamp() * 1000),
                "context_profile_json": event.context_profile_json,
            },
        }

        # Project the user message (turn = one user message)
        self._project_message(event, turn_pending)

    def _accumulate_turn_complete(
        self,
        event: TurnCompleteEvent,
        pending: PendingEntity,
        turn_pending: PendingEntity,
    ) -> None:
        turn_pending.turn_update = {
            "op": "upsert_turn",
            "data": {
                "turn_id": event.turn_id,
                "session_id": event.session_id,
                "trace_id": event.trace_id,
                "turn_index": 0,  # Not updated
                "user_input": "",  # Not updated
                "status": event.status,
                "started_at_ms": 0,  # Not updated
                "finished_at_ms": int(event.timestamp.timestamp() * 1000),
                "final_answer": event.final_answer,
            },
        }
        # Update session status; counters are maintained from start/LLM events.
        turn_pending.session_update = {
            "op": "upsert_session",
            "data": {
                "session_id": event.session_id,
                "user_input": "",
                "status": "idle",
                "started_at_ms": 0,
                "agent_mode": "",
                "updated_at_ms": int(event.timestamp.timestamp() * 1000),
            },
        }

    def _accumulate_frame_start(
        self,
        event: FrameStartEvent,
        pending: PendingEntity,
        turn_pending: PendingEntity,
    ) -> None:
        # Create frame
        pending.frame_create = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                "iteration": event.iteration,
                "status": "running_llm",
                "started_at_ms": int(event.timestamp.timestamp() * 1000),
                "context_snapshot_id": event.context_snapshot_id,
                "history_through_seq": event.history_through_seq,
                "prompt_hash": event.prompt_hash,
            },
        }
        turn_pending.turn_frames_delta += 1
        turn_pending.session_frames_delta += 1
        # NOTE: Context snapshot is created from TurnStartEvent, not here.
        # FrameStartEvent only references the snapshot and records frame-specific data.

    def _accumulate_frame_complete(
        self,
        event: FrameCompleteEvent,
        pending: PendingEntity,
        _turn_pending: PendingEntity,
    ) -> None:
        pending.frame_update = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                # Don't override iteration from create
                "status": event.status,
                "started_at_ms": 0,
                "finished_at_ms": int(event.timestamp.timestamp() * 1000),
                "duration_ms": event.duration_ms,
                "model_name": event.model_name,
                "finish_reason": event.finish_reason,
                "error_message": event.error_message,
            },
        }

    def _accumulate_llm_complete(
        self,
        event: LLMCompleteEvent,
        pending: PendingEntity,
        turn_pending: PendingEntity,
    ) -> None:
        input_tokens = event.input_tokens or 0
        output_tokens = event.output_tokens or 0
        pending.frame_update = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                "status": "running_tools",
                "model_name": event.model,
            },
        }
        pending.frame_input_tokens_delta += input_tokens
        pending.frame_output_tokens_delta += output_tokens
        turn_pending.turn_input_tokens_delta += input_tokens
        turn_pending.turn_output_tokens_delta += output_tokens
        turn_pending.session_input_tokens_delta += input_tokens
        turn_pending.session_output_tokens_delta += output_tokens

        # Project the assistant message (with its tool calls)
        self._project_message(event, turn_pending)

    def _accumulate_tool_result(
        self,
        event: ToolResultEvent,
        pending: PendingEntity,
        turn_pending: PendingEntity,
    ) -> None:
        # Update frame status if tool failed
        if not event.success:
            pending.frame_update = {
                "op": "upsert_frame",
                "data": {
                    "frame_id": event.frame_id,
                    "session_id": event.session_id,
                    "turn_id": event.turn_id,
                    "trace_id": event.trace_id,
                    "status": "running_tools",
                    "error_message": event.error.get("message")
                    if event.error
                    else "Tool failed",
                },
            }

        # Project the tool result message (success, failure, or
        # interruption are all data)
        self._project_message(event, turn_pending)

        # Persist file observations (baselines for conflict detection)
        for obs in event.file_observations:
            raw_source = obs.get("source", "unknown")
            obs_source = (
                raw_source.get("value", "unknown")
                if isinstance(raw_source, dict)
                else str(raw_source)
            )
            if obs_source in ("read", "snapshot", "agent_create"):
                if not event.workspace_id:
                    raise ValueError(
                        "workspace_id is required for file baseline upserts"
                    )
                pending.baseline_upserts.append(
                    {
                        "op": "upsert_file_baseline",
                        "data": {
                            # Workspace identity from the gateway
                            # (matches the resume load path)
                            "workspace_id": event.workspace_id,
                            "observation": obs,
                            "turn_id": event.turn_id,
                            "frame_id": event.frame_id,
                            "session_id": event.session_id,
                        },
                    }
                )

    def _accumulate_approval_required(
        self,
        event: ApprovalRequiredEvent,
        pending: PendingEntity,
        _turn_pending: PendingEntity,
    ) -> None:
        # Frame goes to paused_approval
        pending.frame_update = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                "status": "running_tools",
            },
        }

    def _accumulate_approval_resolved(
        self,
        event: ApprovalGrantedEvent | ApprovalDeniedEvent,
        pending: PendingEntity,
        _turn_pending: PendingEntity,
    ) -> None:
        # Frame resumes running_tools
        pending.frame_update = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                "status": "running_tools",
                "pending_call_id": None,
            },
        }

    def _accumulate_error(
        self, event: ErrorEvent, pending: PendingEntity, _turn_pending: PendingEntity
    ) -> None:
        pending.frame_update = {
            "op": "upsert_frame",
            "data": {
                "frame_id": event.frame_id,
                "session_id": event.session_id,
                "turn_id": event.turn_id,
                "trace_id": event.trace_id,
                "status": "failed",
                "finished_at_ms": int(event.timestamp.timestamp() * 1000),
                "error_message": event.error_message,
            },
        }

    _EVENT_HANDLERS: ClassVar[dict[type[RuntimeEvent], Callable[..., None]]] = {
        SessionStartEvent: _accumulate_session_start,
        TurnStartEvent: _accumulate_turn_start,
        TurnCompleteEvent: _accumulate_turn_complete,
        FrameStartEvent: _accumulate_frame_start,
        FrameCompleteEvent: _accumulate_frame_complete,
        LLMCompleteEvent: _accumulate_llm_complete,
        ToolResultEvent: _accumulate_tool_result,
        ApprovalRequiredEvent: _accumulate_approval_required,
        ApprovalGrantedEvent: _accumulate_approval_resolved,
        ApprovalDeniedEvent: _accumulate_approval_resolved,
        ErrorEvent: _accumulate_error,
    }

    def _queue_stored_event(self, stored: StoredEvent) -> None:
        """Accumulate stored event for writing in prepare_batch."""
        key = f"{stored.turn_id}:{stored.frame_id or 'turn'}"
        pending = self._pending.setdefault(key, PendingEntity())
        pending.events.append(
            {
                "op": "event",
                "data": {
                    "trace_id": stored.trace_id,
                    "session_id": stored.session_id,
                    "turn_id": stored.turn_id,
                    "frame_id": stored.frame_id,
                    "seq": stored.seq,
                    "event_type": stored.type,
                    "data_json": stored.payload_json,
                    "timestamp_ms": stored.timestamp_ms,
                },
            }
        )

    def _generate_trace_id(self) -> str:
        """Generate a trace ID."""
        return f"tr_{datetime.now(UTC).timestamp() * 1000000:.0f}"

    @staticmethod
    def _merge_data(
        create: dict[str, Any] | None,
        update: dict[str, Any] | None,
        required_fields: set[str] | None = None,
    ) -> dict[str, Any]:
        """Merge create and update data, preserving required fields from create."""
        if not create and not update:
            return {}
        if not create:
            assert update is not None
            return update.get("data", {})
        if not update:
            assert create is not None
            return create.get("data", {})

        create_data = create.get("data", {})
        update_data = update.get("data", {})

        # Start with create data
        merged = dict(create_data)

        # Override with update values, but preserve required fields from create
        for key, value in update_data.items():
            # Skip if value is None (explicitly not set)
            if value is None:
                continue
            # For required fields, only override if update has a non-empty value
            if required_fields and key in required_fields:
                if value not in ("", 0):
                    merged[key] = value
            else:
                merged[key] = value

        return merged

    def prepare_batch(self) -> list[dict[str, Any]]:
        """Prepare a batch of operations from accumulated mutations.

        Returns:
            List of operation dicts ready for ExecutionStore.write_batch_direct()
        """
        if not self._pending:
            return []

        batch: list[dict[str, Any]] = []
        keys_to_clear: list[str] = []

        # Process turn keys first (they have session/turn creates)
        # then frame keys (they depend on session/turn existing)
        turn_keys = [k for k in self._pending.keys() if k.endswith(":turn")]
        frame_keys = [k for k in self._pending.keys() if not k.endswith(":turn")]

        for key in turn_keys + frame_keys:
            pending = self._pending[key]
            is_turn_key = key.endswith(":turn")

            self._emit_session_upsert(batch, pending)
            self._emit_turn_upsert(batch, pending)
            self._emit_snapshot_upsert(batch, pending)
            self._emit_frame_upsert(batch, pending)
            self._emit_frame_token_deltas(batch, pending)
            if is_turn_key:
                self._emit_turn_totals(batch, pending)
                self._emit_session_totals(batch, pending)

            self._emit_messages(batch, pending)
            self._emit_associations(batch, pending)
            self._emit_baselines(batch, pending)
            self._emit_events(batch, pending)

            # Determine if this key should be cleared *before* consuming the
            # update fields, since completion is detected via finished_at_ms.
            should_clear = self._should_clear_key(pending, is_turn_key)

            # Consume entity mutation data that was just emitted so incomplete
            # entities are not re-appended on the next flush. Completed keys are
            # removed below; incomplete keys keep their (now empty) entity fields
            # so subsequent updates can still be written as update-only upserts.
            pending.turn_create = None
            pending.turn_update = None
            pending.frame_create = None
            pending.frame_update = None

            if should_clear:
                keys_to_clear.append(key)

        # Clear only completed entries
        for key in keys_to_clear:
            del self._pending[key]

        return batch

    def _emit_session_upsert(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if pending.session_create or pending.session_update:
            data = self._merge_data(
                pending.session_create,
                pending.session_update,
                required_fields={
                    "session_id",
                    "user_input",
                    "status",
                    "started_at_ms",
                    "agent_mode",
                    "model_name",
                    "metadata_json",
                    "context_profile_json",
                },
            )
            batch.append({"op": "upsert_session", "data": data})
            pending.session_create = None
            pending.session_update = None

    def _emit_turn_upsert(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if pending.turn_create or pending.turn_update:
            data = self._merge_data(
                pending.turn_create,
                pending.turn_update,
                required_fields={
                    "turn_id",
                    "session_id",
                    "trace_id",
                    "turn_index",
                    "user_input",
                    "status",
                    "started_at_ms",
                },
            )
            batch.append({"op": "upsert_turn", "data": data})

    def _emit_snapshot_upsert(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if pending.snapshot_create or pending.snapshot_update:
            data = self._merge_data(
                pending.snapshot_create,
                pending.snapshot_update,
                required_fields={"snapshot_id", "turn_id", "content_hash"},
            )
            batch.append({"op": "upsert_context_snapshot", "data": data})
            pending.snapshot_create = None
            pending.snapshot_update = None

    def _emit_frame_upsert(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if pending.frame_create or pending.frame_update:
            data = self._merge_data(
                pending.frame_create,
                pending.frame_update,
                required_fields={
                    "frame_id",
                    "session_id",
                    "turn_id",
                    "trace_id",
                    "iteration",
                    "status",
                    "started_at_ms",
                },
            )
            batch.append({"op": "upsert_frame", "data": data})

    def _emit_frame_token_deltas(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        # Counter deltas are atomic and follow their owning row's upsert.
        if pending.frame_input_tokens_delta or pending.frame_output_tokens_delta:
            batch.append(
                {
                    "op": "increment_frame_tokens",
                    "data": {
                        "frame_id": pending.frame_id,
                        "input_tokens": pending.frame_input_tokens_delta,
                        "output_tokens": pending.frame_output_tokens_delta,
                    },
                }
            )
            pending.frame_input_tokens_delta = 0
            pending.frame_output_tokens_delta = 0

    def _emit_turn_totals(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if (
            pending.turn_input_tokens_delta
            or pending.turn_output_tokens_delta
            or pending.turn_frames_delta
        ):
            batch.append(
                {
                    "op": "increment_turn_totals",
                    "data": {
                        "turn_id": pending.turn_id,
                        "total_input_tokens": pending.turn_input_tokens_delta,
                        "total_output_tokens": pending.turn_output_tokens_delta,
                        "total_frames": pending.turn_frames_delta,
                    },
                }
            )
            pending.turn_input_tokens_delta = 0
            pending.turn_output_tokens_delta = 0
            pending.turn_frames_delta = 0

    def _emit_session_totals(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        if (
            pending.session_input_tokens_delta
            or pending.session_output_tokens_delta
            or pending.session_turns_delta
            or pending.session_frames_delta
        ):
            batch.append(
                {
                    "op": "increment_session_totals",
                    "data": {
                        "session_id": pending.session_id,
                        "total_input_tokens": pending.session_input_tokens_delta,
                        "total_output_tokens": pending.session_output_tokens_delta,
                        "total_turns": pending.session_turns_delta,
                        "total_frames": pending.session_frames_delta,
                    },
                }
            )
            pending.session_input_tokens_delta = 0
            pending.session_output_tokens_delta = 0
            pending.session_turns_delta = 0
            pending.session_frames_delta = 0

    def _emit_messages(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        # Messages (always append)
        for msg in pending.messages:
            batch.append({"op": "message", "data": msg})
        pending.messages.clear()

    def _emit_associations(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        # Associations
        for assoc in pending.associate_tools:
            batch.append({"op": "associate_tool", "data": assoc})
        for assoc in pending.associate_templates:
            batch.append({"op": "associate_template", "data": assoc})
        pending.associate_tools.clear()
        pending.associate_templates.clear()

    def _emit_baselines(
        self, batch: list[dict[str, Any]], pending: PendingEntity
    ) -> None:
        # File baseline upserts
        for baseline in pending.baseline_upserts:
            batch.append(baseline)
        pending.baseline_upserts.clear()

    def _emit_events(self, batch: list[dict[str, Any]], pending: PendingEntity) -> None:
        # Events (accumulated in _queue_stored_event)
        for evt in pending.events:
            batch.append(evt)

    @staticmethod
    def _should_clear_key(pending: PendingEntity, is_turn_key: bool) -> bool:
        # Determine if this key should be cleared
        # Turn keys are cleared when turn is complete (has turn_update with finished_at_ms)
        # Frame keys are cleared when frame is complete (has frame_update with finished_at_ms)
        if is_turn_key:
            return bool(
                pending.turn_update
                and pending.turn_update.get("data", {}).get("finished_at_ms")
            )
        return bool(
            pending.frame_update
            and pending.frame_update.get("data", {}).get("finished_at_ms")
        )

    def close(self) -> list[dict[str, Any]]:
        """Prepare final batch for any remaining pending operations."""
        return self.prepare_batch()

    def clear_flushed_events(self) -> None:
        """Clear event records that have just been made durable.

        Entity mutation data is consumed (cleared) by ``prepare_batch`` when its
        upsert is emitted, so incomplete entities are not re-emitted on every
        flush. Only the already-written event rows are dropped here, preventing
        ``close()`` from re-appending duplicate events.
        """
        for pending in self._pending.values():
            pending.events.clear()
