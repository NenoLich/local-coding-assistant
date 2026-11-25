"""Resume plan construction.

Single entry point the RuntimeManager calls before handing control to
``FrameAgent.resume_turn``: loads the resume batch from the store, rebuilds
the turn/frame, validates the restored context, and returns a typed plan.

The planner is deliberately shaped to the actual resume flow (the session is
already cached, one turn and one frame matter). Reconstructing the full
session graph is intentionally not part of this flow - that shape belongs to
the future replay flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from local_coding_assistant.runtime.engine.frame import RuntimeFrame, build_frame

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import SessionState
    from local_coding_assistant.runtime.session.session_manager import SessionManager
    from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.storage.contracts import ResumeBatch

from .exceptions import ResumeValidationError
from .validator import (
    RESUMABLE_FRAME_STATUSES,
    ResumeValidationInput,
    validate_resume_context,
    validate_tool_support,
)

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context import ContextManager


@dataclass(slots=True)
class ResumePlan:
    """A validated, ready-to-execute resume: entities plus soft warnings."""

    session: SessionState
    turn: RuntimeTurn
    frame: RuntimeFrame | None
    warnings: list[str] = field(default_factory=list)


def select_resumable_frame(
    batch: ResumeBatch, frame_id: str | None
) -> tuple[RuntimeFrame | None, list[str]]:
    """Pick the frame to resume from the batch.

    Args:
        batch: Loaded resume batch (frames + the resumed frame's events)
        frame_id: Explicit frame requested, or None for the turn's last frame

    Returns:
        Tuple of (frame to resume or None to start a new frame, warnings)

    Raises:
        ResumeValidationError: If an explicitly requested frame is missing
            from the turn, or is not in a resumable state.
    """
    if frame_id is not None:
        stored = next((f for f in batch.frames if f.id == frame_id), None)
        if stored is None:
            raise ResumeValidationError(
                f"Frame {frame_id} not found in turn {batch.turn.id}",
                details={"frame_id": frame_id, "turn_id": batch.turn.id},
            )
        if stored.status not in RESUMABLE_FRAME_STATUSES:
            raise ResumeValidationError(
                f"Frame {frame_id} has terminal status {stored.status} and "
                "cannot be resumed",
                details={"frame_id": frame_id, "status": stored.status},
            )
    else:
        stored = batch.frames[-1] if batch.frames else None

    if stored is None:
        return None, []

    if stored.status not in RESUMABLE_FRAME_STATUSES:
        return None, [
            f"Last frame {stored.id} has terminal status {stored.status}; "
            "a new frame will be started"
        ]

    snapshot_id = batch.snapshot.id if batch.snapshot else None
    if (
        stored.context_snapshot_id
        and snapshot_id
        and stored.context_snapshot_id != snapshot_id
    ):
        return None, [
            f"Frame {stored.id} was built against context snapshot "
            f"{stored.context_snapshot_id}, not the turn snapshot "
            f"{snapshot_id}; a new frame will be started"
        ]

    frame = build_frame(stored, batch.frame_events.get(stored.id, []))
    return frame, []


async def build_resume_plan(
    *,
    store: IExecutionStore,
    context_manager: ContextManager,
    session_manager: SessionManager,
    session_id: str,
    turn_id: str | None = None,
    turn_index: int | None = None,
    frame_id: str | None = None,
    exposed_tool_names: set[str] | None = None,
) -> ResumePlan:
    """Load and validate everything needed to resume an interrupted turn.

    Refreshes the pinned context bundle from the turn's snapshot (raising if
    the snapshot's tool/template/repo refs no longer resolve) and selects the
    frame to resume.

    Raises:
        ResumeValidationError: If the session/turn/frame cannot be found or
            the restored context is stale.
    """
    session = session_manager.get_cached(session_id)
    if session is None:
        raise ResumeValidationError(
            f"Session {session_id} is not loaded in cache; cannot resume",
            details={"session_id": session_id},
        )

    batch = await store.load_resume_batch(
        session_id, turn_id=turn_id, turn_index=turn_index, frame_id=frame_id
    )
    if batch is None:
        raise ResumeValidationError(
            f"Session or turn not found for resume (session={session_id}, "
            f"turn_id={turn_id}, turn_index={turn_index})",
            details={"session_id": session_id, "turn_id": turn_id},
        )

    if batch.session.status != "running":
        raise ResumeValidationError(
            f"Session {session_id} is idle (no interrupted turn). "
            "Use resume_session_with_message() to continue with a new message.",
            details={"session_id": session_id, "status": batch.session.status},
        )

    turn = RuntimeTurn.from_batch(batch)

    frame, frame_warnings = select_resumable_frame(batch, frame_id)
    if frame_id is not None and frame is None:
        raise ResumeValidationError(
            f"Frame {frame_id} cannot be resumed for turn {turn.turn_id}",
            details={"frame_id": frame_id},
        )

    # Pre-validation (no extra I/O): fatal integrity checks first, so stale
    # snapshot refs surface as a sharp ResumeValidationError with structured
    # details instead of a lower-level resolution failure.
    validate_resume_context(
        ResumeValidationInput(
            snapshot=turn.context_snapshot, missing_refs=batch.missing_refs
        )
    )

    # Refresh the pinned bundle from the turn snapshot. The context manager
    # re-checks ref resolution as a backstop for refs vanishing mid-turn.
    bundle = await context_manager.update_bundle_cache_from_snapshot(
        session_id=session.id,
        snapshot=turn.context_snapshot,
        config_snapshot=turn.context_config,
    )

    # Recompute the pinned tool set's version over the stored rows to detect
    # schema drift since the snapshot was created.
    current_tool_set_version = context_manager.compute_tool_set_version(
        bundle.refs.tools
    )

    result = validate_resume_context(
        ResumeValidationInput(
            snapshot=turn.context_snapshot,
            missing_refs=batch.missing_refs,
            frame=frame,
            context_config=turn.context_config,
            exposed_tool_names=exposed_tool_names,
            history_sequence=session.history_sequence,
            current_tool_set_version=current_tool_set_version,
        )
    )

    return ResumePlan(
        session=session,
        turn=turn,
        frame=frame,
        warnings=[*frame_warnings, *result.warnings],
    )


async def build_approval_resume_plan(
    *,
    store: IExecutionStore,
    session: SessionState,
    turn: RuntimeTurn,
    frame_id: str,
    exposed_tool_names: set[str] | None = None,
) -> ResumePlan:
    """Rebuild the paused frame for an approval response.

    Approvals are a cache-only concept: the cached RuntimeTurn (with its
    pending approvals) and the session's context entities were resolved live
    in this process, so snapshot ref- and schema-validation is deliberately
    skipped here. The only live check that matters is whether the tool calls
    the paused frame will re-execute are still exposed.
    """
    batch = await store.load_resume_batch(
        session.id, turn_id=turn.turn_id, frame_id=frame_id
    )
    if batch is None:
        raise ResumeValidationError(
            f"Session {session.id} not found for approval response",
            details={"session_id": session.id},
        )

    frame, frame_warnings = select_resumable_frame(batch, frame_id)
    if frame is None:
        raise ResumeValidationError(
            f"Paused frame {frame_id} cannot be resumed for turn {turn.turn_id}",
            details={"frame_id": frame_id, "turn_id": turn.turn_id},
        )

    warnings = list(frame_warnings)
    warnings.extend(validate_tool_support(frame, exposed_tool_names))

    return ResumePlan(
        session=session,
        turn=turn,
        frame=frame,
        warnings=warnings,
    )
