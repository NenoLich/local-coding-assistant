"""Sharp pre-resume validation.

Validates only what restoring a paused turn can silently get wrong. The live
flow is not validated (a fresh snapshot is built from live state, so it is
consistent by construction) and dynamic context mode after reconstruction is
not validated (its refs are re-resolved from live config, not from the
snapshot).

Checks, in order of severity:

- **Snapshot schema version** - fatal: the snapshot was written by a newer
  (or incompatible) schema and cannot be interpreted.
- **Snapshot refs still exist** - fatal for tools/templates/repo (their
  absence would silently drop them from the resumed prompt), warning for
  memories (best-effort by nature).
- **Frame to resume** - fatal if an explicitly requested frame is missing;
  a missing/terminal/snapshot-mismatched frame degrades to starting a new
  frame with a warning (safe fallback).
- **Restored tool calls still supported** - warning: names referenced by the
  resumed frame's tool calls that are no longer exposed by the gateway. The
  gateway prepare-phase still enforces per-call; this is a pre-flight signal.
- **History covers the frame boundary** - warning: the session transcript is
  shorter than the frame's ``history_through_seq``.
- **Config support** - extension point for future checks (e.g. models or
  providers that no longer exist). Today it performs no checks.

Deliberately NOT validated: content-hash re-render (expensive, brittle, and
belongs to the future replay flow), turn-config vs current-config drift
(resume intentionally replays the stored config), and message-boundary
completeness (a store invariant: transcript messages commit in the same
transaction as their events).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextResolutionMode,
    ContextSnapshot,
    RuntimeContextConfig,
)
from local_coding_assistant.runtime.engine.frame import RuntimeFrame
from local_coding_assistant.storage.contracts import MissingRefs

from .exceptions import MissingContextReferencesError, ResumeValidationError

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context.manager import ContextManager
    from local_coding_assistant.storage import IExecutionStore

SUPPORTED_SNAPSHOT_SCHEMA_VERSION = 1

# ToolSetRef.version default; snapshots/profiles pinned before versioned
# tool sets existed carry this constant instead of a computed hash.
UNPINNED_VERSION = "1.0.0"

# Frame statuses the engine state machine can resume from.
RESUMABLE_FRAME_STATUSES = frozenset({"running_llm", "running_tools"})


@dataclass(slots=True)
class ResumeValidationInput:
    """Everything the validator needs to judge a resume."""

    snapshot: ContextSnapshot | None
    missing_refs: MissingRefs
    frame: RuntimeFrame | None = None
    context_config: RuntimeContextConfig | None = None
    exposed_tool_names: set[str] | None = None
    history_sequence: int = 0
    current_tool_set_version: str | None = None


@dataclass(slots=True)
class ResumeValidationResult:
    """Outcome of a resume validation pass."""

    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        """Fatal problems raise ResumeValidationError instead of landing here."""
        return True


def validate_snapshot_schema(snapshot: ContextSnapshot | None) -> list[str]:
    """Fail when the snapshot was written by an incompatible schema."""
    if snapshot is None:
        raise ResumeValidationError(
            "Turn has no context snapshot to resume from",
            details={"reason": "missing_snapshot"},
        )
    if snapshot.schema_version > SUPPORTED_SNAPSHOT_SCHEMA_VERSION:
        raise ResumeValidationError(
            f"Context snapshot schema version {snapshot.schema_version} is not "
            f"supported (max supported: {SUPPORTED_SNAPSHOT_SCHEMA_VERSION})",
            details={
                "schema_version": snapshot.schema_version,
                "supported": SUPPORTED_SNAPSHOT_SCHEMA_VERSION,
            },
        )
    return []


def validate_missing_refs(missing: MissingRefs) -> list[str]:
    """Fail on missing tool/template/repo refs; warn on missing memories."""
    fatal = {
        "tool_ids": missing.tool_ids,
        "template_ids": missing.template_ids,
        "repo_hashes": missing.repo_hashes,
    }
    fatal = {kind: ids for kind, ids in fatal.items() if ids}
    if fatal:
        raise ResumeValidationError(
            "Context snapshot references no longer exist in store; "
            "the pinned prompt context cannot be reconstructed",
            details={"missing_refs": fatal},
        )

    warnings: list[str] = []
    if missing.memory_ids:
        warnings.append(
            f"{len(missing.memory_ids)} pinned memory entr(ies) no longer exist "
            "and will be skipped"
        )
    return warnings


def validate_frame(frame: RuntimeFrame | None) -> list[str]:
    """Warn when the frame about to be resumed has nothing behind it."""
    if frame is None:
        return []
    warnings: list[str] = []
    if frame.status.value not in RESUMABLE_FRAME_STATUSES:
        warnings.append(
            f"Frame {frame.id} has terminal status {frame.status.value}; "
            "a new frame will be started instead"
        )
    if not frame.actions:
        warnings.append(
            f"Frame {frame.id} has no restored events; it will restart from "
            f"status {frame.status.value}"
        )
    return warnings


def validate_tool_support(
    frame: RuntimeFrame | None, exposed_tool_names: set[str] | None
) -> list[str]:
    """Warn when the frame's restored tool calls are no longer exposed."""
    if frame is None or exposed_tool_names is None:
        return []

    referenced = {
        tool_call.name
        for action in frame.actions
        if action.kind.value == "llm_message"
        for tool_call in action.tool_calls
    }
    unsupported = sorted(referenced - exposed_tool_names)
    if not unsupported:
        return []
    return [
        f"Tool(s) {', '.join(unsupported)} referenced by frame {frame.id} are "
        "no longer exposed; those calls will fail if re-executed"
    ]


def validate_tool_set_drift(
    pinned_version: str | None,
    current_version: str | None,
    tool_names: list[str] | None = None,
) -> list[str]:
    """Warn when the pinned tool set's schemas drifted from stored state.

    The version hash covers (tool id, name, argument schema), so a changed
    schema on a still-existing tool row is detected. Exposure itself is not
    checked here - a tool can be legitimately hidden without row changes.

    Legacy refs pinned before versioned tool sets existed (version "1.0.0")
    are skipped: there is no baseline to compare against.
    """
    if not pinned_version or pinned_version == UNPINNED_VERSION:
        return []
    if not current_version or pinned_version == current_version:
        return []
    label = f" ({', '.join(tool_names)})" if tool_names else ""
    return [
        f"Pinned tool set{label} drifted since it was pinned: version "
        f"{pinned_version} -> {current_version}; restored tool calls were "
        "made against the old schemas"
    ]


def validate_history_coverage(
    frame: RuntimeFrame | None, history_sequence: int
) -> list[str]:
    """Warn when the transcript is shorter than the frame's boundary."""
    if frame is None:
        return []
    if frame.history_through_seq > history_sequence:
        return [
            f"Session history ends at seq {history_sequence} but frame "
            f"{frame.id} was built through seq {frame.history_through_seq}; "
            "earlier context may be missing"
        ]
    return []


def validate_config_support(config: RuntimeContextConfig | None) -> list[str]:
    """Extension point for config-support checks.

    Returns warnings only. Intended for future checks such as a pinned LLM
    model that no longer exists among the configured providers.
    """
    return []


def validate_resume_context(data: ResumeValidationInput) -> ResumeValidationResult:
    """Run all pre-resume checks.

    Raises:
        ResumeValidationError: on a fatal problem (unsupported snapshot
            schema, missing tool/template/repo refs).
    """
    warnings: list[str] = []
    warnings.extend(validate_snapshot_schema(data.snapshot))
    warnings.extend(validate_missing_refs(data.missing_refs))
    warnings.extend(validate_frame(data.frame))
    warnings.extend(validate_tool_support(data.frame, data.exposed_tool_names))
    warnings.extend(
        validate_tool_set_drift(
            data.snapshot.tool_set_ref.version if data.snapshot else None,
            data.current_tool_set_version,
            data.snapshot.tool_set_ref.tool_names if data.snapshot else None,
        )
    )
    warnings.extend(validate_history_coverage(data.frame, data.history_sequence))
    warnings.extend(validate_config_support(data.context_config))
    return ResumeValidationResult(warnings=warnings)


async def validate_session_profile(
    *,
    context_manager: ContextManager,
    store: IExecutionStore,
    profile: ContextProfile,
) -> ResumeValidationResult:
    """Validate a stored session context profile before reusing it.

    A resumed-with-message turn (and branched sessions) re-resolve the stored
    profile's refs - the same reference fields a turn snapshot pins - so they
    need the same integrity checks: missing tool/template/repo refs are
    fatal (they would silently drop context from the new turn's prompt),
    tool schema drift is a warning.

    Dynamic mode skips these checks by design: its bundle is rebuilt from
    live config every turn, so no stored refs are involved.

    Raises:
        ResumeValidationError: on fatal problems (via MissingContextReferencesError
            from the ref-resolution backstop, or explicit checks below).
    """
    if context_manager.get_resolution_mode() != ContextResolutionMode.STATIC:
        return ResumeValidationResult(warnings=[])

    missing = await store.check_context_refs(
        tool_ids=profile.tool_set_ref.tool_ids,
        template_ids=profile.template_set_ref.template_ids,
        repo_hashes=[profile.repo_snapshot_ref.repo_snapshot_hash]
        if profile.repo_snapshot_ref
        else None,
    )
    warnings = validate_missing_refs_for_profile(profile, missing)

    # Recompute the tool set version over the stored rows the profile pins:
    # if a row's schema changed since the profile was built, warn.
    if (
        profile.tool_set_ref.version
        and profile.tool_set_ref.version != UNPINNED_VERSION
    ):
        refs = await context_manager.resolve_profile_refs(profile)
        current_version = context_manager.compute_tool_set_version(refs.tools)
        warnings.extend(
            validate_tool_set_drift(
                profile.tool_set_ref.version,
                current_version,
                profile.tool_set_ref.tool_names,
            )
        )

    return ResumeValidationResult(warnings=warnings)


def validate_missing_refs_for_profile(
    profile: ContextProfile, missing: MissingRefs
) -> list[str]:
    """Missing-ref verdicts for a stored session profile (tools/templates/repo fatal)."""
    fatal = {
        "tool_ids": missing.tool_ids,
        "template_ids": missing.template_ids,
        "repo_hashes": missing.repo_hashes,
    }
    fatal = {kind: ids for kind, ids in fatal.items() if ids}
    if fatal:
        raise MissingContextReferencesError(
            "Session context profile references artifacts that no longer "
            "exist in store; its prompt context cannot be reconstructed",
            missing_refs=[f"tool:{t}" for t in missing.tool_ids]
            + [f"template:{t}" for t in missing.template_ids]
            + [f"repo:{h}" for h in missing.repo_hashes],
            context_id=f"session_profile_v{profile.version}",
        )
    return []
