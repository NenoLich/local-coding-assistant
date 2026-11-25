"""Context resolution contracts for runtime use.

This module defines the resolved references and bundle types used during
runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from local_coding_assistant.repository import ProjectInfo, RepoMapData
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextSnapshotStatus,
)

# === Resolved References (for runtime use) ===


@dataclass
class ResolvedRefs:
    """Fully resolved snapshot references, cached per turn/session bundle.

    Contains the fully resolved data for tools, templates, repo, and memory.
    These are stable, versioned references that can be shared across turns
    in static mode.
    """

    tools: list[dict[str, Any]]
    templates: dict[str, str]
    repo_map_data: RepoMapData | None
    project_info: ProjectInfo | None
    target_project_root: str
    memories: list[str]
    template_int_ids: list[int]
    # Identity for cache invalidation
    tool_set_version: str = ""
    template_set_version: str = ""
    repo_snapshot_hash: str = ""
    memory_snapshot_version: str = ""


@dataclass
class SessionContextBundle:
    """Complete resolved context bundle for a session.

    In static mode, this is reused across turns. In dynamic mode,
    a new bundle is created per turn. Contains the session's ContextProfile
    plus the resolved refs.
    """

    profile: ContextProfile
    refs: ResolvedRefs
    created_at_ms: int = field(
        default_factory=lambda: int(
            __import__("datetime").datetime.now(__import__("datetime").UTC).timestamp()
            * 1000
        )
    )


# === Reconstruction Result ===


@dataclass
class ReconstructionResult:
    """Result of a reconstruction operation."""

    status: ContextSnapshotStatus
    snapshot: Any | None  # ContextSnapshot
    profile: ContextProfile | None
    missing_refs: list[str]
    compatibility_warnings: list[str]

    @property
    def is_complete(self) -> bool:
        return self.status == ContextSnapshotStatus.COMPLETE

    @property
    def is_incomplete(self) -> bool:
        return self.status == ContextSnapshotStatus.INCOMPLETE

    @property
    def is_corrupt(self) -> bool:
        return self.status == ContextSnapshotStatus.CORRUPT


__all__ = [
    "ReconstructionResult",
    "ResolvedRefs",
    "SessionContextBundle",
]
