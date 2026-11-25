"""Context profile and snapshot contracts.

This module defines the core context contracts for prompt assembly and
session/turn context management:
- Reference types: ToolSetRef, TemplateSetRef, RepoSnapshotRef, MemorySnapshotRef, MessageBoundary
- Config snapshot: RuntimeContextConfig
- Resolution policy: ContextResolutionMode
- Context Profile (per-session baseline)
- Context Snapshot (per-turn immutable)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# === Reference Types ===


class ToolSetRef(BaseModel):
    """Reference to a specific tool set version."""

    tool_ids: list[int] = Field(default_factory=list, description="Global tool IDs")
    tool_names: list[str] = Field(
        default_factory=list, description="Tool names for human readability"
    )
    version: str = Field(default="1.0.0", description="Tool set version")


class TemplateSetRef(BaseModel):
    """Reference to a specific template set version."""

    template_ids: list[int] = Field(
        default_factory=list, description="Global template IDs"
    )
    template_paths: list[str] = Field(
        default_factory=list, description="Template paths for human readability"
    )
    version: str = Field(default="1.0.0", description="Template set version")


class RepoSnapshotRef(BaseModel):
    """Reference to repository state at snapshot time.

    Minimal reference - only the project root and content hash.
    Full data is stored in repo_snapshots table keyed by hash.
    """

    target_project_root: str = Field(..., description="Target project root path")
    repo_snapshot_hash: str = Field(..., description="Content hash of repo snapshot")


class MemorySnapshotRef(BaseModel):
    """Reference to memory state at snapshot time."""

    memory_ids: list[str] = Field(
        default_factory=list, description="Memory entry identifiers"
    )
    version: str = Field(default="1.0.0", description="Memory snapshot version")


class MessageBoundary(BaseModel):
    """Message sequence boundaries for history reconstruction."""

    session_id: str = Field(..., description="Session these messages belong to")
    start_seq: int = Field(..., description="Starting message sequence number")
    end_seq: int = Field(..., description="Ending message sequence number (inclusive)")


# === Config Snapshot ===


class RuntimeContextConfig(BaseModel):
    """Configuration snapshot for deterministic prompt assembly replay.

    Contains only the config relevant to prompt assembly (not execution limits).
    No custom dict - all fields are explicit.
    """

    llm_model: str | None = Field(default=None, description="LLM model name")
    llm_temperature: float | None = Field(default=None, description="LLM temperature")
    llm_max_tokens: int | None = Field(default=None, description="LLM max tokens")
    tool_call_mode: str | None = Field(default=None, description="Tool call mode")
    execution_mode: str | None = Field(default=None, description="Execution mode")
    agent_mode: bool | None = Field(default=None, description="Agent mode enabled")
    graph_mode: bool | None = Field(default=None, description="Graph mode enabled")
    streaming: bool | None = Field(default=None, description="Streaming enabled")
    sandbox_enabled: bool | None = Field(default=None, description="Sandbox enabled")
    # Environment / platform fields
    llm_provider: str | None = Field(
        default=None, description="Resolved LLM provider name for the configured model"
    )
    personality: str | None = Field(
        default=None, description="Agent personality override"
    )
    platform: str = Field(
        default="", description="Host platform identifier (sys.platform)"
    )
    is_git_repo: bool = Field(
        default=False, description="Whether the target project root is a git repository"
    )
    cwd: str = Field(default="", description="Current working directory")
    # AGENTS.md / prompt config fields (mirrored from config.prompt)
    agents_md_enabled: bool = Field(
        default=True, description="AGENTS.md discovery enabled"
    )
    agents_md_walk_up: bool = Field(
        default=True, description="Walk up from cwd to discover AGENTS.md"
    )
    agents_md_filenames: list[str] = Field(
        default_factory=lambda: ["AGENTS.md", "CLAUDE.md", "LOCCA.md"],
        description="AGENTS.md filenames to discover, closest to cwd first",
    )
    agents_md_global_paths: list[str] = Field(
        default_factory=lambda: [
            "~/.config/locca/AGENTS.md",
            "~/.claude/CLAUDE.md",
        ],
        description="Global AGENTS.md paths",
    )
    config_instructions: list[str] = Field(
        default_factory=list, description="Array of instruction strings or URLs"
    )
    # Explicit prompt template config fields (no custom dict)
    prompt_templates: dict[str, str] = Field(
        default_factory=dict, description="Prompt template paths"
    )
    trim_blocks: bool = Field(default=True, description="Jinja2 trim_blocks")
    lstrip_blocks: bool = Field(default=True, description="Jinja2 lstrip_blocks")


# === Resolution Policy (mirrored from config.schemas.RuntimeConfig) ===


class ContextResolutionMode(str, Enum):
    """Context resolution mode for session context management."""

    STATIC = "static"  # Create profile once, reuse across turns
    DYNAMIC = "dynamic"  # Re-resolve profile from live config each turn


# === Context Profile (per-session baseline) ===


class ContextProfile(BaseModel):
    """Session-level context baseline. Owned by SessionState & StoredSession.

    Represents the combined config + refs that form the starting point for turns.
    In static mode: created once, preserved across turns.
    In dynamic mode: updated each turn from live config + latest refs.
    """

    runtime_context_config: RuntimeContextConfig
    tool_set_ref: ToolSetRef
    template_set_ref: TemplateSetRef
    repo_snapshot_ref: RepoSnapshotRef | None = None
    memory_snapshot_ref: MemorySnapshotRef | None = None
    version: int = Field(default=1, description="Profile version, increments on update")
    updated_at_ms: int = Field(
        default=0, description="Last update time in UTC epoch milliseconds"
    )


# === Context Snapshot (per-turn immutable) ===


class ContextSnapshotStatus(str, Enum):
    """Status of a context snapshot for reconstruction purposes."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    CORRUPT = "corrupt"


class ContextSnapshot(BaseModel):
    """Immutable prompt assembly context for a specific turn.

    Captures refs at turn start + message boundary for history reconstruction.
    The raw prompt is NEVER stored here.
    Config (RuntimeContextConfig) lives in the Turn, but we also
    store it here for validation and TurnStartEvent propagation.
    """

    id: str = Field(..., description="Unique snapshot identifier")
    turn_id: str = Field(..., description="Owning turn identifier")
    schema_version: int = Field(
        default=1, description="Schema version for compatibility"
    )
    # Refs (copied from session's ContextProfile at turn creation, or resolved from store on resume)
    tool_set_ref: ToolSetRef = Field(default_factory=ToolSetRef)
    template_set_ref: TemplateSetRef = Field(default_factory=TemplateSetRef)
    repo_snapshot_ref: RepoSnapshotRef | None = None
    memory_snapshot_ref: MemorySnapshotRef | None = None
    # History boundary for this turn (messages available at turn start)
    message_boundary: MessageBoundary | None = None
    # Validation
    content_hash: str = Field(..., description="SHA256 hash of reconstructed prompt")
    payload_json: str | None = Field(
        default=None, description="Optional compact structured payload"
    )
    created_at_ms: int = Field(
        ..., description="Creation time in UTC epoch milliseconds"
    )

    def to_stored_dict(self) -> dict[str, Any]:
        """Convert to lightweight dict for storage references."""
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "schema_version": self.schema_version,
            "tool_set_ref": self.tool_set_ref.model_dump_json()
            if self.tool_set_ref.tool_ids
            else None,
            "template_set_ref": self.template_set_ref.model_dump_json()
            if self.template_set_ref.template_ids
            else None,
            "repo_snapshot_ref": self.repo_snapshot_ref.model_dump_json()
            if self.repo_snapshot_ref
            else None,
            "memory_snapshot_ref": self.memory_snapshot_ref.model_dump_json()
            if self.memory_snapshot_ref
            else None,
            "message_boundary_json": self.message_boundary.model_dump_json()
            if self.message_boundary
            else None,
            "content_hash": self.content_hash,
            "payload_json": self.payload_json,
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_stored_dict(cls, data: dict[str, Any]) -> ContextSnapshot:
        """Reconstruct from stored dict (may be incomplete if refs are missing)."""
        return cls(
            id=data["id"],
            turn_id=data["turn_id"],
            schema_version=data.get("schema_version", 1),
            tool_set_ref=ToolSetRef.model_validate_json(data["tool_set_ref"])
            if data.get("tool_set_ref")
            else ToolSetRef(),
            template_set_ref=TemplateSetRef.model_validate_json(
                data["template_set_ref"]
            )
            if data.get("template_set_ref")
            else TemplateSetRef(),
            repo_snapshot_ref=RepoSnapshotRef.model_validate_json(
                data["repo_snapshot_ref"]
            )
            if data.get("repo_snapshot_ref")
            else None,
            memory_snapshot_ref=MemorySnapshotRef.model_validate_json(
                data["memory_snapshot_ref"]
            )
            if data.get("memory_snapshot_ref")
            else None,
            message_boundary=MessageBoundary.model_validate_json(
                data["message_boundary_json"]
            )
            if data.get("message_boundary_json")
            else None,
            content_hash=data["content_hash"],
            payload_json=data.get("payload_json"),
            created_at_ms=data["created_at_ms"],
        )


__all__ = [
    "ContextProfile",
    "ContextResolutionMode",
    "ContextSnapshot",
    "ContextSnapshotStatus",
    "MemorySnapshotRef",
    "MessageBoundary",
    "RepoSnapshotRef",
    "RuntimeContextConfig",
    "TemplateSetRef",
    "ToolSetRef",
]
