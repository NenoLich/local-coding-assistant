"""Context resolution and management."""

from local_coding_assistant.runtime.context.manager import ContextManager
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextResolutionMode,
    ContextSnapshot,
    ContextSnapshotStatus,
    MemorySnapshotRef,
    MessageBoundary,
    RepoSnapshotRef,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.context.provider import MemoryProvider
from local_coding_assistant.runtime.context.resolution import (
    ReconstructionResult,
    ResolvedRefs,
    SessionContextBundle,
)

__all__ = [
    "ContextManager",
    "ContextProfile",
    "ContextResolutionMode",
    "ContextSnapshot",
    "ContextSnapshotStatus",
    "MemoryProvider",
    "MemorySnapshotRef",
    "MessageBoundary",
    "ReconstructionResult",
    "RepoSnapshotRef",
    "ResolvedRefs",
    "RuntimeContextConfig",
    "SessionContextBundle",
    "TemplateSetRef",
    "ToolSetRef",
]
