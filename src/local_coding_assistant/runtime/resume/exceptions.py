"""Resume flow exceptions."""

from __future__ import annotations

from local_coding_assistant.core.exceptions import LocalAssistantError


class ResumeValidationError(LocalAssistantError):
    """Raised when a paused turn cannot be safely resumed.

    Carries structured details (e.g. stale snapshot references) so callers
    can render exactly what is unsupported instead of a generic failure.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message, subsystem="resume")
        self.details = details or {}


class MissingContextReferencesError(LocalAssistantError):
    """Raised when pinned context references no longer resolve from store.

    Raised by ContextManager when resolving refs for a stored context
    (turn snapshot or session profile) whose tools/templates/repo snapshot
    have disappeared - the resumed prompt would silently lose context.
    """

    def __init__(self, message: str, missing_refs: list[str], context_id: str) -> None:
        super().__init__(message, subsystem="context")
        self.details = {"missing_refs": missing_refs, "context_id": context_id}
        self.missing_refs = missing_refs
        self.context_id = context_id
