"""Pydantic models for serve command API - re-exports from runtime contracts."""

# Re-export the approval response form from runtime contracts
# to avoid duplication
# Additional request/response models specific to the server API
from pydantic import BaseModel

from local_coding_assistant.runtime.events.contracts import ApprovalResponseForm


class SessionStatusResponse(BaseModel):
    """Session status response."""

    session_id: str
    turn_id: str | None = None
    trace_id: str | None = None
    status: str


class TurnStatusResponse(BaseModel):
    """Turn status response."""

    session_id: str
    turn_id: str | None = None
    trace_id: str | None = None
    status: str


class ResumeSessionRequest(BaseModel):
    """Request to resume a session."""

    user_input: str | None = None
    turn_id: str | None = None
    turn_index: int | None = None
    frame_id: str | None = None


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    user_input: str
    trace_id: str | None = None
    agent_mode: str = "frame_agent"
    model_name: str | None = None
    metadata_json: str | None = None


__all__ = [
    "ApprovalResponseForm",
    "CreateSessionRequest",
    "ResumeSessionRequest",
    "SessionStatusResponse",
    "TurnStatusResponse",
]
