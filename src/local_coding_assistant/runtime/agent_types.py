from __future__ import annotations

from pydantic import BaseModel

from local_coding_assistant.runtime.session import SessionState


class AgentRequest(BaseModel):
    """Standardized request contract for agent execution.

    Encapsulates all per-request parameters for agent mode execution,
    separating them from global configuration defaults.
    """

    user_input: str
    session: SessionState
    agent_mode: str | None = None
    model_override: str | None = None
    temperature_override: float | None = None
    max_tokens_override: int | None = None
    tool_call_mode_override: str | None = None
    sandbox_session_override: str | None = None
    streaming: bool | None = None
    max_iterations: int | None = None
