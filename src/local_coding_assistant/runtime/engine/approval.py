"""Approval coordination for tool execution pause/resume.

Provides in-memory approval tokens for pause/resume without DB persistence.
This replaces the legacy handler_context-based approval flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from local_coding_assistant.tools_platform.contracts import ApprovalResponse
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.engine.approval")


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """A pending approval request for tool execution."""

    token: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk_class: str
    session_id: str
    frame_id: str
    turn_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(init=False)
    timeout_seconds: int = 300  # 5 minutes default
    # Response fields (populated on resume)
    response: ApprovalResponse | None = None
    responded_at: datetime | None = None

    def __post_init__(self) -> None:
        """Set expiration time."""
        self.expires_at = self.created_at + timedelta(seconds=self.timeout_seconds)

    @property
    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        return datetime.now(UTC) > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if approval is still pending."""
        return self.status == ApprovalStatus.PENDING and not self.is_expired
