"""Config-driven models for RuntimeLimits.

These models are serializable for resume/replay and can be stored in the turn config.
Continuation policy configuration is now managed through config.schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RuntimeLimitsConfig(BaseModel):
    """Combined runtime limits configuration.

    Uses ContinuationPolicyConfig from config.schemas for dynamic resolution.
    """

    continuation_policy: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeLimitsConfig:
        return cls.model_validate(data)


class LimitViolationReason(str, Enum):
    """Structured reason for limit violations - replaces string parsing."""

    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"  # noqa: S105
    DEADLINE_EXCEEDED = "deadline_exceeded"
    LOOP_DETECTED = "loop_detected"
    FRAME_LIMIT_REACHED = "frame_limit_reached"
    GENERATION_RETRY_EXHAUSTED = "generation_retry_exhausted"
    COMPACTION_RETRY_EXHAUSTED = "compaction_retry_exhausted"


@dataclass
class LimitCheckResult:
    """Structured result of limit checking - replaces (bool, str) tuple."""

    within_limits: bool
    reason: LimitViolationReason | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls) -> LimitCheckResult:
        return cls(within_limits=True)

    @classmethod
    def violated(
        cls,
        reason: LimitViolationReason,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LimitCheckResult:
        return cls(
            within_limits=False,
            reason=reason,
            detail=detail,
            metadata=metadata or {},
        )
