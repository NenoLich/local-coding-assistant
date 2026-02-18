from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RunError(BaseModel):
    """Represents a runtime error surfaced to the CLI/reporting layer."""

    message: str
    kind: str | None = None


class RunMetrics(BaseModel):
    """Aggregated metrics for a run."""

    tokens_used: int | None = None
    total_latency_ms: float | None = None
    tool_calls: int | None = None


class RuntimeEvent(BaseModel):
    """A structured event emitted during runtime execution."""

    type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    """Normalized runtime report for CLI output and trace export."""

    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4()}")
    session_id: str | None = None
    mode: str = "regular"
    status: str = "success"
    final_answer: str | None = None
    message: str | None = None
    finish_reason: str | None = None
    models_used: list[str] = Field(default_factory=list)
    tokens_used: int | None = None
    iterations: int | None = None
    frames: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metrics: RunMetrics | None = None
    errors: list[RunError] = Field(default_factory=list)
    events: list[RuntimeEvent] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.model_fields

    def get(self, key: str, default: Any | None = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_fields.keys()

    def items(self):
        for key in self.model_fields:
            yield key, getattr(self, key)

    @field_validator("final_answer", mode="before")
    @classmethod
    def stringify_final_answer(cls, value: Any) -> Any:
        if value is not None:
            return str(value)
        return None

    @field_validator("message", mode="before")
    @classmethod
    def stringify_message(cls, value: Any) -> Any:
        if value is not None:
            return str(value)
        return None

    @classmethod
    def from_legacy(cls, payload: dict[str, Any]) -> RunReport:
        mode = "frame" if payload.get("frames") else "regular"
        final_answer = payload.get("final_answer") or payload.get("message")
        message = payload.get("message") or payload.get("final_answer")
        model_used = payload.get("model_used")

        return cls(
            run_id=payload.get("run_id", f"run_{uuid.uuid4()}"),
            session_id=payload.get("session_id"),
            mode=payload.get("mode", mode),
            status=payload.get("status", "success"),
            final_answer=final_answer,
            message=message,
            models_used=[model_used] if isinstance(model_used, str) else [],
            tokens_used=payload.get("tokens_used"),
            iterations=payload.get("iterations"),
            frames=payload.get("frames"),
            history=payload.get("history"),
            tool_calls=payload.get("tool_calls"),
        )
