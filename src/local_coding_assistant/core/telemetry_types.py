"""Shared telemetry models for tool calls and sandbox execution."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FileChangeType(str, Enum):
    """Types of file change events."""

    MODIFIED = "modified"
    CREATED = "created"
    DELETED = "deleted"


class FileChange(BaseModel):
    """Represents a file change with path and change type."""

    path: str
    change_type: FileChangeType


class ResourceType(str, Enum):
    """Type of resource being measured."""

    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    CUSTOM = "custom"


class ResourceMetric(BaseModel):
    """Base class for resource metrics."""

    type: ResourceType
    name: str
    value: float | int | dict[str, Any]
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallTrace(BaseModel):
    """Unified representation of a single tool call trace."""

    call_id: str
    tool_name: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    success: bool = False
    error: str | None = None
    input: dict[str, Any] | None = None
    output: Any | None = None
    resource_metrics: list[ResourceMetric] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_call_id: str | None = None
    child_call_ids: list[str] = Field(
        default_factory=list
    )  # Add child relationship tracking
    execution_mode: str | None = None
    source: str | None = None


class ExecutionEnvelope(BaseModel):
    """Execution wrapper metadata for sandbox runs."""

    tool_name: str
    session_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: float | None = None
    success: bool = False
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None
    file_changes: list[FileChange] = Field(default_factory=list)
    return_code: int | None = None
    system_metrics: list[ResourceMetric] = Field(default_factory=list)


class PresentationOutput(BaseModel):
    """User-facing output derived from tool execution."""

    final_answer: Any | None = None
    format: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
