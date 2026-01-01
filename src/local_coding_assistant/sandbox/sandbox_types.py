"""Type definitions for the sandbox environment."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SandboxExecutionRequest(BaseModel):
    """Request to execute code in the sandbox."""

    code: str
    session_id: str
    timeout: int = 30
    env_vars: dict[str, str] = Field(default_factory=dict)
    persistence: bool = False


class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"


class ResourceMetric(BaseModel):
    """Base class for resource metrics."""

    type: ResourceType
    name: str
    value: float | int | dict[str, Any]
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCallMetric(BaseModel):
    """Metrics for a single tool call."""

    tool_name: str
    call_id: str
    start_time: datetime
    end_time: datetime
    duration: float  # in seconds
    success: bool
    error: str | None = None
    resource_metrics: list[ResourceMetric] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SandboxExecutionResponse(BaseModel):
    """Response from sandbox execution with detailed resource metrics."""

    success: bool = False
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration: float = 0.0  # Total execution time in seconds
    files_created: list[str] = Field(default_factory=list)
    files_modified: list[str] = Field(default_factory=list)
    return_code: int = 0
    final_answer: Any = Field(
        None,
        description="If set, contains the final answer that should be returned to the user",
    )

    # New resource tracking fields
    tool_calls: list[ToolCallMetric] = Field(
        default_factory=list,
        description="Detailed metrics for each tool call during execution",
    )
    system_metrics: list[ResourceMetric] = Field(
        default_factory=list, description="System-level resource metrics"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the execution started",
    )
    end_time: datetime | None = Field(None, description="When the execution completed")

    def add_tool_call_metric(self, tool_call: ToolCallMetric) -> None:
        """Add a tool call metric to the response."""
        self.tool_calls.append(tool_call)

    def add_system_metric(self, metric: ResourceMetric) -> None:
        """Add a system-level resource metric."""
        self.system_metrics.append(metric)

    def finalize(self) -> None:
        """Finalize the response by setting end time and calculating duration."""
        self.end_time = datetime.now(UTC)
        self.duration = (self.end_time - self.start_time).total_seconds()
