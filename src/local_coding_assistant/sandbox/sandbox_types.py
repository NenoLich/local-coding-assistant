"""Type definitions for the sandbox environment."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.core.telemetry_types import (
    ResourceMetric,
    ToolCallTrace,
)


class SandboxExecutionRequest(BaseModel):
    """Request to execute code in the sandbox."""

    code: str
    session_id: str
    timeout: int = 30
    env_vars: dict[str, str] = Field(default_factory=dict)
    persistence: bool = False


class ToolCallMetric(ToolCallTrace):
    """Backward-compatible alias for tool call metrics."""

    pass


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
