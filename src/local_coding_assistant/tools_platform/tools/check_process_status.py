"""Check process status tool implementation.

Inspects a LOCCA-owned process by handle, returning lifecycle state,
return code, elapsed/idle time, last output, bounded output deltas,
and resource information.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    OperationEffect,
    ToolRiskClass,
)


class ResourceUsage(BaseModel):
    """Resource usage metrics for a process."""

    cpu_percent: float | None = Field(default=None, description="CPU usage percentage")
    memory_bytes: int | None = Field(default=None, description="Memory usage in bytes")
    disk_read_bytes: int | None = Field(default=None, description="Disk read bytes")
    disk_write_bytes: int | None = Field(default=None, description="Disk write bytes")
    network_sent_bytes: int | None = Field(
        default=None, description="Network sent bytes"
    )
    network_recv_bytes: int | None = Field(
        default=None, description="Network received bytes"
    )


class ProcessMetadata(BaseModel):
    """Additional process metadata."""

    command: list[str] | None = Field(
        default=None, description="Command that started the process"
    )
    working_directory: str | None = Field(default=None, description="Working directory")
    environment: dict[str, str] | None = Field(
        default=None, description="Environment variables"
    )


class CheckProcessStatusInput(BaseModel):
    """Input model for check_process_status tool."""

    handle_id: str = Field(
        description="Process handle ID from a previous run_command call"
    )


class ProcessStatusOutput(BaseModel):
    """Output model for check_process_status tool."""

    handle_id: str = Field(description="Process handle ID")
    state: str = Field(
        description="Process state (spawning, running, completed, failed, cancelled, timed_out)"
    )
    return_code: int | None = Field(description="Return code if completed")
    exit_signal: str | None = Field(
        description="Signal that terminated process (if applicable)"
    )
    os_pid: int | None = Field(description="OS process ID (for diagnostics only)")
    backend: str = Field(description="Backend that owns this process")
    start_time: str | None = Field(description="ISO timestamp when process was spawned")
    last_activity: str | None = Field(description="ISO timestamp of last output")
    elapsed_seconds: float | None = Field(
        description="Elapsed execution time in seconds"
    )
    idle_seconds: float | None = Field(
        description="Idle time since last output in seconds"
    )
    resource_usage: ResourceUsage = Field(description="Resource usage metrics")
    metadata: ProcessMetadata = Field(description="Additional process metadata")

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal diagnostics)."""
        return {
            "handle_id": self.handle_id,
            "state": self.state,
            "return_code": self.return_code,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "ProcessStatusOutputLLM",
            handle_id=(str, ...),
            state=(str, ...),
            return_code=(int | None, None),
            elapsed_seconds=(float | None, None),
        )
        return llm_output.model_json_schema()


async def check_process_status_impl(
    backend: Any,
    input_model: CheckProcessStatusInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> ProcessStatusOutput:
    """Implementation of check_process_status tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        ProcessStatusOutput with process status
    """
    # Use backend's check_process_status method
    status_data = await backend.check_process_status(input_model.handle_id)

    return ProcessStatusOutput(**status_data)


def register_check_process_status(registry: Any) -> None:
    """Register check_process_status tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        ConflictRule,
        LockLifetime,
        TimeoutProfile,
        compute_process_status_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="check_process_status",
            description="Inspects a LOCCA-owned process by handle ID. Returns lifecycle state, return code, elapsed/idle time, last output timestamp, and resource usage. Only works for processes started by run_command in the current session.",
        )
        .with_risk_class(ToolRiskClass.LOW)
        .with_effects(OperationEffect.READ)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.READ,
                resource_locks=compute_process_status_locks,
                conflict_rule=ConflictRule.WAIT,
                lifetime=LockLifetime.EXECUTION,
            )
        )
        .with_timeout_profile(
            TimeoutProfile(
                soft_timeout=timedelta(seconds=5),
                hard_timeout=timedelta(seconds=10),
                idle_timeout=timedelta(seconds=30),
            )
        )
        .with_implementation(
            factory=check_process_status_impl,
            input_model=CheckProcessStatusInput,
            output_model=ProcessStatusOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
