"""Run command tool implementation.

Executes commands with argv-first execution, shell mode flag, working directory,
and resource policy (stdin disabled). Returns process handle for long-running
commands and streaming output chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    OperationEffect,
    ProcessState,
    ToolRiskClass,
)


class RunCommandInput(BaseModel):
    """Input model for run_command tool."""

    argv: list[str] = Field(
        description="Command and arguments as a list (argv-first execution)"
    )
    request_shell: bool = Field(
        default=False,
        description="Request shell execution (requires approval)",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory relative to workspace root",
    )
    request_network: bool = Field(
        default=False,
        description="Request network access (requires approval)",
    )


class RunCommandOutput(BaseModel):
    """Output model for run_command tool."""

    handle_id: str | None = Field(
        description="Process handle ID for long-running commands"
    )
    return_code: int | None = Field(description="Process return code if completed")
    state: str = Field(description="Process state (running, completed, failed, etc.)")
    stdout: str = Field(description="Collected stdout output")
    stderr: str = Field(description="Collected stderr output")
    truncated: bool = Field(description="Whether output was truncated")
    stdout_bytes: int = Field(default=0, description="Internal stdout byte count")
    stderr_bytes: int = Field(default=0, description="Internal stderr byte count")
    elapsed_seconds: float = Field(default=0.0, description="Internal elapsed seconds")

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal tracking fields)."""
        return {
            "handle_id": self.handle_id,
            "return_code": self.return_code,
            "state": self.state,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "truncated": self.truncated,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "RunCommandOutputLLM",
            handle_id=(str | None, None),
            return_code=(int | None, None),
            state=(str, ...),
            stdout=(str, ...),
            stderr=(str, ...),
            truncated=(bool, ...),
        )
        return llm_output.model_json_schema()


async def run_command_impl(
    backend: Any,
    input_model: RunCommandInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> RunCommandOutput:
    """Implementation of run_command tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        RunCommandOutput with execution results
    """
    import time

    # Resolve working directory
    cwd_path = None
    if input_model.cwd:
        cwd_path = Path(input_model.cwd)

    # Start timing
    start_time = time.time()

    # Collect output chunks
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    handle_id: str | None = None
    return_code: int | None = None
    final_state = ProcessState.RUNNING
    truncated = False
    stdout_bytes = 0
    stderr_bytes = 0

    # Execute command with streaming
    try:
        async for chunk in backend.execute_command(
            argv=input_model.argv,
            cwd=cwd_path,
            env=None,
            timeout_seconds=None,
        ):
            if chunk.channel == "stdout":
                stdout_chunks.append(chunk.payload)
                stdout_bytes = chunk.cumulative_bytes
            elif chunk.channel == "stderr":
                stderr_chunks.append(chunk.payload)
                stderr_bytes = chunk.cumulative_bytes
            elif chunk.channel == "status":
                # Final status chunk
                status_data = chunk.payload
                handle_id = status_data.get("handle_id")
                return_code = status_data.get("return_code")
                final_state = status_data.get("state", ProcessState.RUNNING)
                truncated = status_data.get("truncated", False)
                stdout_bytes = status_data.get("stdout_bytes", 0)
                stderr_bytes = status_data.get("stderr_bytes", 0)
    except FileNotFoundError:
        # Executable could not be resolved via PATH: report a failed result
        # rather than propagating the subprocess error to the caller.
        final_state = ProcessState.FAILED

    # Calculate elapsed time
    elapsed = time.time() - start_time

    # Combine output
    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)

    return RunCommandOutput(
        handle_id=handle_id,
        return_code=return_code,
        state=final_state,
        stdout=stdout,
        stderr=stderr,
        truncated=truncated,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        elapsed_seconds=elapsed,
    )


def register_run_command(registry: Any) -> None:
    """Register run_command tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        ConflictRule,
        LockLifetime,
        TimeoutProfile,
        compute_command_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="run_command",
            description="Executes a command with argv-first execution. \
            Argv should be a valid list. \
            Shell mode and network access require approval. Working directory must be within workspace. \
            Stdin is disabled. Returns process handle for long-running commands.",
        )
        .with_risk_class(ToolRiskClass.HIGH)
        .with_effects(OperationEffect.PROCESS, OperationEffect.WRITE)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.PROCESS,
                resource_locks=compute_command_locks,
                capacity_group="processes",
                conflict_rule=ConflictRule.WAIT,
                lifetime=LockLifetime.DETACHED_PROCESS,
            )
        )
        .with_timeout_profile(
            TimeoutProfile(
                soft_timeout=timedelta(seconds=10),
                hard_timeout=timedelta(seconds=300),
                idle_timeout=timedelta(seconds=30),
                heartbeat_interval=timedelta(seconds=5),
                cancellation_grace=timedelta(seconds=5),
            )
        )
        .with_implementation(
            factory=run_command_impl,
            input_model=RunCommandInput,
            output_model=RunCommandOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
