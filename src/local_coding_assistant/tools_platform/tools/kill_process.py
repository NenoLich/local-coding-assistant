"""Kill process tool implementation.

Gracefully cancels a LOCCA-owned process by handle, waits for a grace period,
terminates the process tree if needed, and returns the final state.
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


class KillProcessInput(BaseModel):
    """Input model for kill_process tool."""

    handle_id: str = Field(
        description="Process handle ID from a previous run_command call"
    )


class KillProcessOutput(BaseModel):
    """Output model for kill_process tool."""

    handle_id: str = Field(description="Process handle ID")
    state: str = Field(description="Final process state after cancellation attempt")
    return_code: int | None = Field(
        description="Return code if process had already completed"
    )
    exit_signal: str | None = Field(description="Signal used for termination")
    terminated: bool = Field(
        description="Whether the process was terminated by this call"
    )
    already_completed: bool = Field(
        description="Whether the process was already completed"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict."""
        return {
            "handle_id": self.handle_id,
            "state": self.state,
            "return_code": self.return_code,
            "exit_signal": self.exit_signal,
            "terminated": self.terminated,
            "already_completed": self.already_completed,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "KillProcessOutputLLM",
            handle_id=(str, ...),
            state=(str, ...),
            return_code=(int | None, None),
            exit_signal=(str | None, None),
            terminated=(bool, ...),
            already_completed=(bool, ...),
        )
        return llm_output.model_json_schema()


async def kill_process_impl(
    backend: Any,
    input_model: KillProcessInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> KillProcessOutput:
    """Implementation of kill_process tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        KillProcessOutput with cancellation results
    """
    from local_coding_assistant.tools_platform.contracts import ProcessState

    # Check current status first. An invalid handle raises ToolValidationError
    # from the backend, which propagates to the caller.
    status_data = await backend.check_process_status(input_model.handle_id)
    current_state = status_data["state"]
    already_completed = current_state in (
        ProcessState.COMPLETED.value,
        ProcessState.FAILED.value,
        ProcessState.CANCELLED.value,
        ProcessState.TIMED_OUT.value,
    )

    # Attempt cancellation (backend handles grace period internally)
    try:
        await backend.cancel_execution(input_model.handle_id)
        terminated = True
    except Exception:
        # Cancellation failed
        terminated = False

    # Check final status
    try:
        final_status_data = await backend.check_process_status(input_model.handle_id)
        final_state = final_status_data["state"]
        return_code = final_status_data.get("return_code")
        exit_signal = final_status_data.get("exit_signal")
    except Exception:
        # If we can't check final status, use current state
        final_state = current_state or "unknown"
        return_code = None
        exit_signal = None

    return KillProcessOutput(
        handle_id=input_model.handle_id,
        state=final_state,
        return_code=return_code,
        exit_signal=exit_signal,
        terminated=terminated,
        already_completed=already_completed,
    )


def register_kill_process(registry: Any) -> None:
    """Register kill_process tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        ConflictRule,
        LockLifetime,
        TimeoutProfile,
        compute_process_kill_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="kill_process",
            description="Gracefully cancels a LOCCA-owned process by handle ID. Waits for a grace period (5 seconds by default) before force termination. Returns final process state. Only works for processes started by run_command in the current session.",
        )
        .with_risk_class(ToolRiskClass.MEDIUM)
        .with_effects(OperationEffect.WRITE)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=compute_process_kill_locks,
                conflict_rule=ConflictRule.REJECT,
                lifetime=LockLifetime.EXECUTION,
            )
        )
        .with_timeout_profile(
            TimeoutProfile(
                soft_timeout=timedelta(seconds=5),
                hard_timeout=timedelta(seconds=15),
                idle_timeout=timedelta(seconds=30),
            )
        )
        .with_implementation(
            factory=kill_process_impl,
            input_model=KillProcessInput,
            output_model=KillProcessOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
