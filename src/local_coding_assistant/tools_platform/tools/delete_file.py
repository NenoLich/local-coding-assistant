"""Delete file tool implementation.

Deletes a file with atomic operations and conflict detection.
Verifies version against the workspace baseline (populated by reads).
Returns detailed change record.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DeleteFileInput(BaseModel):
    """Input model for delete_file tool."""

    path: str = Field(description="Path to file relative to workspace root")
    recursive: bool = Field(default=False, description="Delete directories recursively")
    dry_run: bool = Field(
        default=False, description="Preview deletion without executing"
    )


class DeleteFileOutput(BaseModel):
    """Output model for delete_file tool."""

    success: bool = Field(description="Whether file deletion was successful")
    path: str = Field(description="Path to the deleted file")
    error: str | None = Field(description="Error message if unsuccessful")
    would_delete: list[str] | None = Field(
        default=None, description="Paths that would be deleted (dry_run only)"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal versioning)."""
        return {
            "success": self.success,
            "path": self.path,
            "error": self.error,
            "would_delete": self.would_delete,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "DeleteFileOutputLLM",
            success=(bool, ...),
            path=(str, ...),
            error=(str | None, None),
            would_delete=(list[str] | None, None),
        )
        return llm_output.model_json_schema()


async def delete_file_impl(
    backend: Any,
    input_model: DeleteFileInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> DeleteFileOutput:
    """Implementation of delete_file tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Optional session ID for tracking
        turn_id: Optional turn ID for checkpoint tracking

    Returns:
        DeleteFileOutput with change details
    """
    result_dict = await backend.delete_file(
        path=input_model.path,
        recursive=input_model.recursive,
        dry_run=input_model.dry_run,
        session_id=session_id,
        turn_id=turn_id,
    )

    return DeleteFileOutput(
        success=result_dict.get("success", False),
        path=result_dict.get("path", input_model.path),
        error=result_dict.get("error", "Unknown error"),
        would_delete=result_dict.get("would_delete"),
    )


def register_delete_file(registry: Any) -> None:
    """Register delete_file tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        BackendType,
        ConcurrencyScope,
        ConflictRule,
        LockLifetime,
        OperationEffect,
        TimeoutProfile,
        ToolRiskClass,
        compute_file_delete_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="delete_file",
            description="Deletes a file or directory at the specified path. Supports recursive deletion and dry-run preview. Verifies version against the workspace baseline. Performs atomic deletion with conflict detection. Returns hash, size, and line count before deletion. Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.HIGH)
        .with_effects(OperationEffect.WRITE)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=compute_file_delete_locks,
                conflict_rule=ConflictRule.REQUIRE_FRESH,
                lifetime=LockLifetime.COMMIT,
            )
        )
        .with_timeout_profile(
            TimeoutProfile(
                soft_timeout=timedelta(seconds=5),
                hard_timeout=timedelta(seconds=30),
                idle_timeout=timedelta(seconds=30),
            )
        )
        .with_implementation(
            factory=delete_file_impl,
            input_model=DeleteFileInput,
            output_model=DeleteFileOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
