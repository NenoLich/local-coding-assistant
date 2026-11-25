"""Create file tool implementation.

Creates a new file with atomic operations and conflict detection.
Fails if file already exists. Returns detailed change record.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateFileInput(BaseModel):
    """Input model for create_file tool."""

    path: str = Field(description="Path to file relative to workspace root")
    content: str = Field(description="File content to write")
    overwrite: bool = Field(
        default=False, description="Overwrite file if it already exists"
    )


class CreateFileOutput(BaseModel):
    """Output model for create_file tool."""

    success: bool = Field(description="Whether file creation was successful")
    path: str = Field(description="Path to the created file")
    after_hash: str = Field(description="SHA256 hash after creation")
    after_size: int = Field(description="File size in bytes after creation")
    after_lines: int = Field(description="Line count after creation")
    error: str | None = Field(description="Error message if unsuccessful")

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal versioning)."""
        return {
            "success": self.success,
            "path": self.path,
            "error": self.error,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "CreateFileOutputLLM",
            success=(bool, ...),
            path=(str, ...),
            error=(str | None, None),
        )
        return llm_output.model_json_schema()


async def create_file_impl(
    backend: Any,
    input_model: CreateFileInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> CreateFileOutput:
    """Implementation of create_file tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Optional session ID for tracking
        turn_id: Optional turn ID for checkpoint tracking

    Returns:
        CreateFileOutput with change details
    """
    result_dict = await backend.create_file(
        path=input_model.path,
        content=input_model.content,
        overwrite=input_model.overwrite,
        session_id=session_id,
        turn_id=turn_id,
    )

    success = result_dict.get("success", False)
    path = result_dict.get("path", input_model.path)
    after_hash = result_dict.get("after_hash", "")
    after_size = result_dict.get("after_size", 0)
    after_lines = result_dict.get("after_lines", 0)
    error = result_dict.get("error", "Unknown error")

    return CreateFileOutput(
        success=success,
        path=path,
        after_hash=after_hash,
        after_size=after_size,
        after_lines=after_lines,
        error=error,
    )


def register_create_file(registry: Any) -> None:
    """Register create_file tool with the registry.

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
        compute_file_create_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="create_file",
            description="Creates a new file with the specified content. Fails if file already exists unless overwrite=true. Performs atomic write with conflict detection. Returns hash, size, and line count. Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.MEDIUM)
        .with_effects(OperationEffect.WRITE)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=compute_file_create_locks,
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
            factory=create_file_impl,
            input_model=CreateFileInput,
            output_model=CreateFileOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
