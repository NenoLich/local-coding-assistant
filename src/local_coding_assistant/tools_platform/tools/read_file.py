"""Read file tool implementation.

Reads file content with optional line range and byte limit.
Returns line-numbered content with continuation token for pagination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import (
    BackendType,
    ConcurrencyScope,
    FileObservation,
    OperationEffect,
    ToolRiskClass,
    compute_file_read_locks,
)


class ReadFileInput(BaseModel):
    """Input model for read_file tool."""

    path: str = Field(description="Path to file relative to workspace root")
    offset: int = Field(
        default=0, description="Start line (1-indexed, 0 means beginning of file)"
    )
    limit: int | None = Field(
        default=None, description="Maximum number of lines to read (None = all)"
    )


class ReadFileOutput(BaseModel):
    """Output model for read_file tool."""

    content: str = Field(description="File content with line numbers")
    total_lines: int | None = Field(description="Total line count when known")
    next_offset: int | None = Field(
        description="Next offset for pagination if truncated (1-indexed)"
    )
    truncated: bool = Field(description="Whether content was truncated")
    file_observation: FileObservation | None = Field(
        default=None, description="File observation for conflict detection"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal conflict detection)."""
        return {
            "content": self.content,
            "total_lines": self.total_lines,
            "next_offset": self.next_offset,
            "truncated": self.truncated,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "ReadFileOutputLLM",
            content=(str, ...),
            total_lines=(int | None, None),
            next_offset=(int | None, None),
            truncated=(bool, ...),
        )
        return llm_output.model_json_schema()


async def read_file_impl(
    backend: Any,
    input_model: ReadFileInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> ReadFileOutput:
    """Implementation of read_file tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Optional session ID for tracking
        turn_id: Optional turn ID for checkpoint tracking

    Returns:
        ReadFileOutput with file content
    """
    # Use backend's read_file method
    # offset is 1-indexed (0 = start), convert to 0-indexed for backend
    start_line = input_model.offset if input_model.offset > 0 else 1
    end_line = None
    if input_model.limit is not None:
        end_line = start_line + input_model.limit - 1

    content = await backend.read_file(
        path=Path(input_model.path),
        start_line=start_line,
        end_line=end_line,
        byte_limit=262144,  # 256KB internal limit
    )

    # Observe file for conflict detection
    file_observation = None
    if hasattr(backend, "observe_file"):
        try:
            result = await backend.observe_file(
                Path(input_model.path), session_id=session_id
            )
            if isinstance(result, FileObservation):
                file_observation = result
        except Exception:  # noqa: S110 - observation failure shouldn't break the read
            # Observation failure shouldn't break the read
            pass

    # Count total lines if reading entire file
    total_lines = None
    if input_model.offset == 0 and input_model.limit is None:
        lines = content.splitlines()
        total_lines = len(lines)

    # Add line numbers (1-indexed)
    lines_with_numbers = []
    start = start_line
    for idx, line in enumerate(content.splitlines(keepends=True), start=start):
        lines_with_numbers.append(f"{idx}\t{line}")

    numbered_content = "".join(lines_with_numbers)

    # Check if truncated (by internal byte limit on raw content)
    byte_limit = 262144
    truncated = len(content.encode("utf-8")) >= byte_limit

    # Build next_offset for byte-limit truncation or limit-based pagination,
    # pointing to the next line to read.
    next_offset: int | None = None
    if truncated or input_model.limit is not None:
        next_offset = start + len(content.splitlines())

    return ReadFileOutput(
        content=numbered_content,
        total_lines=total_lines,
        next_offset=next_offset,
        truncated=truncated,
        file_observation=file_observation,
    )


def register_read_file(registry: Any) -> None:
    """Register read_file tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        ConflictRule,
        LockLifetime,
        TimeoutProfile,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="read_file",
            description="Reads file content with optional line offset and limit. Returns line-numbered content (1-indexed). Max 256KB. Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.SAFE)
        .with_effects(OperationEffect.READ)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.READ,
                resource_locks=compute_file_read_locks,
                conflict_rule=ConflictRule.WAIT,
                lifetime=LockLifetime.EXECUTION,
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
            factory=read_file_impl,
            input_model=ReadFileInput,
            output_model=ReadFileOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
