"""Match paths tool implementation.

Matches files using glob patterns from a declared workspace root,
with hidden, ignored, file-type, and result-limit controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MatchPathsInput(BaseModel):
    """Input model for match_paths tool."""

    pattern: str = Field(description="Glob pattern to match files")
    path: str = Field(default=".", description="Search path relative to workspace root")
    include_hidden: bool = Field(
        default=False, description="Include hidden files/directories"
    )
    include_ignored: bool = Field(
        default=False, description="Include git-ignored files"
    )
    extension: str | None = Field(
        default=None, description="Filter by file extension without dot (e.g., 'py')"
    )
    max_results: int = Field(
        default=200, description="Maximum number of results to return"
    )
    offset: int = Field(default=0, description="Number of results to skip (pagination)")


class MatchPathResult(BaseModel):
    """A single matched path result."""

    path: str = Field(description="File path relative to workspace")
    type: str = Field(description="Type: 'file' or 'directory'")
    size: int | None = Field(
        default=None, description="File size in bytes (files only)"
    )


class MatchPathsOutput(BaseModel):
    """Output model for match_paths tool."""

    matches: list[MatchPathResult] = Field(description="List of matched paths")
    total_matches: int = Field(description="Total number of matches found")
    truncated: bool = Field(description="Whether results were truncated")

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict."""
        return {
            "matches": [m.model_dump() for m in self.matches],
            "total_matches": self.total_matches,
            "truncated": self.truncated,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "MatchPathsOutputLLM",
            matches=(list[MatchPathResult], ...),
            total_matches=(int, ...),
            truncated=(bool, ...),
        )
        return llm_output.model_json_schema()


async def match_paths_impl(
    backend: Any,
    input_model: MatchPathsInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> MatchPathsOutput:
    """Implementation of match_paths tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        MatchPathsOutput with matched paths
    """
    result = await backend.match_paths(
        pattern=input_model.pattern,
        path=Path(input_model.path),
        include_hidden=input_model.include_hidden,
        include_ignored=input_model.include_ignored,
        file_type=input_model.extension,
        max_results=input_model.max_results,
    )

    # Apply offset for pagination
    matches = result.matches
    if input_model.offset > 0:
        matches = matches[input_model.offset :]

    total_matches = result.total_matches
    truncated = result.truncated

    return MatchPathsOutput(
        matches=matches,
        total_matches=total_matches,
        truncated=truncated,
    )


def register_match_paths(registry: Any) -> None:
    """Register match_paths tool with the registry.

    Args:
        registry: ToolRegistry instance
    """
    from datetime import timedelta

    from local_coding_assistant.tools_platform.contracts import (
        BackendType,
        ConcurrencyScope,
        ConflictRule,
        LockLifetime,
        LockMode,
        OperationEffect,
        ResourceLock,
        TimeoutProfile,
        ToolRiskClass,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="match_paths",
            description="Matches files using glob patterns from workspace root. Returns file paths with type and size. Max 200 results. Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.SAFE)
        .with_effects(OperationEffect.READ)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.READ,
                resource_locks=[ResourceLock(key="workspace", mode=LockMode.SHARED)],
                capacity_group="search_reads",
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
            factory=match_paths_impl,
            input_model=MatchPathsInput,
            output_model=MatchPathsOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
