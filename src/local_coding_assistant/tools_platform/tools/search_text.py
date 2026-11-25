"""Search text tool implementation.

Searches for literal or regex patterns in files using ripgrep when available,
with a Python fallback for identical bounded output.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.tools_platform.contracts import FileObservation


class PatternType(str, Enum):
    """Pattern type for search."""

    LITERAL = "literal"
    REGEX = "regex"


class SearchTextInput(BaseModel):
    """Input model for search_text tool."""

    query: str = Field(description="Search query (literal or regex)")
    path: str = Field(default=".", description="Search path relative to workspace root")
    pattern: PatternType = Field(
        default=PatternType.LITERAL, description="Pattern type: literal or regex"
    )
    includes: list[str] = Field(
        default_factory=list, description="Glob patterns to include"
    )
    excludes: list[str] = Field(
        default_factory=list, description="Glob patterns to exclude"
    )
    case_sensitive: bool = Field(default=False, description="Case-sensitive search")
    context_lines: int = Field(
        default=2, description="Number of context lines before/after match"
    )
    max_results: int = Field(
        default=200, description="Maximum number of matches to return"
    )
    offset: int = Field(default=0, description="Number of matches to skip (pagination)")
    include_gitignored: bool = Field(
        default=False, description="Include git-ignored files"
    )


class SearchTextMatch(BaseModel):
    """A single search match result."""

    path: str = Field(description="File path relative to workspace")
    line_number: int = Field(description="Line number of match")
    column: int = Field(description="Column number of match")
    line_content: str = Field(description="Line containing the match")
    context_before: list[str] = Field(
        default_factory=list, description="Lines before match"
    )
    context_after: list[str] = Field(
        default_factory=list, description="Lines after match"
    )


class SearchTextOutput(BaseModel):
    """Output model for search_text tool."""

    matches: list[SearchTextMatch] = Field(description="List of matches")
    total_matches: int = Field(description="Total number of matches found")
    truncated: bool = Field(description="Whether results were truncated")
    next_offset: int | None = Field(
        description="Next offset for pagination if truncated"
    )
    file_observations: list[FileObservation] = Field(
        default_factory=list, description="File observations for conflict detection"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal conflict detection)."""
        return {
            "matches": [m.model_dump() for m in self.matches],
            "total_matches": self.total_matches,
            "truncated": self.truncated,
            "next_offset": self.next_offset,
        }

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "SearchTextOutputLLM",
            matches=(list[SearchTextMatch], ...),
            total_matches=(int, ...),
            truncated=(bool, ...),
            next_offset=(int | None, None),
        )
        return llm_output.model_json_schema()


async def search_text_impl(
    backend: Any,
    input_model: SearchTextInput,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> SearchTextOutput:
    """Implementation of search_text tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for conflict detection
        turn_id: Turn ID for conflict detection

    Returns:
        SearchTextOutput with matches
    """
    result = await backend.search_text(
        query=input_model.query,
        path=Path(input_model.path),
        pattern=input_model.pattern.value,
        includes=input_model.includes,
        excludes=input_model.excludes,
        case_sensitive=input_model.case_sensitive,
        context_lines=input_model.context_lines,
        max_matches=input_model.max_results,
        max_bytes=262144,  # Internal byte limit
        include_ignored=input_model.include_gitignored,
    )

    # Skip already-returned matches for pagination
    matches = result.matches
    if input_model.offset > 0:
        matches = matches[input_model.offset :]

    # Observe matched files for conflict detection
    file_observations: list[FileObservation] = []
    if hasattr(backend, "observe_file") and hasattr(result, "matches"):
        seen_paths: set[str] = set()
        for match in result.matches:
            if match.path not in seen_paths:
                seen_paths.add(match.path)
                try:
                    obs = await backend.observe_file(
                        Path(match.path), session_id=session_id
                    )
                    if isinstance(obs, FileObservation):
                        file_observations.append(obs)
                except Exception:  # noqa: S110 - observation failure shouldn't break the read
                    pass

    # Build next_offset when truncated
    next_offset: int | None = None
    if result.truncated:
        next_offset = input_model.offset + len(matches)

    return SearchTextOutput(
        matches=matches,
        total_matches=result.total_matches,
        truncated=result.truncated,
        next_offset=next_offset,
        file_observations=file_observations,
    )


def register_search_text(registry: Any) -> None:
    """Register search_text tool with the registry.

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
            tool_id="search_text",
            description="Searches for literal or regex patterns in files. Returns matches with file paths, line numbers, and context. Max 200 results, 256KB output. Only within the working directory.",
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
                soft_timeout=timedelta(seconds=10),
                hard_timeout=timedelta(seconds=60),
                idle_timeout=timedelta(seconds=30),
            )
        )
        .with_implementation(
            factory=search_text_impl,
            input_model=SearchTextInput,
            output_model=SearchTextOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
