"""Replace exact text tool implementation.

Performs exact-match text replacement with version verification and atomic writes.
Requires exact match count, verifies version against the workspace baseline
(populated by reads), and returns detailed change record.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReplaceExactInput(BaseModel):
    """Input model for replace_exact tool."""

    path: str = Field(description="Path to file relative to workspace root")
    old_text: str = Field(description="Exact text to replace (must match exactly)")
    new_text: str = Field(description="New text to replace with")
    expected_match_count: int = Field(
        description="Expected number of old_text matches (must match exactly)"
    )


class ReplaceExactOutput(BaseModel):
    """Output model for replace_exact tool."""

    success: bool = Field(description="Whether replacement was successful")
    match_count: int = Field(description="Number of matches found and replaced")
    changed: bool = Field(description="Whether file content actually changed")
    diff: str | None = Field(description="Unified diff of changes (if successful)")
    before_hash: str = Field(description="SHA256 hash before replacement")
    after_hash: str = Field(description="SHA256 hash after replacement")
    before_size: int = Field(description="File size in bytes before replacement")
    after_size: int = Field(description="File size in bytes after replacement")
    before_lines: int = Field(description="Line count before replacement")
    after_lines: int = Field(description="Line count after replacement")
    error: str | None = Field(description="Error message if unsuccessful")
    conflict: dict[str, Any] | None = Field(
        default=None,
        description="Structured conflict report when the expected version is stale",
    )
    # Conflict guidance fields (populated when conflict detected)
    external_diff: str | None = Field(
        default=None, description="Diff showing what changed on disk since baseline"
    )
    proposed_diff: str | None = Field(
        default=None, description="Diff showing what the proposed edit would do"
    )
    recommendation: str | None = Field(
        default=None, description="Recommended action: re_read_file or retry_or_adapt"
    )

    def to_llm_dict(self) -> dict[str, Any]:
        """Return LLM-friendly dict (excludes internal versioning/conflict)."""
        base = {
            "success": self.success,
            "match_count": self.match_count,
            "changed": self.changed,
            "diff": self.diff,
            "error": self.error,
        }
        # Include conflict guidance for LLM decision making
        if self.conflict:
            base["external_diff"] = self.external_diff
            base["proposed_diff"] = self.proposed_diff
            base["recommendation"] = self.recommendation
        return base

    @classmethod
    def model_llm_schema(cls) -> dict[str, Any]:
        """Return LLM-friendly JSON schema."""
        from pydantic import create_model

        llm_output = create_model(
            "ReplaceExactOutputLLM",
            success=(bool, ...),
            match_count=(int, ...),
            changed=(bool, ...),
            diff=(str | None, None),
            error=(str | None, None),
        )
        return llm_output.model_json_schema()


async def replace_exact_impl(
    backend: Any,
    input_model: ReplaceExactInput,
    session_id: str,
    turn_id: str | None = None,
) -> ReplaceExactOutput:
    """Implementation of replace_exact tool.

    Args:
        backend: ExecutionBackend instance
        input_model: Validated input parameters
        session_id: Session ID for tracking
        turn_id: Optional turn ID for checkpoint tracking

    Returns:
        ReplaceExactOutput with change details
    """
    result_dict = await backend.replace_exact(
        path=input_model.path,
        old_text=input_model.old_text,
        new_text=input_model.new_text,
        expected_match_count=input_model.expected_match_count,
        session_id=session_id,
        turn_id=turn_id,
    )

    success = result_dict.get("success", False)
    match_count = result_dict.get("match_count", 0)
    before_hash = result_dict.get("before_hash", "")
    after_hash = result_dict.get("after_hash", "")
    before_size = result_dict.get("before_size", 0)
    after_size = result_dict.get("after_size", 0)
    before_lines = result_dict.get("before_lines", 0)
    after_lines = result_dict.get("after_lines", 0)
    error = result_dict.get("error", "Unknown error")
    conflict = result_dict.get("conflict")
    changed = result_dict.get("changed", False)
    diff = result_dict.get("diff")
    external_diff = result_dict.get("external_diff")
    proposed_diff = result_dict.get("proposed_diff")
    recommendation = result_dict.get("recommendation")

    return ReplaceExactOutput(
        success=success,
        match_count=match_count,
        changed=changed,
        diff=diff,
        before_hash=before_hash,
        after_hash=after_hash,
        before_size=before_size,
        after_size=after_size,
        before_lines=before_lines,
        after_lines=after_lines,
        error=error,
        conflict=conflict,
        external_diff=external_diff,
        proposed_diff=proposed_diff,
        recommendation=recommendation,
    )


def register_replace_exact(registry: Any) -> None:
    """Register replace_exact tool with the registry.

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
        compute_file_write_locks,
    )
    from local_coding_assistant.tools_platform.definitions import ToolBuilder

    builder = (
        ToolBuilder(
            tool_id="replace_exact",
            description="Replaces exact text matches in a file. \
            Verifies version against the workspace baseline and match count. \
            Performs atomic write with conflict detection. Returns before/after hashes and line counts. \
            Only within the working directory.",
        )
        .with_risk_class(ToolRiskClass.MEDIUM)
        .with_effects(OperationEffect.WRITE)
        .with_backends(BackendType.HOST, BackendType.DOCKER)
        .with_concurrency_scope(
            ConcurrencyScope(
                effect=OperationEffect.WRITE,
                resource_locks=compute_file_write_locks,
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
            factory=replace_exact_impl,
            input_model=ReplaceExactInput,
            output_model=ReplaceExactOutput,
        )
    )

    definition, implementation = builder.build_with_implementation()
    registry.register(definition, implementation)
