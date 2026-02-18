"""
Data types for partial response handlers.

This module defines the input and output data structures
for handling partial responses due to truncation or tool failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from local_coding_assistant.agent.llm.models import LLMResult
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.session import SessionState


class HandlerErrorType(str, Enum):
    PARSING_ERROR = "parsing_error"
    TRUNCATION = "truncation"
    TOOL_FAILURES = "tool_failures"
    CONTENT_ERROR = "content_error"


@dataclass
class HandlerContext:
    """Input data for partial response handlers."""

    session: SessionState
    llm_response: LLMResult | None = None
    reasoning: str | None = None
    reasoning_tokens: int | None = None
    current_iteration: int = 0
    max_attempts: int = 2
    attempt_count: int = 0
    error_type: HandlerErrorType | None = None
    error_message: str | None = None
    raw_response: str | None = None
    failed_tools: list[dict[str, Any]] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] | None = field(default_factory=list)
    handler_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerOutput:
    """Output data from partial response handlers."""

    status: ExecutionStatus
    template_path: (
        str | None
    )  # Template path for rendering (e.g., "handlers/parsing_retry.jinja2")
    adjusted_llm_options: dict[str, Any] | None
    should_retry: bool
    error_message: str | None
    handler_context: dict[str, Any] | None = None
    retry_strategy: str | None = None  # Retry strategy from tool error classifier
