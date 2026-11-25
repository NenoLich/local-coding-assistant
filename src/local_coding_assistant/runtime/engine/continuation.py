"""FrameContinuation - explicit frame-to-frame continuation context.

Each frame produces a continuation for the NEXT frame.
Contains ONLY directives for the next frame (compaction, retry, approval).
Transcript messages are never built here: they are projected independently
from the runtime event stream (see MessageService.build_message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from local_coding_assistant.tools_platform.contracts import ToolErrorCategory


@dataclass
class ToolObservation:
    """Single tool result for continuation context (directives only)."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output: Any | None = None
    error: str | None = None
    error_category: ToolErrorCategory | None = None
    side_effects_may_have_occurred: bool = False
    safe_to_retry: bool = False


@dataclass
class FrameContinuation:
    """
    Frame-scoped continuation directives for the NEXT frame.

    Each frame produces a continuation for the NEXT frame.
    Does NOT include transcript messages - those are projected from runtime
    events (MessageService.build_message), one message per event.

    Directives:
    - compaction_instruction: For COMPACT_AND_CONTINUE action
    - retry params: For RETRY_GENERATION action
    - approval state: For PAUSE_FOR_APPROVAL action
    - system_injections: Additional system instructions
    """

    # === Compaction (COMPACT_AND_CONTINUE action) ===
    compaction_instruction: str | None = None
    compaction_count: int = 0

    # === LLM retry params (RETRY_GENERATION action) ===
    max_tokens_adjustment: int | None = None
    temperature_adjustment: float | None = None
    generation_retry_count: int = 0
    # Backoff (seconds) to wait before re-executing the frame; set for
    # transient provider-error retries (Retry-After or exponential).
    delay_before_retry: float | None = None

    # === Truncation handling ===
    truncation_retry_count: int = 0
    partial_content: str | None = None

    # === Approval (PAUSE_FOR_APPROVAL action) ===
    pending_approval_token: str | None = None
    pending_approval_tool: str | None = None

    # === Tool observations for next frame's context (directives, not messages) ===
    tool_observations: list[ToolObservation] = field(default_factory=list)

    # === System injections ===
    system_injections: list[str] = field(default_factory=list)

    def to_prompt_context_updates(self) -> dict[str, Any]:
        """Convert to PromptContext field updates for context manager."""
        return {
            "metadata": {
                "compaction_count": self.compaction_count,
                "generation_retry_count": self.generation_retry_count,
                "max_tokens_adjustment": self.max_tokens_adjustment,
                "temperature_adjustment": self.temperature_adjustment,
                "pending_approval": self.pending_approval_token is not None,
                "tool_observations": [
                    {
                        "call_id": obs.call_id,
                        "tool_name": obs.tool_name,
                        "arguments": obs.arguments,
                        "success": obs.success,
                        "output": obs.output,
                        "error": obs.error,
                    }
                    for obs in self.tool_observations
                ],
            },
            "system_injections": self.system_injections,
        }

    def add_tool_observations(
        self, observations: list[ToolObservation]
    ) -> FrameContinuation:
        """Add tool observations for CONTINUE action."""
        new = FrameContinuation(**self.__dict__)
        new.tool_observations = list(self.tool_observations) + observations
        return new

    def with_compaction(self, instruction: str) -> FrameContinuation:
        """Add compaction for COMPACT_AND_CONTINUE action."""
        new = FrameContinuation(**self.__dict__)
        new.compaction_instruction = instruction
        new.compaction_count = self.compaction_count + 1
        return new

    def with_retry_params(
        self,
        max_tokens_adj: int | None = None,
        temp_adj: float | None = None,
        retry_count: int = 0,
        delay_before_retry: float | None = None,
    ) -> FrameContinuation:
        """Add retry params for RETRY_GENERATION action."""
        new = FrameContinuation(**self.__dict__)
        new.max_tokens_adjustment = max_tokens_adj
        new.temperature_adjustment = temp_adj
        new.generation_retry_count = retry_count
        new.delay_before_retry = delay_before_retry
        return new

    def with_approval(self, token: str, tool: str) -> FrameContinuation:
        """Add approval state for PAUSE_FOR_APPROVAL action."""
        new = FrameContinuation(**self.__dict__)
        new.pending_approval_token = token
        new.pending_approval_tool = tool
        return new

    def with_system_injection(self, injection: str) -> FrameContinuation:
        """Add system instruction injection."""
        new = FrameContinuation(**self.__dict__)
        new.system_injections = [*list(self.system_injections), injection]
        return new

    def with_truncation_retry(self, partial_content: str) -> FrameContinuation:
        """Add truncation retry state."""
        new = FrameContinuation(**self.__dict__)
        new.truncation_retry_count = self.truncation_retry_count + 1
        new.partial_content = partial_content
        return new


__all__ = ["FrameContinuation", "ToolObservation"]
