from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from local_coding_assistant.agent.llm import LLMToolCall
from local_coding_assistant.core.telemetry_types import FileChange, ToolCallTrace
from local_coding_assistant.runtime.runtime_types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
    ToolSpec,
)


class ActionKind(str, Enum):
    LLM_MESSAGE = "llm_message"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


class ActionRecord(BaseModel):
    """A single action taken during an execution frame."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    kind: ActionKind
    name: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    # === LLM-SPECIFIC ===
    llm_metrics: LLMMetrics | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)

    # === TOOL-SPECIFIC ===
    tool_trace: ToolCallTrace | None = None  # Contains input/output/metrics

    # === LEGACY SUPPORT ===
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def input(self) -> Any:
        """Legacy support - get from tool_trace."""
        return self.tool_trace.input if self.tool_trace else None

    @property
    def output(self) -> Any:
        """Legacy support - get from tool_trace."""
        return self.tool_trace.output if self.tool_trace else None


class LLMMetrics(BaseModel):
    """Metrics for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # Move reasoning tokens here
    latency_ms: float = 0.0
    model: str | None = None  # Add model field as first-class


class ExecutionMetrics(BaseModel):
    """Aggregated metrics for an execution frame."""

    llm: LLMMetrics = Field(default_factory=LLMMetrics)
    total_latency_ms: float = 0.0
    # Remove tool_calls - will be generated on-demand from actions


class ExecutionStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  # Requires handler intervention (truncation, parsing, etc.)
    FAILED = "failed"
    BLOCKED = "blocked"


class ContinuationStrategy(str, Enum):
    """Strategy for continuing execution after partial response."""

    CONTINUE_REASONING = (
        "continue_reasoning"  # Continue with same context, accumulate reasoning
    )
    RESTART_REASONING = (
        "restart_reasoning"  # Restart reasoning with accumulated context
    )
    EXTEND_AND_CONTINUE = "extend_and_continue"  # Increase max_tokens and continue
    CONTINUE_CONTENT = "continue_content"  # Continue from truncated content
    CONTINUE_TOOL_CALLS = "continue_tool_calls"  # Continue incomplete tool calls
    CONTINUE_WITH_HISTORY = (
        "continue_with_history"  # Update session history and use simple continuation
    )


class ExecutionResult(BaseModel):
    """The normalized result of an execution frame."""

    status: ExecutionStatus
    final_answer: str | None = None
    finish_reason: str | None = None
    total_latency_ms: float | None = None
    total_tokens: float | None = None
    error_message: str | None = None
    handler_context: dict[str, Any] | None = Field(default=None)
    # File operation fields - populated from execution envelope
    agent_file_changes: list[FileChange] = Field(default_factory=list)


class ExecutionFrame(BaseModel):
    """
    A single agent reasoning step.
    Captures complete context, input, execution, and output for full observability.
    """

    # === IDENTIFICATION ===
    id: str = Field(default_factory=lambda: f"frame_{uuid.uuid4()}")
    session_id: str  # Link to overall session
    iteration: int = 1  # Which iteration in the agent loop

    # === COMPLETE INPUT ===
    prompt_context: PromptContext  # Everything needed for input
    rendered_prompt: RenderedPrompt  # Template output

    # === EXECUTION STATE ===
    model_response_raw: str | None = None  # Raw LLM response
    actions: list[ActionRecord] = Field(default_factory=list)  # All actions taken
    result: ExecutionResult | None = None  # Normalized outcome

    # === TIMING ===
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    # === CONVENIENCE PROPERTIES ===
    @property
    def agent_profile(self) -> AgentProfile | None:
        """Get the agent profile from prompt context."""
        return self.prompt_context.agent_profile

    @property
    def execution_mode(self) -> ExecutionMode:
        """Get the execution mode from prompt context."""
        return self.prompt_context.execution_mode

    @property
    def user_input(self) -> str:
        """Get the user input from prompt context."""
        return self.prompt_context.user_input

    @property
    def tools(self) -> list[ToolSpec]:
        """Get the tools from prompt context."""
        return self.prompt_context.tools

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get the history from prompt context."""
        return self.prompt_context.history

    @property
    def full_prompt(self) -> str:
        """Complete prompt as single string."""
        messages = self.rendered_prompt.to_full_prompt()
        return "\n".join([json.dumps(msg) for msg in messages])

    @property
    def duration_ms(self) -> float | None:
        """Execution duration in milliseconds."""
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def is_successful(self) -> bool:
        """Quick success check."""
        return self.result is not None and self.result.status == ExecutionStatus.SUCCESS

    # === ON-DEMAND DATA ACCESS ===
    def get_tool_results(self) -> list[dict[str, Any]]:
        """Generate tool results from actions on-demand."""
        tool_results = []
        for action in self.actions:
            if action.kind == ActionKind.TOOL_CALL and action.tool_trace:
                # Create dict with all relevant ToolCallTrace data
                tool_result = {
                    "tool_name": action.tool_trace.tool_name,
                    "tool_args": action.tool_trace.input or {},
                    "success": action.tool_trace.success,
                    "result": action.tool_trace.output,
                    "error_message": action.tool_trace.error,
                    "execution_time_ms": action.tool_trace.duration_ms,
                    "call_id": action.tool_trace.call_id,
                    "start_time": action.tool_trace.start_time,
                    "end_time": action.tool_trace.end_time,
                    "parent_call_id": action.tool_trace.parent_call_id,
                    "child_call_ids": action.tool_trace.child_call_ids,
                    "execution_mode": action.tool_trace.execution_mode,
                    "source": action.tool_trace.source,
                    "resource_metrics": [
                        m.model_dump() for m in action.tool_trace.resource_metrics
                    ],
                    "metadata": action.tool_trace.metadata,
                }
                tool_results.append(tool_result)
        return tool_results

    def get_tool_metrics(self) -> list[dict[str, Any]]:
        """Generate tool metrics from actions on-demand."""
        tool_metrics = []
        for action in self.actions:
            if action.kind == ActionKind.TOOL_CALL and action.tool_trace:
                metrics = {
                    "tool_name": action.tool_trace.tool_name,
                    "call_id": action.tool_trace.call_id,
                    "execution_time_ms": action.tool_trace.duration_ms or 0.0,
                    "start_time": action.tool_trace.start_time,
                    "end_time": action.tool_trace.end_time,
                    "success": action.tool_trace.success,
                    "error_message": action.tool_trace.error,
                    "parent_call_id": action.tool_trace.parent_call_id,
                    "resource_metrics": [
                        m.model_dump() for m in action.tool_trace.resource_metrics
                    ],
                    "metadata": action.tool_trace.metadata,
                }
                tool_metrics.append(metrics)
        return tool_metrics

    def get_reasoning(self) -> str | None:
        """Get reasoning from LLM action metadata on-demand."""
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.metadata:
                return action.metadata.get("reasoning")
        return None

    def get_reasoning_tokens(self) -> int:
        """Get reasoning tokens from LLM action metadata on-demand."""
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.llm_metrics:
                return getattr(action.llm_metrics, "reasoning_tokens", 0)
        return 0

    def get_llm_metrics(self) -> LLMMetrics | None:
        """Get LLM metrics from LLM action on-demand."""
        for action in self.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.llm_metrics:
                return action.llm_metrics
        return None

    # === ACTION MANAGEMENT ===
    def add_action(
        self, kind: ActionKind, name: str | None = None, _input: Any = None
    ) -> ActionRecord:
        """Add a new action to this frame."""
        action = ActionRecord(kind=kind, name=name)
        # Store input in metadata for non-tool actions, or initialize tool_trace for tools
        if kind == ActionKind.TOOL_CALL and _input is not None:
            # Will be populated later with full ToolCallTrace
            action.metadata["temp_input"] = _input
        elif _input is not None:
            action.metadata["input"] = _input

        self.actions.append(action)
        return action

    def _complete_llm_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        """Complete an LLM message action."""
        if metadata:
            action.tool_calls = metadata.get("tool_calls", [])
            llm_metrics: LLMMetrics
            if action.llm_metrics is not None:
                llm_metrics = action.llm_metrics
            else:
                llm_metrics = LLMMetrics()
                action.llm_metrics = llm_metrics
            llm_metrics.model = metadata.get("model")
            llm_metrics.prompt_tokens = metadata.get("prompt_tokens", 0)
            llm_metrics.completion_tokens = metadata.get("completion_tokens", 0)
            llm_metrics.reasoning_tokens = metadata.get("reasoning_tokens", 0)
            llm_metrics.total_tokens = metadata.get("total_tokens", 0)
            llm_metrics.latency_ms = metadata.get("latency_ms", 0.0)
            # Store reasoning in metadata
            if "reasoning" in metadata:
                action.metadata["reasoning"] = metadata["reasoning"]
        action.finished_at = datetime.now(UTC)

    def _complete_tool_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        """Complete a tool call action."""
        if isinstance(output, ToolCallTrace):
            action.tool_trace = output
        elif metadata and "tool_trace" in metadata:
            action.tool_trace = metadata["tool_trace"]
        else:
            # Create ToolCallTrace from legacy data
            action.tool_trace = ToolCallTrace(
                call_id=metadata.get("call_id", "unknown") if metadata else "unknown",
                tool_name=action.name or "unknown",
                input=action.metadata.get("temp_input", {}),
                output=output,
                success=metadata.get("success", True) if metadata else True,
                error=metadata.get("error") if metadata else None,
                parent_call_id=metadata.get("parent_call_id") if metadata else None,
            )

        if action.metadata.get("temp_input"):
            del action.metadata["temp_input"]
        action.finished_at = datetime.now(UTC)

    def _complete_other_action(
        self, action: ActionRecord, output: Any, metadata: dict[str, Any] | None
    ) -> None:
        """Complete other action types."""
        action.metadata["output"] = output
        if metadata:
            action.metadata.update(metadata)
        action.finished_at = datetime.now(UTC)

    def complete_action(
        self, action_id: str, output: Any, metadata: dict[str, Any] | None = None
    ):
        """Complete an existing action."""
        for action in self.actions:
            if action.id == action_id:
                if action.kind == ActionKind.LLM_MESSAGE:
                    self._complete_llm_action(action, output, metadata)
                elif action.kind == ActionKind.TOOL_CALL:
                    self._complete_tool_action(action, output, metadata)
                else:
                    self._complete_other_action(action, output, metadata)
                break
