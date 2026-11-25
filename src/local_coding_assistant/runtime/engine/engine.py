"""RuntimeEngine - pure frame execution state machine.

Drives frame execution through explicit state transitions based on frame.status:
- RUNNING_LLM -> (LLM complete) -> RUNNING_TOOLS / COMPLETED / FAILED / PARTIAL
- RUNNING_TOOLS -> (tools complete) -> COMPLETED / PARTIAL / FAILED

Supports resume from any state by reading frame.status.
Approval handled by re-executing tool with confirmation_token/approval_response in ToolCallRequest.

Limit checking strategy:
- Deadline: checked at start of each main loop iteration
- Token budget: checked after each LLM chunk (during streaming)
- Loop detection: checked after each tool call is recorded
- Frame cap: checked by FrameAgent before creating new frame
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from local_coding_assistant.agent.llm.models import LLMOptions, LLMTask
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ActionRecord,
    ExecutionStatus,
    RuntimeFrame,
)
from local_coding_assistant.runtime.events.contracts import (
    ApprovalRequiredEvent,
    FrameCompleteEvent,
    FrameStartEvent,
    LLMChunkEvent,
    LLMCompleteEvent,
    LLMStartEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from local_coding_assistant.runtime.session.turn import RuntimeTurn
from local_coding_assistant.tools_platform.contracts import (
    TERMINAL_TOOL_STATUSES,
    ToolCallRequest,
    ToolEvent,
    ToolEventType,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.agent.llm import LLMToolCall
    from local_coding_assistant.agent.llm.service import LLMService
    from local_coding_assistant.runtime.events.contracts import ApprovalRequestForm
    from local_coding_assistant.runtime.events.pipeline import EventPipeline

logger = get_logger("runtime.engine")


class IExecutionEngine(ABC):
    """Abstract frame execution interface.

    The only implementation is the live RuntimeEngine; a read-only replay
    engine does not exist yet (when it does, it can share this interface).
    """

    @abstractmethod
    async def execute_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,  # ResolvedPromptContext
        rendered: Any,  # RenderedPrompt
    ) -> RuntimeFrame:
        """
        Execute a single frame through the state machine.

        Behavior determined by frame.status:
        - RUNNING_LLM: Execute LLM phase
        - RUNNING_TOOLS: Execute tools phase (resume from approval)
        - COMPLETED/FAILED/PARTIAL/BLOCKED: Return as-is (terminal)

        Args:
            frame: Frame to execute (status determines phase)
            turn: Turn with limits, approvals, config
            resolved: Resolved prompt context
            rendered: Rendered prompt

        Returns:
            Updated frame after execution
        """
        ...


class RuntimeEngine(IExecutionEngine):
    """Pure frame execution state machine.

    State transitions driven explicitly by frame.status:
    RUNNING_LLM -> RUNNING_TOOLS -> COMPLETED/FAILED/PARTIAL/BLOCKED
         |              |
         |              └-> (error) -> FAILED
         |              └-> (approval needed) -> re-execute tool with approval_response
         └-> (no tool calls) -> COMPLETED
         └-> (partial/truncation) -> PARTIAL
         └-> (blocked) -> BLOCKED

    Tool actions are event-sourced: the gateway emits TOOL_CALL_START before
    execution and TOOL_CALL_COMPLETE/ERROR/DENIED after, and the engine opens
    and closes TOOL_CALL actions exclusively from those events. The tools
    phase only reads action state, to skip tool calls whose actions already
    reached a terminal status (e.g. completed before a crash).
    """

    def __init__(
        self,
        llm_service: LLMService,
        tool_gateway: ToolGateway,
        event_pipeline: EventPipeline,
    ) -> None:
        self._llm_service = llm_service
        self._tool_gateway = tool_gateway
        self._event_pipeline = event_pipeline
        self._logger = get_logger("runtime.engine")

        # Frame currently being executed - tool events are matched against it
        self._current_frame: RuntimeFrame | None = None

        # Register tool event callback to convert gateway events to runtime events
        self._tool_gateway.register_event_callback(self._on_tool_event)

    def _workspace_id(self, frame: RuntimeFrame) -> str:
        """Resolve the active workspace identity from the gateway.

        The gateway's WorkspaceManager derives a stable workspace_id from the
        project root; tool calls and persisted baselines must use the same
        identity (the session id is NOT a workspace id).

        Args:
            frame: The frame being executed (fallback identity only)

        Returns:
            The gateway workspace id, or the frame session id if unavailable
        """
        workspace_manager = getattr(self._tool_gateway, "workspace_manager", None)
        workspace_id = getattr(workspace_manager, "workspace_id", None)
        if isinstance(workspace_id, str) and workspace_id:
            return workspace_id
        return frame.session_id

    async def execute_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,  # ResolvedPromptContext
        rendered: Any,  # RenderedPrompt
    ) -> RuntimeFrame:
        """Execute a single frame through the state machine.

        Resumes from frame.status - enables restore/resume from any state.

        Args:
            frame: The frame to execute (lightweight, carries status)
            turn: RuntimeTurn with limits, approvals, and snapshot refs
            resolved: Resolved prompt context from context manager
            rendered: Rendered prompt from context manager

        Returns:
            Updated frame with execution results

        Note: Limit violations are reflected in frame status, not raised as exceptions.
        """
        self._logger.info(
            "Executing frame",
            frame_id=frame.id,
            turn_id=frame.turn_id,
            iteration=frame.iteration,
            current_status=frame.status.value,
        )

        # Track the active frame so tool events can open/close its actions
        self._current_frame = frame

        # Emit frame start only on fresh execution (not resume)
        if frame.status == ExecutionStatus.RUNNING_LLM and not frame.actions:
            await self._emit_frame_start(frame, turn)

        try:
            frame = await self._run_frame_loop(frame, turn, resolved, rendered)
        except Exception as e:
            await self._handle_frame_error(frame, e)
        finally:
            # Mark frame as finished if terminal
            if frame.status in (
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.PARTIAL,
                ExecutionStatus.BLOCKED,
            ):
                await self._emit_frame_complete(frame)

            self._current_frame = None

        return frame

    async def _run_frame_loop(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,  # ResolvedPromptContext
        rendered: Any,  # RenderedPrompt
    ) -> RuntimeFrame:
        """Run the frame state-machine loop until a terminal or break condition.

        Dispatches based on current frame.status each iteration:
        RUNNING_LLM -> LLM phase; RUNNING_TOOLS -> tools phase;
        terminal -> stop; unknown -> fail and stop.
        """
        # Pure state machine: dispatch based on current frame.status
        while True:
            # Check for interrupt at start of each loop iteration
            if turn.interrupt_event.is_set():
                self._logger.info(
                    "Interrupt detected, pausing frame execution",
                    frame_id=frame.id,
                    turn_id=turn.turn_id,
                )
                break

            # Check deadline at start of each loop iteration
            deadline_result = turn.limits.deadline_tracker.check_deadline()
            if not deadline_result.within_limits:
                frame.status = ExecutionStatus.FAILED
                frame.result = frame.complete(
                    error_message=deadline_result.detail
                    or "Execution deadline exceeded",
                )
                break

            frame, should_continue = await self._advance_frame(
                frame, turn, resolved, rendered
            )
            if not should_continue:
                break

        return frame

    async def _advance_frame(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,  # ResolvedPromptContext
        rendered: Any,  # RenderedPrompt
    ) -> tuple[RuntimeFrame, bool]:
        """Advance the frame one state-machine step.

        Returns the (possibly updated) frame and whether execution should
        continue looping.
        """
        if frame.status == ExecutionStatus.RUNNING_LLM:
            frame = await self._execute_llm_phase(frame, turn, resolved, rendered)
            return frame, True

        if frame.status == ExecutionStatus.RUNNING_TOOLS:
            frame, approval_break = await self._execute_tools_phase(frame, turn)
            # approval_break means execution was paused for approval
            # and should yield control back to caller for resume
            return frame, not approval_break

        if frame.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.PARTIAL,
            ExecutionStatus.BLOCKED,
        ):
            # Terminal state - frame is done
            return frame, False

        self._logger.warning(
            "Unknown frame status, treating as failed",
            frame_id=frame.id,
            status=frame.status.value,
        )
        frame.status = ExecutionStatus.FAILED
        frame.complete(
            error_message=f"Unknown status: {frame.status.value}",
            finish_reason="Unknown status",
        )
        return frame, False

    async def _handle_frame_error(self, frame: RuntimeFrame, e: Exception) -> None:
        """Classify and record a frame-level execution error."""
        self._logger.error(
            "Frame execution error",
            frame_id=frame.id,
            error=str(e),
            exc_info=True,
        )

        # Classify error as transient (retryable) or non-transient
        is_transient = self._is_transient_error(e)
        from local_coding_assistant.providers.error_classification import (
            classify_llm_error,
        )

        classified = classify_llm_error(e)

        if is_transient:
            # Keep frame in running state for retry via continuation policy
            # Store error in LLM action metadata for policy to consume
            llm_action = next(
                (a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE), None
            )
            if llm_action:
                llm_action.metadata["transient_error"] = str(e)
                llm_action.metadata["error_type"] = type(e).__name__
                llm_action.metadata["error_kind"] = classified.kind.value
                llm_action.metadata["status_code"] = classified.status_code
                llm_action.metadata["retry_after"] = classified.retry_after

            # Don't mark as FAILED - let continuation policy decide RETRY_GENERATION
            # Keep current status (RUNNING_LLM or RUNNING_TOOLS)
            frame.complete(
                error_message=str(e),
                metadata={
                    "transient_error": True,
                    "error_type": type(e).__name__,
                    "error_kind": classified.kind.value,
                    "status_code": classified.status_code,
                    "retry_after": classified.retry_after,
                },
            )

        else:
            frame.status = ExecutionStatus.FAILED
            if not frame.result:
                frame.complete(error_message=str(e), finish_reason="Error")
            else:
                frame.result.error_message = str(e)

    async def _execute_llm_phase(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        resolved: Any,  # ResolvedPromptContext
        rendered: Any,  # RenderedPrompt
    ) -> RuntimeFrame:
        """Execute the LLM phase of the frame.

        Transitions:
        - On success with tool_calls: frame.status = RUNNING_TOOLS
        - On success without tool_calls: frame.status = COMPLETED
        - On partial (truncation): frame.status = PARTIAL
        - On error: frame.status = FAILED
        - On blocked: frame.status = BLOCKED

        Args:
            frame: The frame to execute (must have status=RUNNING_LLM)
            turn: RuntimeTurn with limits, approvals
            resolved: Resolved prompt context
            rendered: Rendered prompt

        Returns:
            Updated frame after LLM execution
        """
        self._logger.debug("Starting LLM phase", frame_id=frame.id)

        if not rendered:
            raise ValueError("Rendered prompt is required")

        llm_task = self._build_llm_task(frame, rendered)
        llm_options = self._build_llm_options(turn, resolved)

        # Emit LLM start event
        await self._emit_llm_start(frame)

        # Add LLM action
        llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="llm_generation")

        try:
            (
                full_content,
                reasoning,
                tool_calls,
                metadata,
                interrupted,
            ) = await self._process_llm_stream(frame, llm_task, llm_options, turn)
            model = metadata["model"]
            provider = metadata["provider"]
            llm_finish_reason = metadata["llm_finish_reason"]
            total_tokens = metadata["total_tokens"] or 0
            prompt_tokens = metadata["prompt_tokens"] or 0
            completion_tokens = metadata["completion_tokens"] or 0
            reasoning_tokens = metadata["reasoning_tokens"] or 0
            cached_tokens = metadata["cached_tokens"] or 0
            cache_write_tokens = metadata["cache_write_tokens"] or 0
            latency_ms = metadata["latency_ms"]

            # Complete LLM action
            frame.complete_action(
                llm_action.id,
                full_content,
                metadata={
                    "tool_calls": tool_calls,
                    "reasoning": reasoning,
                    "model": model,
                    "provider": provider,
                    "total_tokens": total_tokens or 0,
                    "prompt_tokens": prompt_tokens or 0,
                    "completion_tokens": completion_tokens or 0,
                    "reasoning_tokens": reasoning_tokens or 0,
                    "cached_tokens": cached_tokens or 0,
                    "cache_write_tokens": cache_write_tokens or 0,
                    "latency_ms": latency_ms or 0.0,
                },
            )

            frame.model_response_raw = full_content

            if interrupted:
                # We return frame with running_llm status and don't emit llm_complete event
                # to trigger repeated llm call on resume
                return frame

            # Record token budget using structured result
            if prompt_tokens > 0 or completion_tokens > 0 or reasoning_tokens > 0:
                input_tokens = prompt_tokens
                output_tokens = completion_tokens + reasoning_tokens
                budget_result = turn.limits.record_tokens(input_tokens, output_tokens)
                if not budget_result.within_limits:
                    frame.status = ExecutionStatus.FAILED
                    frame.complete(
                        error_message=budget_result.detail or "Token budget exceeded",
                        finish_reason="Token budget exceeded",
                    )
                    self._logger.warning(
                        "Token budget exceeded during LLM phase",
                        frame_id=frame.id,
                        input_tokens_used=turn.limits.token_budget.input_tokens_used,
                        output_tokens_used=turn.limits.token_budget.output_tokens_used,
                        input_max_tokens=turn.limits.token_budget.config.input_max_tokens,
                        output_max_tokens=turn.limits.token_budget.config.output_max_tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    return frame

            # Emit LLM complete event
            tool_calls_dicts = [
                tc if isinstance(tc, dict) else tc.model_dump() for tc in tool_calls
            ]
            await self._emit_llm_complete(
                frame,
                content=full_content,
                reasoning=reasoning,
                tool_calls=tool_calls_dicts,
                model=model,
                provider=provider,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens + reasoning_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                latency_ms=latency_ms,
                finish_reason=llm_finish_reason,
            )

            # Determine next state based on LLM output
            # finish_reason comes from the chunk's finish_reason (carried as
            # llm_finish_reason in metadata). metadata["finish_reason"] is a
            # legacy alias that was never populated; use llm_finish_reason.
            if llm_finish_reason == "length":
                frame.status = ExecutionStatus.PARTIAL
                frame.complete(
                    final_answer=full_content,
                    error_message="Response truncated (max tokens reached)",
                    finish_reason="Response truncated (max tokens reached)",
                )
                self._logger.warning("LLM response truncated", frame_id=frame.id)

            elif llm_finish_reason in {"content_filter", "safety", "blocked"}:
                frame.status = ExecutionStatus.BLOCKED
                frame.complete(
                    error_message=f"Content blocked: {llm_finish_reason}",
                    finish_reason=llm_finish_reason,
                )

            elif tool_calls:
                frame.status = ExecutionStatus.RUNNING_TOOLS
                self._logger.debug(
                    "LLM produced tool calls, transitioning to tools phase",
                    frame_id=frame.id,
                    tool_count=len(tool_calls),
                )

            else:
                frame.status = ExecutionStatus.COMPLETED
                frame.complete(
                    final_answer=full_content,
                    finish_reason="LLM completed without tool calls",
                )
                self._logger.debug(
                    "LLM completed without tool calls", frame_id=frame.id
                )

        except Exception as e:
            self._logger.error(
                "LLM execution error",
                frame_id=frame.id,
                error=str(e),
                exc_info=True,
            )
            from local_coding_assistant.providers.error_classification import (
                classify_llm_error,
            )

            classified = classify_llm_error(e)
            frame.complete_action(
                llm_action.id,
                None,
                metadata={
                    "error": str(e),
                    "error_kind": classified.kind.value,
                    "status_code": classified.status_code,
                    "retry_after": classified.retry_after,
                    "transient": classified.transient,
                },
            )

            if classified.transient:
                # Propagate to the frame-level handler, which keeps the frame
                # resumable (RUNNING_LLM) so the continuation policy can
                # decide RETRY_GENERATION. Returning here would instead
                # re-enter the LLM phase in the state-machine loop.
                raise
            frame.status = ExecutionStatus.FAILED
            frame.complete(error_message=str(e), finish_reason="Error")

        return frame

    def _build_llm_task(self, frame: RuntimeFrame, rendered: Any) -> LLMTask:
        """Build the :class:`LLMTask` from a rendered prompt.

        The system_reminder (authoritative runtime hint) is carried in task
        metadata and later merged by the driver into the last user/tool message
        as a structured text part. It is never prepended to the prompt here and
        never written to canonical history.
        """
        system_prompt = (
            "\n\n".join(rendered.system_messages) if rendered.system_messages else None
        )
        user_prompt = (
            "\n\n".join(rendered.user_messages) if rendered.user_messages else ""
        )

        system_reminder = (
            rendered.metadata.get("system_reminder") if rendered.metadata else None
        )

        history = rendered.history or []
        tool_schemas = rendered.tool_schemas or []

        llm_task_metadata: dict[str, Any] = {
            "frame_id": frame.id,
            "turn_id": frame.turn_id,
        }
        if system_reminder:
            llm_task_metadata["system_reminder"] = system_reminder

        return LLMTask(
            prompt=user_prompt,
            context=history,
            system_prompt=system_prompt,
            tools=tool_schemas,
            metadata=llm_task_metadata,
        )

    def _build_llm_options(self, turn: RuntimeTurn, resolved: Any) -> LLMOptions:
        """Build :class:`LLMOptions` from the turn's context config."""
        context_config = getattr(turn, "context_config", None)
        context_metadata = resolved.metadata if resolved and resolved.metadata else {}

        model = getattr(context_config, "llm_model", None) if context_config else None
        temperature = (
            getattr(context_config, "llm_temperature", None) if context_config else None
        )
        max_tokens = (
            getattr(context_config, "llm_max_tokens", None) if context_config else None
        )
        streaming = (
            getattr(context_config, "streaming", None) if context_config else None
        )

        return LLMOptions(
            model=model or context_metadata.get("model"),
            stream=True if streaming is None else bool(streaming),
            temperature=temperature,
            max_tokens=max_tokens,
            policy=context_metadata.get("model_policy"),
        )

    async def _process_llm_stream(
        self,
        frame: RuntimeFrame,
        llm_task: LLMTask,
        llm_options: LLMOptions,
        turn: RuntimeTurn,
    ) -> tuple[str, str, list, dict[str, Any], bool]:
        """Process LLM streaming and collect content, reasoning, tool calls, and metadata.

        Args:
            frame: The frame being executed
            llm_task: The LLM task to execute
            llm_options: LLM options for the request
            turn: RuntimeTurn with limits for enforcement

        Returns:
            Tuple of (full_content, reasoning, tool_calls, metadata_dict, interrupted)
        """
        full_content = ""
        reasoning = ""
        tool_calls = []
        model = None
        provider = None
        finish_reason = ""
        llm_finish_reason = "stop"
        total_tokens: int | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        reasoning_tokens: int | None = None
        latency_ms: float | None = None
        cached_tokens: int | None = None
        cache_write_tokens: int | None = None

        async for chunk in self._llm_service.stream(llm_task, options=llm_options):
            if chunk.content:
                full_content += chunk.content
                await self._emit_llm_chunk(frame, chunk.content, None, None)

            if chunk.reasoning:
                reasoning += chunk.reasoning
                await self._emit_llm_chunk(frame, None, chunk.reasoning, None)

            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
                tool_calls_dicts = [tc.model_dump() for tc in chunk.tool_calls]
                await self._emit_llm_chunk(frame, None, None, tool_calls_dicts)

            # Capture metadata from last chunk
            (
                model,
                provider,
                llm_finish_reason,
                total_tokens,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                cached_tokens,
                cache_write_tokens,
                latency_ms,
            ) = self._update_chunk_metadata(
                chunk,
                model,
                llm_finish_reason,
                total_tokens,
                prompt_tokens,
                completion_tokens,
                reasoning_tokens,
                latency_ms,
                cached_tokens,
                cache_write_tokens,
            )

            # Check for interrupt
            if turn.interrupt_event.is_set():
                metadata = {
                    "model": llm_options.model,
                    "provider": provider,
                    "llm_finish_reason": llm_finish_reason,
                    "finish_reason": "interrupt",
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "cached_tokens": cached_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "latency_ms": latency_ms,
                }
                return full_content, reasoning, tool_calls, metadata, True

            # Check token budget after each chunk (streaming)
            if total_tokens is not None and total_tokens > 0:
                budget_result = turn.limits.token_budget.check_budget(
                    prompt_tokens or 0,
                    (completion_tokens or 0) + (reasoning_tokens or 0),
                )
                if not budget_result.within_limits:
                    frame.status = ExecutionStatus.FAILED
                    frame.complete(
                        error_message=budget_result.detail
                        or "Token budget exceeded during streaming",
                        finish_reason="Token budget exceeded",
                    )
                    # Return early with what we have so far
                    metadata = {
                        "model": model,
                        "provider": provider,
                        "llm_finish_reason": llm_finish_reason,
                        "finish_reason": finish_reason,
                        "total_tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                        "latency_ms": latency_ms,
                    }
                    return full_content, reasoning, tool_calls, metadata, False

        metadata = {
            "model": model,
            "provider": provider,
            "llm_finish_reason": llm_finish_reason,
            "finish_reason": finish_reason,
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "latency_ms": latency_ms,
        }

        return full_content, reasoning, tool_calls, metadata, False

    def _update_chunk_metadata(
        self,
        chunk: Any,
        model: str | None,
        finish_reason: str,
        total_tokens: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None,
        latency_ms: float | None,
        cached_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> tuple[
        str | None,
        str | None,
        str,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        float | None,
    ]:
        """Update metadata from a streaming chunk.

        Args:
            chunk: The streaming chunk
            model: Current model value
            finish_reason: Current finish reason
            total_tokens: Current total tokens
            prompt_tokens: Current prompt tokens
            completion_tokens: Current completion tokens
            reasoning_tokens: Current reasoning tokens
            latency_ms: Current latency
            cached_tokens: Current cached (prefix) tokens
            cache_write_tokens: Current cache write tokens

        Returns:
            Tuple of updated metadata values
        """
        model = cast(str | None, getattr(chunk, "model", model))
        provider = cast(str | None, getattr(chunk, "provider", None))
        finish_reason = cast(str, getattr(chunk, "finish_reason", finish_reason))

        fields = self._read_chunk_fields(chunk)

        if fields["total_tokens"] is not None:
            total_tokens = fields["total_tokens"]
        if fields["prompt_tokens"] is not None:
            prompt_tokens = fields["prompt_tokens"]
        if fields["completion_tokens"] is not None:
            completion_tokens = fields["completion_tokens"]
        if fields["reasoning_tokens"] is not None:
            reasoning_tokens = fields["reasoning_tokens"]
        if fields["cached_tokens"] is not None:
            cached_tokens = fields["cached_tokens"]
        if fields["cache_write_tokens"] is not None:
            cache_write_tokens = fields["cache_write_tokens"]
        if fields["latency_ms"] is not None:
            latency_ms = fields["latency_ms"]

        return (
            model,
            provider,
            finish_reason,
            total_tokens,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            cached_tokens,
            cache_write_tokens,
            latency_ms,
        )

    def _read_chunk_fields(self, chunk: Any) -> dict[str, int | None]:
        """Read token/latency fields from a chunk's attributes, usage, or metadata.

        Individual chunk attributes take precedence; the ``usage`` dict is
        consulted only when no attribute was present. Returns a dict keyed by
        field name with ``None`` for absent values.
        """
        fields: dict[str, int | None] = {
            "total_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "cached_tokens": None,
            "cache_write_tokens": None,
            "latency_ms": None,
        }
        found_any = False
        for name in fields:
            value = getattr(chunk, name, None)
            if value is not None:
                fields[name] = value
                found_any = True

        if not found_any:
            usage = getattr(chunk, "usage", None)
            if usage and isinstance(usage, dict):
                fields["total_tokens"] = usage.get("total_tokens")
                fields["prompt_tokens"] = usage.get("prompt_tokens")
                fields["completion_tokens"] = usage.get("completion_tokens")
                fields["reasoning_tokens"] = usage.get("reasoning_tokens")
                prompt_tokens_details = usage.get("prompt_tokens_details") or {}
                if isinstance(prompt_tokens_details, dict):
                    fields["cached_tokens"] = prompt_tokens_details.get("cached_tokens")
                    fields["cache_write_tokens"] = prompt_tokens_details.get(
                        "cache_write_tokens"
                    )

        if fields["latency_ms"] is None:
            metadata = getattr(chunk, "metadata", None)
            if metadata and isinstance(metadata, dict):
                fields["latency_ms"] = metadata.get("latency_ms")

        return fields

    async def _execute_tools_phase(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
    ) -> tuple[RuntimeFrame, bool]:
        """Execute the tools phase of the frame.

        Reads tool calls from the completed LLM action.
        Handles approval by re-executing with confirmation_token/approval_response
        stored in turn.pending_approvals.
        Handles retryable tool failures by re-entering the tools phase.

        Tool actions are opened and closed exclusively by tool events emitted
        by the gateway (see _on_tool_event). This method reads action
        state: tool calls whose actions already reached a terminal status
        (e.g. completed before a crash) are skipped and their recorded
        outcome is reused instead of re-executing them.

        Retry logic:
        - Retryable failures (timeout, backend failure): record retry, keep frame
          in RUNNING_TOOLS, re-enter tools phase on next loop iteration
        - Non-retryable failures: emit terminal event, clear retry state
        - Retries exhausted: emit terminal event, clear retry state

        Transitions:
        - All tools succeed: frame.status = COMPLETED
        - Some tools fail (non-retryable or retries exhausted): frame.status = PARTIAL
        - Loop detected: frame.status = FAILED
        - Approval needed: return (frame, True) to yield control
        - Interrupt: return (frame, False) with frame in RUNNING_TOOLS

        Args:
            frame: The frame with LLM tool calls (must have status=RUNNING_TOOLS)
            turn: RuntimeTurn with approvals, limits, and context

        Returns:
            Tuple of updated frame after tool execution and approval_break flag
        """
        self._logger.debug("Starting tools phase", frame_id=frame.id)

        # Get tool calls from LLM action
        llm_action = next(
            (a for a in frame.actions if a.kind == ActionKind.LLM_MESSAGE),
            None,
        )

        if not llm_action or not llm_action.tool_calls:
            self._logger.debug("No tool calls to execute", frame_id=frame.id)
            frame.status = ExecutionStatus.COMPLETED
            frame.complete()
            return frame, False

        tool_calls = llm_action.tool_calls  # Keep as LLMToolCall objects

        # Process each tool call: check recorded outcome, loop detection, retry state, or build request
        pending_requests, recorded_outcomes, stop = self._process_tool_call_loop(
            frame, turn, tool_calls
        )
        if stop:
            # Loop detected - frame already marked FAILED
            return frame, False

        # Execute tools via gateway
        results = await self._tool_gateway.execute_many(
            pending_requests,
            interrupt_event=turn.interrupt_event,
        )

        # Process results: classify retryable vs non-retryable
        retryable_pending, all_terminal = self._classify_tool_results(
            frame, turn, pending_requests, results
        )

        # If any retryable failures pending, ensure events for completed/non-retryable
        # but DON'T emit events for retryable ones yet - they'll be retried
        await self._ensure_terminal_tool_events(
            frame,
            [r for r in pending_requests if r not in retryable_pending],
            [
                res
                for req, res in zip(pending_requests, results, strict=True)
                if req not in retryable_pending
            ],
        )

        # Collect any tools requiring approval
        approval_forms = self._collect_approval_forms(
            frame, turn, pending_requests, results
        )

        if approval_forms is not None:
            self._logger.info(
                "Tool execution paused for approval",
                approval_forms=approval_forms,
            )
            await self._emit_approval_required(frame, turn, approval_forms)
            return frame, True

        if turn.interrupt_event.is_set():
            self._logger.info(
                "Tool execution interrupted",
                frame_id=frame.id,
                cancelled_calls=sum(
                    1
                    for r in results
                    if r.status == ToolExecutionStatus.CANCELLED.value
                ),
            )
            return frame, False

        if not all_terminal:
            # Has retryable failures - keep frame in RUNNING_TOOLS to re-enter
            # on next loop iteration. Don't finalize yet.
            self._logger.debug(
                "Tools phase has retryable failures, will re-enter",
                frame_id=frame.id,
                retryable_count=len(retryable_pending),
            )
            return frame, False

        # All tools terminal - finalize
        return (
            self._finalize_tools_status(
                frame, turn, pending_requests, results, recorded_outcomes
            ),
            False,
        )

    def _process_tool_call_loop(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        tool_calls: list[LLMToolCall],
    ) -> tuple[list[ToolCallRequest], dict[str, str], bool]:
        """Process each tool call: check recorded outcome, loop detection, retry state, or build request.

        Returns (pending_requests, recorded_outcomes, stop). stop is True when a
        loop was detected (frame already marked FAILED) and execution should halt.
        """
        pending_requests: list[ToolCallRequest] = []
        recorded_outcomes: dict[str, str] = {}

        for tool_call in tool_calls:
            call_id = tool_call.id
            if not call_id:
                # No call_id - build request normally
                pending_requests.append(
                    self._build_tool_request(tool_call, frame, turn)
                )
                continue

            # Check if already has a terminal outcome (from previous execution)
            outcome = self._get_tool_call_recorded_outcome(tool_call, frame)
            if outcome:
                # Tool call already has a terminal outcome - skip execution
                recorded_outcomes[call_id] = outcome
                # Clear retry state since this call is terminal
                turn.limits.clear_tool_retry(call_id)
                self._logger.info(
                    "Skipping tool call with recorded terminal outcome",
                    frame_id=frame.id,
                    call_id=call_id,
                    status=outcome,
                )
                continue

            # Check if this call is being retried (has retry count > 0)
            retry_count = turn.limits.get_tool_retry_count(call_id)
            if retry_count > 0:
                # This is a retry - include in pending for re-execution
                self._logger.info(
                    "Retrying tool call",
                    frame_id=frame.id,
                    call_id=call_id,
                    attempt=retry_count + 1,
                )
                pending_requests.append(
                    self._build_tool_request(tool_call, frame, turn)
                )
                continue

            # Check loop detection
            if self._check_tool_call_loop(tool_call, turn, frame):
                return pending_requests, recorded_outcomes, True

            # New tool call - build request for execution
            pending_requests.append(self._build_tool_request(tool_call, frame, turn))

        return pending_requests, recorded_outcomes, False

    def _handle_tool_result(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        call_id: str,
        request: ToolCallRequest,
        result: ToolExecutionResult,
    ) -> bool:
        """Classify a single tool result.

        Returns True if the call should be retried (re-entry needed), False if terminal.
        """
        if result.status == ToolExecutionStatus.COMPLETED.value:
            # Success - clear retry state
            turn.limits.clear_tool_retry(call_id)
            return False

        if result.status == ToolExecutionStatus.PENDING_APPROVAL.value:
            # Approval needed - don't clear retry state, will re-execute after approval
            return False

        if result.error and result.error.retryable:
            # Retryable failure - check if we can retry
            if turn.limits.can_retry_tool(call_id):
                # Record retry attempt
                attempt = turn.limits.record_tool_retry(call_id)
                # Store result for potential resume
                turn.limits.store_tool_result(call_id, result)
                self._logger.info(
                    "Tool call failed with retryable error, will retry",
                    frame_id=frame.id,
                    call_id=call_id,
                    tool_id=request.tool_id,
                    error_code=result.error.code,
                    attempt=attempt,
                    max_retries=turn.limits.tool_retry_guard.max_retries,
                )
                return True
            else:
                # Retries exhausted - treat as terminal non-retryable
                turn.limits.clear_tool_retry(call_id)
                self._logger.warning(
                    "Tool retry exhausted, treating as terminal failure",
                    frame_id=frame.id,
                    call_id=call_id,
                    tool_id=request.tool_id,
                )
                return False

        # Non-retryable failure - clear retry state
        turn.limits.clear_tool_retry(call_id)
        return False

    def _classify_tool_results(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        pending_requests: list[ToolCallRequest],
        results: list[ToolExecutionResult],
    ) -> tuple[list[ToolCallRequest], bool]:
        """Classify tool results into retryable-pending vs terminal.

        Returns (retryable_pending, all_terminal).
        """
        retryable_pending: list[ToolCallRequest] = []
        all_terminal = True
        for request, result in zip(pending_requests, results, strict=True):
            call_id = request.call_id
            if self._handle_tool_result(frame, turn, call_id, request, result):
                retryable_pending.append(request)
                all_terminal = False
        return retryable_pending, all_terminal

    def _collect_approval_forms(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        requests: list[ToolCallRequest],
        results: list[ToolExecutionResult],
    ) -> list[ApprovalRequestForm] | None:
        """Store pending approvals in the turn and collect batch event forms.

        Returns:
            List of ApprovalRequestForm if any call needs approval, else None.
        """
        from local_coding_assistant.runtime.engine.approval import (
            ApprovalRequest,
            ApprovalStatus,
        )
        from local_coding_assistant.runtime.events.contracts import (
            ApprovalRequestForm,
        )

        approval_forms: list[ApprovalRequestForm] = []

        for request, result in zip(requests, results, strict=True):
            if result.status != ToolExecutionStatus.PENDING_APPROVAL:
                continue
            if not (result.decision and result.decision.confirmation_token):
                continue
            if result.decision and result.decision.confirmation_token:
                token = result.decision.confirmation_token
                turn.pending_approvals[request.call_id] = ApprovalRequest(
                    token=token,
                    tool_call_id=request.call_id,
                    tool_name=request.tool_id,
                    arguments=request.raw_arguments,
                    risk_class="high",  # Would need to look up from tool registry
                    session_id=frame.session_id,
                    frame_id=frame.id,
                    turn_id=frame.turn_id,
                    status=ApprovalStatus.PENDING,
                    timeout_seconds=turn.limits_config.continuation_policy.get(
                        "approval_timeout_seconds", 300
                    ),
                )
                # Extract rule info from decision
                rule_id = result.decision.policy_rule
                rule_description = result.decision.reason
                proposed_override = None
                if result.decision.proposed_rule:
                    proposed_override = result.decision.proposed_rule.description
                approval_forms.append(
                    ApprovalRequestForm(
                        approval_token=token,
                        tool_call_id=request.call_id,
                        tool_name=request.tool_id,
                        arguments=request.raw_arguments,
                        rule_id=rule_id,
                        rule_description=rule_description,
                        proposed_override=proposed_override,
                    )
                )

        return approval_forms or None

    def _finalize_tools_status(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        requests: list[ToolCallRequest],
        results: list[ToolExecutionResult],
        recorded_outcomes: dict[str, str],
    ) -> RuntimeFrame:
        """Derive the frame status from tool outcomes.

        TOOL_CALL action records are already opened/closed by gateway tool
        events (see _on_tool_event) - this only clears pending approvals and
        folds executed results with outcomes recorded before this phase
        (e.g. skipped terminal calls).
        """
        all_success = True
        final_answer = frame.model_response_raw or ""
        error_message: str = ""

        for request, result in zip(requests, results, strict=True):
            # Clear pending approval on completion (success or failure)
            turn.pending_approvals.pop(request.call_id, None)
            if result.status != ToolExecutionStatus.COMPLETED.value:
                all_success = False
                error_msg = result.error.message if result.error else "Unknown error"
                error_message += f"\n[Tool {request.tool_id} failed: {error_msg}]"

        # Fold in outcomes recorded before this phase (skipped terminal calls)
        for call_id, recorded_status in recorded_outcomes.items():
            if recorded_status != "completed":
                all_success = False
                error_message += (
                    f"\n[Tool call {call_id} recorded as {recorded_status}]"
                )

        frame.status = (
            ExecutionStatus.COMPLETED if all_success else ExecutionStatus.PARTIAL
        )
        frame.complete(
            final_answer=final_answer,
            error_message=error_message if error_message else None,
        )
        return frame

    def _find_tool_action(
        self, frame: RuntimeFrame, call_id: str
    ) -> ActionRecord | None:
        """Find the TOOL_CALL action for a call_id (read-only lookup)."""
        return next(
            (
                a
                for a in frame.actions
                if a.kind == ActionKind.TOOL_CALL
                and a.tool_execution_data.get("call_id") == call_id
            ),
            None,
        )

    async def _ensure_terminal_tool_events(
        self,
        frame: RuntimeFrame,
        requests: list[ToolCallRequest],
        results: list[ToolExecutionResult],
    ) -> None:
        """Close the action/event gap for results that emitted no events.

        Most outcomes arrive as gateway tool events (TOOL_CALL_COMPLETE /
        ERROR / CANCELLED) and are already terminal by the time results
        return. Some gateway paths however produce a terminal result
        without emitting events - notably prepare-phase failures, where the
        call never reached execution. Without this backstop the transcript
        would have no record that the call was attempted and never failed.

        Approval-pending results are skipped on purpose: nothing executed,
        so there must be no message for them.

        Args:
            frame: Current frame
            requests: Requests dispatched to the gateway (same order)
            results: Results returned by the gateway (same order)
        """
        for request, result in zip(requests, results, strict=True):
            if result.status == ToolExecutionStatus.PENDING_APPROVAL.value:
                continue

            action = self._find_tool_action(frame, request.call_id)
            if (
                action
                and action.tool_execution_data.get("status") in TERMINAL_TOOL_STATUSES
            ):
                # Already closed by a real tool event
                continue

            if not action:
                action = frame.add_action(ActionKind.TOOL_CALL, name=request.tool_id)

            data = action.tool_execution_data
            data.update(
                {
                    "tool_id": request.tool_id,
                    "status": result.status,
                    "output": result.output,
                    "error": result.error.model_dump() if result.error else None,
                    "timing": result.timing or {},
                    "changeset": None,
                    "warnings": [],
                    "backend": result.backend.value if result.backend else None,
                    "decision": result.decision.model_dump()
                    if result.decision
                    else None,
                    "call_id": request.call_id,
                    "input": request.raw_arguments,
                }
            )
            action.finished_at = datetime.now(UTC)

            self._logger.warning(
                "Synthesized tool events",
                frame_id=frame.id,
                call_id=request.call_id,
                status=result.status,
            )

            synthetic_start_event = ToolEvent(
                type=ToolEventType.TOOL_CALL_START,
                call_id=request.call_id,
                session_id=frame.session_id,
                frame_id=frame.id,
                trace_id=frame.trace_id,
                data={
                    "tool_id": request.tool_id,
                    "arguments": request.raw_arguments,
                    "backend": result.backend.value if result.backend else None,
                },
            )
            await self._emit_tool_start_event(
                frame,
                synthetic_start_event,
            )

            synthetic_result_event = ToolEvent(
                type=ToolEventType.TOOL_CALL_COMPLETE
                if result.status == ToolExecutionStatus.COMPLETED.value
                else ToolEventType.TOOL_CALL_ERROR,
                call_id=request.call_id,
                session_id=frame.session_id,
                frame_id=frame.id,
                trace_id=frame.trace_id,
                data={
                    "tool_id": request.tool_id,
                    "status": result.status,
                    "output": result.output,
                    "error": result.error.model_dump() if result.error else None,
                    "timing": result.timing or {},
                    "backend": result.backend.value if result.backend else None,
                    "arguments": request.raw_arguments,
                },
            )
            await self._emit_tool_result_event(
                frame,
                synthetic_result_event,
                success=result.status == ToolExecutionStatus.COMPLETED.value,
            )

    def _get_tool_call_recorded_outcome(
        self, tool_call: LLMToolCall, frame: RuntimeFrame
    ) -> str | None:
        """Check if a tool call already has a recorded terminal outcome.

        Args:
            tool_call: The LLMToolCall to check
            frame: The current frame

        Returns:
            The recorded status string if terminal, else None
        """
        call_id = tool_call.id
        if not call_id:
            return None

        action = self._find_tool_action(frame, call_id)
        if not action:
            return None

        recorded_status = action.tool_execution_data.get("status")
        if recorded_status in TERMINAL_TOOL_STATUSES:
            return recorded_status

        return None

    def _check_tool_call_loop(
        self, tool_call: LLMToolCall, turn: RuntimeTurn, frame: RuntimeFrame
    ) -> bool:
        """Check if a tool call triggers loop detection.

        Records the tool call and checks the loop guard. If a loop is
        detected, marks the frame as failed.

        Args:
            tool_call: The LLMToolCall to check
            turn: RuntimeTurn with limits
            frame: The current frame

        Returns:
            True if loop detected (frame marked as failed), False otherwise
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        call_id = tool_call.id or str(uuid.uuid4())

        turn.record_tool_call(call_id, tool_name, tool_args)
        loop_result = turn.limits.loop_guard.check_loop()

        if not loop_result.within_limits:
            frame.status = ExecutionStatus.FAILED
            frame.complete(
                error_message=loop_result.detail or "Loop detected in tool calls",
                finish_reason="Loop detected in tool calls",
            )
            return True

        return False

    def _build_tool_request(
        self, tool_call: LLMToolCall, frame: RuntimeFrame, turn: RuntimeTurn
    ) -> ToolCallRequest:
        """Build a single ToolCallRequest from a tool call.

        Does NOT create actions - TOOL_CALL actions are opened/closed by
        gateway tool events (see _on_tool_event).

        Args:
            tool_call: The LLMToolCall to convert
            frame: The current frame
            turn: RuntimeTurn with pending approvals

        Returns:
            ToolCallRequest object
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        call_id = tool_call.id or str(uuid.uuid4())
        pending = turn.pending_approvals.get(call_id)
        approval_response = pending.response if pending else None
        confirmation_token = pending.token if pending else None

        return ToolCallRequest(
            tool_id=tool_name,
            call_id=call_id,
            session_id=frame.session_id,
            turn_id=turn.turn_id,
            frame_id=frame.id,
            raw_arguments=tool_args,
            workspace_id=self._workspace_id(frame),
            confirmation_token=confirmation_token,
            approval_response=approval_response,
        )

    def _build_tool_requests(
        self,
        tool_calls: list[LLMToolCall],
        frame: RuntimeFrame,
        turn: RuntimeTurn,
    ) -> tuple[list[ToolCallRequest], bool]:
        """Build ToolCallRequest objects from tool calls.

        Does NOT create actions - TOOL_CALL actions are opened/closed by
        gateway tool events (see _on_tool_event).

        Args:
            tool_calls: List of LLMToolCall objects
            frame: The current frame
            turn: RuntimeTurn with pending approvals and limits

        Returns:
            Tuple of list of ToolCallRequest objects and boolean flag for loop detection
        """
        requests: list[ToolCallRequest] = []

        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.arguments
            call_id = tool_call.id or str(uuid.uuid4())
            pending = turn.pending_approvals.get(call_id)
            approval_response = pending.response if pending else None
            confirmation_token = pending.token if pending else None

            # Check loop detection after recording - BLOCK dispatch if detected
            turn.record_tool_call(call_id, tool_name, tool_args)
            loop_result = turn.limits.loop_guard.check_loop()
            if not loop_result.within_limits:
                frame.status = ExecutionStatus.FAILED
                frame.complete(
                    error_message=loop_result.detail or "Loop detected in tool calls",
                    finish_reason="Loop detected in tool calls",
                )
                # Do NOT create a request - this prevents the tool from being dispatched
                # Return empty requests to stop execution
                return [], True

            request = ToolCallRequest(
                tool_id=tool_name,
                call_id=call_id,
                session_id=frame.session_id,
                turn_id=turn.turn_id,
                frame_id=frame.id,
                raw_arguments=tool_args,
                workspace_id=self._workspace_id(frame),
                confirmation_token=confirmation_token,
                approval_response=approval_response,
            )

            requests.append(request)

        return requests, False

    async def _on_tool_event(self, event: Any) -> None:
        """Convert gateway ToolEvents into frame action lifecycle + runtime events.

        This is the single writer of TOOL_CALL actions:
        - TOOL_CALL_START opens an action (idempotent per call_id)
        - TOOL_CALL_COMPLETE / TOOL_CALL_ERROR / TOOL_CALL_CANCELLED close it
          with the full outcome (cancelled is terminal: the tool did not run
          to completion, so skipping on resume is safe)
        - TOOL_CALL_DENIED with outcome=deny closes it as terminal;
          outcome=ask is left to the approval flow (action stays open)

        Runtime events are emitted for observability/persistence after the
        frame mutation, so the in-memory frame is always consistent when the
        gateway continues execution.
        """
        if not isinstance(event, ToolEvent):
            return

        frame = self._current_frame
        if frame is None:
            # No frame in progress (e.g. CLI tool invocation) - observe only
            return
        if event.frame_id and event.frame_id != frame.id:
            self._logger.debug(
                "Ignoring tool event for different frame",
                event_frame_id=event.frame_id,
                current_frame_id=frame.id,
            )
            return

        if event.type == ToolEventType.TOOL_CALL_START:
            self._open_tool_action(frame, event)
            await self._emit_tool_start_event(frame, event)

        elif event.type == ToolEventType.TOOL_CALL_COMPLETE:
            self._close_tool_action(frame, event, status="completed")
            await self._emit_tool_result_event(frame, event, success=True)

        elif event.type == ToolEventType.TOOL_CALL_ERROR:
            # Don't emit ToolResultEvent yet - tools phase will decide final outcome
            # (retry vs terminal) and emit appropriately via _ensure_terminal_tool_events
            # NOTE: ToolResultEvent emission deferred to tools phase
            pass

        elif event.type == ToolEventType.TOOL_CALL_CANCELLED:
            self._close_tool_action(frame, event, status="cancelled")
            await self._emit_tool_result_event(frame, event, success=False)

        elif event.type == ToolEventType.TOOL_CALL_DENIED:
            if event.data.get("outcome") == "deny":
                self._close_tool_action(frame, event, status="denied")
                await self._emit_tool_result_event(frame, event, success=False)
            else:
                # outcome == "ask": awaiting approval, not terminal
                self._logger.debug(
                    "Tool call awaiting approval",
                    call_id=event.call_id,
                    reason=event.data.get("reason"),
                )

    def _open_tool_action(self, frame: RuntimeFrame, event: ToolEvent) -> ActionRecord:
        """Open a TOOL_CALL action from a TOOL_CALL_START event.

        Idempotent per call_id: if an action already exists for the call
        (e.g. opened before an approval pause), it is reused.
        """
        existing = self._find_tool_action(frame, event.call_id)
        if existing is not None:
            return existing

        tool_id = event.data.get("tool_id", "unknown")
        action = frame.add_action(ActionKind.TOOL_CALL, name=tool_id)
        action.tool_execution_data.update(
            {
                "tool_id": tool_id,
                "call_id": event.call_id,
                "input": event.data.get("arguments", {}),
                "status": "running",
            }
        )
        return action

    def _close_tool_action(
        self, frame: RuntimeFrame, event: ToolEvent, status: str
    ) -> ActionRecord:
        """Close a TOOL_CALL action from a terminal tool event.

        Fills the action with the full outcome payload from the event so the
        frame carries the authoritative result without any engine-side
        bookkeeping of return values.
        """
        action = self._find_tool_action(frame, event.call_id)
        if action is None:
            # Defensive: completion without a recorded start
            action = frame.add_action(
                ActionKind.TOOL_CALL, name=event.data.get("tool_id", "unknown")
            )

        data = action.tool_execution_data
        tool_id = event.data.get("tool_id", "unknown")
        error = event.data.get("error")

        data.update(
            {
                "tool_id": tool_id,
                "call_id": event.call_id,
                "input": event.data.get("arguments", {}),
                "status": status,
                "output": event.data.get("output"),
                "error": error,
                "timing": event.data.get("timing", {}),
                "output_chunks": event.data.get("output_chunks", []),
                "warnings": event.data.get("warnings", []),
                "changeset": event.data.get("changeset"),
                "backend": event.data.get("backend"),
                "trace_id": event.data.get("trace_id"),
                "decision": event.data.get("decision"),
            }
        )
        action.finished_at = datetime.now(UTC)

        return action

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if an error is transient (retryable)."""
        from local_coding_assistant.providers.error_classification import (
            classify_llm_error,
        )

        classified = classify_llm_error(error)
        return classified.transient

    # === Event emission helpers ===

    async def _emit_frame_start(self, frame: RuntimeFrame, turn: RuntimeTurn) -> None:
        event = FrameStartEvent(
            trace_id=turn.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            iteration=frame.iteration,
            context_snapshot_id=frame.context_snapshot_id or "",
            context_hash=frame.prompt_hash or "",
            message_boundary=None,
            history_through_seq=frame.history_through_seq,
            prompt_hash=frame.prompt_hash,
        )
        await self._event_pipeline.emit(event)

    async def _emit_frame_complete(self, frame: RuntimeFrame) -> None:
        model_name = None
        tool_call_count = 0
        for action in frame.actions:
            if action.kind == ActionKind.LLM_MESSAGE:
                model_name = (
                    action.llm_metrics.model if action.llm_metrics is not None else None
                )
            elif action.kind == ActionKind.TOOL_CALL:
                tool_call_count += 1

        event = FrameCompleteEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            status=frame.status.value,
            finish_reason=frame.result.finish_reason if frame.result else None,
            duration_ms=int(frame.duration_ms) if frame.duration_ms else None,
            total_input_tokens=frame.result.input_tokens if frame.result else 0,
            total_output_tokens=frame.result.output_tokens if frame.result else 0,
            model_name=model_name,
            error_message=frame.result.error_message if frame.result else None,
            llm_message=frame.result.final_answer if frame.result else None,
            tool_call_count=tool_call_count,
        )
        await self._event_pipeline.emit(event)

    async def _emit_llm_start(self, frame: RuntimeFrame) -> None:
        event = LLMStartEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            model=None,
            prompt_tokens_estimate=None,
        )
        await self._event_pipeline.emit(event)

    async def _emit_llm_chunk(
        self,
        frame: RuntimeFrame,
        content: str | None,
        reasoning: str | None,
        tool_calls: list | None,
    ) -> None:
        event = LLMChunkEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )
        await self._event_pipeline.emit(event)

    async def _emit_llm_complete(
        self,
        frame: RuntimeFrame,
        content: str,
        reasoning: str | None,
        tool_calls: list[dict[str, Any]],
        model: str | None,
        provider: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        reasoning_tokens: int | None,
        cached_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        latency_ms: float | None = None,
        finish_reason: str | None = None,
    ) -> None:
        event = LLMCompleteEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            content=content,
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )
        await self._event_pipeline.emit(event)

    async def _emit_tool_start_event(
        self, frame: RuntimeFrame, event: ToolEvent
    ) -> None:
        tevent = ToolStartEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            tool_call_id=event.call_id,
            tool_name=event.data.get("tool_id", "unknown"),
            arguments=event.data.get("arguments", {}),
            execution_mode=event.data.get("execution_mode", "unknown"),
        )
        await self._event_pipeline.emit(tevent)

    async def _emit_tool_result_event(
        self, frame: RuntimeFrame, event: ToolEvent, success: bool
    ) -> None:
        # Extract file observations from tool output (for baseline persistence)
        output = event.data.get("output")
        file_observations = []
        if output is not None:
            if hasattr(output, "file_observation") and output.file_observation:
                file_observations.append(
                    output.file_observation.model_dump(mode="json")
                )
            if hasattr(output, "file_observations") and output.file_observations:
                file_observations.extend(
                    [obs.model_dump(mode="json") for obs in output.file_observations]
                )

        tevent = ToolResultEvent(
            trace_id=frame.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            workspace_id=self._workspace_id(frame),
            tool_call_id=event.call_id,
            tool_name=event.data.get("tool_id", "unknown"),
            success=success,
            output=output,
            error=event.data.get("error"),
            timing=event.data.get("timing"),
            execution_mode=event.data.get("execution_mode"),
            file_changes=[],
            file_observations=file_observations,
            backend=event.data.get("backend"),
            arguments=event.data.get("arguments"),
        )
        await self._event_pipeline.emit(tevent)

    async def _emit_approval_required(
        self,
        frame: RuntimeFrame,
        turn: RuntimeTurn,
        approval_forms: list[ApprovalRequestForm],
    ) -> None:
        event = ApprovalRequiredEvent(
            trace_id=turn.trace_id,
            turn_id=frame.turn_id,
            frame_id=frame.id,
            session_id=frame.session_id,
            approval_requests=approval_forms,
            timeout_seconds=turn.limits_config.continuation_policy.get(
                "approval_timeout_seconds", 300
            ),
        )
        await self._event_pipeline.emit(event)
