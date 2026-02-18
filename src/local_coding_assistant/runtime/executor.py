from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from local_coding_assistant.agent.llm import (
    LLMOptions,
    LLMResult,
    LLMService,
    LLMTask,
    LLMToolCall,
)
from local_coding_assistant.core.telemetry_types import ToolCallTrace
from local_coding_assistant.runtime.execution_types import (
    ActionKind,
    ActionRecord,
    ExecutionFrame,
    ExecutionResult,
    ExecutionStatus,
)
from local_coding_assistant.runtime.handlers.tool_error_classifier import (
    ToolErrorClassifier,
)
from local_coding_assistant.tools.types import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.core.protocols import IToolManager
    from local_coding_assistant.runtime.context_manager import ContextManager

logger = get_logger("runtime.executor")


class RuntimeExecutor:
    """
    Executes an ExecutionFrame by orchestrating LLM calls and tool executions.
    """

    def __init__(
        self,
        llm_service: LLMService,
        tool_manager: IToolManager,
        context_manager: ContextManager | None = None,
        config_manager: Any | None = None,
    ):
        self._llm_service = llm_service
        self._tool_manager = tool_manager
        self._context_manager = context_manager
        self._config_manager = config_manager
        self._error_classifier = ToolErrorClassifier()

    def _prepare_llm_request(self, frame: ExecutionFrame) -> tuple[LLMTask, LLMOptions]:
        """Prepare LLM task and options from frame."""
        llm_task = LLMTask(
            prompt=self._combine_sections(frame.rendered_prompt.user_messages)
            if frame.rendered_prompt.user_messages
            else frame.user_input,
            context=frame.rendered_prompt.history,
            tools=frame.rendered_prompt.tool_schemas,
            system_prompt=self._combine_sections(frame.rendered_prompt.system_messages)
            if frame.rendered_prompt.system_messages
            else None,
        )

        options = LLMOptions(
            policy=frame.agent_profile.model_policy if frame.agent_profile else None,
            stream=False,  # Streaming handled separately
        )

        return llm_task, options

    def _handle_llm_error(
        self,
        frame: ExecutionFrame,
        llm_action: ActionRecord,
        llm_error: Exception,
        start_time: float,
    ):
        """Handle LLM generation errors."""
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.error("LLM generation failed for frame %s: %s", frame.id, str(llm_error))

        frame.complete_action(
            llm_action.id,
            output=str(llm_error),
            metadata={
                "model": "unknown",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

        error_msg = str(llm_error).lower()
        if any(
            keyword in error_msg
            for keyword in ["rate limit", "quota", "unauthorized", "permission"]
        ):
            status = ExecutionStatus.BLOCKED
        else:
            status = ExecutionStatus.FAILED

        frame.result = ExecutionResult(
            status=status,
            finish_reason=None,
            error_message=str(llm_error),
            total_latency_ms=latency_ms,
        )
        frame.finished_at = datetime.now(UTC)

    def _handle_llm_response(
        self,
        frame: ExecutionFrame,
        llm_action: ActionRecord,
        response: LLMResult,
        start_time: float,
    ):
        """Handle successful LLM response."""
        frame.model_response_raw = response.content
        tool_calls = response.tool_calls

        reasoning_tokens = response.reasoning_tokens
        if self._should_capture_reasoning() and response.reasoning:
            truncated, _ = self._truncate_reasoning(response.reasoning)
            reasoning = truncated
        else:
            reasoning = None

        frame.complete_action(
            llm_action.id,
            output=response.content,
            metadata={
                "model": response.model,
                "tool_calls": tool_calls,
                "total_tokens": response.total_tokens,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "finish_reason": response.finish_reason,
                "reasoning": reasoning,
                "reasoning_tokens": reasoning_tokens,
                "latency_ms": response.metadata.get("provider_metadata", {}).get(
                    "latency_ms"
                ),
            },
        )

    def _determine_initial_status(
        self, response: LLMResult
    ) -> tuple[ExecutionStatus, dict | None]:
        """Determine initial execution status based on LLM response."""
        initial_status = ExecutionStatus.SUCCESS
        handler_context = None
        if response.finish_reason == "length":
            initial_status = ExecutionStatus.PARTIAL
            handler_context = {
                "error_type": "truncation",
                "reasoning": response.reasoning,
                "finish_reason": response.finish_reason,
            }
        elif (
            response.metadata.get("content_error")
            and response.metadata.get("content_error", {}).get("error_type")
            == "tool_call_parsing_error"
        ):
            content_error_info = response.metadata.get("content_error", {})
            error_message = content_error_info.get("message", "Unknown content error")
            logger.warning(
                "LLM content generation failed (parsing/format): %s", error_message
            )

            initial_status = ExecutionStatus.PARTIAL
            handler_context = {
                "error_type": "parsing_error",
                "message": error_message,
                "raw_response": content_error_info.get(
                    "raw_response", response.content
                ),
                "reasoning": response.reasoning,
                "raw_tool_calls": content_error_info.get("tool_calls_attempted", {}),
            }
        elif response.finish_reason in {"content_filter", "safety", "blocked"}:
            initial_status = ExecutionStatus.BLOCKED

        return initial_status, handler_context

    async def _handle_tool_calls(
        self, frame: ExecutionFrame, response: LLMResult, result: ExecutionResult
    ):
        """Handle tool calls if present."""
        if response.tool_calls:
            for tool_call in response.tool_calls:
                await self._execute_tool_call(frame, tool_call, result)

    def _finalize_result(
        self, frame: ExecutionFrame, result: ExecutionResult, response: LLMResult
    ):
        """Finalize the execution result."""
        if (
            not response.tool_calls
            and response.content
            and result.status == ExecutionStatus.SUCCESS
        ):
            result.final_answer = response.content

        if result.status == ExecutionStatus.PARTIAL:
            error_type = (
                result.handler_context.get("error_type")
                if result.handler_context
                else "unknown"
            )
            result.error_message = (
                f"Execution paused for handler intervention: {error_type}"
            )

    async def execute(
        self, frame: ExecutionFrame, use_streaming: bool = False
    ) -> ExecutionFrame:
        """
        Executes the frame: sends prompt to LLM, handles tool calls, and returns updated frame.
        """
        # Only set started_at if it's not already set (for fresh frames)
        if not frame.started_at:
            frame.started_at = datetime.now(UTC)
        logger.info("Executing frame %s", frame.id)

        try:
            # 1. Prepare LLM Request and LLM Options
            llm_task, options = self._prepare_llm_request(frame)

            # 2. Record LLM Action
            llm_action = frame.add_action(ActionKind.LLM_MESSAGE, name="generate")

            # 3. Call LLM
            start_time = time.perf_counter()
            try:
                response = await self._generate_llm_result(
                    llm_task, options=options, streaming=use_streaming
                )
            except Exception as llm_error:
                self._handle_llm_error(frame, llm_action, llm_error, start_time)
                return frame

            # 4. Handle LLM Response
            self._handle_llm_response(frame, llm_action, response, start_time)

            # 5. Determine initial status
            initial_status, handler_context = self._determine_initial_status(response)

            result = ExecutionResult(
                status=initial_status,
                finish_reason=response.finish_reason,
                handler_context=handler_context,
            )

            # 6. Handle Tool Calls if any
            await self._handle_tool_calls(frame, response, result)

            # 7. Finalize result
            self._finalize_result(frame, result, response)

            frame.result = result

        except Exception as e:
            logger.exception("Error executing frame %s", frame.id)
            frame.result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                finish_reason=None,
                error_message=str(e),
            )
        finally:
            frame.finished_at = datetime.now(UTC)
            if frame.result and hasattr(frame.result, "total_latency_ms"):
                frame.result.total_latency_ms = (
                    frame.finished_at - frame.started_at
                ).total_seconds() * 1000

        return frame

    async def _execute_tool_call(
        self, frame: ExecutionFrame, tool_call: LLMToolCall, result: ExecutionResult
    ):
        """Executes a single tool call and updates the frame and result."""
        tool_action = frame.add_action(
            ActionKind.TOOL_CALL, name=tool_call.name, _input=tool_call.arguments
        )
        execution_mode = frame.prompt_context.execution_mode
        tool_start_time = time.perf_counter()

        try:
            self._validate_tool_call(frame, tool_call, result)

            request = ToolExecutionRequest(
                tool_name=tool_call.name, payload=tool_call.arguments
            )

            tool_response = await self._tool_manager.execute_async(request)
            if tool_response.result:
                logger.debug(
                    f"Tool call {tool_call.name} completed with response: {tool_response.result}"
                )

            tool_end_time = time.perf_counter()
            tool_latency_ms = (tool_end_time - tool_start_time) * 1000

            if frame.prompt_context.tool_call_mode == "ptc":
                self._handle_ptc_mode(
                    frame, tool_call, tool_response, result, tool_action, execution_mode
                )
            else:
                self._handle_classic_mode(
                    frame,
                    tool_call,
                    tool_response,
                    tool_action,
                    execution_mode,
                    tool_start_time,
                    tool_end_time,
                    tool_latency_ms,
                )

            self._update_final_answer(result, tool_response)

            if not tool_response.success and result.status == ExecutionStatus.SUCCESS:
                result.status = ExecutionStatus.PARTIAL

        except Exception as e:
            self._handle_tool_execution_error(
                frame, tool_call, tool_action, execution_mode, tool_start_time, e
            )

    def _validate_tool_call(
        self, frame: ExecutionFrame, tool_call: LLMToolCall, result: ExecutionResult
    ):
        """Validates that the tool call is for an exposed tool."""
        exposed_tool_names = []
        for tool_schema in frame.rendered_prompt.tool_schemas:
            if isinstance(tool_schema, dict) and "function" in tool_schema:
                exposed_tool_names.append(tool_schema["function"].get("name", ""))

        if tool_call.name not in exposed_tool_names:
            error_msg = f"LLM initiated tool call of {tool_call.name}, which was not exposed to LLM"
            # Set BLOCKED status instead of raising - this is a policy/configuration issue
            result.status = ExecutionStatus.PARTIAL
            result.error_message = error_msg
            result.handler_context = {
                "error_type": "tool_failures",
                "message": error_msg,
            }
            raise ValueError(error_msg)

    def _handle_ptc_mode(
        self,
        frame: ExecutionFrame,
        tool_call: LLMToolCall,
        tool_response: ToolExecutionResponse,
        result: ExecutionResult,
        tool_action,
        execution_mode,
    ):
        """Handles tool call execution in PTC (sandbox) mode."""
        sandbox_tool_calls = tool_response.tool_calls or []
        child_call_ids: list[str] = []
        for call in sandbox_tool_calls:
            if isinstance(call, dict):
                call = ToolCallTrace(**call)
            call.parent_call_id = tool_action.id
            call.execution_mode = execution_mode
            child_call_ids.append(call.call_id)
            child_action = frame.add_action(
                ActionKind.TOOL_CALL,
                name=call.tool_name,
                _input=call.input,
            )
            frame.complete_action(
                child_action.id,
                output=call,  # Pass ToolCallTrace directly
                metadata={
                    "tool_trace": call,  # Store full ToolCallTrace
                },
            )

        self._handle_ptc_envelope(
            frame,
            tool_call,
            tool_response,
            result,
            tool_action,
            child_call_ids,
            execution_mode,
        )

    def _handle_ptc_envelope(
        self,
        frame: ExecutionFrame,
        tool_call: LLMToolCall,
        tool_response: ToolExecutionResponse,
        result: ExecutionResult,
        tool_action,
        child_call_ids: list[str],
        execution_mode,
    ):
        """Handles PTC mode with envelope data."""
        if tool_response.envelope is None:
            wrapper_metadata = {
                "success": tool_response.success,
                "execution_time_ms": tool_response.execution_time_ms,
                "child_call_ids": child_call_ids,
            }

            frame.complete_action(
                tool_action.id,
                output=tool_response.result
                if tool_response.success
                else tool_response.error_message,
                metadata=wrapper_metadata,
            )

            return

        if tool_response.success:
            output = (
                tool_response.result
                or tool_response.envelope.stdout
                or tool_response.envelope.stderr
            )
        else:
            output = tool_response.envelope.stderr or tool_response.error_message
        logger.debug(
            f"Sandbox tool call {tool_call.name} completed with output: {output}"
        )

        if result.files_created is None:
            result.files_created = []
        if result.files_modified is None:
            result.files_modified = []
        result.files_created.extend(tool_response.envelope.files_created or [])
        result.files_modified.extend(tool_response.envelope.files_modified or [])

        tool_trace = ToolCallTrace(
            call_id=tool_action.id,
            tool_name=tool_call.name,
            start_time=tool_response.envelope.start_time,
            end_time=tool_response.envelope.end_time,
            duration_ms=tool_response.envelope.duration_ms,
            success=tool_response.envelope.success,
            error=tool_response.envelope.stderr or tool_response.envelope.error,
            input=tool_call.arguments,
            output=output,
            resource_metrics=tool_response.envelope.system_metrics,
            child_call_ids=child_call_ids,
            execution_mode=execution_mode,
        )
        frame.complete_action(
            tool_action.id,
            output=tool_trace,
            metadata={
                "tool_trace": tool_trace,
            },
        )

    def _handle_classic_mode(
        self,
        frame: ExecutionFrame,
        tool_call: LLMToolCall,
        tool_response: ToolExecutionResponse,
        tool_action,
        execution_mode,
        tool_start_time,
        tool_end_time,
        tool_latency_ms,
    ):
        """Handles tool call execution in classic mode."""
        # Create ToolCallTrace for classic tool call
        tool_trace = ToolCallTrace(
            call_id=tool_action.id,
            tool_name=tool_call.name,
            start_time=datetime.fromtimestamp(tool_start_time, UTC),
            end_time=datetime.fromtimestamp(tool_end_time, UTC),
            duration_ms=tool_latency_ms,
            success=tool_response.success,
            error=tool_response.error_message if not tool_response.success else None,
            input=tool_call.arguments,
            output=tool_response.result
            if tool_response.success
            else tool_response.error_message,
            resource_metrics=[],
            child_call_ids=[],
            execution_mode=execution_mode,
        )

        frame.complete_action(
            tool_action.id,
            output=tool_trace,  # Store ToolCallTrace directly
        )

    def _update_final_answer(
        self, result: ExecutionResult, tool_response: ToolExecutionResponse
    ):
        """Updates the final answer in the result if the tool response indicates it's final."""
        if tool_response.is_final:
            # Use PresentationOutput for smart formatting if available
            if tool_response.output and tool_response.output.final_answer is not None:
                final_value = tool_response.output.final_answer
                format_ = tool_response.output.format
                if format_ == "json":
                    import json

                    result.final_answer = json.dumps(final_value)
                else:
                    # Default to string conversion for 'text' or unknown formats
                    result.final_answer = str(final_value)
            else:
                # Fallback to raw result
                result.final_answer = str(tool_response.result)

    def _handle_tool_execution_error(
        self,
        frame: ExecutionFrame,
        tool_call: LLMToolCall,
        tool_action,
        execution_mode,
        tool_start_time,
        error: Exception,
    ):
        """Handles errors during tool execution."""
        logger.error("Error executing tool %s: %s", tool_call.name, str(error))
        tool_end_time = time.perf_counter()
        tool_latency_ms = (tool_end_time - tool_start_time) * 1000

        # Record failed tool using ToolCallTrace
        failed_tool_trace = ToolCallTrace(
            call_id=tool_action.id,
            tool_name=tool_call.name,
            start_time=datetime.fromtimestamp(tool_start_time, UTC),
            end_time=datetime.fromtimestamp(tool_end_time, UTC),
            duration_ms=tool_latency_ms,
            success=False,
            error=str(error),
            input=tool_call.arguments,
            output=str(error),
            execution_mode=execution_mode,
        )

        frame.complete_action(
            tool_action.id,
            output=failed_tool_trace,  # Store ToolCallTrace directly
            metadata={
                "success": False,
                "error": str(error),
            },
        )

    @staticmethod
    def _combine_sections(sections: list[str] | None) -> str:
        if not sections:
            return ""
        return "\n\n".join(s for s in sections if s).strip()

    def _should_capture_reasoning(self) -> bool:
        if not self._config_manager:
            return False
        llm_config = self._config_manager.global_config.llm
        return bool(getattr(llm_config, "capture_reasoning", False))

    def _truncate_reasoning(self, reasoning: str) -> tuple[str, int]:
        total_chars = len(reasoning)
        max_chars = 0
        if self._config_manager:
            max_chars = (
                getattr(
                    self._config_manager.global_config.llm, "reasoning_max_chars", 0
                )
                or 0
            )
        if max_chars <= 0 or total_chars <= max_chars:
            return reasoning, total_chars
        return reasoning[-max_chars:], total_chars

    @staticmethod
    def _extract_reasoning_tokens(usage: dict[str, Any] | None) -> int | None:
        if not usage:
            return None
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            tokens = completion_details.get("reasoning_tokens")
            if tokens is not None:
                try:
                    return int(tokens)
                except (TypeError, ValueError):
                    return None
        return None

    async def _generate_llm_result(  # noqa: C901
        self,
        task: LLMTask,
        *,
        options: LLMOptions,
        streaming: bool,
    ) -> LLMResult:
        if not streaming:
            return await self._llm_service.generate(task, options=options)

        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        tool_calls: list[LLMToolCall] = []
        usage: dict[str, Any] | None = None
        metadata: dict[str, Any] = {}
        model = "unknown"
        provider = "unknown"
        finish_reason: str | None = None

        async for chunk in self._llm_service.stream(task, options=options):
            if chunk.content:
                content_chunks.append(chunk.content)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.reasoning:
                reasoning_chunks.append(chunk.reasoning)
            if chunk.usage:
                usage = chunk.usage
            if chunk.metadata:
                metadata = chunk.metadata
            if chunk.model:
                model = chunk.model
            if chunk.provider:
                provider = chunk.provider
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        prompt_tokens, completion_tokens, total_tokens = self._extract_usage_metrics(
            usage
        )

        reasoning = None
        if reasoning_chunks and self._should_capture_reasoning():
            raw_reasoning = "".join(reasoning_chunks)
            reasoning, _ = self._truncate_reasoning(raw_reasoning)

        reasoning_tokens = self._extract_reasoning_tokens(usage)

        return LLMResult(
            content="".join(content_chunks),
            model=model,
            provider=provider,
            finish_reason=finish_reason,
            reasoning=reasoning,
            reasoning_tokens=reasoning_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=tool_calls,
            metadata={
                "usage": usage,
                "provider_metadata": metadata,
            },
        )

    @staticmethod
    def _extract_usage_metrics(
        usage: dict[str, Any] | None,
    ) -> tuple[int | None, int | None, int | None]:
        if not usage:
            return None, None, None

        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens
