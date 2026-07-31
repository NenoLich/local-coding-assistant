from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from local_coding_assistant.prompt import PromptComposer
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.execution_types import (
    ActionKind,
    ExecutionFrame,
    ExecutionStatus,
)
from local_coding_assistant.runtime.executor import RuntimeExecutor
from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.runtime.handlers.handler_types import HandlerContext
from local_coding_assistant.runtime.reporting import RunMetrics, RunReport
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.agent.llm import LLMService
    from local_coding_assistant.core.protocols import IConfigManager, IToolManager
    from local_coding_assistant.runtime.context_manager import ContextManager
    from local_coding_assistant.runtime.session import SessionState

logger = get_logger("agent.frame_agent")


class FrameAgent:
    """
    An agent that operates by executing ExecutionFrames.
    This agent standardizes each step into a frame, allowing for better
    observability and comparison.
    """

    def __init__(
        self,
        llm_service: LLMService,
        tool_manager: IToolManager,
        context_manager: ContextManager,
        config_manager: IConfigManager,
        name: str = "frame_agent",
        max_iterations: int = 5,
        max_continuation_attempts: int = 2,
    ):
        self.name = name
        self.max_iterations = max_iterations
        self._executor = RuntimeExecutor(
            llm_service,
            tool_manager,
            config_manager,
        )
        self._context_manager = context_manager
        self._config_manager = config_manager
        self._composer = PromptComposer(config_manager=config_manager)
        self.history: list[ExecutionFrame] = []
        self.session_id = f"agent_{name}_{int(time.time())}"

        # Initialize handler integration
        self.handler_integration = HandlerIntegration()

        # Track continuation attempts for proper retry logic
        self._continuation_attempts = 0
        self.max_continuation_attempts = max_continuation_attempts

    async def run(
        self, user_input: str, session: SessionState
    ) -> AsyncIterator[ExecutionEvent]:
        """
        Runs the agent loop until a final answer is reached or max iterations exceeded, yielding execution events.
        """
        logger.info("Starting FrameAgent loop for session %s", session.id)
        start_time = time.perf_counter()

        current_user_input = user_input
        current_iteration = 1
        last_answer = None

        yield ExecutionEvent(EventType.TURN_START, session.id, None)

        while current_iteration <= self.max_iterations:
            logger.info("Iteration %d/%d", current_iteration, self.max_iterations)

            # Build and execute frame, yielding events
            frame_generator = self._build_and_execute_frame(
                session, current_user_input, current_iteration
            )
            completed_frame = None
            async for event in frame_generator:
                yield event
                if event.type == EventType.FRAME_COMPLETE:
                    completed_frame = event.data["frame"]

            # Process the result
            (
                should_continue,
                new_iteration,
                final_answer,
            ) = await self._process_frame_result(
                completed_frame, session, current_iteration
            )

            if should_continue:
                current_user_input = ""

            if final_answer:
                last_answer = final_answer
                break

            if not should_continue:
                break

            current_iteration = new_iteration

        report = await self._finalize_run(
            session, current_iteration, last_answer, start_time
        )

        yield ExecutionEvent(
            EventType.TURN_COMPLETE, session.id, None, {"report": report}
        )

    async def _build_and_execute_frame(
        self, session: SessionState, current_user_input: str, current_iteration: int
    ) -> AsyncGenerator[ExecutionEvent]:
        """Build context, create frame, execute it and yield events, return completed frame."""
        # 1. Build Context and Frame
        # Resolve tool call mode - for now default to what context manager supports
        mode = self._config_manager.global_config.runtime.tool_call_mode

        # Get handler context from session metadata if available
        handler_context = session.metadata.get("handler_context")

        prompt_context = self._context_manager.build_context(
            session=session,
            user_input=current_user_input,
            tool_call_mode=mode,
            agent_mode=True,
            handler_context=handler_context,
        )

        rendered_prompt = self._composer.render(prompt_context)

        # Construct ExecutionFrame with new structure
        frame = ExecutionFrame(
            session_id=session.id,
            iteration=current_iteration,
            prompt_context=prompt_context,
            rendered_prompt=rendered_prompt,
        )

        # 2. Execute Frame and yield events
        async for event in self._executor.execute(frame):
            yield event

    async def _process_frame_result(
        self,
        completed_frame: ExecutionFrame | None,
        session: SessionState,
        current_iteration: int,
    ) -> tuple[bool, int, str | None]:
        """Process the result of a frame execution and return whether to continue and updated state."""
        if completed_frame is None:
            logger.error("No completed frame received")
            return False, current_iteration, None

        self.history.append(completed_frame)

        # 3. Process Result
        if not completed_frame.result or (
            completed_frame.result.status != ExecutionStatus.SUCCESS
            and completed_frame.result.status != ExecutionStatus.PARTIAL
        ):
            logger.error(
                "Frame execution failed: %s",
                completed_frame.result.error_message
                if completed_frame.result
                else "No result",
            )
            return False, current_iteration, None

        if completed_frame.result.status == ExecutionStatus.PARTIAL:
            should_continue, new_iteration = await self._handle_partial_response(
                completed_frame, session, current_iteration
            )
            return (
                should_continue,
                new_iteration if should_continue else current_iteration,
                None,
            )

        # Update session history based on frame actions
        self._update_session(session, completed_frame)

        # Reset attempt count on successful execution (non-partial response)
        if self._continuation_attempts > 0:
            logger.debug("Partial response sequence completed successfully")
            self._continuation_attempts = 0
            # Clear handler context from session metadata
            session.metadata.pop("handler_context", None)

        if completed_frame.result.final_answer:
            return False, current_iteration, completed_frame.result.final_answer

        # If there were tool calls, we continue loop with updated session history
        return True, current_iteration + 1, None

    async def _finalize_run(
        self,
        session: SessionState,
        current_iteration: int,
        last_answer: str | None,
        start_time: float,
    ) -> RunReport:
        """Finalize the run, calculate metrics and return the report."""
        logger.info(
            f"Frame agent run has ended on {current_iteration}/{self.max_iterations} iteration"
        )
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        models_used = set()
        last_frame = None
        total_tokens: int | None = None
        for frame in self.history:
            last_frame = frame
            metrics = frame.get_llm_metrics()
            if metrics:
                total_tokens = (total_tokens or 0) + (metrics.total_tokens or 0)
                if metrics.model is not None:
                    models_used.add(metrics.model)

        status = "success"
        finish_reason = None
        if last_frame and last_frame.result:
            status = (
                last_frame.result.status.value
                if hasattr(last_frame.result.status, "value")
                else str(last_frame.result.status)
            )
            finish_reason = last_frame.result.finish_reason
        elif last_answer is None:
            status = "failed"

        # Build report
        frames = [f.model_dump() for f in self.history]
        report = RunReport(
            run_id=f"run_{uuid.uuid4()}",
            session_id=session.id,
            mode="frame",
            status=status,
            final_answer=last_answer,
            message=last_answer,
            finish_reason=finish_reason,
            models_used=list(models_used),
            iterations=current_iteration,
            frames=frames,
            tokens_used=total_tokens,
            metrics=RunMetrics(
                tokens_used=total_tokens, total_latency_ms=total_latency_ms
            ),
        )

        return report

    def _update_session(self, session: SessionState, frame: ExecutionFrame):
        """Syncs the results from the executed frame back to the session state."""
        # Add user prompt
        if user_prompt := frame.rendered_prompt.get_user_prompt():
            session.add_user_message(user_prompt)

        # Add assistant message if LLM spoke
        if llm_response := frame.model_response_raw:
            session.add_assistant_message(content=llm_response)

        for action in frame.actions:
            if action.kind == ActionKind.LLM_MESSAGE and action.tool_calls:
                # Format tool_calls to Responses API structure
                for tc in action.tool_calls:
                    # Handle LLMToolCall objects
                    formatted_tool_call = {
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc, "name", ""),
                            "arguments": getattr(tc, "arguments", "{}"),
                        },
                    }
                    # Include extra_content if present
                    extra_content = getattr(tc, "extra_content", {})
                    if extra_content:
                        formatted_tool_call["extra_content"] = extra_content
                    session.add_assistant_message(tool_call=formatted_tool_call)

            # Add tool messages for each parent tool call in the frame
            if action.kind == ActionKind.TOOL_CALL and not action.metadata.get(
                "parent_call_id"
            ):
                call_id = (
                    getattr(action.tool_trace, "call_id", action.id)
                    if action.tool_trace
                    else action.id
                )
                result = (
                    action.output.get("result", "")
                    if isinstance(action.output, dict)
                    else str(action.output)
                )
                session.add_tool_message(
                    call_id=call_id,
                    result=str(result),
                )

    async def _handle_partial_response(
        self,
        completed_frame: ExecutionFrame,
        session: SessionState,
        current_iteration: int,
    ) -> tuple[bool, int]:
        # Reset attempt count for new partial response sequence
        if self._continuation_attempts == 0:
            logger.debug("Starting new partial response handling sequence")

        if completed_frame.result is None:
            return False, current_iteration

        failed_tools = self._collect_failed_tools(completed_frame.get_tool_results())

        # Create HandlerContext for the handler integration
        handler_context_for_handler = HandlerContext(
            session=session,
            llm_response=None,
            reasoning=completed_frame.get_reasoning(),
            reasoning_tokens=completed_frame.get_reasoning_tokens(),
            current_iteration=current_iteration,
            max_attempts=self.max_continuation_attempts,
            attempt_count=self._continuation_attempts,
            error_type=completed_frame.result.handler_context.get("error_type")
            if completed_frame.result.handler_context
            else None,
            error_message=completed_frame.result.handler_context.get("message")
            if completed_frame.result.handler_context
            else None,
            raw_response=completed_frame.result.handler_context.get("raw_response")
            if completed_frame.result.handler_context
            else None,
            failed_tools=failed_tools,
            raw_tool_calls=completed_frame.result.handler_context.get("raw_tool_calls")
            if completed_frame.result.handler_context
            else None,
            handler_data=completed_frame.result.handler_context.get("handler_data", {})
            if completed_frame.result.handler_context
            else {},
        )

        handler_output = await self.handler_integration.handle_partial_response(
            execution_status=completed_frame.result.status,
            handler_context=handler_context_for_handler,
        )
        # Check if we should continue
        if not self.handler_integration.should_continue_execution(handler_output):
            logger.warning(
                "Handler integration decided not to continue: %s",
                handler_output.error_message,
            )
            self._continuation_attempts = 0
            # Clear handler context from session metadata
            session.metadata.pop("handler_context", None)
            return False, current_iteration
        # Update session with current handler context
        if session.metadata.get("handler_context"):
            self._update_session(session, completed_frame)
        # Store handler context in session metadata for next iteration
        session.metadata["handler_context"] = {
            **(handler_output.handler_context or {}),
            "template_path": handler_output.template_path,
        }

        # Handle retry strategy (wait/delay if needed)
        if handler_output.retry_strategy == "retry_with_backoff":
            import asyncio

            wait_time = min(2**self._continuation_attempts, 10)
            logger.info("Waiting %d seconds before retry (backoff strategy)", wait_time)
            await asyncio.sleep(wait_time)
        elif handler_output.retry_strategy == "escalate":
            logger.warning("Escalating tool failure - not retrying")
            self._continuation_attempts = 0
            session.metadata.pop("handler_context", None)
            return False, current_iteration
        # Adjust options if needed
        if handler_output.adjusted_llm_options:
            self._config_manager.set_session_overrides(
                handler_output.adjusted_llm_options
            )
        self._continuation_attempts += 1
        return True, current_iteration + 1

    def _collect_failed_tools(self, tool_results: list[Any]) -> list[dict[str, Any]]:
        """Collect and classify failed tool results."""
        failed_tools = []

        for tool_result in tool_results:
            if not tool_result.get("success", True):
                # Extract tool information
                tool_name = tool_result.get("tool_name", "unknown")
                tool_args = tool_result.get("tool_args", {})
                error_message = tool_result.get("error_message", "Unknown error")

                # Classify the error
                failure_info = self.handler_integration.classify_tool_error(
                    tool_name=tool_name, tool_args=tool_args, error=error_message
                )

                failed_tools.append(failure_info)

        return failed_tools

    def get_frames(self) -> list[ExecutionFrame]:
        return self.history
