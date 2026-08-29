"""
This module provides `RuntimeManager`, which is responsible for orchestrating
end-to-end query handling across the LLM and tools using a per-run session.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from local_coding_assistant.agent.llm import (
    LLMOptions,
    LLMResult,
    LLMService,
    LLMTask,
    LLMToolCall,
)
from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.prompt import PromptComposer
from local_coding_assistant.repository import RepositoryContextService
from local_coding_assistant.runtime.agent_types import AgentRequest
from local_coding_assistant.runtime.context_manager import ContextManager
from local_coding_assistant.runtime.events import EventType, ExecutionEvent
from local_coding_assistant.runtime.handlers.handler_types import (
    HandlerContext,
    HandlerErrorType,
)
from local_coding_assistant.runtime.reporting import RunMetrics, RunReport
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.types import (
    ToolExecutionRequest,
    ToolExecutionResponse,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools.tool_manager import ToolManager

from local_coding_assistant.runtime.dashboard_integration import (
    collect_event_for_dashboard,
)
from local_coding_assistant.runtime.execution_types import ExecutionStatus
from local_coding_assistant.runtime.handlers.handler_integration import (
    HandlerIntegration,
)
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.runtime_manager")


class RuntimeManager:
    def __init__(
        self,
        config_manager: IConfigManager,
        llm_service: LLMService | None = None,
        tool_manager: IToolManager | ToolManager | None = None,
        repository_context_service: RepositoryContextService | None = None,
    ) -> None:
        """Initialize the runtime manager.

        Args:
            config_manager: The config manager to use for configuration (required)
            llm_service: The LLM service to use for generating responses
            tool_manager: The tool manager to use for executing tools
            repository_context_service: The repository context service for code analysis
        """
        if config_manager is None:
            raise ValueError("config_manager is required")

        self._llm_service = llm_service
        self._tool_manager = tool_manager
        self._repository_context_service = repository_context_service
        self.config_manager = config_manager
        self._context_manager = ContextManager(
            config_manager=config_manager,
            tool_manager=tool_manager,
            repository_context_service=repository_context_service,
        )
        self._prompt_composer = PromptComposer(config_manager=config_manager)

        # Initialize handler integration for partial response handling
        self._handler_integration = HandlerIntegration()

        # Ensure config manager has global configuration loaded
        if (
            not hasattr(self.config_manager, "global_config")
            or self.config_manager.global_config is None
        ):
            self.config_manager.load_global_config()

        self.session: SessionState | None = None

    def start(self) -> None:
        """Start the runtime (no-op placeholder)."""
        logger.debug("RuntimeManager.start() called (no-op)")

    def stop(self) -> None:
        """Stop the runtime (no-op placeholder)."""
        logger.debug("RuntimeManager.stop() called (no-op)")

    async def orchestrate(
        self,
        text: str,
        *,
        agent_mode: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_call_mode: str | None = None,
        sandbox_session: str | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Unified entrypoint: run a single query and return structured output.

        Args:
            text: The input text/query to process
            agent_mode: Optional agent mode override ('default_loop', 'graph', 'frame', or None)
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            tool_call_mode: Optional tool call mode ('classic', 'ptc' or 'reasoning_only'). If None, uses the mode from config.
            sandbox_session: Optional session ID for sandbox persistence.

        Returns:
            Structured output with session_id, message, model_used, tokens_used,
            tool_calls, and history.
            Uses agent_mode to determine which agent implementation to use.
        """
        session = self._setup_session()

        request = AgentRequest(
            user_input=text,
            session=session,
            agent_mode=agent_mode,
            model_override=model,
            temperature_override=temperature,
            max_tokens_override=max_tokens,
            tool_call_mode_override=tool_call_mode,
            sandbox_session_override=sandbox_session,
        )

        # Set session overrides based on request
        if (
            request.agent_mode is not None
            and request.agent_mode
            != self.config_manager.global_config.runtime.agent_mode
        ):
            self.config_manager.set_session_overrides(
                {"runtime.agent_mode": request.agent_mode}
            )

        if request.tool_call_mode_override is not None:
            self.config_manager.set_session_overrides(
                {"runtime.tool_call_mode": request.tool_call_mode_override}
            )

            if request.tool_call_mode_override == "ptc":
                self.config_manager.set_session_overrides({"sandbox.enabled": True})
                if request.sandbox_session_override:
                    self.config_manager.set_session_overrides(
                        {
                            "sandbox.session_id": request.sandbox_session_override,
                            "sandbox.persistence": True,
                        }
                    )

        # Only set model override if provided
        if request.model_override:
            self.config_manager.set_session_overrides(
                {"llm.model_name": request.model_override}
            )

        if self.config_manager.global_config.runtime.agent_mode != "no_agent":
            # Lazy initialize repository service if needed
            if self._repository_context_service:
                self._repository_context_service.ensure_initialized()

            async for event in self._run_agent_mode(request):
                await collect_event_for_dashboard(event)
                yield event
            return

        # Regular mode execution
        async for event in self._run_regular_mode(request):
            await collect_event_for_dashboard(event)
            yield event
        return

    def _setup_session(self) -> SessionState:
        """Setup and configure the session for the current request."""
        # Resolve session and reset if non-persistent
        if self.session is None:
            self.session = SessionState()

        session = self.session

        # Check if persistent_sessions is False or not set (default to False)
        if not self.config_manager.global_config.runtime.persistent_sessions:
            session.reset()

        return session

    async def _handle_direct_tool_call(
        self, text: str
    ) -> tuple[dict[str, Any] | None, str]:
        """Handle direct tool invocation from user input.

        Args:
            text: The user input text

        Returns:
            Tuple of (tool_outputs, processed_text)
        """
        if text.startswith("tool:"):
            from local_coding_assistant.core.exceptions import ToolRegistryError

            if self._tool_manager is None:
                logger.warning(
                    "Tool functionality is not available (tool_manager is None)"
                )
                return None, "Tool functionality is not available"

            try:
                _, rest = text.split(":", 1)
                name, payload_str = rest.strip().split(" ", 1)
                payload = json.loads(payload_str)

                # Use ToolExecutionRequest for better error handling
                request = ToolExecutionRequest(tool_name=name, payload=payload)
                response = await self._tool_manager.execute_async(request)

                if not response.success:
                    error_msg = (
                        f"Tool {name} execution failed: {response.error_message}"
                    )
                    logger.error(error_msg)
                    return None, error_msg

                tool_outputs = {name: response.result}
                logger.debug("Tool invoked: %s => %s", name, response.result)

                # Include execution time in the response
                exec_time = (
                    f"{response.execution_time_ms:.2f}ms"
                    if response.execution_time_ms
                    else "unknown"
                )
                processed_text = f"Tool {name} executed successfully in {exec_time}"
                return tool_outputs, processed_text

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool payload: {e!s}"
                logger.error(error_msg)
                return None, error_msg
            except ToolRegistryError:
                # Re-raise ToolRegistryError to be handled by the caller
                raise
            except Exception as e:
                logger.error("Tool invocation failed", error=str(e), exc_info=True)
                return None, f"Tool invocation failed: {e!s}"

        return None, text

    async def _prepare_llm_request(
        self,
        text: str,
        session: SessionState,
        tool_outputs: dict[str, Any] | None = None,
    ) -> tuple[LLMTask, str]:
        """Prepare the LLM request with context and configuration.

        Args:
            text: The input text/query to process
            session: The current session state
            tool_outputs: Optional tool outputs from previous steps


        Returns:
            LLMTask and policy name for model fallback
        """
        # Get tool_call_mode from config
        mode = self.config_manager.global_config.runtime.tool_call_mode

        prompt_context = self._context_manager.build_context(
            session=session,
            user_input=text,
            tool_call_mode=mode,
            agent_mode=False,
            handler_context=session.metadata.get("handler_context", None),
            agent_file_changes=session.metadata.get("agent_file_changes", None),
        )
        rendered_prompt = self._prompt_composer.render(prompt_context)
        system_prompt = self._combine_sections(rendered_prompt.system_messages)
        prompt_text = self._combine_sections(rendered_prompt.user_messages) or text

        request = LLMTask(
            prompt=prompt_text,
            context=rendered_prompt.history,
            tools=rendered_prompt.tool_schemas,
            tool_outputs=tool_outputs or {},
            system_prompt=system_prompt or session.system_prompt,
        )

        policy = (
            prompt_context.agent_profile.model_policy
            if prompt_context.agent_profile
            else ""
        )

        return request, policy

    def _build_history_entries(self, session: SessionState) -> list[dict[str, Any]]:
        history = [m.model_dump() for m in session.history]
        for tool_call in session.tool_calls:
            content = ""
            if tool_call.result is not None:
                content = json.dumps(tool_call.result, ensure_ascii=True)
            history.append(
                {
                    "role": "tool",
                    "content": content,
                    "metadata": {
                        "name": tool_call.name,
                        "args": tool_call.args,
                    },
                }
            )
        return history

    def _build_regular_report(
        self,
        session: SessionState,
        response: LLMResult,
        *,
        status: str = "success",
    ) -> RunReport:
        """Build a normalized run report for regular mode."""
        tool_calls = [tc.model_dump() for tc in session.tool_calls]
        history = self._build_history_entries(session)
        metrics = RunMetrics(tokens_used=response.total_tokens)

        return RunReport(
            run_id=f"run_{uuid.uuid4()}",
            session_id=session.id,
            mode="regular",
            status=status,
            final_answer=response.content,
            message=response.content,
            finish_reason=response.finish_reason,
            models_used=[response.model],
            tokens_used=response.total_tokens,
            tool_calls=tool_calls,
            history=history,
            metrics=metrics,
        )

    async def _handle_llm_tool_calls(
        self,
        session: SessionState,
        response: LLMResult,
        exposed_tools: list[dict[str, Any]],
    ) -> None:
        """Handle tool calls initiated by the LLM."""
        if not response.tool_calls:
            return

        if self._tool_manager is None:
            await self._handle_missing_tool_manager(session, response.tool_calls)
            return

        exposed_tool_names = []
        for exposed_tool in exposed_tools:
            tool_name = exposed_tool.get("function", {}).get("name", "")
            if tool_name:
                exposed_tool_names.append(tool_name)

        for tool_call in response.tool_calls:
            if tool_call.name not in exposed_tool_names:
                logger.warning(
                    "Tool call '%s' ignored: Tool not exposed to LLM", tool_call.name
                )
                return
            await self._process_single_tool_call(tool_call, session)

    async def _handle_missing_tool_manager(
        self, session: SessionState, tool_calls: list[LLMToolCall]
    ) -> None:
        """Handle tool calls when tool manager is not available."""
        logger.warning("Tool calls received but tool manager is not available")
        for tool_call in tool_calls:
            func_name = tool_call.name

            session.add_tool_message(
                call_id=tool_call.id if tool_call.id else "unknown",
                result="Tool functionality is not available",
            )
            logger.warning(
                "Tool call '%s' ignored: Tool manager not available", func_name
            )

    async def _process_single_tool_call(
        self, tool_call: LLMToolCall, session: SessionState
    ) -> None:
        """Process a single tool call from the LLM response."""
        call_id: str = tool_call.id if tool_call.id else "unknown"
        func_name = tool_call.name
        args = tool_call.arguments
        session_id = args.get("session_id", "")

        formatted_tool_call = {
            "id": getattr(tool_call, "id", ""),
            "type": "function",
            "function": {
                "name": getattr(tool_call, "name", ""),
                "arguments": getattr(tool_call, "arguments", "{}"),
            },
        }
        # Include extra_content if present
        extra_content = getattr(tool_call, "extra_content", {})
        if extra_content:
            formatted_tool_call["extra_content"] = extra_content
        session.add_assistant_message(tool_call=formatted_tool_call)

        if tool_call.type == "code" and not session_id:
            args["session_id"] = self.config_manager.global_config.sandbox.session_id

        try:
            tool_result = await self._execute_tool(func_name, args)

            self._log_tool_success(func_name, tool_result)
        except Exception as e:
            await self._handle_tool_error(e, func_name, call_id, session)
            return

        result = tool_result.get("result", "")
        session.add_tool_message(call_id=call_id, result=str(result))

    async def _execute_tool(
        self, func_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single tool and return its result."""
        request = ToolExecutionRequest(tool_name=func_name, payload=args)
        if self._tool_manager is None:
            return {"error": "Tool functionality is not available"}

        response = await self._tool_manager.execute_async(request)
        exec_time = (
            f"{response.execution_time_ms:.2f}ms"
            if response.execution_time_ms
            else "unknown"
        )

        if response.is_final:
            logger.info(
                "Final answer received from tool %s in %s", func_name, exec_time
            )
            return {
                "result": response.result,
                "is_final": True,
                "format": getattr(response, "format", "text"),
                "metadata": getattr(response, "metadata", {}),
            }

        if response.success:
            logger.debug("Tool %s executed successfully in %s", func_name, exec_time)

        else:
            logger.error(
                "Tool %s execution failed", func_name, error=response.error_message
            )

        return response.dried_out()

    async def _handle_tool_error(
        self,
        error: Exception,
        func_name: str,
        call_id: str,
        session: SessionState,
    ) -> None:
        """Handle general tool execution errors."""
        error_msg = str(error)
        logger.error("Tool call '%s' failed", func_name, error=error_msg)
        session.add_tool_message(
            call_id=call_id,
            result=error_msg,
        )

    def _log_tool_success(self, func_name: str, tool_result: Any) -> None:
        """Log successful tool execution."""
        logger.debug(f"LLM-initiated tool call of {func_name}", tool_result=tool_result)

    async def _run_regular_mode(
        self,
        request: AgentRequest,
    ) -> AsyncIterator[ExecutionEvent]:
        """Run a single query in regular mode with handler integration."""
        session = request.session

        yield ExecutionEvent(EventType.TURN_START, session.id, None)

        user_message, tool_outputs = await self._record_user_message(
            request.user_input, session
        )

        attempts = 0
        max_attempts = 2
        llm_request = None
        response = None
        status = "success"
        base_options = {
            "model": request.model_override,
            "temperature": request.temperature_override,
            "max_tokens": request.max_tokens_override,
        }

        while attempts < max_attempts:
            attempts += 1
            llm_request, policy = await self._prepare_llm_request(
                user_message, session, tool_outputs
            )
            llm_service = self._require_llm_service()
            model_val = base_options.get("model", request.model_override)
            options = LLMOptions(
                model=cast(str | None, model_val),
                temperature=base_options.get(
                    "temperature", request.temperature_override
                ),
                max_tokens=base_options.get("max_tokens", request.max_tokens_override),
                policy=policy,
            )
            logger.debug("Calling llm service with options", options=options)
            response = await llm_service.generate(llm_request, options=options)
            logger.debug("LLM returned response; len=%d", len(response.content))

            error_type = None
            # Determine execution status based on finish reason
            execution_status = ExecutionStatus.SUCCESS
            if response.finish_reason == "length":
                execution_status = ExecutionStatus.PARTIAL
                error_type = "truncation"
            elif response.metadata.get("content_error"):
                execution_status = ExecutionStatus.PARTIAL
                error_type = "parsing_error"
            elif response.finish_reason in {"content_filter", "safety", "blocked"}:
                execution_status = ExecutionStatus.BLOCKED

            # Handle partial responses using handler integration
            if execution_status == ExecutionStatus.PARTIAL:
                # Create HandlerContext for the handler integration
                handler_context = HandlerContext(
                    session=session,
                    llm_response=None,
                    reasoning=response.reasoning,
                    reasoning_tokens=response.reasoning_tokens,
                    max_attempts=max_attempts,
                    attempt_count=attempts,
                    error_type=HandlerErrorType(error_type) if error_type else None,
                    error_message=response.metadata.get(
                        "error_message", "Unknown error"
                    ),
                    raw_response=response.content,
                    raw_tool_calls=response.metadata.get("raw_tool_calls", []),
                )

                handler_output = (
                    await self._handler_integration.handle_partial_response(
                        execution_status=execution_status,
                        handler_context=handler_context,
                    )
                )

                # Check if we should continue
                if self._handler_integration.should_continue_execution(handler_output):
                    # Store handler context in session metadata for next iteration
                    session.metadata["handler_context"] = {
                        **(handler_output.handler_context or {}),
                        "template_path": handler_output.template_path,
                    }
                    # Adjust options for retry if needed using new method
                    base_options = self._handler_integration.get_adjusted_llm_options(
                        base_options=base_options, handler_output=handler_output
                    )
                    continue
                else:
                    status = "failed"
                    break

            # For other statuses, break the loop
            if execution_status == ExecutionStatus.BLOCKED:
                status = "blocked"
                break
            elif execution_status == ExecutionStatus.SUCCESS:
                status = "success"
                break

            # If we get here, continue to next attempt
            if attempts < max_attempts:
                continue
            break

        # Record assistant message
        assert response is not None, "Response should be set after LLM call"
        session.add_user_message(request.user_input)
        session.add_assistant_message(response.content)

        # Handle LLM-initiated tool calls and build report
        assert llm_request is not None, "Request should be set before LLM call"
        await self._handle_llm_tool_calls(session, response, llm_request.tools)
        report = self._build_regular_report(session, response, status=status)

        logger.info("Runtime finished query; session_id=%s", session.id)

        yield ExecutionEvent(
            EventType.TURN_COMPLETE, session.id, None, {"report": report}
        )

    async def _run_agent_mode(
        self,
        request: AgentRequest,
    ) -> AsyncIterator[ExecutionEvent]:
        """Run the runtime in agent mode, delegating to AgentLoop, LangGraphAgent or FrameAgent."""
        # Determine which agent implementation to use
        runtime_config = self.config_manager.global_config.runtime
        current_agent_mode = request.agent_mode or runtime_config.agent_mode
        stream_mode = (
            request.streaming
            if request.streaming is not None
            else runtime_config.stream
        )

        if current_agent_mode == "frame":
            async for event in self._run_frame_agent_mode(request):
                yield event
        elif current_agent_mode == "graph":
            session = request.session
            report = await self._run_langgraph_agent_mode(
                request.user_input,
                request.model_override,
                request.temperature_override,
                request.max_tokens_override,
                stream_mode,
            )
            yield ExecutionEvent(
                EventType.TURN_COMPLETE, session.id, None, {"report": report}
            )
        else:
            session = request.session
            report = await self._run_legacy_agent_mode(
                request.user_input,
                request.model_override,
                request.temperature_override,
                request.max_tokens_override,
                stream_mode,
            )
            yield ExecutionEvent(
                EventType.TURN_COMPLETE, session.id, None, {"report": report}
            )

    async def _run_frame_agent_mode(
        self,
        request: AgentRequest,
    ) -> AsyncIterator[ExecutionEvent]:
        """Run the runtime in agent mode using the new FrameAgent."""
        from local_coding_assistant.agent.frame_agent import FrameAgent

        if self._llm_service is None or self._tool_manager is None:
            session = request.session
            report = RunReport(
                mode="frame",
                status="failed",
                final_answer=None,
                message=None,
                errors=[],
            )
            yield ExecutionEvent(
                EventType.TURN_COMPLETE, session.id, None, {"report": report}
            )
            return

        # Setup session for the agent
        session = request.session

        # Create frame agent
        agent = FrameAgent(
            llm_service=self._llm_service,
            tool_manager=self._tool_manager,
            context_manager=self._context_manager,
            config_manager=self.config_manager,
            name="orchestrated_agent",
            max_iterations=request.max_iterations or 5,
        )

        # Run the agent and yield events
        async for event in agent.run(request.user_input, session):
            yield event

    async def _run_langgraph_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> RunReport:
        """Run the runtime in agent mode using LangGraphAgent."""
        # Import LangGraphAgent locally to avoid circular imports
        from local_coding_assistant.agent.langgraph_agent import (
            AgentState,
            LangGraphAgent,
        )

        if self._llm_service is None:
            return RunReport(
                mode="graph",
                status="failed",
                final_answer=None,
                message=None,
                errors=[],
            )

        # Create LangGraph agent with runtime components
        agent = LangGraphAgent(
            llm_service=self._llm_service,
            tool_manager=self._tool_manager if self._tool_manager is not None else None,
            name="runtime_langgraph_agent",
            streaming=streaming,
        )

        # Create initial state with the user input
        initial_state = AgentState()
        initial_state.user_input = text
        initial_state.max_iterations = 10  # Could be made configurable

        if streaming:
            # Run in streaming mode
            final_answer = None
            history = []
            async for state, _ in agent.run_stream(initial_state):
                history = state.history
                if state.final_answer:
                    final_answer = state.final_answer
                    break

            return RunReport(
                run_id=f"run_{uuid.uuid4()}",
                session_id=initial_state.session_id,
                mode="graph",
                status="success" if final_answer else "failed",
                final_answer=final_answer,
                message=final_answer,
                iterations=len(history),
                history=history,
            )
        else:
            # Run in non-streaming mode
            final_answer = await agent.run(initial_state)
            history = agent.get_history()

            return RunReport(
                run_id=f"run_{uuid.uuid4()}",
                session_id=initial_state.session_id,
                mode="graph",
                status="success" if final_answer else "failed",
                final_answer=final_answer,
                message=final_answer,
                iterations=initial_state.iteration,
                history=history,
            )

    async def _run_legacy_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> RunReport:
        """Run the runtime in agent mode using legacy AgentLoop."""
        # Import AgentLoop locally to avoid circular imports
        from local_coding_assistant.agent.agent_loop import AgentLoop

        if self._llm_service is None:
            return RunReport(
                mode="legacy",
                status="failed",
                final_answer=None,
                message=None,
                errors=[],
            )

        # Create agent loop with runtime components
        agent_loop = AgentLoop(
            llm_service=self._llm_service,
            tool_manager=self._tool_manager if self._tool_manager is not None else None,
            name="runtime_agent",
            streaming=streaming,
        )

        # Run the agent loop
        final_answer = await agent_loop.run()

        # Return structured result
        history = agent_loop.get_history()
        return RunReport(
            run_id=f"run_{uuid.uuid4()}",
            session_id=agent_loop.session_id,
            mode="legacy",
            status="success" if final_answer else "failed",
            final_answer=final_answer,
            message=final_answer,
            iterations=agent_loop.current_iteration,
            history=history,
        )

    async def _record_user_message(
        self, text: str, session: SessionState
    ) -> tuple[str, dict[str, Any] | None]:
        """Handle direct tool calls and record the effective user message."""
        tool_outputs, processed_text = await self._handle_direct_tool_call(text)

        if tool_outputs:
            for _, result in tool_outputs.items():
                session.add_tool_message(call_id="id000", result=str(result))

        return processed_text, tool_outputs

    def _require_llm_service(self) -> LLMService:
        """Ensure an LLM service with generation capability is available."""
        if not self._llm_service or not hasattr(self._llm_service, "generate"):
            raise RuntimeError(
                "LLM service is not available or does not support generation"
            )

        return self._llm_service

    def _should_capture_reasoning(self) -> bool:
        llm_config = self.config_manager.global_config.llm
        return bool(getattr(llm_config, "capture_reasoning", False))

    @staticmethod
    def _combine_sections(sections: list[str] | None) -> str:
        if not sections:
            return ""
        return "\n\n".join(s for s in sections if s).strip()

    async def run_programmatic_tool_call(
        self,
        code: str,
        session_id: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> ToolExecutionResponse:
        """Execute code using the programmatic tool calling interface.

        Args:
            code: The Python code to execute.
            session_id: Optional session ID (uses current session if None).
            env_vars: Optional environment variables.

        Returns:
            The execution result.
        """
        if self._tool_manager is None:
            return ToolExecutionResponse(
                success=False,
                tool_name="execute_python_code",
                tool_args={"code": code},
                error_message="Tool functionality is not available",
                execution_time_ms=0.0,
            )

        if session_id is None:
            session_id = self.config_manager.global_config.sandbox.session_id

        try:
            # Now we can call it directly as it's part of the interface
            return await self._tool_manager.run_programmatic_tool_call(
                code=code, session_id=session_id, env_vars=env_vars
            )

        except Exception as e:
            logger.error("Programmatic tool call failed", error=str(e), exc_info=True)
            return ToolExecutionResponse(
                success=False,
                tool_name="execute_python_code",
                tool_args={"code": code},
                error_message=str(e),
                execution_time_ms=0.0,
            )
