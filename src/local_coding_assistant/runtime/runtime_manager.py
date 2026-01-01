"""
This module provides `RuntimeManager`, which is responsible for orchestrating
end-to-end query handling across the LLM and tools using a per-run session.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from local_coding_assistant.agent.llm_manager import (
    LLMManager,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from local_coding_assistant.core.protocols import IConfigManager, IToolManager
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolExecutionRequest,
    ToolExecutionResponse,
)

if TYPE_CHECKING:
    from local_coding_assistant.tools.tool_manager import ToolManager

from local_coding_assistant.utils.logging import get_logger


class RuntimeManager:
    def __init__(
        self,
        config_manager: IConfigManager,
        llm_manager: LLMManager | None = None,
        tool_manager: IToolManager | ToolManager | None = None,
    ) -> None:
        """Initialize the runtime manager.

        Args:
            config_manager: The config manager to use for configuration (required)
            llm_manager: The LLM manager to use for generating responses
            tool_manager: The tool manager to use for executing tools
        """
        if config_manager is None:
            raise ValueError("config_manager is required")

        self._llm_manager = llm_manager
        self._tool_manager = tool_manager
        self._log = get_logger("runtime.runtime_manager")
        self.config_manager = config_manager

        # Ensure config manager has global configuration loaded
        if (
            not hasattr(self.config_manager, "global_config")
            or self.config_manager.global_config is None
        ):
            self.config_manager.load_global_config()

        self.session: SessionState | None = None

    def start(self) -> None:
        """Start the runtime (no-op placeholder)."""
        self._log.debug("RuntimeManager.start() called (no-op)")

    def stop(self) -> None:
        """Stop the runtime (no-op placeholder)."""
        self._log.debug("RuntimeManager.stop() called (no-op)")

    async def orchestrate(
        self,
        text: str,
        *,
        agent_mode: bool = False,
        graph_mode: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_call_mode: str | None = None,
        sandbox_session: str | None = None,
    ) -> dict[str, Any]:
        """Unified entrypoint: run a single query and return structured output.

        Args:
            text: The input text/query to process
            agent_mode: If True, delegate to AgentLoop for autonomous operation
            graph_mode: If True, use LangGraph-based agent instead of legacy AgentLoop
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            tool_call_mode: Optional tool call mode ('classic' or 'ptc'). If None, uses the mode from config.
            sandbox_session: Optional session ID for sandbox persistence.

        Returns:
            Structured output with session_id, message, model_used, tokens_used,
            tool_calls, and history. In agent_mode, returns agent-specific format.
        """
        # Update the config with the provided tool_call_mode if specified
        if tool_call_mode is not None:
            self.config_manager.set_session_overrides(
                {"runtime.tool_call_mode": tool_call_mode}
            )

            if tool_call_mode == "ptc":
                self.config_manager.set_session_overrides({"sandbox.enabled": True})

                # Set up the session with sandbox_session if provided
                if sandbox_session:
                    self.config_manager.set_session_overrides(
                        {
                            "sandbox.session_id": sandbox_session,
                            "sandbox.persistence": True,
                        }
                    )

        if agent_mode or graph_mode:
            return await self._run_agent_mode(
                text, model, temperature, max_tokens, graph_mode
            )

        # Regular mode execution
        return await self._run_regular_mode(text, model, temperature, max_tokens)

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
                self._log.warning(
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
                    self._log.error(error_msg)
                    return None, error_msg

                tool_outputs = {name: response.result}
                self._log.debug("Tool invoked: %s => %s", name, response.result)

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
                self._log.error(error_msg)
                return None, error_msg
            except ToolRegistryError:
                # Re-raise ToolRegistryError to be handled by the caller
                raise
            except Exception as e:
                self._log.error("Tool invocation failed: %s", e)
                return None, f"Tool invocation failed: {e!s}"

        return None, text

    async def _prepare_llm_request(
        self,
        text: str,
        session: SessionState,
        tool_outputs: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[LLMRequest, dict[str, Any]]:
        """Prepare the LLM request with context and configuration.

        Args:
            text: The input text/query to process
            session: The current session state
            tool_outputs: Optional tool outputs from previous steps
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            A tuple of (LLMRequest, overrides)
        """
        # Get tool_call_mode from config
        mode = self.config_manager.global_config.runtime.tool_call_mode

        # Get tools based on the current mode
        tools = self._get_available_tools()
        system_prompt = session.system_prompt or ""

        # In PTC mode, we need to add tool documentation to the system prompt
        if mode == "ptc" and self._tool_manager:
            sandbox_tools_prompt = self._tool_manager.get_sandbox_tools_prompt()
            if sandbox_tools_prompt:
                system_prompt = f"""{system_prompt}
 You are operating in Programmatic Tool Calling (PTC) mode.

In this mode, you DO NOT call tools using JSON.
Instead, you write a complete Python program that will be executed in a secure sandbox.

────────────────────────────────────────
HOW TO EXECUTE CODE
────────────────────────────────────────
When computation is needed, emit a Python program using the tool:

execute_python_code

The program must:
- Be valid Python
- Run top-to-bottom
- Use only allowed imports
- Call available tool functions directly
- Print intermediate values if helpful
- End by calling `final_answer(...)`

Example structure:

```python
from tools_api import sum_tool, final_answer

result = sum_tool(a=2, b=5)
print("Intermediate result:", result)

final_answer(f"The result is {{result}}")
```

The value passed to final_answer(...) will be shown to the user.

────────────────────────────────────────
AVAILABLE TOOL FUNCTIONS
────────────────────────────────────────
The following Python functions are available for use. You must import them from tools_api to be able to use them.
 
{sandbox_tools_prompt}

When using these tools, you need to import and call them directly in your code.

────────────────────────────────────────
ENVIRONMENT CONSTRAINTS
────────────────────────────────────────
You are running inside a restricted sandbox.

Allowed:

Standard Python syntax

Data processing logic

Loops, conditionals, functions

Imports from the standard library such as:
math, json, datetime, pathlib, collections, itertools, time

Forbidden:

eval(), exec(), compile()

import or dynamic imports

Accessing files outside /workspace

Modifying system files

Arbitrary shell execution

Network access unless explicitly provided via tools

Violating these rules will cause execution to fail.

────────────────────────────────────────
IMPORTANT
────────────────────────────────────────

Do NOT describe what you would do — write the code.

Do NOT return explanations outside of final_answer.

If a task cannot be completed safely, explain why using final_answer.
"""

        request = LLMRequest(
            prompt=text,
            context=self._build_llm_context(session),
            tools=tools,
            tool_outputs=tool_outputs or {},
            system_prompt=system_prompt,
        )

        overrides = self._apply_overrides(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

        return request, overrides

    def _build_llm_context(self, session: SessionState) -> dict[str, Any]:
        """Construct the contextual payload for LLM interactions."""
        return {
            "session_id": session.id,
            "history": [m.model_dump() for m in session.history],
            "tool_calls": [tc.model_dump() for tc in session.tool_calls],
            "metadata": session.metadata,
        }

    def _build_result(
        self, session: SessionState, response: LLMResponse
    ) -> dict[str, Any]:
        """Build the result dictionary from session and response."""
        return {
            "session_id": session.id,
            "message": response.content,
            "model_used": response.model_used,
            "tokens_used": response.tokens_used,
            "tool_calls": [tc.model_dump() for tc in session.tool_calls],
            "history": [m.model_dump() for m in session.history],
        }

    async def _handle_llm_tool_calls(
        self, session: SessionState, response: LLMResponse
    ) -> None:
        """Handle tool calls initiated by the LLM."""
        if not response.tool_calls:
            return

        if self._tool_manager is None:
            await self._handle_missing_tool_manager(session, response.tool_calls)
            return

        for tool_call in response.tool_calls:
            await self._process_single_tool_call(tool_call, session)

    async def _handle_missing_tool_manager(
        self, session: SessionState, tool_calls: list[ToolCall]
    ) -> None:
        """Handle tool calls when tool manager is not available."""
        self._log.warning("Tool calls received but tool manager is not available")
        for tool_call in tool_calls:
            func_name = tool_call.name
            args = tool_call.arguments

            session.add_tool_message(
                name=func_name,
                args=args,
                result={"error": "Tool functionality is not available"},
            )
            self._log.warning(
                "Tool call '%s' ignored: Tool manager not available", func_name
            )

    async def _process_single_tool_call(
        self, tool_call: ToolCall, session: SessionState
    ) -> None:
        """Process a single tool call from the LLM response."""
        func_name = tool_call.name
        args = tool_call.arguments
        session_id = args.get("session_id", "")

        if tool_call.type == "code" and not session_id:
            args["session_id"] = self.config_manager.global_config.sandbox.session_id

        try:
            tool_result = await self._execute_tool(func_name, args)

            self._log_tool_success(func_name, tool_result)
        except Exception as e:
            await self._handle_tool_error(e, func_name, args, session)
            return

        session.add_tool_message(name=func_name, args=args, result=tool_result)

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
            self._log.info(
                "Final answer received from tool %s in %s", func_name, exec_time
            )
            return {
                "result": response.result,
                "is_final": True,
                "format": getattr(response, "format", "text"),
                "metadata": getattr(response, "metadata", {}),
            }

        if response.success:
            self._log.debug("Tool %s executed successfully in %s", func_name, exec_time)

        else:
            self._log.error(
                "Tool %s execution failed: %s", func_name, response.error_message
            )

        return response.dried_out()

    async def _handle_tool_error(
        self,
        error: Exception,
        func_name: str,
        args: dict[str, Any],
        session: SessionState,
    ) -> None:
        """Handle general tool execution errors."""
        error_msg = str(error)
        self._log.error("Tool call '%s' failed: %s", func_name, error_msg)
        session.add_tool_message(
            name=func_name,
            args=args,
            result={"error": error_msg},
        )

    def _log_tool_success(self, func_name: str, tool_result: Any) -> None:
        """Log successful tool execution."""
        self._log.debug("LLM-initiated tool call: %s => %s", func_name, tool_result)

    async def _run_regular_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a single query in regular mode."""
        session = self._setup_session()

        user_message, tool_outputs = await self._record_user_message(text, session)

        request, overrides = await self._prepare_llm_request(
            user_message,
            session,
            tool_outputs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_manager = self._require_llm_manager()
        response = await llm_manager.generate(
            request, overrides=overrides if overrides else None
        )
        self._log.debug("LLM returned response; len=%d", len(response.content))

        # Record assistant message
        session.add_assistant_message(response.content)

        # Handle LLM-initiated tool calls and build result
        result = self._build_result(session, response)
        await self._handle_llm_tool_calls(session, response)

        self._log.info("Runtime finished query; session_id=%s", session.id)
        return result

    async def _run_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool | None = None,
        graph_mode: bool = False,
    ) -> dict[str, Any]:
        """Run the runtime in agent mode, delegating to AgentLoop or LangGraphAgent."""
        # Handle configuration overrides
        _overrides = self._apply_overrides(
            model=model, temperature=temperature, max_tokens=max_tokens
        )

        # Determine which agent implementation to use
        runtime_config = self.config_manager.global_config.runtime
        use_graph_mode = graph_mode or runtime_config.use_graph_mode
        stream_mode = streaming if streaming is not None else runtime_config.stream

        if use_graph_mode:
            return await self._run_langgraph_agent_mode(
                text, model, temperature, max_tokens, stream_mode
            )
        else:
            return await self._run_legacy_agent_mode(
                text, model, temperature, max_tokens, stream_mode
            )

    async def _run_langgraph_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Run the runtime in agent mode using LangGraphAgent."""
        # Import LangGraphAgent locally to avoid circular imports
        from local_coding_assistant.agent.langgraph_agent import (
            AgentState,
            LangGraphAgent,
        )

        if self._llm_manager is None:
            return {"error": "LLM manager is not available"}

        # Create LangGraph agent with runtime components
        agent = LangGraphAgent(
            llm_manager=self._llm_manager,
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

            return {
                "final_answer": final_answer,
                "iterations": len(history),
                "history": history,
                "session_id": initial_state.session_id,
                "streaming": True,
            }
        else:
            # Run in non-streaming mode
            final_answer = await agent.run(initial_state)
            history = agent.get_history()

            return {
                "final_answer": final_answer,
                "iterations": initial_state.iteration,
                "history": history,
                "session_id": initial_state.session_id,
                "streaming": False,
            }

    async def _run_legacy_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Run the runtime in agent mode using legacy AgentLoop."""
        # Import AgentLoop locally to avoid circular imports
        from local_coding_assistant.agent.agent_loop import AgentLoop

        if self._llm_manager is None:
            return {"error": "LLM manager is not available"}

        # Create agent loop with runtime components
        agent_loop = AgentLoop(
            llm_manager=self._llm_manager,
            tool_manager=self._tool_manager if self._tool_manager is not None else None,
            name="runtime_agent",
            streaming=streaming,
        )

        # Run the agent loop
        final_answer = await agent_loop.run()

        # Return structured result
        return {
            "final_answer": final_answer,
            "iterations": agent_loop.current_iteration,
            "history": agent_loop.get_history(),
            "session_id": agent_loop.session_id,
        }

    def _get_available_tools(self) -> list[dict[str, Any]] | None:
        """Return registered tools in OpenAI function-calling format.

        In PTC mode, only returns sandbox tools. In classic mode, returns all tools.
        """
        if self._tool_manager is None:
            return None

        # Get tool_call_mode from config
        mode = self.config_manager.global_config.runtime.tool_call_mode

        tool_specs: list[dict[str, Any]] = []
        category = ToolCategory.PTC if mode == "ptc" else None
        for entry in self._tool_manager.list_tools(
            available_only=True, category=category
        ):
            resolved = self._resolve_tool_entry(entry)
            if resolved is None:
                continue

            name, tool_obj = resolved
            spec = self._build_tool_spec(name, tool_obj)
            if spec is not None:
                tool_specs.append(spec)

        return tool_specs or None

    async def _record_user_message(
        self, text: str, session: SessionState
    ) -> tuple[str, dict[str, Any] | None]:
        """Handle direct tool calls and record the effective user message."""
        tool_outputs, processed_text = await self._handle_direct_tool_call(text)

        if tool_outputs:
            for tool_name, result in tool_outputs.items():
                session.add_tool_message(name=tool_name, args={}, result=result)

        session.add_user_message(processed_text)
        self._log.debug("Recorded user message; session_id=%s", session.id)

        return processed_text, tool_outputs

    def _apply_overrides(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Merge provided overrides with current session overrides."""
        base_overrides = self._current_overrides()
        overrides = base_overrides.copy()

        if model is not None:
            overrides["llm.model_name"] = model
        if temperature is not None:
            overrides["llm.temperature"] = temperature
        if max_tokens is not None:
            overrides["llm.max_tokens"] = max_tokens

        if overrides and overrides != base_overrides:
            self.config_manager.set_session_overrides(overrides)
        elif overrides and not base_overrides:
            self.config_manager.set_session_overrides(overrides)

        return overrides

    def _current_overrides(self) -> dict[str, Any]:
        """Return a defensive copy of current session overrides."""
        base = getattr(self.config_manager, "session_overrides", {}) or {}

        if isinstance(base, dict):
            return base.copy()

        if isinstance(base, Iterable):
            return dict(base)

        return {}

    def _require_llm_manager(self) -> LLMManager:
        """Ensure an LLM manager with generation capability is available."""
        if not self._llm_manager or not hasattr(self._llm_manager, "generate"):
            raise RuntimeError(
                "LLM manager is not available or does not support generation"
            )

        return self._llm_manager

    def _resolve_tool_entry(self, entry: Any) -> tuple[str, Any] | None:
        """Normalize tool entries from the tool manager iterator."""
        if isinstance(entry, tuple) and len(entry) == 2:
            name, tool_obj = entry
        else:
            name = getattr(entry, "name", str(entry))
            tool_obj = entry

        if not hasattr(tool_obj, "description"):
            return None

        return name, tool_obj

    def _build_tool_spec(self, name: str, tool_obj: Any) -> dict[str, Any] | None:
        """Create the OpenAI function definition for a tool."""
        try:
            parameters = self._extract_tool_parameters(tool_obj)
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log.warning("Failed to extract tool parameters: %s", exc)
            parameters = {"type": "object", "properties": {}, "required": []}

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool_obj.description,
                "parameters": parameters,
            },
        }

    def _extract_tool_parameters(self, tool_obj: Any) -> dict:
        """Extract parameters from a tool object.

        Args:
            tool_obj: The tool object to extract parameters from.
                     Can be a ToolInfo instance or any object with a 'parameters' attribute.

        Returns:
            Dictionary with parameter schema in OpenAI format:
            {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "..."},
                    "param2": {"type": "number", "description": "..."}
                },
                "required": ["param1"]
            }
        """
        # If tool_obj has a 'parameters' attribute, use that directly
        if hasattr(tool_obj, "parameters") and tool_obj.parameters is not None:
            return tool_obj.parameters

        # Fallback to empty schema if no parameters found
        self._log.debug(
            "No parameters found for tool: %s, using empty schema",
            getattr(tool_obj, "__name__", str(tool_obj)),
        )
        return {"type": "object", "properties": {}, "required": []}

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
            self._log.error(f"Programmatic tool call failed: {e}")
            return ToolExecutionResponse(
                success=False,
                tool_name="execute_python_code",
                error_message=str(e),
                execution_time_ms=0.0,
            )
