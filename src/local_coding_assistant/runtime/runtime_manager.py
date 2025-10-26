"""
This module provides `RuntimeManager`, which is responsible for orchestrating
end-to-end query handling across the LLM and tools using a per-run session.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from local_coding_assistant.agent.llm_manager import (
    LLMManager,
    LLMRequest,
    LLMResponse,
)
from local_coding_assistant.config import get_config_manager
from local_coding_assistant.runtime.session import SessionState

if TYPE_CHECKING:
    from local_coding_assistant.tools.tool_manager import ToolManager

from local_coding_assistant.utils.logging import get_logger


class RuntimeManager:
    def __init__(
        self, llm_manager: LLMManager, tool_manager: ToolManager, config_manager=None
    ) -> None:
        """Initialize the runtime with its dependencies.

        Args:
            llm_manager: An LLM manager providing `generate()` method.
            tool_manager: A tool manager capable of being iterated or invoked.
            config_manager: ConfigManager instance for resolving configuration.
                          If None, uses the global config manager.
        """
        self._llm_manager = llm_manager
        self._tool_manager = tool_manager
        self._log = get_logger("runtime.runtime_manager")
        self.config_manager = config_manager or get_config_manager()

        # Ensure config manager has global configuration loaded
        if self.config_manager.global_config is None:
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
    ) -> dict[str, Any]:
        """Unified entrypoint: run a single query and return structured output.

        Args:
            text: The input text/query to process
            agent_mode: If True, delegate to AgentLoop for autonomous operation
            graph_mode: If True, use LangGraph-based agent instead of legacy AgentLoop
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Structured output with session_id, message, model_used, tokens_used,
            tool_calls, and history. In agent_mode, returns agent-specific format.
        """
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
        runtime_config = self.config_manager.resolve().runtime
        if not runtime_config.persistent_sessions:
            session.reset()

        return session

    def _handle_direct_tool_call(self, text: str) -> tuple[dict[str, Any] | None, str]:
        """Handle direct tool invocation from user input.

        Args:
            text: The user input text

        Returns:
            Tuple of (tool_outputs, processed_text)
        """
        tool_outputs: dict[str, Any] | None = None

        if text.startswith("tool:"):
            try:
                _, rest = text.split(":", 1)
                name, payload_str = rest.strip().split(" ", 1)
                payload = json.loads(payload_str)
                result = self._tool_manager.run_tool(name, payload)
                tool_outputs = {name: result}
                self._log.debug("Tool invoked: %s => %s", name, result)
                # Return the tool result instead of the original text
                processed_text = f"Tool {name} executed successfully"
            except Exception as e:
                self._log.error("Tool invocation failed: %s", e)
                raise
        else:
            processed_text = text

        return tool_outputs, processed_text

    def _prepare_llm_request(
        self,
        text: str,
        session: SessionState,
        tool_outputs: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[LLMRequest, dict[str, Any]]:
        """Prepare the LLM request with context and configuration."""
        # Prepare tools for LLM request
        available_tools = self._get_available_tools()

        # Create LLM request
        context = {
            "session_id": session.id,
            "history": [
                m.model_dump() for m in session.history
            ],  # Include all history for context
            "tool_calls": [tc.model_dump() for tc in session.tool_calls],
            "metadata": session.metadata,
        }

        request = LLMRequest(
            prompt=text,
            context=context,
            tools=available_tools if available_tools else None,
            tool_outputs=tool_outputs,
            system_prompt=session.system_prompt,
        )

        # Handle configuration overrides
        overrides = self.config_manager.session_overrides.copy()
        if model is not None:
            overrides["llm.model_name"] = model
        if temperature is not None:
            overrides["llm.temperature"] = temperature
        if max_tokens is not None:
            overrides["llm.max_tokens"] = max_tokens

        # Update session configuration if overrides are provided
        if overrides:
            self.config_manager.set_session_overrides(overrides)

        return request, overrides

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

    def _handle_llm_tool_calls(
        self, session: SessionState, response: LLMResponse
    ) -> None:
        """Handle tool calls initiated by the LLM."""
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if "function" in tool_call:
                    func_name = tool_call["function"]["name"]
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        tool_result = self._tool_manager.run_tool(func_name, args)
                        session.add_tool_message(
                            name=func_name, args=args, result=tool_result
                        )
                        self._log.debug(
                            "LLM-initiated tool call: %s => %s", func_name, tool_result
                        )
                    except Exception as e:
                        self._log.error("LLM-initiated tool call failed: %s", e)

    async def _run_regular_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a single query in regular mode."""
        session = self._setup_session()

        # Record user message
        session.add_user_message(text)
        self._log.debug("Recorded user message; session_id=%s", session.id)

        # Handle direct tool invocation: "tool:<name> <json-payload>"
        tool_outputs, processed_text = self._handle_direct_tool_call(text)

        # If there were tool outputs, replace the original message with tool result
        # Otherwise, use the original text for LLM request
        if tool_outputs:
            # Record the original tool call for context
            for tool_name, result in tool_outputs.items():
                session.add_tool_message(name=tool_name, args={}, result=result)
            # Use processed text (tool result) for LLM request
            user_message_for_llm = processed_text
        else:
            # Use original text for LLM request
            user_message_for_llm = text

        # Prepare and execute LLM request
        request, overrides = self._prepare_llm_request(
            user_message_for_llm, session, tool_outputs, model, temperature, max_tokens
        )

        # Ask LLM with context
        response = await self._llm_manager.generate(
            request, overrides=overrides if overrides else None
        )
        self._log.debug("LLM returned response; len=%d", len(response.content))

        # Record assistant message
        session.add_assistant_message(response.content)

        # Handle LLM-initiated tool calls and build result
        result = self._build_result(session, response)
        self._handle_llm_tool_calls(session, response)

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
        overrides = {}
        if model is not None:
            overrides["llm.model_name"] = model
        if temperature is not None:
            overrides["llm.temperature"] = temperature
        if max_tokens is not None:
            overrides["llm.max_tokens"] = max_tokens

        # Update session configuration if overrides are provided
        if overrides:
            self.config_manager.set_session_overrides(overrides)

        # Determine which agent implementation to use
        runtime_config = self.config_manager.resolve().runtime
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

        # Create LangGraph agent with runtime components
        agent = LangGraphAgent(
            llm_manager=self._llm_manager,
            tool_manager=self._tool_manager,
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

        # Create agent loop with runtime components
        agent_loop = AgentLoop(
            llm_manager=self._llm_manager,
            tool_manager=self._tool_manager,
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

    def _get_available_tools(self) -> list[dict[str, Any]]:
        """Get available tools for LLM requests."""
        available_tools = []
        if hasattr(self._tool_manager, "__iter__"):
            for tool in self._tool_manager:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    available_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                            },
                        }
                    )
        return available_tools
