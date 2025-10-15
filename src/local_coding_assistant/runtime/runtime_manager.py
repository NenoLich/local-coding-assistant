"""Runtime management (task loop, process orchestration).

This module provides `RuntimeManager`, which is responsible for orchestrating
end-to-end query handling across the LLM and tools using a per-run session.
"""

from __future__ import annotations

import json
from typing import Any

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest
from local_coding_assistant.config.schemas import RuntimeConfig
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.utils.logging import get_logger


class RuntimeManager:
    def __init__(
        self, llm_manager: LLMManager, tool_manager: ToolManager, config: RuntimeConfig
    ) -> None:
        """Initialize the runtime with its dependencies.

        Args:
            llm_manager: An LLM manager providing `generate()` method.
            tool_manager: A tool manager capable of being iterated or invoked.
            config: Runtime configuration settings.
        """
        self._llm_manager = llm_manager
        self._tool_manager = tool_manager
        self._log = get_logger("runtime.runtime_manager")
        self.config = config
        self.session: SessionState | None = (
            SessionState() if config.persistent_sessions else None
        )

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
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Unified entrypoint: run a single query and return structured output.

        Args:
            text: The input text/query to process
            agent_mode: If True, delegate to AgentLoop for autonomous operation
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Structured output with session_id, message, model_used, tokens_used,
            tool_calls, and history. In agent_mode, returns agent-specific format.
        """
        if agent_mode:
            return await self._run_agent_mode(text, model, temperature, max_tokens)

        # Regular mode - single query execution
        return await self._run_regular_mode(text, model, temperature, max_tokens)

    async def _run_regular_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a single query in regular mode."""
        # Resolve session and reset if non-persistent
        session = self.session or SessionState()
        if not self.config.persistent_sessions:
            session.reset()

        self._log.info("Runtime starting query; model=%s", model or "default")

        # Record user message
        session.add_user_message(text)
        self._log.debug("Recorded user message; session_id=%s", session.id)

        # Handle direct tool invocation: "tool:<name> <json-payload>"
        tool_outputs: dict[str, Any] | None = None
        if text.startswith("tool:"):
            try:
                _, rest = text.split(":", 1)
                name, payload_str = rest.strip().split(" ", 1)
                payload = json.loads(payload_str)
                result = self._tool_manager.run_tool(name, payload)
                session.add_tool_message(name=name, args=payload, result=result)
                tool_outputs = {name: result}
                self._log.debug("Tool invoked: %s => %s", name, result)
            except Exception as e:
                self._log.error("Tool invocation failed: %s", e)
                raise

        # Prepare tools for LLM request
        available_tools = self._get_available_tools()

        # Create LLM request
        context = {
            "session_id": session.id,
            "history": [
                m.model_dump() for m in session.history[:-1]
            ],  # Exclude current user message for compatibility with tests
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
        if model is not None or temperature is not None or max_tokens is not None:
            self._llm_manager.update_config(
                model_name=model, temperature=temperature, max_tokens=max_tokens
            )
            self._log.debug("Applied LLM config overrides for request")

        # Ask LLM with context
        response = await self._llm_manager.generate(request)
        self._log.debug("LLM returned response; len=%d", len(response.content))

        # Record assistant message
        session.add_assistant_message(response.content)

        result: dict[str, Any] = {
            "session_id": session.id,
            "message": response.content,
            "model_used": response.model_used,
            "tokens_used": response.tokens_used,
            "tool_calls": [tc.model_dump() for tc in session.tool_calls],
            "history": [m.model_dump() for m in session.history],
        }

        # Handle LLM-initiated tool calls
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

        self._log.info("Runtime finished query; session_id=%s", session.id)
        return result

    async def _run_agent_mode(
        self,
        text: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Run the runtime in agent mode, delegating to AgentLoop."""
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
