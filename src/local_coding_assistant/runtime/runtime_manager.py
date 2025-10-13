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
    """Coordinate LLM and tools with a session-aware runtime."""

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
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Unified entrypoint: run a single query and return structured output.

        - Uses a persistent session if enabled; otherwise resets a fresh one.
        - Records the user message, optionally invokes a tool directive in text,
          then asks the LLM with structured context and optional tool outputs.
        - Records the assistant message and returns a structured payload.
        """
        # Resolve session and reset if non-persistent
        session = self.session or SessionState()
        if not self.config.persistent_sessions:
            session.reset()

        self._log.info("Runtime starting query; model=%s", model or "default")

        # Record user message
        session.add_user_message(text)
        self._log.debug("Recorded user message; session_id=%s", session.id)

        # Simple tool invocation pipeline: "tool:<name> <json-payload>"
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
                # Let exceptions bubble; tests assert failures are surfaced
                self._log.error("Tool invocation failed: %s", e)
                raise

        # Prepare tools for LLM request
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
            # Update LLM config with overrides (no restoration needed)
            self._llm_manager.update_config(
                model_name=model, temperature=temperature, max_tokens=max_tokens
            )
            self._log.debug("Applied LLM config overrides for request")

        # Ask LLM with context (optionally including tool outputs)
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

        # Add tool calls from LLM response if present
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if "function" in tool_call:
                    func_name = tool_call["function"]["name"]
                    try:
                        # Parse arguments and invoke tool
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
