"""Runtime management (task loop, process orchestration).

This module provides `RuntimeManager`, which is responsible for orchestrating
end-to-end query handling across the LLM and tools using a per-run session.
"""

from __future__ import annotations

import json
from typing import Any

from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.utils.logging import get_logger


class RuntimeManager:
    """Coordinate LLM and tools with a session-aware runtime."""

    def __init__(self, llm: Any, tools: Any, persistent: bool = False) -> None:
        """Initialize the runtime with its dependencies.

        Args:
            llm: An LLM manager providing `ask_with_context()`.
            tools: A tool registry capable of being iterated or invoked.
            persistent: Reuse a default session across calls when True.
        """
        self._llm = llm
        self._tools = tools
        self._log = get_logger(__name__)
        self.persistent = persistent
        self.session: SessionState | None = SessionState() if persistent else None

    def start(self) -> None:
        """Start the runtime (no-op placeholder)."""
        self._log.debug("RuntimeManager.start() called (no-op)")

    def stop(self) -> None:
        """Stop the runtime (no-op placeholder)."""
        self._log.debug("RuntimeManager.stop() called (no-op)")

    def orchestrate(self, text: str, *, model: str | None = None) -> dict[str, Any]:
        """Unified entrypoint: run a single query and return structured output.

        - Uses a persistent session if enabled; otherwise resets a fresh one.
        - Records the user message, optionally invokes a tool directive in text,
          then asks the LLM with structured context and optional tool outputs.
        - Records the assistant message and returns a structured payload.
        """
        # Resolve session and reset if non-persistent
        session = self.session or SessionState()
        if not self.persistent:
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
                result = self._tools.invoke(name, payload)
                session.add_tool_message(name=name, args=payload, result=result)
                tool_outputs = {name: result}
                self._log.debug("Tool invoked: %s => %s", name, result)
            except Exception as e:
                # Let exceptions bubble; tests assert failures are surfaced
                self._log.error("Tool invocation failed: %s", e)
                raise

        # Ask LLM with context (optionally including tool outputs)
        message = self._llm.ask_with_context(
            session, tools=self._tools, model=model, tool_outputs=tool_outputs
        )
        self._log.debug("LLM returned message; len=%d", len(message))

        # Record assistant message
        session.add_assistant_message(message)

        result: dict[str, Any] = {
            "session_id": session.id,
            "message": message,
            "tool_calls": [tc.model_dump() for tc in session.tool_calls],
            "history": [m.model_dump() for m in session.history],
        }
        self._log.info("Runtime finished query; session_id=%s", session.id)
        return result
