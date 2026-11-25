"""Message projection from the runtime event stream.

The transcript is a pure projection of runtime events: every message is
derived from exactly one event as soon as the event exists - cancellations
and failures are data too. Frame and turn lifecycle boundaries never take
part in transcript construction, so pause/resume/crash cannot duplicate,
reorder, or lose messages.

Event -> message mapping:
- TurnStartEvent   -> user message     (dedupe key ``user:{turn_id}``)
- LLMCompleteEvent -> assistant message (key ``assistant:{frame_id}``);
                      nothing is written when the generation produced
                      neither content nor tool calls
- ToolResultEvent  -> tool message     (key ``tool:{turn_id}:{tool_call_id}``)
                      with output (completed), error text (failed), or an
                      interruption note (cancelled)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from local_coding_assistant.runtime.events.contracts import (
    LLMCompleteEvent,
    RuntimeEvent,
    ToolResultEvent,
    TurnStartEvent,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session.session import Message

logger = get_logger("runtime.message_service")


def _normalize_tool_calls(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize tool call dicts from an event into canonical transcript form.

    Stored in a simple internal form: ``{"id", "name", "arguments"}``. The
    driver layer is responsible for normalizing to the provider-specific wire
    format (e.g. OpenAI's ``{"id", "type": "function", "function": {...}}``)
    right before sending to the LLM.
    """
    calls: list[dict[str, Any]] = []
    for tc in raw or []:
        if not isinstance(tc, dict):
            continue
        function = tc.get("function")
        if not isinstance(function, dict):
            function = {}
        call: dict[str, Any] = {
            "id": str(tc.get("id", f"call_{id(tc)}")),
            "name": str(function.get("name", tc.get("name", ""))),
            "arguments": function.get("arguments", tc.get("arguments", {})),
        }
        extra_content = tc.get("extra_content")
        if extra_content:
            call["extra_content"] = extra_content
        calls.append(call)
    return calls


def _stringify_output(output: Any) -> str:
    """Render a tool output as transcript content."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    # If output is a Pydantic model with to_llm_dict, use it for LLM-friendly output
    if hasattr(output, "to_llm_dict") and callable(output.to_llm_dict):
        output = output.to_llm_dict()
    if isinstance(output, dict | list):
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(output)
    return str(output)


class MessageService:
    """Projects runtime events into canonical transcript messages.

    This is the only component that converts runtime events into
    conversation messages. It does not decide whether to continue, and it
    never reads frames: each message is written exactly once, when its
    source data arrives.
    """

    def __init__(self) -> None:
        self._logger = get_logger("runtime.message_service")

    def build_message(self, event: RuntimeEvent) -> tuple[Message, str] | None:
        """Project an event into ``(message, dedupe_key)``.

        Returns None for events with nothing to record - a failed or
        interrupted LLM generation writes nothing instead of an empty
        assistant message (debugging lives in ErrorEvent/frame records).
        """
        from local_coding_assistant.runtime.session.session import Message

        if isinstance(event, TurnStartEvent):
            return (
                Message(role="user", content=event.user_input),
                f"user:{event.turn_id}",
            )

        if isinstance(event, LLMCompleteEvent):
            tool_calls = _normalize_tool_calls(event.tool_calls)
            content = event.content or ""
            if not content and not tool_calls:
                # No assistant data produced (failed/transient generation)
                return None
            return (
                Message(
                    role="assistant",
                    content=content or None,
                    tool_calls=tool_calls or None,
                ),
                f"assistant:{event.frame_id}",
            )

        if isinstance(event, ToolResultEvent):
            from local_coding_assistant.tools_platform.errors import ToolErrorCode

            error = event.error or {}
            if event.success:
                content = _stringify_output(event.output)
            elif error.get("code") == ToolErrorCode.INTERRUPTED.value:
                # Cancellation is data: record that the tool did not finish
                content = (
                    "Tool call interrupted: "
                    f"{error.get('message', 'cancelled before completion')}"
                )
            else:
                content = f"Error: {error.get('message', 'Unknown error')}"
            return (
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=event.tool_call_id,
                ),
                f"tool:{event.turn_id}:{event.tool_call_id}",
            )

        return None
