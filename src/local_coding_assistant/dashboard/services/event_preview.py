"""Shared event preview field extraction utilities for the dashboard."""

from __future__ import annotations

from typing import Any

_PREVIEW_KEYS: dict[str, tuple[str, ...]] = {
    "llm_complete": (
        "model",
        "provider",
        "finish_reason",
        "input_tokens",
        "output_tokens",
    ),
    "tool_start": (
        "tool_call_id",
        "tool_name",
    ),
    "tool_result": (
        "tool_call_id",
        "tool_name",
        "success",
    ),
    "turn_complete": (
        "status",
        "final_answer",
        "total_input_tokens",
        "total_output_tokens",
        "total_frames",
        "finish_reason",
    ),
    "frame_complete": (
        "status",
        "finish_reason",
        "tool_call_count",
    ),
    "turn_start": ("user_input", "turn_index", "agent_mode", "model_name"),
    "frame_start": ("iteration",),
    "session_start": ("user_input", "agent_mode", "model_name"),
    "error": ("error_type", "error_message", "recoverable"),
    "continuation_decision": ("decision", "reason"),
}


def preview_for_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Extract important preview fields for the timeline."""
    preview: dict[str, Any] = {}

    for key in _PREVIEW_KEYS.get(event_type, ()):
        if key in data:
            preview[key] = data[key]

    handler = _PREVIEW_HANDLERS.get(event_type)
    if handler:
        handler(data, preview)

    return preview


def _handle_llm_complete(data: dict[str, Any], preview: dict[str, Any]) -> None:
    if latency := data.get("latency_ms"):
        preview["llm_exec_time"] = f"{float(latency / 1000):.3f} sec"
    if content := data.get("content"):
        preview["content"] = str(content)[:250]
    if tool_calls := data.get("tool_calls"):
        preview["tool_calls"] = str(tool_calls)[:250]


def _handle_turn_complete(data: dict[str, Any], preview: dict[str, Any]) -> None:
    if final_answer := data.get("final_answer"):
        preview["final_answer"] = str(final_answer)[:250]


def _handle_tool_result(data: dict[str, Any], preview: dict[str, Any]) -> None:
    for field in ("output", "error", "stdout", "stderr"):
        if (preview_key := data.get(field)) is not None:
            preview[field] = str(preview_key)[:250]
    if (timing := data.get("timing")) is not None:
        if (preview_key := timing.get("elapsed_seconds")) is not None:
            preview["tool_exec_time"] = f"{preview_key:.3f} sec"


def _handle_frame_complete(data: dict[str, Any], preview: dict[str, Any]) -> None:
    if (preview_key := data.get("duration_ms")) is not None:
        preview["duration"] = f"{float(preview_key / 1000):.3f} sec"
    if (preview_key := data.get("error_message")) is not None:
        preview["error_message"] = str(preview_key)[:250]


def _handle_approval_required(data: dict[str, Any], preview: dict[str, Any]) -> None:
    preview["approval_requests"] = len(data.get("approval_requests", []))


def _handle_approval_granted(data: dict[str, Any], preview: dict[str, Any]) -> None:
    preview["approval_token"] = data.get("approval_token")
    preview["tool_call_id"] = data.get("tool_call_id")


def _handle_approval_denied(data: dict[str, Any], preview: dict[str, Any]) -> None:
    preview["approval_token"] = data.get("approval_token")
    preview["tool_call_id"] = data.get("tool_call_id")
    preview["reason"] = data.get("reason")


_PREVIEW_HANDLERS: dict[str, Any] = {
    "llm_complete": _handle_llm_complete,
    "turn_complete": _handle_turn_complete,
    "tool_result": _handle_tool_result,
    "frame_complete": _handle_frame_complete,
    "approval_required": _handle_approval_required,
    "approval_granted": _handle_approval_granted,
    "approval_denied": _handle_approval_denied,
}
