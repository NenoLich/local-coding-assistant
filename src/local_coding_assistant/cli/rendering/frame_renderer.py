from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.tree import Tree

from local_coding_assistant.runtime.reporting import RunReport


def _stringify(value: Any, limit: int = 160) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    try:
        rendered = json.dumps(value, ensure_ascii=True, default=str)
    except TypeError:
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _as_dict(frame: Any) -> dict[str, Any]:
    if isinstance(frame, dict):
        return frame
    if hasattr(frame, "model_dump"):
        return frame.model_dump()
    return {}


def render_frame(report: RunReport, console: Console, verbose: bool) -> None:  # noqa: C901
    frames = report.frames or []
    root = Tree(f"Frames ({len(frames)}) - status: {report.status}")

    if not frames:
        root.add("No frames captured")
        console.print(root)
        return

    for frame in frames:
        frame_dict = _as_dict(frame)
        iteration = frame_dict.get("iteration", "?")
        result = frame_dict.get("result", {}) or {}
        status = result.get("status", "unknown")
        if hasattr(status, "value"):
            status = status.value
        if report.metrics is not None:
            llm_tokens = report.metrics.tokens_used
            duration = report.metrics.total_latency_ms

            label = f"Frame {iteration} [{status}]"
            if duration is not None:
                label += f" {duration:.1f}ms"
            if llm_tokens is not None:
                label += f" tokens={llm_tokens}"

        else:
            label = f"Frame {iteration} [{status}]"

        frame_node = root.add(label)

        actions = frame_dict.get("actions", []) or []
        if actions:
            actions_node = frame_node.add(f"Actions ({len(actions)})")
            for action in actions:
                action_kind = action.get("kind", "unknown")
                action_name = action.get("name") or "-"
                action_node = actions_node.add(f"{action_kind}: {action_name}")
                if verbose:
                    # Extract input based on action kind
                    if action_kind == "tool_call":
                        input_val = action.get("tool_trace", {}).get("input")
                    else:
                        input_val = action.get("metadata", {}).get("input")

                    # Extract output based on action kind
                    if action_kind == "tool_call":
                        output_val = action.get("tool_trace", {}).get("output")
                    else:
                        output_val = action.get("metadata", {}).get("output")

                    action_node.add(f"input: {_stringify(input_val)}")
                    action_node.add(f"output: {_stringify(output_val)}")
                    metadata = action.get("metadata") or {}
                    if metadata:
                        action_node.add(f"meta: {_stringify(metadata)}")

        final_answer = result.get("final_answer")
        if final_answer:
            frame_node.add(f"final_answer: {_stringify(final_answer)}")
        error_message = result.get("error_message")
        if error_message:
            frame_node.add(f"error: {_stringify(error_message)}")

    console.print(root)
