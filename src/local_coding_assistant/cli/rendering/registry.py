from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console

from local_coding_assistant.cli.rendering.frame_renderer import render_frame
from local_coding_assistant.cli.rendering.json_renderer import render_json
from local_coding_assistant.cli.rendering.plain import render_plain
from local_coding_assistant.cli.rendering.summary import render_summary
from local_coding_assistant.runtime.reporting import RunReport

Renderer = Callable[[RunReport, Console, bool], None]

_RENDERERS: dict[str, Renderer] = {
    "plain": render_plain,
    "json": render_json,
    "frame": render_frame,
    "summary": render_summary,
}


def coerce_report(report: RunReport | dict[str, Any]) -> RunReport:
    if isinstance(report, RunReport):
        return report
    return RunReport.from_legacy(report)


def dump_report(report: RunReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, RunReport):
        return report.model_dump()
    return report


def render_report(
    report: RunReport | dict[str, Any],
    *,
    _format: str,
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    renderer = _RENDERERS.get(_format, render_plain)
    resolved = coerce_report(report)
    active_console = console or Console()
    renderer(resolved, active_console, verbose)
