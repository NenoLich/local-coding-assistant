from __future__ import annotations

from rich.console import Console

from local_coding_assistant.runtime.reporting import RunReport


def render_plain(report: RunReport, console: Console, verbose: bool) -> None:
    message = report.message or report.final_answer or ""
    console.print(message)
