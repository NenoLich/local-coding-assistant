from __future__ import annotations

from rich.console import Console
from rich.table import Table

from local_coding_assistant.runtime.reporting import RunReport


def render_summary(report: RunReport, console: Console, verbose: bool) -> None:
    table = Table(title="Run Summary", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Run ID", report.run_id)
    table.add_row("Session ID", report.session_id or "-")
    table.add_row("Mode", report.mode)
    table.add_row("Status", report.status)
    table.add_row(
        "Models", ", ".join(report.models_used) if report.models_used else "-"
    )
    table.add_row("Tokens", str(report.tokens_used or "-"))
    table.add_row("Iterations", str(report.iterations or "-"))
    table.add_row("Final Answer", report.final_answer or "-")

    console.print(table)
