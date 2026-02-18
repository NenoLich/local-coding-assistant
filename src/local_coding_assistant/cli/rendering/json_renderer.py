from __future__ import annotations

import json

from rich.console import Console

from local_coding_assistant.runtime.reporting import RunReport


def render_json(report: RunReport, console: Console, verbose: bool) -> None:
    payload = report.model_dump()
    console.print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))
