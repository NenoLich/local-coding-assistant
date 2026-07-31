from __future__ import annotations

from pathlib import Path

from local_coding_assistant.cli.rendering import render_report

GOLDEN_DIR = Path(__file__).parent / "golden"


def _read_golden(name: str) -> str:
    return (GOLDEN_DIR / name).read_text()


def test_frame_renderer_golden(rich_console, sample_run_report):
    render_report(
        sample_run_report,
        _format="frame",
        verbose=True,
        console=rich_console,
    )
    output = rich_console.export_text()
    assert output == _read_golden("frame_renderer.txt")


def test_summary_renderer_golden(rich_console, sample_run_report):
    render_report(
        sample_run_report,
        _format="summary",
        verbose=False,
        console=rich_console,
    )
    output = rich_console.export_text()
    assert output == _read_golden("summary_renderer.txt")
