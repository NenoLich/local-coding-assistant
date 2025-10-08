from __future__ import annotations

import json

from typer.testing import CliRunner

from local_coding_assistant.cli.main import app

runner = CliRunner()


def test_run_query_echoes_plain_text():
    result = runner.invoke(app, ["run", "query", "hello world", "--log-level", "INFO"])
    assert result.exit_code == 0
    # Expect our CLI prints a Response: header and then the echo from LLMManager
    assert "Response:" in result.stdout
    assert "[LLMManager] Echo: hello world" in result.stdout


def test_run_query_tool_directive_verbose_and_info():
    payload = json.dumps({"a": 5, "b": 6})
    result = runner.invoke(
        app,
        [
            "run",
            "query",
            f"tool:sum {payload}",
            "--log-level",
            "INFO",
            "-v",
        ],
    )
    assert result.exit_code == 0
    # Expect Response header
    assert "Response:" in result.stdout
    # Expect echo including the user content; LLMManager in integration prints with tool outputs phrase
    assert "[LLMManager] Echo:" in result.stdout
    assert "with tool outputs" in result.stdout
