"""Integration-style tests for the `tool run-multiple` CLI command."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_coding_assistant.cli.commands import tool as tool_cli


class ConsoleCapture:
    """Lightweight substitute for Rich console used in tests."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, *args, **kwargs) -> None:  # pragma: no cover - exercised via CLI
        text = " ".join(str(arg) for arg in args)
        self.messages.append(text)

    @property
    def combined(self) -> str:
        return "\n".join(self.messages)


@pytest.fixture
def run_multiple_env(monkeypatch):
    """Provide a patched CLI context and capture console output."""

    console_capture = ConsoleCapture()
    fake_tool_manager = SimpleNamespace(name="fake-tool-manager")

    def fake_create(
        cls, log_level: str, *, config_file: str | None = None, sandbox: bool = False
    ):
        return SimpleNamespace(tool_manager=fake_tool_manager)

    monkeypatch.setattr(tool_cli.ToolCLIContext, "create", classmethod(fake_create))
    monkeypatch.setattr(tool_cli, "console", console_capture)

    return SimpleNamespace(console=console_capture, tool_manager=fake_tool_manager)


def test_run_multiple_sequential_displays_plan_results_summary(
    cli_runner, run_multiple_env, monkeypatch
):
    """Sequential execution should display plan, per-tool output, and summary."""

    captured_specs: dict[str, list[tuple[str, dict]]] = {}

    async def fake_run_tools(**kwargs):
        captured_specs["value"] = kwargs["tool_specs"]
        return [
            ("alpha_tool", True, {"result": 42}),
            ("beta_tool", False, "boom"),
        ]

    monkeypatch.setattr(tool_cli, "_run_tools", fake_run_tools)

    result = cli_runner.invoke(
        tool_cli.app, ["run-multiple", "alpha_tool:value=3;beta_tool"]
    )

    assert result.exit_code == 0
    assert captured_specs["value"] == [
        ("alpha_tool", {"value": 3}),
        ("beta_tool", {}),
    ]

    combined_output = run_multiple_env.console.combined
    assert "[bold]Execution Plan (sequential):[/bold]" in combined_output
    assert "1. [bold]alpha_tool[/bold]" in combined_output
    assert "2. [bold]beta_tool[/bold]" in combined_output
    assert "[bold]Summary:[/bold] 1 of 2 tools executed successfully" in combined_output
    assert "result: 42" in combined_output  # YAML output from _display_tool_result
    assert "[red]Error: The following tools failed: beta_tool[/red]" in combined_output
    assert "[red]Error details: boom[/red]" in combined_output


def test_run_multiple_parallel_displays_parallel_plan(
    cli_runner, run_multiple_env, monkeypatch
):
    """Parallel mode should call _run_tools with parallel flag and show plan summary."""

    captured_kwargs: dict[str, object] = {}

    async def fake_run_tools(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            ("calc_tool", True, "done"),
            ("weather_tool", True, "ok"),
        ]

    monkeypatch.setattr(tool_cli, "_run_tools", fake_run_tools)

    result = cli_runner.invoke(
        tool_cli.app, ["run-multiple", "--parallel", "calc_tool;weather_tool"]
    )

    assert result.exit_code == 0
    assert captured_kwargs.get("parallel") is True
    assert captured_kwargs.get("tool_specs") == [
        ("calc_tool", {}),
        ("weather_tool", {}),
    ]

    combined_output = run_multiple_env.console.combined
    assert "[bold]Execution Plan (parallel):[/bold]" in combined_output
    assert "[bold]Summary:[/bold] 2 of 2 tools executed successfully" in combined_output


def test_run_multiple_parallel_limit_validation(
    cli_runner, run_multiple_env, monkeypatch
):
    """_validate_tool_specs should stop execution when more than 4 tools run in parallel."""

    async def fail_if_called(**_):
        raise AssertionError("run_tools should not be invoked when validation fails")

    monkeypatch.setattr(tool_cli, "_run_tools", fail_if_called)

    result = cli_runner.invoke(
        tool_cli.app,
        ["run-multiple", "--parallel", "one;two;three;four;five"],
    )

    assert result.exit_code == 1
    assert "Cannot run 5 tools in parallel" in result.output
    assert run_multiple_env.console.messages == ["\n[red]Error: 1[/red]"]


def test_run_multiple_requires_at_least_one_tool(
    cli_runner, run_multiple_env, monkeypatch
):
    """Validation should fail gracefully when specs do not include any tools."""

    async def fail_if_called(**_):
        raise AssertionError("run_tools should not be invoked when validation fails")

    monkeypatch.setattr(tool_cli, "_run_tools", fail_if_called)

    result = cli_runner.invoke(tool_cli.app, ["run-multiple", ";;;"])

    assert result.exit_code == 1
    assert "No valid tool specifications provided" in result.output
    assert run_multiple_env.console.messages == ["\n[red]Error: 1[/red]"]
