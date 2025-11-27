"""End-to-end tests for the tool CLI commands."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from local_coding_assistant.cli.main import app


class DummyToolManager:
    """Minimal tool manager stub used by CLI tests."""

    def __init__(self, *, result: dict | None = None, tool_ids: set[str] | None = None):
        self._result = result or {"success": True, "result": {"message": "ok"}}
        self._tool_ids = tool_ids or {"echo_tool"}
        self.reload_called = False

    def get_tool(self, tool_id: str) -> object:
        if tool_id not in self._tool_ids:
            raise KeyError(tool_id)
        return object()

    def list_tools(self, available_only: bool = False) -> list:
        return []

    def execute(self, request):
        return SimpleNamespace(model_dump=lambda: self._result)

    def reload_tools(self) -> None:
        self.reload_called = True


@pytest.mark.usefixtures("cli_runner")
class TestToolCommands:
    """Test suite for `tool` CLI subcommands."""

    def test_tool_add_creates_config_and_confirms_load(
        self,
        cli_runner,
        test_configs,
        patch_tool_bootstrap,
    ) -> None:
        tool_manager = DummyToolManager(tool_ids={"sample_tool"})
        patch_tool_bootstrap.return_value = {"tools": tool_manager}

        result = cli_runner.invoke(
            app,
            [
                "tool",
                "add",
                "sample_tool",
                "--description",
                "Sample tool configured via CLI",
                "--path",
                "tests/mock_tools/sample_tool.py",
                "--config-file",
                str(test_configs["local"]),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "Successfully added tool" in result.stdout

        config = yaml.safe_load(test_configs["local"].read_text(encoding="utf-8"))
        assert config["tools"][0]["id"] == "sample_tool"
        assert config["tools"][0]["path"] == "tests/mock_tools/sample_tool.py"

    def test_tool_run_outputs_execution_result(
        self,
        cli_runner,
        patch_tool_bootstrap,
    ) -> None:
        expected_payload = {"success": True, "result": {"message": "Echo result"}}
        tool_manager = DummyToolManager(result=expected_payload)
        patch_tool_bootstrap.return_value = {"tools": tool_manager}

        result = cli_runner.invoke(
            app,
            ["tool", "run", "echo_tool", "text=hello"],
        )

        assert result.exit_code == 0, result.stdout
        assert "Executing tool: echo_tool" in result.stdout
        assert "success: true" in result.stdout.lower()
        assert "Echo result" in result.stdout

    def test_tool_list_handles_empty_registry(
        self,
        cli_runner,
        patch_tool_bootstrap,
    ) -> None:
        tool_manager = DummyToolManager()
        tool_manager.list_tools = MagicMock(return_value=[])
        patch_tool_bootstrap.return_value = {"tools": tool_manager}

        result = cli_runner.invoke(app, ["tool", "list"])

        assert result.exit_code == 0, result.stdout
        assert "No tools found" in result.stdout

    def test_tool_remove_updates_configuration(
        self,
        cli_runner,
        test_configs,
        patch_tool_bootstrap,
    ) -> None:
        test_configs["local"].write_text(
            yaml.safe_dump(
                {
                    "tools": [
                        {
                            "id": "obsolete_tool",
                            "path": "tests/mock_tools/obsolete.py",
                            "description": "This tool should be removed",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        tool_manager = DummyToolManager(tool_ids=set())
        patch_tool_bootstrap.return_value = {"tools": tool_manager}

        result = cli_runner.invoke(
            app,
            [
                "tool",
                "remove",
                "obsolete_tool",
                "--config-file",
                str(test_configs["local"]),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert "Successfully removed tool" in result.stdout

        config = yaml.safe_load(test_configs["local"].read_text(encoding="utf-8"))
        assert config["tools"] == []

    def test_tool_validate_reports_errors(
        self,
        cli_runner,
        test_configs,
    ) -> None:
        test_configs["local"].write_text(
            yaml.safe_dump({"invalid": "structure"}),
            encoding="utf-8",
        )

        result = cli_runner.invoke(
            app,
            [
                "tool",
                "validate",
                "--config-file",
                str(test_configs["local"]),
            ],
        )

        assert result.exit_code == 1
        assert "Configuration validation failed" in result.stdout

    def test_tool_reload_invokes_manager(
        self,
        cli_runner,
        patch_tool_bootstrap,
    ) -> None:
        tool_manager = DummyToolManager()
        patch_tool_bootstrap.return_value = {"tools": tool_manager}

        result = cli_runner.invoke(app, ["tool", "reload"])

        assert result.exit_code == 0, result.stdout
        assert tool_manager.reload_called is True
        assert "Successfully reloaded" in result.stdout
