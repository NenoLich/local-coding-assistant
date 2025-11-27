"""Integration tests for the tools subsystem and CLI commands."""
from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from local_coding_assistant.cli.commands import tool as tool_cli
from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.core.exceptions import ToolRegistryError
from local_coding_assistant.tools.tool_manager import ToolManager
from local_coding_assistant.tools.types import ToolExecutionRequest


@pytest.fixture
def tool_test_env(test_configs):
    """Create an isolated tool configuration environment using test_configs."""
    # Create a tool module creator function
    def create_tool_module(tool_id: str, multiplier: int = 2) -> Path:
        class_name = "".join(part.capitalize() for part in tool_id.split("_"))
        module_path = test_configs["modules_dir"] / f"{tool_id}.py"
        module_path.write_text(
            textwrap.dedent(
                f"""
                from pydantic import BaseModel
                from local_coding_assistant.tools.base import Tool

                class {class_name}(Tool):
                    class Input(BaseModel):
                        value: int

                    class Output(BaseModel):
                        result: int

                    def run(self, input_data: Input) -> Output:
                        return self.Output(result=input_data.value * {multiplier})
                """
            ),
            encoding="utf-8",
        )
        return module_path

    # Return a namespace with the same interface as before
    return SimpleNamespace(
        config_dir=test_configs["config_dir"],
        default_config=test_configs["default"],
        local_config=test_configs["local"],
        create_tool_module=create_tool_module
    )




def test_tool_manager_loads_tools_from_config_and_executes(tool_test_env, test_configs):
    """Ensure ToolManager loads YAML-defined tools and executes them."""

    module_path = tool_test_env.create_tool_module("sample_tool", multiplier=3)
    disabled_module = tool_test_env.create_tool_module("disabled_tool")

    # Write tool configurations to the default config
    tool_test_env.default_config.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "id": "sample_tool",
                        "description": "Sample tool for integration tests",
                        "path": str(module_path),
                        "enabled": True,
                    },
                    {
                        "id": "disabled_tool",
                        "description": "Disabled tool should not build runtime",
                        "path": str(disabled_module),
                        "enabled": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    
    # Use the config manager with test configs
    config_manager = ConfigManager(
        tool_config_paths=[tool_test_env.default_config, tool_test_env.local_config]
    )
    tool_manager = ToolManager(config_manager=config_manager, auto_load=True)

    info = tool_manager.get_tool_info("sample_tool")
    assert info is not None and info.available is True

    result = asyncio.run(tool_manager.run_tool_async("sample_tool", {"value": 4}))
    assert result == {"result": 12}

    stats = tool_manager.get_execution_stats()["sample_tool"]
    assert stats["call_count"] == 1
    assert stats["error_count"] == 0

    disabled_info = tool_manager.get_tool_info("disabled_tool")
    assert disabled_info is not None
    assert disabled_info.enabled is False
    assert disabled_info.available is False

    with pytest.raises(ToolRegistryError):
        tool_manager.run_tool("disabled_tool", {"value": 1})


def test_tool_manager_execute_reports_success_and_errors(tool_test_env):
    """Verify ToolManager.execute wraps results and validation errors."""

    module_path = tool_test_env.create_tool_module("exec_tool", multiplier=2)
    tool_test_env.default_config.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "id": "exec_tool",
                        "description": "Tool used to exercise execute()",
                        "path": str(module_path),
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    
    # Use the config manager with the test config files
    config_manager = ConfigManager(
        tool_config_paths=[tool_test_env.default_config, tool_test_env.local_config]
    )
    manager = ToolManager(config_manager=config_manager, auto_load=True)

    success = manager.execute(
        ToolExecutionRequest(tool_name="exec_tool", payload={"value": 5})
    )
    assert success.success is True
    assert success.result == {"result": 10}
    assert success.execution_time_ms is not None and success.execution_time_ms > 0

    failure = manager.execute(ToolExecutionRequest(tool_name="exec_tool", payload={}))
    assert failure.success is False
    assert failure.error_message is not None
    assert "Invalid input" in failure.error_message


def test_cli_tool_run_executes_registered_tool(tool_test_env, cli_runner, monkeypatch):
    """The `tool run` CLI command should execute tools via ToolManager."""

    module_path = tool_test_env.create_tool_module("cli_tool", multiplier=4)
    tool_test_env.default_config.write_text(
        yaml.safe_dump(
            {
                "tools": [
                    {
                        "id": "cli_tool",
                        "description": "Tool executed through CLI run command",
                        "path": str(module_path),
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_bootstrap(**_: object) -> dict[str, object]:
        # Create a config manager with the test config files
        config_manager = ConfigManager(
            tool_config_paths=[tool_test_env.default_config, tool_test_env.local_config]
        )
        return {
            "tools": ToolManager(config_manager=config_manager, auto_load=True),
        }

    monkeypatch.setattr(tool_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(tool_cli.app, ["run", "cli_tool", "value=3"])

    assert result.exit_code == 0
    assert "success: true" in result.stdout
    assert "result:" in result.stdout
    assert "result: 12" in result.stdout.replace("\r", "")


def test_cli_tool_add_persists_configuration(tool_test_env, cli_runner, monkeypatch, test_configs):
    """The `tool add` CLI command should persist new tools and load them."""

    module_path = tool_test_env.create_tool_module("added_cli_tool", multiplier=6)
    tool_test_env.default_config.write_text(yaml.safe_dump({"tools": []}), encoding="utf-8")

    def fake_bootstrap(**_: object) -> dict[str, object]:
        # Create a config manager with the test config files
        config_manager = ConfigManager(
            tool_config_paths=[tool_test_env.default_config, tool_test_env.local_config]
        )
        return {
            "tools": ToolManager(config_manager=config_manager, auto_load=True),
        }

    monkeypatch.setattr(tool_cli, "bootstrap", fake_bootstrap)

    result = cli_runner.invoke(
        tool_cli.app,
        [
            "add",
            "added_cli_tool",
            "--description",
            "Tool added via CLI",
            "--path",
            str(module_path),
            "--config-file",
            str(test_configs["local"]),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Successfully" in result.stdout

    config = yaml.safe_load(test_configs["local"].read_text(encoding="utf-8"))
    assert any(entry["id"] == "added_cli_tool" for entry in config.get("tools", []))

    config_manager = ConfigManager(
        tool_config_paths=[test_configs["default"], test_configs["local"]]
    )
    tool_manager = ToolManager(config_manager=config_manager, auto_load=True)
    run_result = tool_manager.run_tool("added_cli_tool", {"value": 2})
    assert run_result == {"result": 12}
