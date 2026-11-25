"""End-to-end tests for the tool CLI commands.

The refactored ``tool`` surface exposes only ``list`` and ``run``. Both go
through ``ToolCLIContext.create(...)`` which returns a context carrying a
``tool_gateway``; the gateway provides ``get_exposed_tools()`` and ``execute()``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from local_coding_assistant.cli.main import app

runner = CliRunner()


class _ToolDef:
    def __init__(
        self,
        tool_id,
        version="1.0",
        input_schema=None,
    ):
        self.id = tool_id
        self.version = version
        self.risk_class = SimpleNamespace(value="low")
        self.supported_backends = []
        self.description = "a tool"
        self.input_schema = input_schema or {}


class _ToolResult:
    def __init__(self, output, error=None):
        self.output = output
        self.error = error


class _Gateway:
    def __init__(
        self,
        tools=(),
        execute_output=None,
        input_schema=None,
    ):
        self._tools = [_ToolDef(t, input_schema=input_schema) for t in tools]
        self.execute_result = _ToolResult(output=execute_output)
        self.workspace_manager = SimpleNamespace(workspace_id="default")
        self.last_request = None

    def get_exposed_tools(self):
        return list(self._tools)

    async def execute(self, request):
        self.last_request = request
        return self.execute_result


def _make_context(holder):
    ctx = MagicMock()
    ctx.tool_gateway = holder["gateway"]
    return ctx


def _patch_tool_context(holder):
    from local_coding_assistant.cli.commands.tool import ToolCLIContext

    return patch.object(
        ToolCLIContext,
        "create",
        side_effect=lambda *args, **kwargs: _make_context(holder),
    )


class TestToolCommands:
    """Test suite for ``tool`` CLI subcommands."""

    def test_tool_list_handles_empty_registry(self):
        holder = {"gateway": _Gateway()}
        with _patch_tool_context(holder):
            result = runner.invoke(app, ["tool", "list"])

        assert result.exit_code == 0, result.stdout
        assert "No tools found" in result.stdout

    def test_tool_run_outputs_execution_result(self):
        expected_payload = {"success": True, "result": {"message": "Echo result"}}
        holder = {
            "gateway": _Gateway(tools=["echo_tool"], execute_output=expected_payload)
        }
        with _patch_tool_context(holder):
            result = runner.invoke(app, ["tool", "run", "echo_tool", "text=hello"])

        assert result.exit_code == 0, result.stdout
        assert "Executing tool: echo_tool" in result.stdout
        assert "Tool execution result:" in result.stdout
        assert "Echo result" in result.stdout

    def test_tool_run_maps_positional_args_to_input_fields(self):
        holder = {
            "gateway": _Gateway(
                tools=["read_file"],
                input_schema={
                    "properties": {"path": {}, "offset": {}, "limit": {}},
                },
                execute_output={"content": "x"},
            )
        }
        with _patch_tool_context(holder):
            result = runner.invoke(
                app,
                ["tool", "run", "read_file", "pyproject.toml", "0", "50"],
            )

        assert result.exit_code == 0, result.stdout
        req = holder["gateway"].last_request
        assert req is not None
        assert req.raw_arguments == {
            "path": "pyproject.toml",
            "offset": 0,
            "limit": 50,
        }
