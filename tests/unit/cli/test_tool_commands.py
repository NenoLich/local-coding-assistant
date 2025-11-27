"""Unit tests for the tool CLI command helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_coding_assistant.cli.commands import tool as tool_commands
from local_coding_assistant.cli.commands.tool import ToolCLIError
from local_coding_assistant.tools.types import ToolCategory, ToolSource


def test_build_tool_config_regular_tool_validates_and_returns_expected_payload():
    result = tool_commands._build_tool_config(  # noqa: SLF001
        tool_id="echo",
        name="Echo Tool",
        description="Echo back input",
        enabled=True,
        category="utility",
        module="example.tools.echo",
        path=None,
        endpoint=None,
        provider=None,
        permissions=None,
        tags=None,
    )

    assert result["id"] == "echo"
    assert result["name"] == "Echo Tool"
    assert result["module"] == "example.tools.echo"
    assert result["source"] == "external"
    assert result["category"] == "utility"
    assert result["enabled"] is True


@pytest.mark.parametrize(
    "endpoint,provider,module,path,error_message",
    [
        ("https://api.example.com", None, None, None, "Both --endpoint and --provider"),
        ("https://api.example.com", "provider", "pkg.tool", None, "Cannot mix"),
    ],
)
def test_build_tool_config_mcp_validation_errors(endpoint, provider, module, path, error_message):
    with pytest.raises(ToolCLIError, match=error_message):
        tool_commands._build_tool_config(  # noqa: SLF001
            tool_id="remote",
            name=None,
            description="Remote tool",
            enabled=True,
            category="utility",
            module=module,
            path=path,
            endpoint=endpoint,
            provider=provider,
            permissions=None,
            tags=None,
        )


def test_build_tool_config_mcp_success_sets_mcp_source():
    config = tool_commands._build_tool_config(  # noqa: SLF001
        tool_id="remote",
        name=None,
        description="Remote tool",
        enabled=True,
        category="utility",
        module=None,
        path=None,
        endpoint="https://api.example.com",
        provider="acme",
        permissions=["read"],
        tags=["ai"],
    )

    assert config["source"] == "mcp"
    assert config["endpoint"] == "https://api.example.com"
    assert config["provider"] == "acme"
    assert config["permissions"] == ["read"]
    assert config["tags"] == ["ai"]


def test_parse_tool_args_handles_json_payload_and_key_value_pairs():
    payload = tool_commands._parse_tool_args(  # noqa: SLF001
        ["{\"foo\": 1}"]
    )
    assert payload == {"foo": 1}

    payload = tool_commands._parse_tool_args(  # noqa: SLF001
        ["mode=fast", "42", "extra=true", "name=util", "values=[1,2]"]
    )
    assert payload["mode"] == "fast"
    assert payload["extra"] is True
    assert payload["name"] == "util"
    assert payload["values"] == [1, 2]
    assert payload["args"] == [42]


def test_collect_tool_rows_sorts_and_filters_available_tools():
    tool_a = SimpleNamespace(
        name="Alpha",
        category=ToolCategory.UTILITY,
        enabled=True,
        available=True,
        source=ToolSource.BUILTIN,
        description="Alpha tool",
    )
    tool_b = SimpleNamespace(
        name="beta",
        category="coding",
        enabled=True,
        available=False,
        source="external",
        description="Beta tool",
    )

    class FakeManager:
        def __init__(self):
            self.calls = []
            
        def list_tools(self, available_only: bool = False, category: str | None = None):
            self.calls.append(("list_tools", {"available_only": available_only, "category": category}))
            return [tool for tool in [tool_a, tool_b] if not available_only or tool.available]

    # Test with available_only=False (should return all tools)
    manager = FakeManager()
    rows_all = tool_commands._collect_tool_rows(manager, available_only=False)  # noqa: SLF001
    assert [row.name for row in rows_all] == ["Alpha", "beta"]
    assert manager.calls == [("list_tools", {"available_only": False, "category": None})]
    
    # Test with available_only=True (should return only available tools)
    manager = FakeManager()
    rows_available = tool_commands._collect_tool_rows(manager, available_only=True)  # noqa: SLF001
    assert [row.name for row in rows_available] == ["Alpha"]
    assert manager.calls == [("list_tools", {"available_only": True, "category": None})]


def test_execute_tool_prefers_manager_execute_with_tool_execution_request():
    captured_requests = []

    class Response:
        def __init__(self, data):
            self.data = data

        def model_dump(self):
            return self.data

    class Manager:
        def execute(self, request):
            captured_requests.append(request)
            return Response({"status": "ok"})

    result = tool_commands._execute_tool(Manager(), "demo", {"input": "hello"})  # noqa: SLF001

    assert result == {"status": "ok"}
    assert captured_requests
    request = captured_requests[0]
    assert request.tool_name == "demo"
    assert request.payload == {"input": "hello"}


def test_execute_tool_falls_back_to_run_tool_when_execute_missing():
    calls = []

    class Manager:
        def run_tool(self, tool_id, payload):
            calls.append((tool_id, payload))
            return {"result": "ran"}

    result = tool_commands._execute_tool(Manager(), "demo", {"x": 1})  # noqa: SLF001

    assert result == {"result": "ran"}
    assert calls == [("demo", {"x": 1})]


def test_execute_tool_raises_when_manager_has_no_sync_interface():
    with pytest.raises(ToolCLIError, match="does not support synchronous execution"):
        tool_commands._execute_tool(object(), "demo", {})  # noqa: SLF001
