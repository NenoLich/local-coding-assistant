"""Unit tests for the tool CLI command helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
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


def test_render_tool_table_empty_list():
    """Test rendering a table with no tools."""
    # Create a mock Table class to capture the add_row calls
    class MockTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.columns = []
            
        def add_column(self, *args, **kwargs):
            self.columns.append(args[0] if args else None)
            
        def add_row(self, *args):
            self.rows.append(args)

    with (
        patch('local_coding_assistant.cli.commands.tool.console.print') as mock_print,
        patch('local_coding_assistant.cli.commands.tool.Table', new=MockTable) as mock_table
    ):
        tool_commands._render_tool_table([])
        
        # Verify the table was printed once
        assert mock_print.call_count == 1
        
        # Get the table that was printed
        table = mock_print.call_args[0][0]
        
        # Verify the table has no rows
        assert len(table.rows) == 0
        
        # Verify the table has the right number of columns
        assert len(table.columns) == 6  # Name, Category, Enabled, Available, Source, Description


def test_render_tool_table_single_tool():
    """Test rendering a table with a single tool."""
    # Create a mock ToolDisplayRow
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.category = "utility"
    mock_tool.enabled = True
    mock_tool.available = True
    mock_tool.source = "test_source"
    mock_tool.description = "A test tool"

    # Create a mock Table class to capture the add_row calls
    class MockTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.columns = []
            
        def add_column(self, *args, **kwargs):
            self.columns.append(args[0] if args else None)
            
        def add_row(self, *args):
            self.rows.append(args)

    # Mock the _truncate_text function to return the same text
    with (
        patch('local_coding_assistant.cli.commands.tool.console.print') as mock_print,
        patch('local_coding_assistant.cli.commands.tool._truncate_text', 
              side_effect=lambda x, _limit=100: x) as mock_truncate,
        patch('local_coding_assistant.cli.commands.tool.Table', new=MockTable) as mock_table
    ):
        tool_commands._render_tool_table([mock_tool])
        
        # Verify the table was printed once
        assert mock_print.call_count == 1
        
        # Get the table that was printed
        table = mock_print.call_args[0][0]
        
        # Verify the table has one row
        assert len(table.rows) == 1
        
        # Get the row data
        row = table.rows[0]
        
        # Verify the row data is correct
        assert row[0] == "test_tool"
        assert row[1] == "utility"
        assert row[2] == "✓"  # Enabled
        assert row[3] == "✓"  # Available
        assert row[4] == "test_source"
        assert row[5] == "A test tool"


def test_render_tool_table_multiple_tools():
    """Test rendering a table with multiple tools."""
    # Create mock tools
    tool1 = MagicMock()
    tool1.name = "tool1"
    tool1.category = "utility"
    tool1.enabled = True
    tool1.available = True
    tool1.source = "source1"
    tool1.description = "Tool 1"
    
    tool2 = MagicMock()
    tool2.name = "tool2"
    tool2.category = "ai"
    tool2.enabled = False
    tool2.available = True
    tool2.source = "source2"
    tool2.description = "Tool 2"
    
    tool3 = MagicMock()
    tool3.name = "tool3"
    tool3.category = "utility"
    tool3.enabled = True
    tool3.available = False
    tool3.source = "source3"
    tool3.description = "Tool 3"
    
    test_tools = [tool1, tool2, tool3]
    
    # Create a mock Table class to capture the add_row calls
    class MockTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.columns = []
            
        def add_column(self, *args, **kwargs):
            self.columns.append(args[0] if args else None)
            
        def add_row(self, *args):
            self.rows.append(args)
    
    with (
        patch('local_coding_assistant.cli.commands.tool.console.print') as mock_print,
        patch('local_coding_assistant.cli.commands.tool._truncate_text', 
              side_effect=lambda x, _limit=100: x) as mock_truncate,
        patch('local_coding_assistant.cli.commands.tool.Table', new=MockTable) as mock_table
    ):
        tool_commands._render_tool_table(test_tools)
        
        # Verify the table was printed once
        assert mock_print.call_count == 1
        
        # Get the table that was printed
        table = mock_print.call_args[0][0]
        
        # Verify the table has three rows
        assert len(table.rows) == 3
        
        # Get the rows
        row1, row2, row3 = table.rows
        
        # Verify the rows are in the correct order (should be sorted by name)
        assert row1[0] == "tool1"
        assert row2[0] == "tool2"
        assert row3[0] == "tool3"
        
        # Verify the status indicators
        assert row1[2] == "✓"  # tool1 enabled
        assert row2[2] == "✗"  # tool2 disabled
        assert row3[3] == "✗"  # tool3 not available


def test_update_tool_in_config_update_existing():
    """Test updating an existing tool's configuration."""
    # Arrange
    existing_config = {
        "tools": [
            {"id": "tool1", "name": "Old Name", "enabled": True},
            {"id": "tool2", "name": "Tool 2", "enabled": True}
        ]
    }
    new_config = {
        "tools": [
            {"id": "tool1", "name": "New Name", "enabled": False}
        ]
    }

    # Act
    result, action = tool_commands._update_tool_in_config(
        existing_config, new_config, "tool1"
    )

    # Assert
    assert action == "Updated"
    assert len(result["tools"]) == 2
    assert result["tools"][0]["name"] == "New Name"
    assert result["tools"][0]["enabled"] is False
    assert result["tools"][1] == {"id": "tool2", "name": "Tool 2", "enabled": True}


def test_update_tool_in_config_add_new():
    """Test adding a new tool configuration."""
    # Arrange
    existing_config = {"tools": [{"id": "tool1", "name": "Tool 1", "enabled": True}]}
    new_config = {
        "tools": [{"id": "tool2", "name": "New Tool", "enabled": True}]
    }

    # Act
    result, action = tool_commands._update_tool_in_config(
        existing_config, new_config, "tool2"
    )

    # Assert
    assert action == "Added"
    assert len(result["tools"]) == 2
    assert result["tools"][1]["id"] == "tool2"


def test_update_tool_in_config_empty_new_config():
    """Test behavior when new_config has no tools."""
    # Arrange
    existing_config = {"tools": [{"id": "tool1", "name": "Tool 1", "enabled": True}]}
    new_config = {"tools": []}

    # Act
    result, action = tool_commands._update_tool_in_config(
        existing_config, new_config, "tool1"
    )

    # Assert
    assert action == "Updated"  # Still returns Updated but doesn't change anything
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "Tool 1"  # Unchanged


def test_update_tool_in_config_error_handling():
    """Test error handling when config structure is invalid."""
    # Arrange - existing_config is missing the 'tools' key
    existing_config = {}
    new_config = {"tools": [{"id": "tool1", "name": "New Tool"}]}

    # Act & Assert
    with patch("local_coding_assistant.cli.commands.tool.log.warning") as mock_warning:
        result, action = tool_commands._update_tool_in_config(
            existing_config, new_config, "tool1"
        )
        
        # Should still return the new config and "Added" on error
        assert action == "Added"
        assert result == new_config
        mock_warning.assert_called_once()


def test_render_tool_table_edge_cases():
    """Test edge cases like None values and long descriptions."""
    # Create a mock tool with edge cases
    mock_tool = MagicMock()
    mock_tool.name = "edge_case_tool"
    mock_tool.category = None  # Should be replaced with "-"
    mock_tool.enabled = False
    mock_tool.available = False
    mock_tool.source = None  # Should be replaced with "-"
    long_description = "This is a very long description that should be truncated to fit within the default limit of 100 characters. " \
                      "This ensures that the _truncate_text function is working as expected."
    mock_tool.description = long_description
    
    # Create a mock Table class to capture the add_row calls
    class MockTable:
        def __init__(self, *args, **kwargs):
            self.rows = []
            self.columns = []
            self.show_header = True
            self.header_style = "bold"
            
        def add_column(self, *args, **kwargs):
            self.columns.append(args[0] if args else None)
            
        def add_row(self, *args):
            self.rows.append(args)
    
    # Mock _truncate_text to actually truncate the text
    def mock_truncate(text, limit=100):
        if len(text) > limit:
            return text[:limit-3] + "..."
        return text
    
    with (
        patch('local_coding_assistant.cli.commands.tool.console.print') as mock_print,
        patch('local_coding_assistant.cli.commands.tool._truncate_text', 
              side_effect=mock_truncate) as mock_truncate_func,
        patch('local_coding_assistant.cli.commands.tool.Table', new=MockTable) as mock_table
    ):
        tool_commands._render_tool_table([mock_tool])
        
        # Verify the table was printed once
        assert mock_print.call_count == 1
        
        # Get the table that was printed
        table = mock_print.call_args[0][0]
        
        # Verify the table has one row
        assert len(table.rows) == 1
        
        # Get the row data
        row = table.rows[0]
        
        # Verify None values are replaced with "-"
        assert row[1] == "-"  # category
        assert row[4] == "-"  # source
        
        # Verify description is truncated
        assert len(row[5]) <= 103  # 100 chars + "..."
        assert row[5].endswith("...")


def test_list_options():
    """Test that list_options correctly lists all available options."""
    # Mock the console.print function to capture output
    with patch("local_coding_assistant.cli.commands.tool.console.print") as mock_print, patch(
        "local_coding_assistant.cli.commands.tool.get_enum_values"
    ) as mock_get_enum_values:
        # Setup mock return values for enum values
        mock_get_enum_values.side_effect = [
            ["category1", "category2"],  # For ToolCategory
            ["permission1", "permission2"],  # For ToolPermission
            ["tag1", "tag2"],  # For ToolTag
        ]

        # Call the function
        tool_commands.list_options()

        # Verify console.print was called with the expected output
        assert mock_print.call_count >= 3  # At least 3 calls (header + categories + permissions + tags)
        
        # Get all calls to console.print
        calls = [call[0][0] for call in mock_print.call_args_list]
        
        # Verify the header is printed
        assert any("Available Tool Configuration Options" in call for call in calls)
        
        # Verify categories section
        assert any("Categories:" in call for call in calls)
        assert any("- category1" in call for call in calls)
        assert any("- category2" in call for call in calls)
        
        # Verify permissions section
        assert any("Permissions:" in call for call in calls)
        assert any("- permission1" in call for call in calls)
        assert any("- permission2" in call for call in calls)
        
        # Verify tags section
        assert any("Tags:" in call for call in calls)
        assert any("- tag1" in call for call in calls)
        assert any("- tag2" in call for call in calls)
        
        # Verify get_enum_values was called with the correct enum classes
        from local_coding_assistant.tools.types import ToolCategory, ToolPermission, ToolTag
        assert mock_get_enum_values.call_count == 3
        assert mock_get_enum_values.call_args_list[0].args[0] == ToolCategory
        assert mock_get_enum_values.call_args_list[1].args[0] == ToolPermission
        assert mock_get_enum_values.call_args_list[2].args[0] == ToolTag