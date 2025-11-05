"""End-to-end tests for list-tools CLI command."""

import json
from unittest.mock import MagicMock, patch

from local_coding_assistant.cli.main import app


class TestListToolsCommand:
    """Test cases for the list-tools CLI command."""

    def test_list_tools_basic(self, cli_runner):
        """Test basic list-tools functionality."""
        # Mock bootstrap directly in the test
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            # Create mock tools registry with proper attribute access
            mock_tools = []

            # Create search_web tool mock
            search_web_tool = MagicMock()
            search_web_tool.name = "search_web"
            search_web_tool.__name__ = "search_web"
            search_web_tool.__class__.__name__ = "WebSearchTool"
            mock_tools.append(search_web_tool)

            # Create read_file tool mock
            read_file_tool = MagicMock()
            read_file_tool.name = "read_file"
            read_file_tool.__name__ = "read_file"
            read_file_tool.__class__.__name__ = "FileReadTool"
            mock_tools.append(read_file_tool)

            mock_bootstrap.return_value = {"tools": mock_tools}
            mock_bootstrap2.return_value = {"tools": mock_tools}

            result = cli_runner.invoke(app, ["list-tools", "list"])

            assert result.exit_code == 0
            assert "Available tools:" in result.stdout
            assert "search_web" in result.stdout
            assert "read_file" in result.stdout
            assert "WebSearchTool" in result.stdout
            assert "FileReadTool" in result.stdout

    def test_list_tools_json_output(self, cli_runner):
        """Test list-tools with JSON output."""
        # Mock bootstrap directly in the test
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            # Create mock tools registry with proper attribute access
            mock_tools = []

            # Create search_web tool mock
            search_web_tool = MagicMock()
            search_web_tool.name = "search_web"
            search_web_tool.__name__ = "search_web"
            search_web_tool.__class__.__name__ = "WebSearchTool"
            mock_tools.append(search_web_tool)

            # Create read_file tool mock
            read_file_tool = MagicMock()
            read_file_tool.name = "read_file"
            read_file_tool.__name__ = "read_file"
            read_file_tool.__class__.__name__ = "FileReadTool"
            mock_tools.append(read_file_tool)

            mock_bootstrap.return_value = {"tools": mock_tools}
            mock_bootstrap2.return_value = {"tools": mock_tools}

            result = cli_runner.invoke(app, ["list-tools", "list", "--json"])

            assert result.exit_code == 0

            # Parse the JSON output
            output_data = json.loads(result.stdout)
            assert "count" in output_data
            assert "tools" in output_data
            assert output_data["count"] == 2

            tool_names = [tool["name"] for tool in output_data["tools"]]
            assert "search_web" in tool_names
            assert "read_file" in tool_names

    def test_list_tools_with_category_filter(self, cli_runner):
        """Test list-tools with category filter (not implemented yet)."""
        # Mock bootstrap directly in the test
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            # Create mock tools registry with proper attribute access
            mock_tools = []

            # Create search_web tool mock
            search_web_tool = MagicMock()
            search_web_tool.name = "search_web"
            search_web_tool.__name__ = "search_web"
            search_web_tool.__class__.__name__ = "WebSearchTool"
            mock_tools.append(search_web_tool)

            # Create read_file tool mock
            read_file_tool = MagicMock()
            read_file_tool.name = "read_file"
            read_file_tool.__name__ = "read_file"
            read_file_tool.__class__.__name__ = "FileReadTool"
            mock_tools.append(read_file_tool)

            mock_bootstrap.return_value = {"tools": mock_tools}
            mock_bootstrap2.return_value = {"tools": mock_tools}

            result = cli_runner.invoke(app, ["list-tools", "list", "--cat", "web"])

            assert result.exit_code == 0
            assert "Available tools:" in result.stdout
            assert "Category filter requested: web (not implemented)" in result.stdout

    def test_list_tools_empty_registry(self, cli_runner):
        """Test list-tools when no tools are available."""
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            mock_bootstrap.return_value = {"tools": []}
            mock_bootstrap2.return_value = {"tools": []}

            result = cli_runner.invoke(app, ["list-tools", "list"])

            assert result.exit_code == 0
            assert "Available tools:" in result.stdout
            assert "- (none)" in result.stdout

    def test_list_tools_single_tool(self, cli_runner):
        """Test list-tools with only one tool."""
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            # Single tool mock
            single_tool = MagicMock()
            single_tool.__name__ = "single_function"
            single_tool.name = "single_tool"
            single_tool.__class__ = MagicMock(__name__="SingleTool")

            mock_bootstrap.return_value = {"tools": [single_tool]}
            mock_bootstrap2.return_value = {"tools": [single_tool]}

            result = cli_runner.invoke(app, ["list-tools", "list"])

            assert result.exit_code == 0
            assert "Available tools:" in result.stdout
            assert "single_tool" in result.stdout
            assert "SingleTool" in result.stdout

    def test_list_tools_json_single_tool(self, cli_runner):
        """Test list-tools JSON output with single tool."""
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            single_tool = MagicMock()
            single_tool.__name__ = "single_function"
            single_tool.name = "single_tool"
            single_tool.__class__ = MagicMock(__name__="SingleTool")

            mock_bootstrap.return_value = {"tools": [single_tool]}
            mock_bootstrap2.return_value = {"tools": [single_tool]}

            result = cli_runner.invoke(app, ["list-tools", "list", "--json"])

            assert result.exit_code == 0

            output_data = json.loads(result.stdout)
            assert output_data["count"] == 1
            assert len(output_data["tools"]) == 1
            assert output_data["tools"][0]["name"] == "single_tool"

    def test_list_tools_missing_registry(self, cli_runner):
        """Test list-tools when tools registry is missing."""
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            mock_bootstrap.return_value = {}
            mock_bootstrap2.return_value = {}

            result = cli_runner.invoke(app, ["list-tools", "list"])

            assert result.exit_code == 0
            assert "No tools available" in result.stdout

    def test_list_tools_with_log_level(self, cli_runner):
        """Test list-tools with custom log level."""
        # Mock bootstrap directly in the test
        with (
            patch("local_coding_assistant.core.bootstrap") as mock_bootstrap,
            patch(
                "local_coding_assistant.cli.commands.list_tools.bootstrap"
            ) as mock_bootstrap2,
        ):
            # Create mock tools registry with proper attribute access
            mock_tools = []

            # Create search_web tool mock
            search_web_tool = MagicMock()
            search_web_tool.name = "search_web"
            search_web_tool.__name__ = "search_web"
            search_web_tool.__class__.__name__ = "WebSearchTool"
            mock_tools.append(search_web_tool)

            # Create read_file tool mock
            read_file_tool = MagicMock()
            read_file_tool.name = "read_file"
            read_file_tool.__name__ = "read_file"
            read_file_tool.__class__.__name__ = "FileReadTool"
            mock_tools.append(read_file_tool)

            mock_bootstrap.return_value = {"tools": mock_tools}
            mock_bootstrap2.return_value = {"tools": mock_tools}

            result = cli_runner.invoke(
                app, ["list-tools", "list", "--log-level", "DEBUG"]
            )

            assert result.exit_code == 0
            assert "Available tools:" in result.stdout
            assert "search_web" in result.stdout
            assert "read_file" in result.stdout
