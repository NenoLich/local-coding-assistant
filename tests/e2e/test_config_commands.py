"""End-to-end tests for config CLI commands."""

import os

from local_coding_assistant.cli.main import app


class TestConfigCommands:
    """Test cases for configuration CLI commands."""

    def test_config_get_nonexistent_key(self, cli_runner):
        """Test getting a configuration key that doesn't exist."""
        result = cli_runner.invoke(app, ["config", "get", "NONEXISTENT_KEY"])

        assert result.exit_code == 0
        assert "LOCCA_NONEXISTENT_KEY is not set" in result.stdout

    def test_config_get_existing_key(self, cli_runner, mock_env_vars):
        """Test getting a configuration key that exists."""
        # Set a test environment variable
        os.environ["LOCCA_CUSTOM_KEY"] = "custom_value"

        result = cli_runner.invoke(app, ["config", "get", "CUSTOM_KEY"])

        assert result.exit_code == 0
        assert "LOCCA_CUSTOM_KEY=custom_value" in result.stdout

    def test_config_get_all(self, cli_runner, mock_env_vars):
        """Test getting all configuration values."""
        # Set additional test environment variables
        os.environ["LOCCA_ANOTHER_KEY"] = "another_value"

        result = cli_runner.invoke(app, ["config", "get"])

        assert result.exit_code == 0
        assert "All configuration (env, LOCCA_*):" in result.stdout
        assert "LOCCA_TEST_MODE=true" in result.stdout
        assert "LOCCA_LOG_LEVEL=INFO" in result.stdout
        assert "LOCCA_ANOTHER_KEY=another_value" in result.stdout

    def test_config_set_new_key(self, cli_runner):
        """Test setting a new configuration key."""
        result = cli_runner.invoke(app, ["config", "set", "NEW_KEY", "new_value"])

        assert result.exit_code == 0
        assert "Set LOCCA_NEW_KEY=new_value" in result.stdout

        # Verify the environment variable was set
        assert os.environ.get("LOCCA_NEW_KEY") == "new_value"

    def test_config_set_overwrite_existing(self, cli_runner, mock_env_vars):
        """Test setting a configuration key that already exists."""
        # Pre-set a value
        os.environ["LOCCA_EXISTING_KEY"] = "old_value"

        result = cli_runner.invoke(app, ["config", "set", "EXISTING_KEY", "new_value"])

        assert result.exit_code == 0
        assert "Set LOCCA_EXISTING_KEY=new_value" in result.stdout

        # Verify the environment variable was updated
        assert os.environ.get("LOCCA_EXISTING_KEY") == "new_value"

    def test_config_set_special_characters(self, cli_runner):
        """Test setting a configuration key with special characters in value."""
        special_value = 'Value with "quotes" and spaces and special chars: @#$%^&*()'

        result = cli_runner.invoke(app, ["config", "set", "SPECIAL_KEY", special_value])

        assert result.exit_code == 0
        assert f"Set LOCCA_SPECIAL_KEY={special_value}" in result.stdout

        # Verify the environment variable was set correctly
        assert os.environ.get("LOCCA_SPECIAL_KEY") == special_value

    def test_config_set_empty_value(self, cli_runner):
        """Test setting a configuration key with empty value."""
        result = cli_runner.invoke(app, ["config", "set", "EMPTY_KEY", ""])

        assert result.exit_code == 0
        assert "Set LOCCA_EMPTY_KEY=" in result.stdout

        # Verify the environment variable was set to empty string
        assert os.environ.get("LOCCA_EMPTY_KEY") == ""

    def test_config_workflow(self, cli_runner):
        """Test a complete config workflow: set, get, verify."""
        # Set a value
        result = cli_runner.invoke(app, ["config", "set", "WORKFLOW_TEST", "test_value"])
        assert result.exit_code == 0

        # Get the value
        result = cli_runner.invoke(app, ["config", "get", "WORKFLOW_TEST"])
        assert result.exit_code == 0
        assert "LOCCA_WORKFLOW_TEST=test_value" in result.stdout

        # Get all values and verify it's included
        result = cli_runner.invoke(app, ["config", "get"])
        assert result.exit_code == 0
        assert "LOCCA_WORKFLOW_TEST=test_value" in result.stdout
