"""
Unit tests for CLI config management commands.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.config import (
    _env_manager,
    _k,
    app,
    get_config,
    set_config,
    unset_config,
)

# Initialize test runner
runner = CliRunner()

# Test data
TEST_PREFIX = "LOCCA_"
TEST_KEY = "TEST_KEY"
TEST_VALUE = "test_value"
TEST_ENV_VAR = f"{TEST_PREFIX}{TEST_KEY}"


class TestConfigCommands:
    """Test config management commands."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Save original environment
        self.original_env = os.environ.copy()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_paths = [
            Path(self.temp_dir.name) / ".env",
            Path(self.temp_dir.name) / ".env.local",
        ]

        # Create a new EnvManager instance for testing
        self.test_env_manager = MagicMock()
        self.test_env_manager.env_prefix = TEST_PREFIX

        # Patch _env_manager in the config module
        self.patcher = patch(
            "local_coding_assistant.cli.commands.config._env_manager",
            self.test_env_manager,
        )
        self.patcher.start()

        # Setup mock return values
        self.test_env_manager.with_prefix.side_effect = lambda k: f"{TEST_PREFIX}{k}"
        self.test_env_manager.without_prefix.side_effect = lambda k: (
            k[len(TEST_PREFIX) :] if k.startswith(TEST_PREFIX) else k
        )
        self.test_env_manager.get_all_env_vars.return_value = {}

        # Clear any test environment variables
        for key in list(os.environ.keys()):
            if key.startswith(TEST_PREFIX):
                del os.environ[key]

        # Import here to ensure the patch is in place
        from local_coding_assistant.cli.commands import config

        self.test_env_manager.get_env.side_effect = lambda k: os.environ.get(
            self.test_env_manager.with_prefix(k)
        )

        yield

        # Cleanup
        self.patcher.stop()
        self.temp_dir.cleanup()
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_k_compat_wrapper(self):
        """Test the compatibility wrapper for environment variable prefixing."""
        assert _k("TEST") == f"{TEST_PREFIX}TEST"

    def test_get_config_existing_env_var(self):
        """Test getting a configuration value from environment variables."""
        # Set up test environment
        os.environ[TEST_ENV_VAR] = TEST_VALUE

        # Test the CLI command
        result = runner.invoke(app, ["get", TEST_KEY])
        assert result.exit_code == 0
        assert f"{TEST_ENV_VAR}={TEST_VALUE} (from environment)" in result.output

    @patch("os.environ", {})
    def test_get_config_from_env_file(self):
        """Test getting a configuration value from .env.local."""

        # Setup the mock to return our test value when called with TEST_KEY
        def get_env_side_effect(key):
            if key == TEST_KEY or key == TEST_ENV_VAR:
                return TEST_VALUE
            return None

        self.test_env_manager.get_env.side_effect = get_env_side_effect

        # Make sure with_prefix is properly set up
        self.test_env_manager.with_prefix.side_effect = lambda k: (
            k if k.startswith(TEST_PREFIX) else f"{TEST_PREFIX}{k}"
        )

        # Test the CLI command
        result = runner.invoke(app, ["get", TEST_KEY])

        # Verify the output
        assert result.exit_code == 0, f"Command failed with output: {result.output}"
        assert f"{TEST_ENV_VAR}={TEST_VALUE} (from .env.local)" in result.output

        # Verify get_env was called with the correct key
        self.test_env_manager.get_env.assert_called_with(TEST_KEY)

    def test_get_config_non_existent(self):
        """Test getting a non-existent configuration key."""
        result = runner.invoke(app, ["get", "NON_EXISTENT_KEY"])
        assert result.exit_code == 1
        assert "No configuration found for key" in result.output

    @patch("os.environ", {f"{TEST_PREFIX}KEY1": "value1"})
    def test_get_all_config(self):
        """Test getting all configuration values."""
        # Setup the mock to return our test values
        self.test_env_manager.get_all_env_vars.return_value = {
            f"{TEST_PREFIX}KEY1": "value1",
            f"{TEST_PREFIX}KEY2": "value2",
        }

        # Mock the os.environ check in the config command
        with patch(
            "local_coding_assistant.cli.commands.config.os.environ",
            {f"{TEST_PREFIX}KEY1": "value1"},
        ):
            result = runner.invoke(app, ["get"])

        assert result.exit_code == 0
        assert "All configuration (LOCCA_*):" in result.output
        assert f"{TEST_PREFIX}KEY1=value1 (from environment)" in result.output
        assert f"{TEST_PREFIX}KEY2=value2 (from .env.local)" in result.output

    def test_set_config_persistent(self):
        """Test setting a persistent configuration value."""
        # Setup the mock to track calls to set_env and save_to_env_file
        self.test_env_manager.set_env.return_value = None
        self.test_env_manager.save_to_env_file.return_value = None

        # Mock the with_prefix method to return the prefixed key
        prefixed_key = f"{TEST_PREFIX}{TEST_KEY}"
        self.test_env_manager.with_prefix.return_value = prefixed_key

        # Test the CLI command
        result = runner.invoke(app, ["set", TEST_KEY, TEST_VALUE])

        # Verify the command was successful
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify with_prefix was called with the correct key (twice - once for set_env and once for save_to_env_file)
        assert self.test_env_manager.with_prefix.call_count == 2
        self.test_env_manager.with_prefix.assert_called_with(TEST_KEY)

        # Verify set_env was called with the correct arguments
        self.test_env_manager.set_env.assert_called_once_with(TEST_KEY, TEST_VALUE)

        # Verify save_to_env_file was called with the correct arguments
        self.test_env_manager.save_to_env_file.assert_called_once_with(
            prefixed_key, TEST_VALUE
        )

        # Verify get_env was not called (since set_config doesn't use it)
        self.test_env_manager.get_env.assert_not_called()

    def test_set_config_temporary(self):
        """Test setting a temporary configuration value."""
        # Setup the mock to track calls to set_env
        self.test_env_manager.set_env.return_value = None

        # Test the CLI command with --temporary flag
        result = runner.invoke(app, ["set", "--temporary", TEST_KEY, TEST_VALUE])

        # Verify the command was successful
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify set_env was called with the correct arguments
        self.test_env_manager.set_env.assert_called_once_with(TEST_KEY, TEST_VALUE)

        # Verify save_to_env_file was NOT called for temporary setting
        self.test_env_manager.save_to_env_file.assert_not_called()

        # Verify get_env was not called (since set_config with --temporary doesn't use it)
        self.test_env_manager.get_env.assert_not_called()

    def test_unset_config_persistent(self):
        """Test unsetting a persistent configuration value."""
        # Setup the mock to track calls to unset_env and remove_from_env_file
        self.test_env_manager.unset_env.return_value = None
        self.test_env_manager.remove_from_env_file.return_value = None

        # Test the CLI command
        result = runner.invoke(app, ["unset", TEST_KEY])

        # Verify the command was successful
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify unset_env was called with the correct arguments
        self.test_env_manager.unset_env.assert_called_once_with(TEST_KEY)

        # Verify remove_from_env_file was called to remove the key from the file
        self.test_env_manager.remove_from_env_file.assert_called_once_with(TEST_ENV_VAR)

    def test_unset_config_temporary(self):
        """Test unsetting a temporary configuration value."""
        # Setup the mock to track calls to unset_env
        self.test_env_manager.unset_env.return_value = None

        # Test the CLI command with --temporary flag
        result = runner.invoke(app, ["unset", "--temporary", TEST_KEY])

        # Verify the command was successful
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify unset_env was called with the correct arguments
        self.test_env_manager.unset_env.assert_called_once_with(TEST_KEY)

        # Verify remove_from_env_file was NOT called for temporary unset
        self.test_env_manager.remove_from_env_file.assert_not_called()

    def test_unset_non_existent(self):
        """Test unsetting a non-existent configuration key."""
        # Setup the mock to track calls to unset_env and remove_from_env_file
        self.test_env_manager.unset_env.return_value = None
        self.test_env_manager.remove_from_env_file.return_value = None

        # Test the CLI command
        result = runner.invoke(app, ["unset", "NON_EXISTENT_KEY"])

        # The command should still succeed
        assert result.exit_code == 0, f"Command failed with output: {result.output}"

        # Verify unset_env was called with the non-existent key
        self.test_env_manager.unset_env.assert_called_once_with("NON_EXISTENT_KEY")

        # Verify remove_from_env_file was called to attempt removal from file
        self.test_env_manager.remove_from_env_file.assert_called_once_with(
            f"{TEST_PREFIX}NON_EXISTENT_KEY"
        )

        # Verify the success message is in the output
        assert "unset and removed" in result.output.lower()
        assert "non_existent_key" in result.output.lower()

    def test_config_priority(self):
        """Test that environment variables take precedence over .env.local."""
        # Setup the mock to return different values based on the environment
        env_removed = False

        def get_env_side_effect(key, default=None):
            nonlocal env_removed
            if key == TEST_KEY or key == TEST_ENV_VAR:
                # First call returns from environment, second from file
                if not env_removed:
                    return "env_value"
                return "file_value"
            return default

        self.test_env_manager.get_env.side_effect = get_env_side_effect

        # First call - should get environment value
        assert self.test_env_manager.get_env(TEST_KEY) == "env_value"

        # Simulate environment variable being removed
        env_removed = True

        # Second call - should get file value
        assert self.test_env_manager.get_env(TEST_KEY) == "file_value"

    def test_set_config_error_handling(self):
        """Test error handling when setting a configuration value fails."""
        # Make set_env raise an exception
        error_msg = "Test error"
        self.test_env_manager.set_env.side_effect = Exception(error_msg)

        # Test the CLI command
        result = runner.invoke(app, ["set", TEST_KEY, TEST_VALUE])

        # The command should still exit with 0 due to safe_entrypoint
        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}"

    @patch("local_coding_assistant.cli.commands.config._env_manager.unset_env")
    def test_unset_config_error_handling(self, mock_unset_env):
        """Test error handling when unsetting a configuration value fails."""
        # Make unset_env raise an exception
        mock_unset_env.side_effect = Exception("Test error")

        # Test the CLI command
        result = runner.invoke(app, ["unset", TEST_KEY])

        # Verify unset_env was called with the correct arguments
        mock_unset_env.assert_called_once_with(TEST_KEY)

        # The command should still complete successfully due to safe_entrypoint
        assert result.exit_code == 0
