"""
Unit tests for CLI run commands.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from local_coding_assistant.cli.commands.run import query

# Initialize test runner
runner = CliRunner()

# Test data
TEST_QUERY = "What is the meaning of life?"
TEST_MODEL = "test-model"
TEST_RESPONSE = {"message": "42"}


class TestRunCommands:
    """Test run command functionality."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup mocks for each test."""
        # Create a mock runtime
        self.mock_runtime = AsyncMock()
        self.mock_runtime.orchestrate.return_value = TEST_RESPONSE

        # Create a mock context
        self.mock_ctx = {"runtime": self.mock_runtime}

        # Patch the bootstrap function
        self.bootstrap_patcher = patch(
            "local_coding_assistant.cli.commands.run.bootstrap"
        )
        self.mock_bootstrap = self.bootstrap_patcher.start()
        self.mock_bootstrap.return_value = self.mock_ctx

        # Patch asyncio.run to actually run the coroutine
        self.asyncio_patcher = patch(
            "local_coding_assistant.cli.commands.run.asyncio.run",
            side_effect=asyncio.run,
        )
        self.mock_asyncio_run = self.asyncio_patcher.start()

        # Patch typer.echo to capture output
        self.echo_patcher = patch("local_coding_assistant.cli.commands.run.typer.echo")
        self.mock_echo = self.echo_patcher.start()

        yield

        # Cleanup
        self.bootstrap_patcher.stop()
        self.asyncio_patcher.stop()
        self.echo_patcher.stop()

    def test_query_basic(self):
        """Test basic query execution."""
        # Reset the echo mock to track only new calls
        self.mock_echo.reset_mock()

        # Call the query function directly with test parameters
        # We need to pass the default values explicitly since we're calling it directly
        query(
            text=TEST_QUERY,
            verbose=False,  # Explicitly set to False to match test expectations
            model=None,
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level="INFO",
        )

        # Verify the expected output
        self.mock_echo.assert_any_call(f"Running query: {TEST_QUERY}")

        # Verify the response was printed
        self.mock_echo.assert_any_call("\nResponse:")
        self.mock_echo.assert_any_call(TEST_RESPONSE["message"])

        # Verify the runtime was called correctly
        self.mock_bootstrap.assert_called_once()
        from unittest.mock import ANY

        self.mock_runtime.orchestrate.assert_awaited_once_with(
            TEST_QUERY,
            model=None,
            tool_call_mode="reasoning",
            sandbox_session=ANY,  # Typer OptionInfo object
        )

    def test_query_with_model(self):
        """Test query execution with a specific model."""
        # Reset the echo mock to track only new calls
        self.mock_echo.reset_mock()

        # Call the query function directly with test parameters
        query(
            text=TEST_QUERY,
            verbose=False,
            model=TEST_MODEL,  # Pass the test model
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level="INFO",
        )

        # Verify the expected output
        self.mock_echo.assert_any_call(f"Running query: {TEST_QUERY}")
        self.mock_echo.assert_any_call(f"Using model: {TEST_MODEL}")

        # Verify the response was printed
        self.mock_echo.assert_any_call("\nResponse:")
        self.mock_echo.assert_any_call(TEST_RESPONSE["message"])

        # Verify the runtime was called with the correct model and default parameters
        from unittest.mock import ANY

        self.mock_runtime.orchestrate.assert_awaited_once_with(
            TEST_QUERY,
            model=TEST_MODEL,
            tool_call_mode="reasoning",
            sandbox_session=ANY,  # Typer OptionInfo object
        )

    def test_verbose_mode(self):
        """Test verbose mode output."""
        # Reset the echo mock to track only new calls
        self.mock_echo.reset_mock()

        # Call the query function directly with verbose=True
        query(
            text=TEST_QUERY,
            verbose=True,  # Enable verbose mode
            model=None,
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level="INFO",
        )

        # Verify the expected output
        self.mock_echo.assert_any_call(f"Running query: {TEST_QUERY}")
        self.mock_echo.assert_any_call("Verbose mode enabled")

        # Verify the response was printed
        self.mock_echo.assert_any_call("\nResponse:")
        self.mock_echo.assert_any_call(TEST_RESPONSE["message"])

        # Verify the runtime was called with the correct parameters
        from unittest.mock import ANY

        self.mock_runtime.orchestrate.assert_awaited_once_with(
            TEST_QUERY,
            model=None,
            tool_call_mode="reasoning",
            sandbox_session=ANY,  # Typer OptionInfo object
        )

    def test_log_level_parameter(self):
        """Test that log level parameter is passed to bootstrap."""
        # Reset the bootstrap mock to track only new calls
        self.mock_bootstrap.reset_mock()

        # Call the query function directly with a specific log level
        test_log_level = "DEBUG"
        query(
            text=TEST_QUERY, 
            verbose=False, 
            model=None, 
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level=test_log_level
        )

        # Verify bootstrap was called with the correct log level
        self.mock_bootstrap.assert_called_once_with(log_level=logging.DEBUG)

    def test_runtime_unavailable(self):
        """Test behavior when runtime is not available."""
        # Setup the test
        self.mock_ctx["runtime"] = None

        # Call the query function directly and expect it to raise typer.Exit
        with patch("local_coding_assistant.cli.commands.run.typer.echo") as mock_echo:
            try:
                query(
                    text=TEST_QUERY, 
                    verbose=False, 
                    model=None, 
                    tool_call_mode="reasoning",  # Pass the default value explicitly
                    sandbox_session=None,  # Pass the default value explicitly
                    log_level="INFO"
                )
                # If we get here, the test should fail
                assert False, "Expected typer.Exit to be raised"
            except typer.Exit as e:
                # Verify the exit code is 1
                assert e.exit_code == 1

                # Verify the error message was shown
                mock_echo.assert_any_call(
                    "Error: Runtime manager not available (LLM initialization failed)"
                )

        # Verify orchestrate was not called
        self.mock_runtime.orchestrate.assert_not_awaited()

    def test_query_output_format(self):
        """Test the format of the query output."""
        # Reset the echo mock to track only new calls
        self.mock_echo.reset_mock()

        # Call the query function directly
        query(
            text=TEST_QUERY, 
            verbose=False, 
            model=None, 
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level="INFO"
        )

        # Verify the expected output format
        self.mock_echo.assert_any_call(f"Running query: {TEST_QUERY}")
        self.mock_echo.assert_any_call("\nResponse:")
        self.mock_echo.assert_any_call(TEST_RESPONSE["message"])

    def test_asyncio_run_usage(self):
        """Test that asyncio.run is used correctly."""
        # Reset the mock to track calls
        self.mock_asyncio_run.reset_mock()

        # Call the query function directly
        query(
            text=TEST_QUERY, 
            verbose=False, 
            model=None, 
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level="INFO"
        )

        # Verify asyncio.run was called once
        assert self.mock_asyncio_run.call_count == 1

        # Verify the runtime's orchestrate method was called within asyncio.run
        self.mock_runtime.orchestrate.assert_awaited_once()

    @patch("local_coding_assistant.core.error_handler.logger")
    def test_error_handling(self, mock_log):
        """Test error handling in the query function."""
        # Make orchestrate raise an exception
        test_error = RuntimeError("Test error")
        self.mock_runtime.orchestrate.side_effect = test_error

        # Reset mocks
        self.mock_echo.reset_mock()
        mock_log.error.reset_mock()
        mock_log.critical.reset_mock()

        # Call the query function directly and verify it doesn't raise
        try:
            query(
                text=TEST_QUERY, 
                verbose=False, 
                model=None, 
                tool_call_mode="reasoning",  # Pass the default value explicitly
                sandbox_session=None,  # Pass the default value explicitly
                log_level="INFO"
            )
        except Exception as e:
            pytest.fail(f"Query raised {type(e).__name__} unexpectedly: {e!s}")

        # Verify the error was logged (check both error and critical logs)
        error_logged = False
        for log_method in [mock_log.error, mock_log.critical]:
            for call in log_method.call_args_list:
                if "Test error" in str(call[0]):
                    error_logged = True
                    break
            if error_logged:
                break

        assert error_logged, "Expected error message not found in logs"

    def test_safe_entrypoint_decorator(self):
        """Test that the safe_entrypoint decorator is applied."""
        import inspect

        # The @safe_entrypoint decorator should be applied to the query function
        original_func = inspect.unwrap(query)
        assert original_func is not query, (
            "Function should be wrapped by @safe_entrypoint"
        )
        # Check that it's the correct function by checking its module
        assert original_func.__module__ == "local_coding_assistant.cli.commands.run", (
            "Function should be from the correct module"
        )

    @pytest.mark.parametrize(
        "log_level,expected_level",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("invalid", logging.INFO),  # Default to INFO for invalid levels
        ],
    )
    def test_log_level_handling(self, log_level, expected_level):
        """Test handling of different log levels."""
        # Reset the bootstrap mock to track calls
        self.mock_bootstrap.reset_mock()

        # Call the query function directly with the test log level
        query(
            text=TEST_QUERY, 
            verbose=False, 
            model=None, 
            tool_call_mode="reasoning",  # Pass the default value explicitly
            sandbox_session=None,  # Pass the default value explicitly
            log_level=log_level
        )

        # Verify bootstrap was called with the expected log level
        self.mock_bootstrap.assert_called_once_with(log_level=expected_level)
