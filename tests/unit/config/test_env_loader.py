"""Unit tests for EnvLoader functionality."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from local_coding_assistant.config.env_loader import EnvLoader
from local_coding_assistant.core.exceptions import ConfigError


class TestEnvLoader:
    """Test EnvLoader functionality."""

    def test_default_env_paths(self):
        """Test that default .env paths are correctly identified."""
        loader = EnvLoader()

        # Should find paths relative to project root
        paths = loader._get_default_env_paths()
        assert len(paths) == 2
        assert any(".env" in str(path) for path in paths)
        assert any(".env.local" in str(path) for path in paths)

    def test_custom_env_paths(self):
        """Test EnvLoader with custom paths."""
        custom_paths = [Path("/custom/.env"), Path("/custom/.env.local")]
        loader = EnvLoader(env_paths=custom_paths)

        assert loader.env_paths == custom_paths

    def test_custom_env_prefix(self):
        """Test EnvLoader with custom prefix."""
        loader = EnvLoader(env_prefix="CUSTOM_")
        assert loader.env_prefix == "CUSTOM_"

    def test_get_config_from_env_empty(self):
        """Test get_config_from_env with no matching environment variables."""
        loader = EnvLoader()

        # Clear any existing LOCCA_ variables for this test
        with patch.dict(os.environ, {}, clear=True):
            config = loader.get_config_from_env()

        assert config == {}

    def test_get_config_from_env_basic(self):
        """Test get_config_from_env with basic environment variables."""
        loader = EnvLoader()

        test_env = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_LLM__TEMPERATURE": "0.8",
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = loader.get_config_from_env()

        expected = {
            "llm": {"model_name": "gpt-4", "temperature": 0.8},
            "runtime": {"persistent_sessions": True},
        }

        assert config == expected

    def test_get_config_from_env_type_conversion(self):
        """Test that environment variables are properly converted to appropriate types."""
        loader = EnvLoader()

        test_env = {
            "LOCCA_LLM__MAX_TOKENS": "1000",  # Should convert to int
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "false",  # Should convert to bool
            "LOCCA_LLM__TEMPERATURE": "0.7",  # Should convert to float
            "LOCCA_LLM__MODEL_NAME": "gpt-3.5-turbo",  # Should remain string
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = loader.get_config_from_env()

        assert config["llm"]["max_tokens"] == 1000
        assert config["llm"]["temperature"] == 0.7
        assert config["llm"]["model_name"] == "gpt-3.5-turbo"
        assert config["runtime"]["persistent_sessions"] is False

    def test_get_config_from_env_null_values(self):
        """Test handling of null values in environment variables."""
        loader = EnvLoader()

        test_env = {"LOCCA_LLM__API_KEY": "null"}

        with patch.dict(os.environ, test_env, clear=False):
            config = loader.get_config_from_env()

        assert config["llm"]["api_key"] is None

    def test_get_config_from_env_case_insensitive(self):
        """Test that environment variable parsing is case insensitive."""
        loader = EnvLoader()

        test_env = {
            "LOCCA_LLM__MODEL_NAME": "GPT-4",
            "locca_runtime__persistent_sessions": "true",  # Mixed case
        }

        with patch.dict(os.environ, test_env, clear=False):
            config = loader.get_config_from_env()

        assert config["llm"]["model_name"] == "GPT-4"
        assert config["runtime"]["persistent_sessions"] is True

    def test_env_file_loading_success(self):
        """Test successful .env file loading."""
        loader = EnvLoader()

        with patch("dotenv.load_dotenv") as mock_load:
            mock_load.return_value = None

            # Create temporary .env file
            with patch.object(Path, "exists", return_value=True):
                loader.load_env_files()

            # Should have called load_dotenv for each path
            assert mock_load.call_count == 2  # .env and .env.local

    def test_env_file_loading_missing_dotenv(self):
        """Test graceful handling when python-dotenv is not available."""
        loader = EnvLoader()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'dotenv'")
        ):
            # Should not raise an error
            loader.load_env_files()

    def test_env_file_loading_file_error(self):
        """Test error handling when .env file loading fails."""
        loader = EnvLoader()

        with patch("dotenv.load_dotenv") as mock_load:
            mock_load.side_effect = Exception("File not found")

            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(ConfigError, match=r"Failed to load .env file"):
                    loader.load_env_files()

    def test_env_file_loading_no_files(self):
        """Test behavior when no .env files exist."""
        loader = EnvLoader()

        with patch("dotenv.load_dotenv") as mock_load:
            with patch.object(Path, "exists", return_value=False):
                loader.load_env_files()

            # Should not call load_dotenv when files don't exist
            mock_load.assert_not_called()

    def test_nested_value_setting(self):
        """Test the _set_nested_value helper method."""
        loader = EnvLoader()
        data = {}

        # Test simple case
        loader._set_nested_value(data, ["llm", "model_name"], "gpt-4")
        assert data == {"llm": {"model_name": "gpt-4"}}

        # Test deeper nesting
        loader._set_nested_value(data, ["llm", "temperature", "value"], "0.8")
        assert data == {"llm": {"model_name": "gpt-4", "temperature": {"value": 0.8}}}

    def test_convert_env_value_boolean(self):
        """Test _convert_env_value for boolean conversion."""
        loader = EnvLoader()

        assert loader._convert_env_value("true") is True
        assert loader._convert_env_value("false") is False
        assert loader._convert_env_value("TRUE") is True
        assert loader._convert_env_value("FALSE") is False

    def test_convert_env_value_numeric(self):
        """Test _convert_env_value for numeric conversion."""
        loader = EnvLoader()

        assert loader._convert_env_value("42") == 42
        assert loader._convert_env_value("3.14") == 3.14
        assert loader._convert_env_value("1e10") == 1e10

    def test_convert_env_value_string(self):
        """Test _convert_env_value for string values."""
        loader = EnvLoader()

        assert loader._convert_env_value("hello") == "hello"
        assert loader._convert_env_value("") == ""
        assert loader._convert_env_value("null") is None
