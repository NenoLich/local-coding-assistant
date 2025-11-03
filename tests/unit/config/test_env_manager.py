"""Unit tests for EnvManager functionality."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.core.exceptions import ConfigError


class TestEnvManager:
    """Test EnvManager functionality."""

    def test_default_env_paths(self):
        """Test that default .env paths are correctly identified."""
        manager = EnvManager()

        # Should find paths relative to project root
        assert len(manager.env_paths) == 2
        assert any(".env" in str(path) for path in manager.env_paths)
        assert any(".env.local" in str(path) for path in manager.env_paths)

    def test_custom_env_paths(self):
        """Test EnvManager with custom paths."""
        custom_paths = [Path("/custom/.env"), Path("/custom/.env.local")]
        manager = EnvManager(env_paths=custom_paths)

        assert manager.env_paths == custom_paths

    def test_custom_env_prefix(self):
        """Test EnvManager with custom prefix."""
        manager = EnvManager(env_prefix="CUSTOM_")
        assert manager.env_prefix == "CUSTOM_"

    def test_get_config_from_env_empty(self):
        """Test get_config_from_env with no matching environment variables."""
        manager = EnvManager()

        # Clear any existing LOCCA_ variables for this test
        with patch.dict(os.environ, {}, clear=True):
            config = manager.get_config_from_env()

        assert config == {}

    def test_get_config_from_env_basic(self):
        """Test get_config_from_env with basic environment variables."""
        manager = EnvManager()

        test_env = {
            "LOCCA_LLM__MODEL_NAME": "gpt-4",
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "true",
            "LOCCA_LLM__TEMPERATURE": "0.8",
        }

        with patch.dict(os.environ, test_env, clear=True):
            config = manager.get_config_from_env()

        expected = {
            "llm": {"model_name": "gpt-4", "temperature": 0.8},
            "runtime": {"persistent_sessions": True},
        }

        assert config == expected

    def test_get_config_from_env_type_conversion(self):
        """Test that environment variables are properly converted to appropriate types."""
        manager = EnvManager()

        test_env = {
            "LOCCA_LLM__MAX_TOKENS": "1000",  # Should convert to int
            "LOCCA_RUNTIME__PERSISTENT_SESSIONS": "false",  # Should convert to bool
            "LOCCA_LLM__TEMPERATURE": "0.7",  # Should convert to float
            "LOCCA_LLM__MODEL_NAME": "gpt-3.5-turbo",  # Should remain string
        }

        with patch.dict(os.environ, test_env, clear=True):
            config = manager.get_config_from_env()

        assert config["llm"]["max_tokens"] == 1000
        assert config["llm"]["temperature"] == 0.7
        assert config["llm"]["model_name"] == "gpt-3.5-turbo"
        assert config["runtime"]["persistent_sessions"] is False

    def test_get_config_from_env_null_values(self):
        """Test handling of null values in environment variables."""
        manager = EnvManager()

        test_env = {"LOCCA_LLM__API_KEY": "null"}

        with patch.dict(os.environ, test_env, clear=True):
            config = manager.get_config_from_env()

        assert config["llm"]["api_key"] is None

    def test_get_config_from_env_case_insensitive(self):
        """Test that environment variable parsing is case insensitive."""
        manager = EnvManager()

        test_env = {
            "LOCCA_LLM__MODEL_NAME": "GPT-4",
            "locca_runtime__persistent_sessions": "true",  # Mixed case
        }

        with patch.dict(os.environ, test_env, clear=True):
            config = manager.get_config_from_env()

        assert config["llm"]["model_name"] == "GPT-4"
        assert config["runtime"]["persistent_sessions"] is True

    def test_env_file_loading_success(self):
        """Test successful .env file loading."""
        manager = EnvManager()
        
        # Create a list to track which paths were loaded
        loaded_paths = []
        
        def mock_load_dotenv(path, **kwargs):
            loaded_paths.append(Path(path).resolve())
            return None
            
        with patch("dotenv.load_dotenv", side_effect=mock_load_dotenv):
            # Mock path existence
            with patch.object(Path, "exists", return_value=True):
                manager.load_env_files()
                
                # Get the expected paths that should have been loaded
                expected_paths = [Path(p).resolve() for p in manager.env_paths]
                
                # Verify the correct paths were loaded
                assert len(loaded_paths) == len(expected_paths)
                for path in expected_paths:
                    assert path in loaded_paths

    def test_env_file_loading_missing_dotenv(self):
        """Test graceful handling when python-dotenv is not available."""
        manager = EnvManager()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'dotenv'")
        ):
            # Should not raise an error
            manager.load_env_files()

    def test_env_file_loading_file_error(self):
        """Test error handling when .env file loading fails."""
        test_error = Exception("File not found")
        manager = EnvManager()
        
        with patch("dotenv.load_dotenv") as mock_load, \
             patch.object(Path, "exists", return_value=True):
            
            # Configure the mock to raise our test error
            mock_load.side_effect = test_error
            
            # Verify the correct exception is raised with the expected message
            with pytest.raises(ConfigError) as exc_info:
                manager.load_env_files()
                
            # Verify the exception message includes the original error
            assert "Failed to load .env file" in str(exc_info.value)
            assert str(test_error) in str(exc_info.value)

    def test_env_file_loading_no_files(self):
        """Test behavior when no .env files exist."""
        manager = EnvManager()

        with patch.object(Path, "exists", return_value=False):
            # Should not raise an error
            manager.load_env_files()

    def test_with_prefix(self):
        """Test the with_prefix helper method."""
        manager = EnvManager(env_prefix="TEST_")
        assert manager.with_prefix("KEY") == "TEST_KEY"
        assert manager.with_prefix("TEST_KEY") == "TEST_KEY"  # Shouldn't double prefix

    def test_without_prefix(self):
        """Test the without_prefix helper method."""
        manager = EnvManager(env_prefix="TEST_")
        assert manager.without_prefix("TEST_KEY") == "KEY"
        assert manager.without_prefix("KEY") == "KEY"  # No prefix to remove

    def test_get_env(self):
        """Test getting an environment variable with prefix handling."""
        manager = EnvManager(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_API_KEY": "123"}, clear=True):
            assert manager.get_env("API_KEY") == "123"
            assert manager.get_env("TEST_API_KEY") == "123"  # Should work with or without prefix

    def test_set_env(self):
        """Test setting an environment variable with prefix handling."""
        manager = EnvManager(env_prefix="TEST_")
        with patch.dict(os.environ, {}, clear=True):
            manager.set_env("API_KEY", "123")
            assert os.environ["TEST_API_KEY"] == "123"

    def test_unset_env(self):
        """Test unsetting an environment variable with prefix handling."""
        manager = EnvManager(env_prefix="TEST_")
        with patch.dict(os.environ, {"TEST_API_KEY": "123"}, clear=True):
            manager.unset_env("API_KEY")
            assert "TEST_API_KEY" not in os.environ

    def test_save_to_env_file(self, tmp_path):
        """Test saving a key-value pair to an env file."""
        # Create test env file path
        env_path = tmp_path / ".env.local"
        
        # Create EnvManager instance with mocked dotenv
        with patch('dotenv.set_key') as mock_set_key:
            manager = EnvManager()
            
            # Mock get_env_path to return our test path
            with patch.object(manager, 'get_env_path', return_value=env_path):
                # Call the method under test
                manager.save_to_env_file("TEST_KEY", "test_value")
                
                # Get the actual path that was passed to set_key
                call_args = mock_set_key.call_args[0]
                called_path = call_args[0]
                
                # Verify the path points to the same file, regardless of string/Path
                assert Path(called_path).resolve() == env_path.resolve()
                
                # Verify the rest of the arguments
                assert call_args[1:] == ("TEST_KEY", "test_value")
                assert mock_set_key.call_args[1] == {"quote_mode": "never"}
    

    def test_remove_from_env_file(self, tmp_path):
        """Test removing a key from an env file."""
        # Create test env file path and ensure it exists
        env_path = tmp_path / ".env.local"
        env_path.touch()  # Create the file
        
        # Create EnvManager instance with mocked dotenv
        with patch('dotenv.unset_key') as mock_unset_key:
            manager = EnvManager()
            
            # Mock get_env_path to return our test path
            with patch.object(manager, 'get_env_path', return_value=env_path):
                # Call the method under test
                manager.remove_from_env_file("TEST_KEY")
                
                # Verify unset_key was called with the correct arguments
                mock_unset_key.assert_called_once()
                call_args = mock_unset_key.call_args[0]
                called_path = call_args[0]
                
                # Verify the path points to the same file, regardless of string/Path
                assert Path(called_path).resolve() == env_path.resolve()
                assert call_args[1] == "TEST_KEY"
                assert mock_unset_key.call_args[1] == {"quote_mode": "never"}

    def test_get_all_env_vars(self):
        """Test getting all environment variables with the prefix."""
        manager = EnvManager(env_prefix="TEST_")
        test_env = {
            "TEST_KEY1": "value1",
            "TEST_KEY2": "value2",
            "OTHER_PREFIX_KEY": "value3"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            result = manager.get_all_env_vars()
            
        assert result == {"TEST_KEY1": "value1", "TEST_KEY2": "value2"}
        assert "OTHER_PREFIX_KEY" not in result
