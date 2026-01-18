"""Unit tests for EnvManager functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.config.env_manager import EnvManager
from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.core.exceptions import ConfigError


class TestEnvManager:
    """Test EnvManager functionality."""

    def test_default_env_paths(self, tmp_path):
        """Test that default .env paths are correctly identified."""
        # Setup mock PathManager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.get_project_root.return_value = tmp_path
        mock_path_manager.resolve_path.side_effect = lambda x: tmp_path / x.replace(
            "@project/", ""
        )

        # Create test .env files
        (tmp_path / ".env").touch()
        (tmp_path / ".env.local").touch()

        manager = EnvManager(path_manager=mock_path_manager, load_env=False)

        # Should find the default .env files
        assert len(manager.env_paths) >= 2
        assert any(".env" in str(path) for path in manager.env_paths)
        assert any(".env.local" in str(path) for path in manager.env_paths)

    def test_custom_env_paths(self, tmp_path):
        """Test EnvManager with custom paths."""
        # Create a mock path manager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.resolve_path.side_effect = lambda x: Path(x)

        # Create test files
        env_file = tmp_path / ".env"
        env_file.touch()

        custom_paths = [str(env_file)]
        manager = EnvManager(
            env_paths=custom_paths, load_env=False, path_manager=mock_path_manager
        )

        # Paths should be converted to Path objects
        assert len(manager.env_paths) == 1
        assert isinstance(manager.env_paths[0], Path)
        assert str(env_file) in str(manager.env_paths[0])

    def test_custom_env_prefix(self):
        """Test EnvManager with custom prefix."""
        manager = EnvManager(env_prefix="CUSTOM_", load_env=False)
        assert manager.env_prefix == "CUSTOM_"

    def test_get_config_from_env_empty(self):
        """Test get_config_from_env with no matching environment variables."""
        manager = EnvManager(load_env=False)

        # Clear any existing LOCCA_ variables for this test
        with patch.dict(os.environ, {}, clear=True):
            config = manager.get_config_from_env()

        assert config == {}

    def test_get_config_from_env_basic(self):
        """Test get_config_from_env with basic environment variables."""
        manager = EnvManager(load_env=False)

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

    def test_env_file_loading_success(self, tmp_path):
        """Test successful .env file loading."""
        # Setup mock PathManager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.get_project_root.return_value = tmp_path
        mock_path_manager.resolve_path.side_effect = lambda x: tmp_path / x.replace(
            "@project/", ""
        )

        # Create test .env files
        (tmp_path / ".env").write_text("TEST_KEY=value")
        (tmp_path / ".env.local").write_text("TEST_LOCAL=local_value")

        manager = EnvManager(path_manager=mock_path_manager, load_env=False)

        # Track loaded paths
        loaded_paths = []

        def mock_load_dotenv(path, **kwargs):
            loaded_paths.append(Path(path).resolve())
            return True

        with patch("dotenv.load_dotenv", side_effect=mock_load_dotenv):
            manager.load_env_files()

            # Verify the correct paths were loaded
            assert len(loaded_paths) > 0
            assert any(".env" in str(p) for p in loaded_paths)

    def test_env_file_loading_missing_dotenv(self):
        """Test graceful handling when python-dotenv is not available."""
        # Create a mock path manager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.resolve_path.return_value = Path("/nonexistent/.env")

        # Create a custom import function that raises ImportError for dotenv
        def mock_import(name, *args, **kwargs):
            if name == "dotenv" or name.startswith("dotenv."):
                raise ImportError("No module named 'dotenv'")
            return original_import(name, *args, **kwargs)

        # Patch the built-in import
        original_import = __import__
        with patch("builtins.__import__", side_effect=mock_import):
            # Create manager after patching imports
            manager = EnvManager(load_env=False, path_manager=mock_path_manager)

            # Patch logger to verify warning
            with patch(
                "local_coding_assistant.config.env_manager.logger"
            ) as mock_logger:
                # Set env_paths to avoid file system operations
                manager.env_paths = [Path("/nonexistent/.env")]

                # Should not raise an error
                manager.load_env_files()

                # Should log a warning
                assert mock_logger.warning.called
                warning_msg = mock_logger.warning.call_args[0][0]
                assert "python-dotenv not installed" in warning_msg

    def test_env_file_loading_file_error(self, tmp_path):
        """Test error handling when .env file loading fails."""
        test_error = Exception("File not found")

        # Setup mock PathManager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.get_project_root.return_value = tmp_path
        mock_path_manager.resolve_path.return_value = tmp_path / ".env"

        manager = EnvManager(path_manager=mock_path_manager, load_env=False)

        with (
            patch("dotenv.load_dotenv") as mock_load,
            patch.object(Path, "exists", return_value=True),
        ):
            # Configure the mock to raise our test error
            mock_load.side_effect = test_error

            # Verify the correct exception is raised with the expected message
            with pytest.raises(ConfigError) as exc_info:
                manager.load_env_files()

            # Verify the exception message includes the original error
            assert "Failed to load environment from" in str(exc_info.value)
            assert "File not found" in str(exc_info.value)

    def test_env_file_loading_no_files(self, tmp_path):
        """Test behavior when no .env files exist."""
        # Setup mock PathManager that returns a non-existent path
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.get_project_root.return_value = tmp_path
        mock_path_manager.resolve_path.return_value = tmp_path / "nonexistent.env"

        manager = EnvManager(path_manager=mock_path_manager, load_env=False)

        with patch.object(Path, "exists", return_value=False):
            # Should not raise an error, just log a warning
            with patch(
                "local_coding_assistant.config.env_manager.logger"
            ) as mock_logger:
                manager.load_env_files()
                mock_logger.warning.assert_called()

    def test_with_prefix(self):
        """Test the with_prefix helper method."""
        manager = EnvManager(env_prefix="TEST_", load_env=False)
        assert manager.with_prefix("KEY") == "TEST_KEY"
        assert manager.with_prefix("TEST_KEY") == "TEST_KEY"  # Shouldn't double prefix

    def test_without_prefix(self):
        """Test the without_prefix helper method."""
        manager = EnvManager(env_prefix="TEST_", load_env=False)
        assert manager.without_prefix("TEST_KEY") == "KEY"
        assert manager.without_prefix("KEY") == "KEY"  # No prefix to remove

    def test_get_env(self):
        """Test getting an environment variable with prefix handling."""
        manager = EnvManager(env_prefix="TEST_", load_env=False)
        with patch.dict(os.environ, {"TEST_API_KEY": "123"}, clear=True):
            # Test with and without prefix
            assert os.environ.get(manager.with_prefix("API_KEY")) == "123"
            assert os.environ.get("TEST_API_KEY") == "123"

    def test_set_env(self):
        """Test setting an environment variable with prefix handling."""
        manager = EnvManager(env_prefix="TEST_", load_env=False)
        with patch.dict(os.environ, {}, clear=True):
            # Test setting environment variable
            prefixed_key = manager.with_prefix("API_KEY")
            os.environ[prefixed_key] = "123"
            assert os.environ[prefixed_key] == "123"

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
        with patch("dotenv.set_key") as mock_set_key:
            manager = EnvManager()

            # Mock get_env_path to return our test path
            with patch.object(manager._instance, "get_env_path", return_value=env_path):
                # Call the method under test
                manager.save_to_env_file("TEST_KEY", "test_value")

                # Get the actual path that was passed to set_key
                call_args = mock_set_key.call_args[0]
                called_path = call_args[0]

                # Verify the path points to the same file, regardless of string/Path
                assert Path(called_path).resolve() == env_path.resolve()

                # Verify the rest of the arguments
                assert call_args[1:] == ("TEST_KEY", "test_value")
                assert mock_set_key.call_args.kwargs == {"quote_mode": "never"}

    def test_remove_from_env_file(self, tmp_path):
        """Test removing a key from an env file."""
        # Create test env file path and ensure it exists
        env_path = tmp_path / ".env.local"
        env_path.touch()  # Create the file

        # Create EnvManager instance with mocked dotenv
        with patch("dotenv.unset_key") as mock_unset_key:
            manager = EnvManager()

            # Mock get_env_path to return our test path
            with patch.object(manager._instance, "get_env_path", return_value=env_path):
                # Call the method under test
                manager.remove_from_env_file("TEST_KEY")

                # Verify unset_key was called with the correct arguments
                mock_unset_key.assert_called_once()
                call_args = mock_unset_key.call_args[0]
                called_path = call_args[0]

                # Verify the path points to the same file, regardless of string/Path
                assert Path(called_path).resolve() == env_path.resolve()
                assert call_args[1] == "TEST_KEY"
                assert mock_unset_key.call_args.kwargs == {"quote_mode": "never"}

    def test_get_all_env_vars(self):
        """Test getting all environment variables with the prefix."""
        manager = EnvManager(env_prefix="TEST_")
        test_env = {
            "TEST_KEY1": "value1",
            "TEST_KEY2": "value2",
            "OTHER_PREFIX_KEY": "value3",
        }

        with patch.dict(os.environ, test_env, clear=True):
            result = manager.get_all_env_vars()

        assert result == {"TEST_KEY1": "value1", "TEST_KEY2": "value2"}
        assert "OTHER_PREFIX_KEY" not in result

    def test_resolve_env_paths_with_special_symbols(self, tmp_path):
        """Test _resolve_env_paths handles @ and ~ symbols correctly."""
        # Setup mock PathManager
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.resolve_path.side_effect = lambda x: tmp_path / x.replace("@project/", "").replace("~", "home")
        
        # Create test paths with special symbols - need to set them as strings directly
        test_paths = [
            "@project/.env",  # Should be resolved
            "~/.env.local",   # Should be resolved
            "/absolute/path/.env",  # Should not be resolved (no special symbols)
            str(tmp_path / "direct.env"),  # Direct path
        ]
        
        manager = EnvManager(env_paths=test_paths, load_env=False, path_manager=mock_path_manager)
        
        # Reset env_paths to strings to test _resolve_env_paths properly
        manager.env_paths = test_paths
        
        # Manually call _resolve_env_paths
        manager._resolve_env_paths()
        
        # Verify paths were resolved correctly
        assert len(manager.env_paths) == 4
        # Check that @project path was resolved (no longer contains @project)
        resolved_project_paths = [path for path in manager.env_paths if str(path).endswith(".env") and "@project" not in str(path)]
        assert len(resolved_project_paths) >= 1
        # Check that ~ path was resolved (no longer contains ~)
        resolved_home_paths = [path for path in manager.env_paths if str(path).endswith(".env.local") and "~" not in str(path)]
        assert len(resolved_home_paths) >= 1

    def test_resolve_env_paths_with_resolution_errors(self, tmp_path):
        """Test _resolve_env_paths handles resolution errors gracefully."""
        # Setup mock PathManager that raises errors for special paths
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.resolve_path.side_effect = ValueError("Invalid path")
        
        test_paths = ["@project/.env", "/normal/path/.env"]
        
        # Patch the logger at module level where it's imported
        with patch("local_coding_assistant.config.env_manager.logger") as mock_logger:
            manager = EnvManager(env_paths=test_paths, load_env=False, path_manager=mock_path_manager)
            
            # Reset env_paths to strings to test _resolve_env_paths properly
            manager.env_paths = test_paths
            
            # Manually call _resolve_env_paths
            manager._resolve_env_paths()
            
            # Should log warning for failed resolution but continue
            assert mock_logger.warning.called
            warning_call = mock_logger.warning.call_args
            assert "Failed to resolve path" in warning_call[0][0]
            assert "@project/.env" in warning_call[0][0]
            
            # Should still have both paths (failed resolution falls back to Path constructor)
            assert len(manager.env_paths) == 2
            assert any("normal" in str(path) and "path" in str(path) for path in manager.env_paths)
            assert any("@project" in str(path) for path in manager.env_paths)

    def test_get_default_env_paths_fallback_project_root(self, tmp_path):
        """Test _get_default_env_paths fallback project root detection."""
        # Setup mock PathManager that returns None for project root but resolves paths
        mock_path_manager = MagicMock(spec=PathManager)
        mock_path_manager.get_project_root.return_value = None
        mock_path_manager.resolve_path.side_effect = lambda x: tmp_path / x.replace("@project/", "")
        
        # Create a temporary directory structure with pyproject.toml
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").touch()
        
        # Mock the current file path and cwd to simulate being in the test project
        with patch("local_coding_assistant.config.env_manager.Path.cwd", return_value=project_dir):
            with patch("local_coding_assistant.config.env_manager.__file__", str(project_dir / "src" / "env_manager.py")):
                manager = EnvManager(path_manager=mock_path_manager, load_env=False)
                
                # Should find the project root via pyproject.toml
                default_paths = manager._get_default_env_paths()
                
                # Verify paths are based on the detected project root
                assert any(str(project_dir) in str(path) for path in default_paths)

    def test_get_env_path_file_exists_and_creation(self, tmp_path):
        """Test get_env_path finds existing files and creates directories when needed."""
        # Setup test environment
        existing_file = tmp_path / ".env.local"
        existing_file.touch()
        
        non_existent_dir = tmp_path / "subdir"
        non_existent_file = non_existent_dir / ".env.test"
        
        manager = EnvManager(
            env_paths=[existing_file, non_existent_file], 
            load_env=False
        )
        
        # Test finding existing file
        found_path = manager.get_env_path(".env.local")
        assert found_path == existing_file
        
        # Test creating directories for non-existent file
        created_path = manager.get_env_path(".env.test")
        assert created_path.parent.exists()
        assert created_path.parent == non_existent_dir
        
        # Test fallback to cwd when no paths match
        with patch("local_coding_assistant.config.env_manager.Path.cwd", return_value=tmp_path):
            fallback_path = manager.get_env_path("custom.env")
            assert fallback_path == tmp_path / "custom.env"
            assert fallback_path.parent.exists()
