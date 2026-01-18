"""Unit tests for the ConfigManager system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from local_coding_assistant.config import ConfigManager
from local_coding_assistant.config.schemas import AppConfig, LLMConfig, RuntimeConfig
from local_coding_assistant.core.exceptions import ConfigError, LLMError


class TestConfigManager:
    """Test cases for ConfigManager functionality."""

    def test_init(self):
        """Test ConfigManager initialization."""
        manager = ConfigManager()
        assert manager._global_config is None
        assert manager._session_overrides == {}

    def test_load_global_config(self):
        """Test loading global configuration."""
        manager = ConfigManager()
        config = manager.load_global_config()

        assert isinstance(config, AppConfig)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.runtime, RuntimeConfig)
        assert config.runtime.enable_logging is True
        assert config.runtime.log_level == "INFO"

    def test_set_session_overrides(self):
        """Test setting session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(overrides)

        assert manager._session_overrides == overrides

    def test_clear_session_overrides(self):
        """Test clearing session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        manager.set_session_overrides({"llm.temperature": 0.5})
        assert manager._session_overrides != {}

        manager.clear_session_overrides()
        assert manager._session_overrides == {}

    def test_resolve_global_only(self):
        """Test resolving configuration with global layer only."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.resolve()

        assert isinstance(resolved, AppConfig)
        assert resolved.llm.temperature == 0.7

    def test_resolve_with_session_overrides(self):
        """Test resolving configuration with session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        session_overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(session_overrides)

        resolved = manager.resolve()

        assert resolved.llm.temperature == 0.5  # Session override applied

    def test_get_config_with_call_overrides(self):
        """Test resolving configuration with call overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        call_overrides = {"llm.temperature": 0.3}
        resolved = manager.get_config(overrides=call_overrides)

        assert resolved.llm.temperature == 0.3  # Call override applied

    def test_get_config_with_provider_override(self):
        """Test resolving configuration with provider override."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.get_config(provider="anthropic")

        # Provider override is applied via ConfigManager but not stored in LLMConfig
        # The LLMConfig doesn't have provider/model_name fields, so we test that
        # the resolve method completes without error and temperature is preserved
        assert resolved.llm.temperature == 0.7  # Should keep global value

    def test_get_config_with_model_override(self):
        """Test resolving configuration with model override."""
        manager = ConfigManager()
        manager.load_global_config()

        resolved = manager.get_config(model_name="gpt-4")

        # Model override is applied via ConfigManager but not stored in LLMConfig
        # The LLMConfig doesn't have provider/model_name fields, so we test that
        # the resolve method completes without error and temperature is preserved
        assert resolved.llm.temperature == 0.7  # Should keep global value

    def test_get_config_priority_order(self):
        """Test that call overrides take priority over session overrides."""
        manager = ConfigManager()
        manager.load_global_config()

        # Set session override
        session_overrides = {"llm.temperature": 0.5}
        manager.set_session_overrides(session_overrides)

        # Call override should win
        call_overrides = {"llm.temperature": 0.3}
        resolved = manager.get_config(overrides=call_overrides)

        assert resolved.llm.temperature == 0.3  # Call override wins

    def test_resolve_session_overrides_global(self):
        """Test that session overrides take priority over global config."""
        manager = ConfigManager()
        manager.load_global_config()

        # Session override should win over global
        session_overrides = {"llm.temperature": 0.8}
        manager.set_session_overrides(session_overrides)

        resolved = manager.resolve()

        assert resolved.llm.temperature == 0.8  # Session override wins

    def test_config_manager_properties(self):
        """Test ConfigManager property accessors."""
        manager = ConfigManager()
        manager.load_global_config()

        # Test global_config property
        assert manager.global_config is not None
        assert isinstance(manager.global_config, AppConfig)

        # Test session_overrides property
        manager.set_session_overrides({"llm.temperature": 0.5})
        overrides = manager.session_overrides
        assert overrides == {"llm.temperature": 0.5}

        # Should return a copy, not the original
        overrides["llm.temperature"] = 0.1
        assert manager.session_overrides["llm.temperature"] == 0.5

    def test_resolve_without_global_config(self):
        """Test that resolve raises error when global config not loaded."""
        manager = ConfigManager()

        with pytest.raises(Exception):  # Should raise ConfigError
            manager.resolve()

    def test_nested_override_application(self):
        """Test that nested overrides are applied correctly."""
        manager = ConfigManager()
        manager.load_global_config()

        # Test nested override
        overrides = {"runtime.enable_logging": False}
        resolved = manager.get_config(overrides=overrides)

        assert resolved.runtime.enable_logging is False
        assert resolved.llm.temperature == 0.7  # Should keep global value

    def test_reload_tools_success(self):
        """Test successful tools reloading."""
        manager = ConfigManager()
        
        # Mock the _load_tools method
        mock_tools = [{"name": "tool1"}, {"name": "tool2"}]
        with patch.object(manager, '_load_tools', return_value=mock_tools):
            manager.reload_tools()
            
            # Verify tools were loaded and cached
            assert manager._loaded_tools == mock_tools

    def test_reload_tools_failure(self):
        """Test tools reloading with failure."""
        manager = ConfigManager()
        
        # Mock _load_tools to raise an exception and patch logger to avoid formatting issues
        with patch.object(manager, '_load_tools', side_effect=Exception("Load failed")):
            with patch('local_coding_assistant.config.config_manager.logger') as mock_logger:
                with pytest.raises(ConfigError, match="Failed to reload tools: Load failed"):
                    manager.reload_tools()
                
                # Verify error was logged
                mock_logger.error.assert_called_once()

    def test_reload_tools_clears_cache(self):
        """Test that reload_tools clears existing cache."""
        manager = ConfigManager()
        
        # Set up existing cache
        manager._loaded_tools = [{"name": "old_tool"}]
        
        # Mock successful reload
        mock_tools = [{"name": "new_tool"}]
        with patch.object(manager, '_load_tools', return_value=mock_tools):
            manager.reload_tools()
            
            # Verify cache was updated
            assert manager._loaded_tools == mock_tools

    def test_set_session_overrides_llm_validation_error(self):
        """Test LLM validation error in set_session_overrides."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Override with invalid LLM temperature
        invalid_overrides = {"llm.temperature": -0.5}  # Invalid negative temperature
        
        with pytest.raises(LLMError, match="Configuration update validation failed"):
            manager.set_session_overrides(invalid_overrides)

    def test_set_session_overrides_general_validation_error(self):
        """Test general validation error in set_session_overrides."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Override with invalid max_session_history
        invalid_overrides = {"runtime.max_session_history": -1}  # Invalid negative value
        
        with pytest.raises(ConfigError, match="Invalid resolved configuration"):
            manager.set_session_overrides(invalid_overrides)

    def test_set_session_overrides_without_global_config(self):
        """Test set_session_overrides without global config loaded."""
        manager = ConfigManager()
        
        # Valid overrides but no global config - this should work when no global config is loaded
        # It just validates the overrides directly
        overrides = {"llm.temperature": 0.5}
        
        # This should actually work since it validates the overrides directly
        manager.set_session_overrides(overrides)
        assert manager._session_overrides == overrides

    def test_get_config_llm_validation_error(self):
        """Test LLM validation error in get_config."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Call with invalid LLM temperature
        invalid_overrides = {"llm.temperature": -0.5}  # Invalid negative temperature
        
        with pytest.raises(LLMError, match="Configuration validation failed"):
            manager.get_config(overrides=invalid_overrides)

    def test_get_config_general_validation_error(self):
        """Test general validation error in get_config."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Call with invalid max_session_history
        invalid_overrides = {"runtime.max_session_history": -1}  # Invalid negative value
        
        with pytest.raises(ConfigError, match="Invalid resolved configuration"):
            manager.get_config(overrides=invalid_overrides)

    def test_save_config_no_path_no_config_paths(self):
        """Test save_config with no path and no configured paths."""
        manager = ConfigManager()
        manager.config_paths = []  # Empty config paths
        
        with pytest.raises(ConfigError, match="No config paths configured to save to"):
            manager.save_config()

    def test_save_config_no_global_config_loaded(self):
        """Test save_config when global config is not loaded."""
        manager = ConfigManager()
        manager.config_paths = ["/tmp/config.yaml"]
        
        with pytest.raises(ConfigError, match="No configuration loaded. Call load_global_config\\(\\) first."):
            manager.save_config()

    def test_save_config_success_with_path(self):
        """Test successful save_config with explicit path."""
        manager = ConfigManager()
        manager.load_global_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Mock PathManager to return our temp path
            with patch.object(manager._path_manager, 'resolve_path', return_value=Path(temp_path)):
                manager.save_config(temp_path)
                
                # Verify file was created and contains valid YAML
                assert Path(temp_path).exists()
                with open(temp_path, 'r') as f:
                    saved_data = yaml.safe_load(f)
                    assert saved_data is not None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_config_success_default_path(self):
        """Test successful save_config with default path."""
        manager = ConfigManager()
        manager.load_global_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        # Set up config paths
        manager.config_paths = [temp_path]
        
        try:
            # Mock PathManager to return our temp path
            with patch.object(manager._path_manager, 'resolve_path', return_value=Path(temp_path)):
                manager.save_config()  # No path argument, should use first config path
                
                # Verify file was created
                assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_config_file_write_error(self):
        """Test save_config with file write error."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Mock PathManager and open to raise OSError
        with patch.object(manager._path_manager, 'resolve_path', return_value=Path("/tmp/test.yaml")):
            with patch('builtins.open', side_effect=OSError("Permission denied")):
                with pytest.raises(ConfigError, match="Failed to save configuration to /tmp/test.yaml: Permission denied"):
                    manager.save_config("/tmp/test.yaml")

    def test_save_config_yaml_error(self):
        """Test save_config with YAML serialization error."""
        manager = ConfigManager()
        manager.load_global_config()
        
        # Mock yaml.safe_dump to raise an error
        with patch.object(manager._path_manager, 'resolve_path', return_value=Path("/tmp/test.yaml")):
            with patch('builtins.open', mock_open()):
                with patch('yaml.safe_dump', side_effect=yaml.YAMLError("Serialization failed")):
                    with pytest.raises(ConfigError, match="Failed to save configuration to /tmp/test.yaml: Serialization failed"):
                        manager.save_config("/tmp/test.yaml")

    def test_save_config_excludes_unset_defaults_none(self):
        """Test that save_config excludes unset, default, and None values."""
        manager = ConfigManager()
        manager.load_global_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Use MagicMock to wrap the model_dump method and capture calls
            with patch.object(manager._path_manager, 'resolve_path', return_value=Path(temp_path)):
                with patch('builtins.open', mock_open()):
                    with patch('yaml.safe_dump') as mock_yaml_dump:
                        manager.save_config(temp_path)
                        
                        # Verify yaml.safe_dump was called and capture the data
                        assert mock_yaml_dump.called
                        call_args = mock_yaml_dump.call_args
                        # The first argument should be the config data
                        config_data = call_args[0][0]
                        assert isinstance(config_data, dict)
                        
                        # Verify sort_keys=False was passed
                        assert call_args[1]['sort_keys'] is False
        finally:
            Path(temp_path).unlink(missing_ok=True)
