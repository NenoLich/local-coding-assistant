"""Unit tests for LLMManager update_config functionality."""

# NOTE: These tests are for an older version of LLMManager that had update_config functionality.
# The current LLMManager uses ConfigManager for configuration management and doesn't have
# an update_config method. These tests are commented out until the functionality is restored.

# from unittest.mock import MagicMock, patch
#
# import pytest
#
# from local_coding_assistant.agent.llm_manager import LLMManager
# from local_coding_assistant.config.schemas import LLMConfig
# from local_coding_assistant.core.exceptions import LLMError
#
#
# class TestLLMManagerUpdateConfig:
#     """Test LLMManager update_config functionality."""
#
#     def test_update_config_temperature_only(self):
#         """Test update_config with temperature change only (safe update)."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         original_model = llm_manager._current_llm_config.model_name
#
#         # Update only temperature (should not require new client)
#         llm_manager.update_config(temperature=0.8)
#
#         # Config should be updated
#         assert llm_manager._current_llm_config.temperature == 0.8
#         assert llm_manager._current_llm_config.model_name == original_model
#
#     def test_update_config_model_requires_client(self):
#         """Test update_config with model change (requires new client)."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         original_temp = llm_manager._current_llm_config.temperature
#
#         with patch.object(llm_manager, "client", None):
#             # Update model (should require new client)
#             llm_manager.update_config(model_name="gpt-4")
#
#             # Config should be updated
#             assert llm_manager._current_llm_config.model_name == "gpt-4"
#             assert llm_manager._current_llm_config.temperature == original_temp
#
#     def test_update_config_multiple_params(self):
#         """Test update_config with multiple parameters."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         # Update multiple parameters
#         llm_manager.update_config(model_name="gpt-4", temperature=0.8, max_tokens=200)
#
#         assert llm_manager._current_llm_config.model_name == "gpt-4"
#         assert llm_manager._current_llm_config.temperature == 0.8
#         assert llm_manager._current_llm_config.max_tokens == 200
#
#     def test_update_config_invalid_temperature(self):
#         """Test update_config with invalid temperature."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         # Should raise LLMError for invalid temperature
#         with pytest.raises(LLMError, match="Configuration update validation failed"):
#             llm_manager.update_config(temperature=-1)
#
#     def test_update_config_invalid_max_tokens(self):
#         """Test update_config with invalid max_tokens."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         # Should raise LLMError for invalid max_tokens
#         with pytest.raises(LLMError, match="Configuration update validation failed"):
#             llm_manager.update_config(max_tokens=0)
#
#     def test_update_config_no_changes(self):
#         """Test update_config with no parameters provided."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         original_model = llm_manager._current_llm_config.model_name
#         original_temp = llm_manager._current_llm_config.temperature
#
#         # Call with no updates
#         llm_manager.update_config()
#
#         # Config should remain unchanged
#         assert llm_manager._current_llm_config.model_name == original_model
#         assert llm_manager._current_llm_config.temperature == original_temp
#
#     def test_update_config_provider_change(self):
#         """Test update_config with provider change (requires new client)."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         with patch.object(llm_manager, "client", None):
#             # Update provider (should require new client)
#             llm_manager.update_config(provider="anthropic")
#
#             assert llm_manager._current_llm_config.provider == "anthropic"
#
#     def test_update_config_api_key_change(self):
#         """Test update_config with API key change (may require new client)."""
#         config = LLMConfig(model_name="gpt-3.5-turbo", api_key="old_key")
#         llm_manager = LLMManager()
#         llm_manager.config_manager = MagicMock()
#         # Mock the resolve method to return proper config
#         mock_config = MagicMock()
#         mock_config.llm = config
#         llm_manager.config_manager.resolve.return_value = mock_config
#         llm_manager._current_llm_config = config
#
#         with patch.object(llm_manager, "client", None):
#             # Update API key (should require new client)
#             llm_manager.update_config(api_key="new_key")
#
#             assert llm_manager._current_llm_config.api_key == "new_key"
