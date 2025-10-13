"""Unit tests for LLMManager update_config functionality."""

import pytest
from unittest.mock import patch

from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest
from local_coding_assistant.config.schemas import LLMConfig
from local_coding_assistant.core.exceptions import AgentError


class TestLLMManagerUpdateConfig:
    """Test LLMManager update_config functionality."""

    def test_update_config_temperature_only(self):
        """Test update_config with temperature change only (safe update)."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
        llm_manager = LLMManager(config)

        original_model = llm_manager.config.model_name
        original_temp = llm_manager.config.temperature

        # Update only temperature (should not require new client)
        llm_manager.update_config(temperature=0.8)

        # Config should be updated
        assert llm_manager.config.temperature == 0.8
        assert llm_manager.config.model_name == original_model

    def test_update_config_model_requires_client(self):
        """Test update_config with model change (requires new client)."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
        llm_manager = LLMManager(config)

        original_temp = llm_manager.config.temperature

        with patch.object(llm_manager, "client", None):
            # Update model (should require new client)
            llm_manager.update_config(model_name="gpt-4")

            # Config should be updated
            assert llm_manager.config.model_name == "gpt-4"
            assert llm_manager.config.temperature == original_temp

    def test_update_config_multiple_params(self):
        """Test update_config with multiple parameters."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100)
        llm_manager = LLMManager(config)

        # Update multiple parameters
        llm_manager.update_config(model_name="gpt-4", temperature=0.8, max_tokens=200)

        assert llm_manager.config.model_name == "gpt-4"
        assert llm_manager.config.temperature == 0.8
        assert llm_manager.config.max_tokens == 200

    def test_update_config_invalid_temperature(self):
        """Test update_config with invalid temperature."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
        llm_manager = LLMManager(config)

        # Should raise AgentError for invalid temperature
        with pytest.raises(AgentError, match="Configuration update validation failed"):
            llm_manager.update_config(temperature=-1)

    def test_update_config_invalid_max_tokens(self):
        """Test update_config with invalid max_tokens."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
        llm_manager = LLMManager(config)

        # Should raise AgentError for invalid max_tokens
        with pytest.raises(AgentError, match="Configuration update validation failed"):
            llm_manager.update_config(max_tokens=0)

    def test_update_config_no_changes(self):
        """Test update_config with no parameters provided."""
        config = LLMConfig(model_name="gpt-3.5-turbo", temperature=0.7)
        llm_manager = LLMManager(config)

        original_model = llm_manager.config.model_name
        original_temp = llm_manager.config.temperature

        # Call with no updates
        llm_manager.update_config()

        # Config should remain unchanged
        assert llm_manager.config.model_name == original_model
        assert llm_manager.config.temperature == original_temp

    def test_update_config_provider_change(self):
        """Test update_config with provider change (requires new client)."""
        config = LLMConfig(model_name="gpt-3.5-turbo", provider="openai")
        llm_manager = LLMManager(config)

        with patch.object(llm_manager, "client", None):
            # Update provider (should require new client)
            llm_manager.update_config(provider="anthropic")

            assert llm_manager.config.provider == "anthropic"

    def test_update_config_api_key_change(self):
        """Test update_config with API key change (may require new client)."""
        config = LLMConfig(model_name="gpt-3.5-turbo", api_key="old_key")
        llm_manager = LLMManager(config)

        with patch.object(llm_manager, "client", None):
            # Update API key (should require new client)
            llm_manager.update_config(api_key="new_key")

            assert llm_manager.config.api_key == "new_key"
