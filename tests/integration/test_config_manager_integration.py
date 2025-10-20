"""Integration tests for the ConfigManager system."""

import os
from unittest.mock import patch

import pytest

from local_coding_assistant.agent.llm_manager import LLMManager
from local_coding_assistant.config import ConfigManager
from local_coding_assistant.config.schemas import LLMConfig


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with other components."""

    def test_llm_manager_integration(self):
        """Test that LLMManager works with ConfigManager."""
        manager = ConfigManager()
        manager.load_global_config()

        llm_manager = LLMManager(manager)

        # Should be able to get LLM config
        llm_config = llm_manager._get_llm_config()
        assert isinstance(llm_config, LLMConfig)
        assert llm_config.model_name == "gpt-5-mini"

        # Should be able to get config with overrides
        llm_config = llm_manager._get_llm_config(model_name="gpt-4")
        assert llm_config.model_name == "gpt-4"

    @patch.dict("os.environ", {"LOCCA_TEST_MODE": "true"})
    def test_llm_manager_mock_response(self):
        """Test that LLMManager returns mock responses in test mode."""
        from local_coding_assistant.agent.llm_manager import LLMManager, LLMRequest

        manager = ConfigManager()
        manager.load_global_config()

        llm_manager = LLMManager(manager)
        request = LLMRequest(prompt="test prompt")

        response = llm_manager._generate_mock_response(request)

        assert response.content == "[LLMManager] Echo: test prompt"
        assert response.model_used == "gpt-5-mini"
        assert response.tokens_used == 50
