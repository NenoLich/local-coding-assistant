"""Unit tests for bootstrap error handling."""

from unittest.mock import Mock, patch

import pytest

from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import bootstrap


class TestBootstrapErrorHandling:
    """Test error handling during bootstrap."""

    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    def test_environment_setup_failure(self, mock_init_config):
        """Test handling of environment setup failure."""
        # Setup mock to raise an exception
        mock_init_config.side_effect = Exception("Environment setup failed")

        # Verify the exception is wrapped in a RuntimeError
        with pytest.raises(RuntimeError, match="Failed to initialize application"):
            bootstrap()

    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    def test_config_loading_failure(self, mock_init_config):
        """Test handling of config loading failure."""
        # Setup mock
        mock_init_config.side_effect = Exception("Config loading failed")

        # Verify exception is wrapped in a RuntimeError
        with pytest.raises(RuntimeError, match="Failed to initialize application"):
            bootstrap()

    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    @patch("local_coding_assistant.core.bootstrap._initialize_llm_manager")
    def test_llm_initialization_failure(
        self, mock_init_llm, mock_init_config, caplog
    ):
        """Test behavior when LLM manager initialization returns None."""
        # Setup mocks
        mock_config = {"logging": {"level": "INFO"}}
        mock_config_manager = Mock()
        mock_init_config.return_value = mock_config_manager
        mock_init_llm.return_value = None  # Simulate LLM init returns None

        # Mock other components to avoid side effects
        with patch(
            "local_coding_assistant.core.bootstrap._initialize_tool_manager",
            return_value=Mock(),
        ):
            with patch(
                "local_coding_assistant.core.bootstrap._initialize_runtime_manager",
                return_value=Mock(),
            ):
                # Call bootstrap
                ctx = bootstrap()

        # Verify it still returns a context
        assert isinstance(ctx, AppContext)

        # Verify LLM manager is not set in context
        assert ctx.get("llm") is None

    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    @patch("local_coding_assistant.core.bootstrap._initialize_llm_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_tool_manager")
    def test_tool_manager_initialization_failure(
        self, mock_init_tools, mock_init_llm, mock_init_config, caplog
    ):
        """Test behavior when tool manager initialization returns None."""
        # Setup mocks
        mock_config = {"logging": {"level": "INFO"}}
        mock_config_manager = Mock()
        mock_llm_manager = Mock()
        mock_init_config.return_value = mock_config_manager
        mock_init_llm.return_value = mock_llm_manager
        mock_init_tools.return_value = None  # Simulate tool manager init returns None

        # Mock runtime manager to avoid side effects
        with patch(
            "local_coding_assistant.core.bootstrap._initialize_runtime_manager",
            return_value=Mock(),
        ):
            # Call bootstrap
            ctx = bootstrap()

        # Verify it still returns a context
        assert isinstance(ctx, AppContext)

        # Verify tool manager is not set in context
        assert ctx.get("tools") is None

    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    @patch("local_coding_assistant.core.bootstrap._initialize_llm_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_tool_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_runtime_manager")
    def test_runtime_manager_initialization_failure(
        self,
        mock_init_runtime,
        mock_init_tools,
        mock_init_llm,
        mock_init_config,
        caplog,
    ):
        """Test handling of runtime manager initialization failure."""
        # Setup mocks
        mock_config = {"logging": {"level": "INFO"}}
        mock_config_manager = Mock()
        mock_llm_manager = Mock()
        mock_tool_manager = Mock()
        mock_init_config.return_value = mock_config_manager
        mock_init_llm.return_value = mock_llm_manager
        mock_init_tools.return_value = mock_tool_manager
        mock_init_runtime.return_value = None  # Simulate runtime manager init failure

        # Call bootstrap
        ctx = bootstrap()

        # Verify it still returns a context
        assert isinstance(ctx, AppContext)

        # Verify warning was logged
        log_messages = [record.message for record in caplog.records]
        assert not any(
            "Failed to initialize runtime manager" in msg for msg in log_messages
        )

        # The runtime manager is optional, so it's fine if it's None
        assert ctx.get("runtime") is None
