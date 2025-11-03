"""
Unit tests for core bootstrap functionality.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import bootstrap


class TestBootstrapInitialization:
    """Test bootstrap initialization."""

    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap._load_config")
    def test_bootstrap_successful_initialization(
            self,
            mock_load_config,
            mock_get_config_manager,
            mock_get_logger,
            mock_setup_logging,
    ):
        """Test successful bootstrap initialization."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        # Mock config manager
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create a mock LLMManager class
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            # Mock RuntimeManager instance
            mock_runtime_instance = MagicMock()
            mock_runtime_class.return_value = mock_runtime_instance

            # Mock ToolManager instance
            mock_tool_instance = MagicMock()
            mock_tool_class.return_value = mock_tool_instance

            # Call the bootstrap function
            ctx = bootstrap()

            # Verify context is returned
            assert isinstance(ctx, AppContext)

            # Verify services are registered
            assert ctx.get("llm") == mock_llm_instance
            assert ctx.get("runtime") == mock_runtime_instance

            # Verify logging was set up with config-based log level (INFO)
            mock_setup_logging.assert_called_once_with(mock_config, None)

            # Verify LLM manager was initialized with config manager
            mock_llm_class.assert_called_once()
            call_args = mock_llm_class.call_args[0]
            assert len(call_args) >= 1
            assert call_args[0] == mock_config_manager

            # Verify runtime manager was created with the correct arguments
            mock_runtime_class.assert_called_once_with(
                llm_manager=mock_llm_instance,
                tool_manager=mock_tool_instance,
                config_manager=mock_config_manager,
            )

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_with_custom_config_path(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap with custom config path."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "DEBUG"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap(config_path="/custom/config.yaml")

            # Verify custom config path was used
            mock_load_config.assert_called_once_with("/custom/config.yaml")

            # Verify LLM manager was initialized
            mock_llm_class.assert_called_once()
            call_args = mock_llm_class.call_args[0]
            assert len(call_args) >= 1
            assert call_args[0] == mock_config_manager

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_with_custom_log_level(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap with custom log level."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "WARNING"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function with custom log level
            ctx = bootstrap(log_level=logging.DEBUG)

            # Verify custom log level was used
            mock_setup_logging.assert_called_once_with(mock_config, logging.DEBUG)

            # Verify LLM manager was initialized
            mock_llm_class.assert_called_once()
            call_args = mock_llm_class.call_args[0]
            assert len(call_args) >= 1
            assert call_args[0] == mock_config_manager


class TestBootstrapErrorHandling:
    """Test bootstrap error handling."""

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_llm_initialization_failure(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap when LLM manager initialization fails."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_class = MagicMock(side_effect=Exception("LLM initialization failed"))
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class, \
                patch('local_coding_assistant.core.bootstrap.logger') as mock_logger:
            mock_tool_class.return_value = mock_tool_instance

            # Call the function - should not raise an exception
            ctx = bootstrap()

            # Verify LLM manager was called
            mock_llm_class.assert_called_once()

            # Verify runtime manager was not called
            mock_runtime_class.assert_not_called()

            # Verify LLM and runtime are None in the context
            assert ctx.get("llm") is None
            assert ctx.get("runtime") is None

            # Verify config is not set in the context when initialization fails
            assert ctx.get("config") is None

            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("LLM initialization failed" in call for call in warning_calls)

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_tool_manager_initialization_failure(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap when tool manager initialization fails."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_tool_class = MagicMock(side_effect=Exception("Tool manager failed"))
        mock_runtime_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager', mock_tool_class), \
                patch('local_coding_assistant.core.bootstrap.logger') as mock_logger:
            mock_runtime_class.return_value = mock_runtime_instance

            # Call the function
            ctx = bootstrap()

            # Verify LLM manager was called
            mock_llm_class.assert_called_once()

            # Verify tool manager was called and failed
            mock_tool_class.assert_called_once()

            # Verify runtime manager was called with None for tool_manager
            mock_runtime_class.assert_called_once_with(
                llm_manager=mock_llm_instance,
                tool_manager=None,
                config_manager=mock_config_manager
            )

            # Verify context has LLM and runtime, but tools is None
            assert ctx.get("llm") == mock_llm_instance
            assert ctx.get("tools") is None
            assert ctx.get("runtime") == mock_runtime_instance

            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("Tool manager failed" in call for call in warning_calls)
    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_logging_disabled(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap when logging is disabled."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = False
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap()

            # Verify logging was set to CRITICAL level (50)
            mock_setup_logging.assert_called_once_with(mock_config, 50)

            # Verify services are initialized
            assert ctx.get("llm") == mock_llm_instance
            assert ctx.get("runtime") == mock_runtime_instance

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_config_load_failure(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap when config loading fails."""
        # Setup test data - config loading fails
        mock_load_config.return_value = (None, False)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = None
        mock_config_manager.load_global_config.return_value = None
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_class = MagicMock()
        mock_runtime_class = MagicMock()
        mock_tool_class = MagicMock()

        # Mock the provider manager instance that will be used
        mock_provider_manager = MagicMock()
        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager', mock_runtime_class), \
                patch('local_coding_assistant.core.bootstrap.ToolManager', mock_tool_class), \
                patch('local_coding_assistant.core.bootstrap.logger', mock_logger), \
                patch('local_coding_assistant.core.bootstrap.get_config_manager', return_value=mock_config_manager), \
                patch('local_coding_assistant.core.bootstrap._load_config', return_value=(None, False)), \
                patch('local_coding_assistant.providers.provider_manager.provider_manager', mock_provider_manager):
            # Call the function
            ctx = bootstrap()

            # Verify context is still created
            assert isinstance(ctx, AppContext)

# Verify LLM manager was still initialized with config manager and provider manager
            mock_llm_class.assert_called_once()
            args, _ = mock_llm_class.call_args
            assert len(args) == 2
            assert args[0] == mock_config_manager
            # We can't directly compare the provider_manager instances, but we can verify the call

            # Verify runtime manager was called with the correct keyword arguments
            mock_runtime_class.assert_called_once()
            _, kwargs = mock_runtime_class.call_args
            assert kwargs == {
                'llm_manager': mock_llm_class.return_value,
                'tool_manager': mock_tool_class.return_value,
                'config_manager': mock_config_manager
            }

            # Verify tool manager was called
            mock_tool_class.assert_called_once()

            # Verify no warning was logged for config load failure
            mock_logger.warning.assert_not_called()

            # Verify provider manager was passed to LLMManager
            mock_llm_class.assert_called_once_with(mock_config_manager, mock_provider_manager)


class TestBootstrapIntegration:
    """Test bootstrap integration scenarios."""

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_complete_flow(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test complete bootstrap flow."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap()

            # Verify context is returned
            assert isinstance(ctx, AppContext)

            # Verify services are registered
            assert ctx.get("llm") == mock_llm_instance
            assert ctx.get("runtime") == mock_runtime_instance

            # Verify LLM manager was initialized
            mock_llm_class.assert_called_once()
            call_args = mock_llm_class.call_args[0]
            assert len(call_args) >= 1
            assert call_args[0] == mock_config_manager

            # Verify runtime manager was created with the correct arguments
            mock_runtime_class.assert_called_once_with(
                llm_manager=mock_llm_instance,
                tool_manager=mock_tool_instance,
                config_manager=mock_config_manager,
            )

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_without_global_config(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test bootstrap when no global config exists."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        
        # Mock the config manager to return None for global config
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = None
        mock_config_manager.load_global_config.return_value = None
        mock_get_config_manager.return_value = mock_config_manager
        
        # Mock _load_config to return our test config
        mock_load_config.return_value = (mock_config, True)

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap()

            # Verify context is returned
            assert isinstance(ctx, AppContext)
            
            # Verify config manager was used
            assert mock_get_config_manager.call_count == 2  # Called twice: once in bootstrap, once in _initialize_llm_manager
            
            # Verify LLM manager was initialized with the config manager
            mock_llm_class.assert_called_once()
            args, _ = mock_llm_class.call_args
            assert len(args) == 2
            assert args[0] == mock_config_manager
            
            # Verify no warnings were logged
            mock_logger.warning.assert_not_called()


class TestBootstrapContextRegistration:
    """Test context registration in bootstrap."""

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_context_service_registration(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test that services are properly registered in context."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = MagicMock()
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = MagicMock()
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap()

            # Verify all services are registered
            assert ctx.get("llm") == mock_llm_instance
            assert ctx.get("runtime") == mock_runtime_instance
            assert ctx.get("tools") == mock_tool_instance  # tools should be registered

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_context_none_services(
            self,
            mock_setup_logging,
            mock_get_logger,
            mock_get_config_manager,
            mock_load_config,
    ):
        """Test context when some services are None."""
        # Setup test data
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = None  # Simulate no LLM config
        mock_load_config.return_value = (mock_config, True)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create mock instances
        mock_llm_instance = None  # Simulate LLM initialization failure
        mock_llm_class = MagicMock(return_value=mock_llm_instance)
        mock_runtime_instance = None  # Simulate Runtime initialization failure
        mock_tool_instance = MagicMock()

        with patch('local_coding_assistant.core.bootstrap.LLMManager', mock_llm_class), \
                patch('local_coding_assistant.core.bootstrap.RuntimeManager') as mock_runtime_class, \
                patch('local_coding_assistant.core.bootstrap.ToolManager') as mock_tool_class:
            mock_runtime_class.return_value = mock_runtime_instance
            mock_tool_class.return_value = mock_tool_instance

            # Call the function
            ctx = bootstrap()

            # Verify context is still returned
            assert isinstance(ctx, AppContext)

            # Verify None services are handled
            assert ctx.get("llm") is None  # LLM is None due to initialization failure
            assert ctx.get("runtime") is None  # Runtime is None due to initialization failure
            assert ctx.get("tools") == mock_tool_instance  # Tools should still be registered


class TestBootstrapEnvironmentLoading:
    """Test environment loading in bootstrap."""

    @patch("local_coding_assistant.core.bootstrap.EnvManager")
    def test_env_loading_success(self, mock_env_manager_class):
        """Test successful environment loading."""
        mock_env_manager = MagicMock()
        mock_env_manager.load_env_files = MagicMock()
        mock_env_manager_class.return_value = mock_env_manager

        with patch("local_coding_assistant.core.bootstrap._load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.runtime.enable_logging = True
            mock_config.runtime.log_level = "INFO"
            mock_config.llm = MagicMock()
            mock_load_config.return_value = (mock_config, True)
            with patch("local_coding_assistant.core.bootstrap.get_config_manager"):
                with patch("local_coding_assistant.core.bootstrap.get_logger"):
                    with patch("local_coding_assistant.core.bootstrap._setup_logging"):
                        with patch("local_coding_assistant.core.bootstrap.LLMManager"):
                            with patch(
                                "local_coding_assistant.core.bootstrap.RuntimeManager"
                            ):
                                with patch(
                                    "local_coding_assistant.core.bootstrap.ToolManager"
                                ):
                                    with patch("local_coding_assistant.providers.provider_manager.provider_manager"):
                                        bootstrap()

                                        # Verify env manager was created and used
                                        mock_env_manager.load_env_files.assert_called_once()

    @patch("local_coding_assistant.core.bootstrap.EnvManager")
    @patch("builtins.print")  # Mock print function for warnings
    def test_env_loading_failure(self, mock_print, mock_env_manager_class):
        """Test environment loading failure."""
        mock_env_manager = MagicMock()
        mock_env_manager.load_env_files = MagicMock(
            side_effect=Exception("Env load failed")
        )
        mock_env_manager_class.return_value = mock_env_manager

        with patch("local_coding_assistant.core.bootstrap._load_config") as mock_load_config:
            mock_config = MagicMock()
            mock_config.runtime.enable_logging = True
            mock_config.runtime.log_level = "INFO"
            mock_config.llm = MagicMock()
            mock_load_config.return_value = (mock_config, True)
            with patch("local_coding_assistant.core.bootstrap.get_config_manager"):
                with patch("local_coding_assistant.core.bootstrap.get_logger"):
                    with patch("local_coding_assistant.core.bootstrap._setup_logging"):
                        with patch("local_coding_assistant.core.bootstrap.LLMManager"):
                            with patch(
                                "local_coding_assistant.core.bootstrap.RuntimeManager"
                            ):
                                with patch(
                                    "local_coding_assistant.core.bootstrap.ToolManager"
                                ):
                                    with patch("local_coding_assistant.providers.provider_manager.provider_manager"):
                                        # Should not raise exception, just print warning
                                        ctx = bootstrap()

                                        assert isinstance(ctx, AppContext)
                                        mock_print.assert_called_once()
                                        print_calls = [
                                            call.args[0]
                                            for call in mock_print.call_args_list
                                        ]
                                        assert any(
                                            "Failed to load .env files" in call
                                            for call in print_calls
                                        )
