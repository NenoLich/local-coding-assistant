"""
Unit tests for core bootstrap functionality.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import bootstrap


class TestBootstrapInitialization:
    """Test bootstrap initialization."""

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_successful_initialization(
        self,
        mock_setup_logging,
        mock_get_logger,
        mock_get_config_manager,
        mock_load_config,
    ):
        """Test successful bootstrap initialization."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)  # Return tuple (config, is_valid)

        # Mock config manager
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock the global provider_manager import
        with patch(
            "local_coding_assistant.core.bootstrap.provider_manager"
        ) as mock_provider_manager:
            # Mock LLMManager
            with patch(
                "local_coding_assistant.core.bootstrap.LLMManager"
            ) as mock_llm_class:
                mock_llm_instance = MagicMock()
                mock_llm_class.return_value = mock_llm_instance

                # Mock RuntimeManager
                with patch(
                    "local_coding_assistant.core.bootstrap.RuntimeManager"
                ) as mock_runtime_class:
                    mock_runtime_instance = MagicMock()
                    mock_runtime_class.return_value = mock_runtime_instance

                    # Mock ToolManager
                    with patch(
                        "local_coding_assistant.core.bootstrap.ToolManager"
                    ) as mock_tool_class:
                        mock_tool_instance = MagicMock()
                        mock_tool_class.return_value = mock_tool_instance

                        ctx = bootstrap()

                        # Verify context is returned
                        assert isinstance(ctx, AppContext)

                        # Verify services are registered
                        assert ctx.get("llm") == mock_llm_instance
                        assert ctx.get("runtime") == mock_runtime_instance

                        # Verify logging was set up with config-based log level (INFO)
                        mock_setup_logging.assert_called_once_with(mock_config, None)

                        # Verify LLM manager was initialized with config manager and provider manager
                        mock_llm_class.assert_called_once_with(
                            mock_config_manager, mock_provider_manager
                        )

                        # Verify runtime manager was created
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

        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager"
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            with patch(
                "local_coding_assistant.core.bootstrap.RuntimeManager"
            ) as mock_runtime_class:
                mock_runtime_instance = MagicMock()
                mock_runtime_class.return_value = mock_runtime_instance

                with patch(
                    "local_coding_assistant.core.bootstrap.ToolManager"
                ) as mock_tool_class:
                    mock_tool_instance = MagicMock()
                    mock_tool_class.return_value = mock_tool_instance

                    ctx = bootstrap(config_path="/custom/config.yaml")

                    # Verify custom config path was used
                    mock_load_config.assert_called_once_with(
                        "/custom/config.yaml"
                    )

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

        with patch("local_coding_assistant.core.bootstrap.provider_manager"):
            with patch("local_coding_assistant.core.bootstrap.LLMManager"):
                with patch("local_coding_assistant.core.bootstrap.RuntimeManager"):
                    with patch("local_coding_assistant.core.bootstrap.ToolManager"):
                        ctx = bootstrap(log_level=logging.DEBUG)

                        # Verify custom log level was used - use positional args to match actual call signature
                        mock_setup_logging.assert_called_once_with(mock_config, logging.DEBUG)


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

        # Mock LLMManager to raise exception
        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager"
        ) as mock_llm_class:
            mock_llm_class.side_effect = Exception("LLM initialization failed")

            # Mock ToolManager
            with patch(
                "local_coding_assistant.core.bootstrap.ToolManager"
            ) as mock_tool_class:
                mock_tool_instance = MagicMock()
                mock_tool_class.return_value = mock_tool_instance

                with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                    with patch("local_coding_assistant.core.bootstrap.logger") as mock_bootstrap_logger:
                        ctx = bootstrap()

                        # Verify context still created
                        assert isinstance(ctx, AppContext)

                        # Verify LLM manager is None but others are registered
                        assert ctx.get("llm") is None
                        assert ctx.get("tools") == mock_tool_instance
                        assert ctx.get("runtime") is None  # Runtime requires LLM

                        # Verify warning was logged
                        mock_bootstrap_logger.warning.assert_called()
                        assert mock_bootstrap_logger.warning.call_count >= 1
                        warning_calls = [
                            call.args[0] for call in mock_bootstrap_logger.warning.call_args_list
                        ]
                        assert any(
                            "Failed to initialize LLM manager" in call for call in warning_calls
                        )

    @patch("local_coding_assistant.core.bootstrap._load_config")
    @patch("local_coding_assistant.core.bootstrap.get_config_manager")
    @patch("local_coding_assistant.core.bootstrap.get_logger")
    @patch("local_coding_assistant.core.bootstrap._setup_logging")
    def test_bootstrap_tool_manager_failure(
        self,
        mock_setup_logging,
        mock_get_logger,
        mock_get_config_manager,
        mock_load_config,
    ):
        """Test bootstrap when tool manager initialization fails."""
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

        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager"
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            # Mock ToolManager to raise exception
            with patch(
                "local_coding_assistant.core.bootstrap.ToolManager"
            ) as mock_tool_class:
                mock_tool_class.side_effect = Exception("Tool manager failed")

                # Mock SumTool registration to also fail
                with patch("local_coding_assistant.core.bootstrap.SumTool"):
                    with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                        with patch("local_coding_assistant.core.bootstrap.logger") as mock_bootstrap_logger:
                            ctx = bootstrap()

                            # Verify context created but tools is None
                            assert isinstance(ctx, AppContext)
                            assert ctx.get("llm") == mock_llm_instance
                            assert ctx.get("tools") is None
                            assert ctx.get("runtime") is None  # Runtime requires both LLM and tools

                            # Verify warning was logged
                            mock_bootstrap_logger.warning.assert_called()
                            assert mock_bootstrap_logger.warning.call_count >= 1
                            warning_calls = [
                                call.args[0] for call in mock_bootstrap_logger.warning.call_args_list
                            ]
                            assert any(
                                "Failed to initialize tool manager" in call for call in warning_calls
                            )

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

        with patch("local_coding_assistant.core.bootstrap.provider_manager"):
            with patch("local_coding_assistant.core.bootstrap.LLMManager"):
                with patch("local_coding_assistant.core.bootstrap.RuntimeManager"):
                    with patch("local_coding_assistant.core.bootstrap.ToolManager"):
                        ctx = bootstrap()

                        # Verify logging disabled (CRITICAL level) - use positional args to match actual call signature
                        mock_setup_logging.assert_called_once_with(mock_config, 50)

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
        # Simulate config loading failure by returning (None, False)
        mock_load_config.return_value = (None, False)

        mock_config_manager = MagicMock()
        mock_config_manager.global_config = MagicMock()
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Should still create context even if config fails
        with patch("local_coding_assistant.core.bootstrap.provider_manager"):
            ctx = bootstrap()

            assert isinstance(ctx, AppContext)
            # Services might be None or partially initialized depending on failure point


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
        # Mock complete configuration
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

        # Mock all services
        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager"
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            with patch(
                "local_coding_assistant.core.bootstrap.RuntimeManager"
            ) as mock_runtime_class:
                mock_runtime_instance = MagicMock()
                mock_runtime_class.return_value = mock_runtime_instance

                with patch(
                    "local_coding_assistant.core.bootstrap.ToolManager"
                ) as mock_tool_class:
                    mock_tool_instance = MagicMock()
                    mock_tool_class.return_value = mock_tool_instance

                    # Mock SumTool
                    with patch(
                        "local_coding_assistant.core.bootstrap.SumTool"
                    ) as mock_sum_tool:
                        mock_sum_tool_instance = MagicMock()
                        mock_sum_tool.return_value = mock_sum_tool_instance

                        with patch("local_coding_assistant.core.bootstrap.provider_manager") as mock_provider_manager:
                            ctx = bootstrap()

                            # Verify all services initialized and registered
                            assert ctx.get("llm") == mock_llm_instance
                            assert ctx.get("tools") == mock_tool_instance
                            assert ctx.get("runtime") == mock_runtime_instance

                            # Verify tool was registered
                            mock_tool_instance.register_tool.assert_called_once_with(
                                mock_sum_tool_instance
                            )

                            # Verify LLM manager was initialized with config manager and provider manager
                            mock_llm_class.assert_called_once_with(
                                mock_config_manager, mock_provider_manager
                            )

                            # Note: load_global_config is called internally by _load_config
                            # The exact call count depends on the implementation details

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
        mock_config = MagicMock()
        mock_config.runtime.enable_logging = True
        mock_config.runtime.log_level = "INFO"
        mock_config.llm = MagicMock()
        mock_load_config.return_value = (mock_config, True)

        # Config manager has no global config
        mock_config_manager = MagicMock()
        mock_config_manager.global_config = None
        mock_config_manager.load_global_config.return_value = mock_config
        mock_get_config_manager.return_value = mock_config_manager

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager"
        ) as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            with patch(
                "local_coding_assistant.core.bootstrap.RuntimeManager"
            ) as mock_runtime_class:
                mock_runtime_instance = MagicMock()
                mock_runtime_class.return_value = mock_runtime_instance

                with patch(
                    "local_coding_assistant.core.bootstrap.ToolManager"
                ) as mock_tool_class:
                    mock_tool_instance = MagicMock()
                    mock_tool_class.return_value = mock_tool_instance

                    with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                        with patch("local_coding_assistant.core.bootstrap.logger") as mock_bootstrap_logger:
                            ctx = bootstrap()

                            # Should still work and load global config
                            # Note: load_global_config is called internally by _load_config


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

        # Mock services with specific instances for verification
        mock_llm_instance = MagicMock()
        mock_tool_instance = MagicMock()
        mock_runtime_instance = MagicMock()

        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager",
            return_value=mock_llm_instance,
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.RuntimeManager",
                return_value=mock_runtime_instance,
            ):
                with patch(
                    "local_coding_assistant.core.bootstrap.ToolManager",
                    return_value=mock_tool_instance,
                ):
                    with patch("local_coding_assistant.core.bootstrap.SumTool"):
                        with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                            ctx = bootstrap()

                            # Verify exact instances are registered
                            assert ctx.get("llm") is mock_llm_instance
                            assert ctx.get("tools") is mock_tool_instance
                            assert ctx.get("runtime") is mock_runtime_instance

                            # Verify context methods work
                            assert "llm" in ctx
                            assert "tools" in ctx
                            assert "runtime" in ctx
                            assert "nonexistent" not in ctx

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

        # Mock LLM to fail
        with patch(
            "local_coding_assistant.core.bootstrap.LLMManager",
            side_effect=Exception("LLM failed"),
        ):
            with patch(
                "local_coding_assistant.core.bootstrap.ToolManager"
            ) as mock_tool_class:
                mock_tool_instance = MagicMock()
                mock_tool_class.return_value = mock_tool_instance

                with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                    with patch("local_coding_assistant.core.bootstrap.logger") as mock_bootstrap_logger:
                        ctx = bootstrap()

                        # Verify None services are still registered
                        assert ctx.get("llm") is None
                        assert ctx.get("tools") is mock_tool_instance
                        assert ctx.get("runtime") is None

                        # Verify context still works
                        assert "llm" in ctx
                        assert "tools" in ctx
                        assert "runtime" in ctx

                        # Verify warning was logged about missing LLM
                        mock_bootstrap_logger.warning.assert_called()
                        assert mock_bootstrap_logger.warning.call_count >= 1
                        warning_calls = [
                            call.args[0] for call in mock_bootstrap_logger.warning.call_args_list
                        ]
                        assert any(
                            "Skipping runtime manager creation due to missing LLM" in call for call in warning_calls
                        )


class TestBootstrapEnvironmentLoading:
    """Test environment loading in bootstrap."""

    @patch("local_coding_assistant.core.bootstrap.EnvLoader")
    def test_env_loading_success(self, mock_env_loader_class):
        """Test successful environment loading."""
        mock_env_loader = MagicMock()
        mock_env_loader.load_env_files = MagicMock()
        mock_env_loader_class.return_value = mock_env_loader

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
                                    with patch("local_coding_assistant.core.bootstrap.provider_manager"):
                                        bootstrap()

                                        # Verify env loader was created and used
                                        mock_env_loader.load_env_files.assert_called_once()

    @patch("local_coding_assistant.core.bootstrap.EnvLoader")
    @patch("builtins.print")  # Mock print function for warnings
    def test_env_loading_failure(self, mock_print, mock_env_loader_class):
        """Test environment loading failure."""
        mock_env_loader = MagicMock()
        mock_env_loader.load_env_files = MagicMock(
            side_effect=Exception("Env load failed")
        )
        mock_env_loader_class.return_value = mock_env_loader

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
                                    with patch("local_coding_assistant.core.bootstrap.provider_manager"):
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
