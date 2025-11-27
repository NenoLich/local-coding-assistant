"""Unit tests for bootstrap initialization."""

from unittest.mock import Mock, patch

import pytest

from local_coding_assistant.core.app_context import AppContext
from local_coding_assistant.core.bootstrap import bootstrap
from local_coding_assistant.core.dependencies import AppDependencies


class TestBootstrapInitialization:
    """Test bootstrap initialization process."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock config manager."""
        mock = Mock()
        mock.load_global_config.return_value = {"logging": {"level": "INFO"}}
        mock.get_tools.return_value = {}
        mock.global_config = {}
        mock.session_overrides = {}
        return mock
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        return Mock()
    
    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager."""
        return Mock()
    
    @pytest.fixture
    def mock_runtime_manager(self):
        """Create a mock runtime manager."""
        mock = Mock()
        mock.orchestrate.return_value = "Test response"
        return mock
    
    @patch("local_coding_assistant.core.bootstrap._setup_environment")
    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    @patch("local_coding_assistant.core.bootstrap._initialize_llm_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_tool_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_runtime_manager")
    def test_bootstrap_initialization(
        self,
        mock_init_runtime,
        mock_init_tools,
        mock_init_llm,
        mock_init_config,
        mock_setup_env,
        mock_config_manager,
        mock_llm_manager,
        mock_tool_manager,
        mock_runtime_manager,
    ):
        """Test successful bootstrap initialization."""
        # Setup mocks
        mock_init_config.return_value = ({"logging": {"level": "INFO"}}, mock_config_manager)
        mock_init_llm.return_value = mock_llm_manager
        mock_init_tools.return_value = mock_tool_manager
        mock_init_runtime.return_value = mock_runtime_manager

        # Call bootstrap
        ctx = bootstrap()

        # Assertions
        assert isinstance(ctx, AppContext)
        mock_setup_env.assert_called_once()
        
        # Verify component initialization
        mock_init_llm.assert_called_once_with(mock_config_manager, {"logging": {"level": "INFO"}})
        mock_init_tools.assert_called_once_with(mock_config_manager)
        mock_init_runtime.assert_called_once_with(
            config_manager=mock_config_manager,
            llm_manager=mock_llm_manager,
            tool_manager=mock_tool_manager,
        )
        
        # Verify _initialize_config was called with env_manager
        mock_init_config.assert_called_once()
        # Check that env_manager is passed as the third positional argument
        assert len(mock_init_config.call_args[0]) == 3  # config_path, config_manager, env_manager
        assert mock_init_config.call_args[0][0] is None  # config_path
        assert mock_init_config.call_args[0][1] is None  # config_manager
        assert mock_init_config.call_args[0][2] is not None  # env_manager
        
        # Verify context setup
        assert ctx.get("llm") == mock_llm_manager
        assert ctx.get("tools") == mock_tool_manager
        assert ctx.get("runtime") == mock_runtime_manager
        
        # Verify dependencies
        assert hasattr(ctx, 'deps')
        assert isinstance(ctx.deps, AppDependencies)
        assert ctx.deps.config_manager == mock_config_manager
        assert ctx.deps.llm_manager == mock_llm_manager
        assert ctx.deps.tool_manager == mock_tool_manager
        assert ctx.deps.runtime_manager == mock_runtime_manager
        assert ctx.deps.is_initialized()
    
    @patch("local_coding_assistant.core.bootstrap._setup_environment")
    @patch("local_coding_assistant.core.bootstrap._initialize_config")
    @patch("local_coding_assistant.core.bootstrap._initialize_llm_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_tool_manager")
    @patch("local_coding_assistant.core.bootstrap._initialize_runtime_manager")
    def test_bootstrap_with_custom_config_path(
        self,
        mock_init_runtime,
        mock_init_tools,
        mock_init_llm,
        mock_init_config,
        mock_setup_env,
        mock_config_manager,
    ):
        """Test bootstrap with custom config path."""
        # Setup mocks
        mock_init_config.return_value = ({"logging": {"level": "DEBUG"}}, mock_config_manager)
        mock_init_llm.return_value = Mock()
        mock_init_tools.return_value = Mock()
        mock_init_runtime.return_value = Mock()

        # Call bootstrap with custom config path
        config_path = "/path/to/custom/config.yaml"
        ctx = bootstrap(config_path=config_path)

        # Verify config was loaded from custom path with env_manager
        mock_init_config.assert_called_once()
        # Check that env_manager is passed as the third positional argument
        assert len(mock_init_config.call_args[0]) == 3  # config_path, config_manager, env_manager
        assert mock_init_config.call_args[0][0] == config_path  # config_path
        assert mock_init_config.call_args[0][1] is None  # config_manager
        assert mock_init_config.call_args[0][2] is not None  # env_manager
        assert isinstance(ctx, AppContext)
