# Local Coding Assistant (LOCCA)

An AI-powered coding assistant that runs locally with support for LLMs, tools, and advanced agent patterns including observe-plan-act-reflect loops, LangGraph compatibility, and sophisticated opar-agent implementation.

## Features

- **Local-first architecture** with optional cloud backend support
- **Advanced Agent Patterns** - Observe-Plan-Act-Reflect loop implementation
- **LangGraph Compatibility** - Integration with LangGraph for graph-based orchestration
- **LangGraph Opar-Agent** - Sophisticated agent implementation with advanced reasoning capabilities
- **Streaming LLM Support** - Real-time streaming responses and tool calling
- **Provider System** - Declarative provider configs with automatic fallback and health checks
- **CLI-first interface** with comprehensive commands, including provider management
- **Three-Layer Configuration** - Global defaults, session overrides, and call-level overrides
- **Session Management** - Persistent sessions with context awareness
- **Error Handling** - Robust error handling and logging throughout
- **Tool System** - Extensible tool registry with JSON schema validation
- **Programmatic Tool Calling (PTC)** - Advanced tool calling capabilities with sandboxed execution
- **Sandbox Environment** - Secure, isolated execution environment for untrusted code
- **Execution Statistics** - Comprehensive monitoring and metrics for tool execution
- **Dashboard** - Web-based observability and analysis dashboard for ExecutionFrame monitoring
- **Repository Context Service** - AST-backed repository indexing, repo-map generation, symbol search, and metadata extraction for code-aware prompts

## Architecture Overview

LOCCA implements a sophisticated agent architecture with the following key components:

### Agent Loop
- **Observe-Plan-Act-Reflect Pattern** - Advanced reasoning loop for complex tasks
- **Tool Integration** - Seamless tool calling and result processing
- **Session Awareness** - Maintains context across multiple interactions
- **Error Recovery** - Handles failures gracefully with retry mechanisms

### LangGraph Integration
- **Graph-based Orchestration** - AgentLoop can serve as execution engine in LangGraph workflows
- **Node Compatibility** - Implements LangGraph node interface for seamless integration
- **State Management** - Proper state passing between graph nodes
- **LangGraph Opar-Agent** - Advanced agent implementation using LangGraph for sophisticated reasoning

### Core Systems
- **LLM Service** - Unified interface for local and remote LLMs with streaming support, routing, and fallback
- **Provider Manager & Router** - Discovers providers, merges layered configs, and handles fallback routing
- **Tool Registry** - Dynamic tool loading and management with schema validation
- **Runtime Manager** - Session and context management with persistence
- **Configuration System** - Three-layer precedence (global, session, call) with Pydantic schemas
- **Repository Context Service** - Builds a lightweight code index using AST parsing, metadata extraction, repo-map generation, and symbol search
- **Sandbox Manager** - Secure execution environment with resource constraints and isolation
- **Statistics Manager** - Tracks and reports on tool execution metrics and resource usage
- **Dashboard System** - Web-based observability interface with real-time monitoring and analytics

### Repository Context Service

The repository layer is implemented as a first-class service, not just a prompt helper. It indexes tracked files, extracts project metadata, builds a repo map, and supports symbol search using SQLite + FTS-based lookups.

Key capabilities in the current codebase:

- AST-based parsing for Python, JavaScript/TypeScript, Rust, Go, and other supported languages
- Project metadata extraction from `pyproject.toml`, Cargo manifests, TS config, and workspace config files
- Repository map generation with ranked symbols and language distribution snapshots
- Incremental reindexing and file-change monitoring for git-tracked files
- Persistent, temporary, and test database modes via `StorageManager`
- Search APIs for symbol names, types, and file filtering through `RepositoryContextService.search_symbols()`

This service is wired into runtime/context setup and can be used directly from the codebase or via tool integrations when a repository-aware prompt is needed.

```python
from pathlib import Path
from local_coding_assistant.repository import RepositoryContextService

service = RepositoryContextService(
    target_project_root=Path("/path/to/project"),
    config_manager=config_manager,
)

project_info = service.get_project_info()
repo_map = service.get_repo_map_data()
symbol_matches = service.search_symbols("load_config", limit=10)
```

## Configuration

LOCCA ships with a three-layer configuration system managed by `ConfigManager`:

1. **Global layer** – Defaults merged with YAML (`defaults.yaml`, `providers.default.yaml`, optional custom files) and environment variables.
2. **Session layer** – Runtime overrides applied for the duration of a session.
3. **Call layer** – Per-call overrides supplied by CLI flags or API parameters.

Higher layers override lower layers. The default bootstrap loads `.env`, optional `providers.default.yaml`, then merges user overrides from the home directory (`~/.local-coding-assistant/config/providers.local.yaml`).


### Path Management

The `path_manager` utility provides a unified way to handle file paths throughout the application. In the current implementation, the valid runtime aliases include:

- `@project`: Project root directory
- `@config`: Configuration directory (from `LOCCA_CONFIG_DIR`)
- `@data`: Data directory (from `LOCCA_DATA_DIR`)
- `@cache`: Cache directory (from `LOCCA_CACHE_DIR`)
- `@log`: Log directory (from `LOCCA_LOG_DIR`)
- `@module`: Module or package root
- `@tools`: Tools directory
- `@templates`: Template directory

Example usage:
```python
from local_coding_assistant.config.path_manager import PathManager

path_manager = PathManager()
config_path = path_manager.resolve("@config/providers.default.yaml")
log_file = path_manager.join("@log", "app.log")
```

Note: `@root` and `@logs` are not real aliases in the current path manager; use `@project` and `@log` instead.

### .env Files

Environment variables are loaded from multiple `.env` files in the following order (with later files overriding earlier ones):

1. `.env` - Base configuration (always loaded first)
2. `.env.${LOCCA_ENV}` - Environment-specific settings (e.g., `.env.development`)
3. `.env.local` - Local environment overrides (gitignored)

Example structure:
```
.env                    # Base configuration (required)
.env.development        # Development-specific overrides
.env.test              # Test-specific overrides
.env.production        # Production-specific overrides
.env.local # Local overrides (gitignored)
```

Environment variables can reference each other using `${VARIABLE_NAME}` syntax within the same file or in files loaded later in the chain.

### YAML Configuration Files

Configuration is loaded from multiple YAML files with the following precedence (higher overrides lower):

1. Core configuration:
   - `config/defaults.yaml` - Default configuration (base settings)
   - `config/${LOCCA_ENV}.yaml` - Environment-specific overrides
   - `config/local.yaml` - Local overrides (gitignored)

2. Provider configuration:
   - `config/providers.default.yaml` - Default provider configurations
   - `config/providers.${LOCCA_ENV}.yaml` - Environment-specific providers
   - `config/providers.local.yaml` - Local provider overrides (gitignored)

3. Tool configuration:
   - `config/tools.default.yaml` - Default tool configurations
   - `config/tools.${LOCCA_ENV}.yaml` - Environment-specific tool configurations
   - `config/tools.local.yaml` - Local tool overrides (gitignored)

All paths in configuration files can use the `@` aliases (e.g., `@data/models`) which will be resolved by the `path_manager`.

## Development Setup

1. Ensure you have Python 3.12+ installed
2. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | sh
   ```
3. Clone the repository and navigate to the project directory
4. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv_linux/bin/activate  # On Windows: .venv_linux\Scripts\activate
   ```
5. Install development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```
6. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

```
.
# Root directory
├── .github/                  # GitHub Actions workflows
├── config/                   # Configuration files
│   ├── modules/              # External tool modules
│   ├── auto_model_policies.yaml  # Auto-model configuration
│   ├── defaults.yaml         # Default configuration
│   ├── providers.default.yaml # Default provider configurations
│   ├── providers.local.yaml  # Local provider overrides (gitignored)
│   ├── tools.default.yaml    # Default tool configurations
│   └── tools.local.yaml      # Local tool overrides (gitignored)
├── data/                     # Persistent data storage (created at runtime)
├── logs/                     # Application logs (created at runtime)
├── src/                      # Source code
│   └── local_coding_assistant/
│       ├── agent/            # Agent loop and LLM management
│       │   ├── frame_agent.py # Frame-based agent implementation
│       │   ├── langgraph_agent.py # LangGraph opar-agent
│       │   ├── llm/          # LLM service components
│       │   │   ├── service.py # Main LLM service interface
│       │   │   ├── models.py # LLM models and schemas
│       │   │   ├── pipeline.py # Request processing pipeline
│       │   │   ├── routing.py # Provider routing logic
│       │   │   ├── fallback.py # Fallback mechanisms
│       │   │   └── telemetry.py # LLM telemetry and monitoring
│       │   └── agent_loop.py # Observe-Plan-Act-Reflect implementation
│       ├── cli/              # CLI commands and interface
│       │   ├── rendering/    # CLI rendering components
│       │   └── commands/     # Command implementations
│       ├── config/           # Configuration management
│       │   ├── builder.py    # Configuration builder
│       │   ├── cache.py      # Configuration caching
│       │   ├── dependencies.py # Configuration dependencies
│       │   ├── field.py      # Configuration fields
│       │   ├── validation_engine.py # Configuration validation
│       │   ├── config_manager.py # Multi-source configuration management
│       │   ├── providers.default.yaml # Default provider configurations
│       │   ├── providers.local.yaml  # Local provider overrides (gitignored)
│       │   ├── schemas.py    # Pydantic models for configuration
│       │   └── path_manager.py # Path resolution utility
│       ├── core/             # Core infrastructure
│       ├── providers/        # LLM provider implementations
│       ├── runtime/          # Runtime and session management
│       │   ├── execution_types.py # Execution type definitions
│       │   ├── executor.py   # Execution engine
│       │   ├── handlers/     # Execution handlers
│       │   ├── runtime_manager.py # Session orchestration
│       │   └── dashboard_integration.py # Dashboard event integration
│       ├── repository/       # Repository-aware context layer
│       │   ├── ast_parser/   # Tree-sitter symbol and relationship extraction
│       │   ├── metadata/     # Project metadata extraction logic
│       │   ├── call_graph.py # Repo Map ranking and call-graph utilities
│       │   ├── database.py   # SQLite-backed repository index
│       │   ├── repo_map.py   # Repo map builder and formatter
│       │   ├── service.py    # Main RepositoryContextService orchestrator
│       │   ├── storage.py    # Persistent/temp/test storage lifecycle
│       │   ├── symbol_search.py # Full-text symbol lookup
│       │   └── file_monitoring.py # Reindex callbacks and file-change detection
│       ├── sandbox/          # Secure code execution environment
│       │   ├── guest/        # Guest-side sandbox components
│       │   │   ├── agent.py  # Guest agent implementation
│       │   │   ├── resource_tracker.py # Resource usage monitoring
│       │   │   ├── session.py # Session management
│       │   │   └── tools_api.py # Tool API for guest environment
│       │   ├── base.py       # Base sandbox interfaces
│       │   ├── docker_sandbox.py # Docker-based sandbox implementation
│       │   ├── manager.py    # Sandbox lifecycle management
│       │   ├── sandbox_types.py # Type definitions for sandbox
│       │   └── security.py   # Security policies and validations
│       ├── tools/            # Tool system
│       │   ├── base/         # Base tool classes and schemas
│       │   └── tool_registry.py # Tool registration and management
│       ├── dashboard/         # Web dashboard for observability
│       │   ├── app.py        # FastAPI application
│       │   ├── event_collector.py # Event aggregation and storage
│       │   ├── models.py     # Pydantic models for dashboard data
│       │   ├── run_dashboard.py # Dashboard runner script
│       │   ├── routes/       # API and web routes
│       │   │   ├── api.py    # REST API endpoints
│       │   │   ├── main.py   # Main web page routes
│       │   │   └── websocket.py # WebSocket handlers
│       │   └── templates/    # Jinja2 HTML templates
│       └── utils/            # Utility functions
├── tests/                    # Test suite
│   ├── e2e/                  # End-to-end tests
│   │   ├── conftest.py       # Test fixtures
│   │   ├── test_cli_commands.py
│   │   └── test_provider_commands.py
│   ├── integration/          # Integration tests
│   └── unit/                 # Unit tests
│       └── providers/        # Provider-specific unit tests
├── .env                      # Base environment variables
├── .env.development          # Development environment
├── .env.test                 # Test environment
├── .env.production           # Production environment
├── .env.local                # Local overrides (gitignored)
├── .gitignore
├── .pre-commit-config.yaml   # Pre-commit hooks
├── .python-version           # Python version file
├── llms.txt                  # LLM documentation
├── main.py                   # Main entry point
├── pyproject.toml            # Project metadata and dependencies
└── uv.lock                   # Dependency lock file

```

## CLI Usage

Use the installed console script `locca`.

### Basic Commands

```bash
# Run a single query with the agent loop
uv run locca run query "What is the weather like in New York?" --log-level DEBUG -v

# Run with specific model
uv run locca run query "Calculate 2 + 2" --model gpt-4 --temperature 0.7

# Enable streaming responses
uv run locca run query "Explain quantum computing" --streaming

# Route through a specific provider/model for the request only
uv run locca run query "Summarize RFC 9110" --provider openrouter --model "openrouter:google/gemini-pro"
```

### Tool Management

```bash
# List available tools (text and JSON)
uv run locca tool list
uv run locca tool list --json

# Add a new tool
uv run locca tool add --name calculator --path ./my_tools/calc.py

# Remove a tool
uv run locca tool remove calculator

# Run a tool directly
uv run locca tool run calculator --input '{"query": "example"}'

# Validate tool configuration
uv run locca tool validate

# Reload tools (after config changes)
uv run locca tool reload
```

Tools can also be registered by adding them to `config/tools.local.yaml`:

```yaml
tools:
  - name: calculator
    path: ./my_tools/calc.py  # or module: my_tools.calculator
    enabled: true
    config:
      precision: 2
```

### Configuration

```bash
# Get/set config (env-backed, prefix LOCCA_)
uv run locca config set LLM__MODEL_NAME gpt-4
uv run locca config get LLM__MODEL_NAME

# Show current configuration
uv run locca config show

# Manage provider configurations
uv run locca provider list
uv run locca provider add openrouter --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --models qwen/qwen3-coder:free qwen/qwen3-72b-preview --dev
uv run locca provider validate
uv run locca provider remove openrouter --dev
```

### Development Server

```bash
# Start the development server
uv run locca serve start --host 0.0.0.0 --port 8080 --reload
```

### Dashboard

```bash
# Start the dashboard server
uv run locca dashboard serve --host 127.0.0.1 --port 8080

# Start dashboard in background (detached mode)
uv run locca dashboard serve --detach

# Stop dashboard server
uv run locca dashboard stop --port 8080

# Stop all dashboard servers
uv run locca dashboard stop --all
```

The dashboard provides:
- **Real-time monitoring** of active execution sessions
- **Historical analysis** of runs and frames with detailed metrics
- **Analytics dashboard** with charts and performance insights
- **Interactive filtering** and data export capabilities
- **WebSocket-based** live updates during agent execution

Access the dashboard at `http://127.0.0.1:8080` when running.

## Advanced Features

### Agent Patterns

LOCCA implements sophisticated agent patterns with Frame Agent as the main implementation:

- **Frame-based Execution** - Flexible execution patterns using configurable frames
- **Tool Integration** - Seamless integration with the tool system for extended capabilities
- **Session Persistence** - Maintains context across multiple queries
- **Error Recovery** - Graceful handling of failures with retry logic

**Note**: Legacy agent implementations (AgentLoop and LangGraph Agent) are deprecated. Use Frame Agent for all new development.

### LangGraph Integration (Deprecated)

!!! warning "Deprecated"
    LangGraph integration is deprecated. The LangGraph-based agent implementations will be removed in a future version.

Previous LangGraph features:
- Graph-based orchestration (deprecated)
- Node-based execution (deprecated)
- Complex state management (superseded by Frame Agent)

### Provider System

- **Layered Configuration** – Built-in defaults merged with user YAML and environment variables.
- **Automatic Reloads** – CLI commands trigger reloads through `ProviderManager.reload()`.
- **Health Monitoring** – Providers expose health checks cached by `LLMManager`.
- **Fallback Routing** – `ProviderRouter` promotes healthy providers following agent policies.

### Streaming Support

- **Real-time Responses** - Streaming LLM responses for better UX
- **Tool Call Streaming** - Stream tool calls as they're generated
- **Progress Updates** - Live progress reporting during long operations

## Testing

Run tests with uv/taskipy or pytest:

```bash
# Using taskipy (pyproject [tool.taskipy.tasks])
uv run task test-unit
uv run task test-integration
uv run task test-e2e
uv run task test-all

# Or directly with pytest
uv run pytest -v

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/e2e/ -v
```

- **Frame Agent** - Main agent implementation using flexible execution frames
- **LLM Service** - Refactored LLM management with modular components (service, routing, fallback, telemetry)
- **Provider Module** - Declarative provider configs, layered reloads, and CLI management
- **Config Manager v2** - Three-layer hierarchy with validation and runtime overrides
- **Configuration Builder** - Advanced configuration building, caching, and validation engine
- **CLI Provider Commands** - `locca provider add|list|remove|validate|reload`
- **CLI Rendering** - Enhanced CLI output with dedicated rendering components
- **Integration Tests** - Expanded coverage for config merging and provider routing
- **Streaming LLM Responses** - Real-time streaming for better user experience
- **Enhanced Runtime** - New execution engine with handlers and execution types
- **Session & Context Awareness** - Persistent sessions with context management
- **Enhanced Tool System** - JSON schema validation and better error handling
- **Sandbox Tool Processing** - Advanced tool call processing in sandbox environment
- **Centralized Logging** - Structured logging throughout the application
- **Error Handling** - Robust error handling with safe entry points

**Note**: Legacy agent implementations (AgentLoop, LangGraph Agent) are deprecated in favor of Frame Agent.

This project continues to evolve with a focus on reliability, extensibility, and advanced AI patterns.