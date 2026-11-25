![Project Screenshot](LoCCa_stone_letters_cosmic_digits_202607211135.jpeg)
# Local Coding Assistant (LOCCA)

An AI-powered coding assistant that runs locally with support for LLMs, tools, and advanced agent patterns including observe-plan-act-reflect loops, Frame Agent implementation, and sophisticated runtime architecture.

## Features

- **Local-first architecture** with optional cloud backend support
- **Frame Agent** - Main agent implementation using flexible execution frames with continuation policy
- **Streaming LLM Support** - Real-time streaming responses and tool calling via Server-Sent Events
- **Provider System** - Declarative provider configs with automatic fallback and health checks
- **CLI-first interface** with comprehensive commands, including provider management
- **Three-Layer Configuration** - Global defaults, session overrides, and call-level overrides
- **Session Management** - Persistent sessions with context awareness, resume, and branching
- **Error Handling** - Robust error handling and logging throughout
- **Tool System** - Extensible tool registry with JSON schema validation and middleware pipeline
- **Sandbox Environment** - Secure, isolated execution environment for untrusted code (Docker)
- **Execution Statistics** - Comprehensive monitoring and metrics for tool execution
- **Dashboard** - Web-based observability and analysis dashboard with REST API and WebSocket
- **Repository Context Service** - AST-backed repository indexing, repo-map generation, symbol search, and metadata extraction for code-aware prompts
- **Durable Execution Store** - SQLite-backed event store for sessions, turns, frames, and messages

## Architecture Overview

LOCCA implements a sophisticated agent architecture with the following key components:

### Frame Agent (Main Agent)
- **Frame-based Execution** - Flexible execution patterns using configurable frames with continuation policy
- **Tool Integration** - Seamless integration with the tool system for extended capabilities
- **Session Persistence** - Maintains context across multiple interactions with resume and branching
- **Error Recovery** - Graceful handling of failures with retry mechanisms

### Core Systems
- **LLM Service** - Unified interface for local and remote LLMs with streaming support, routing, and fallback
- **Provider Manager & Router** - Discovers providers, merges layered configs, and handles fallback routing
- **Runtime Manager** - Orchestrates end-to-end query handling: sessions, turns, frames, and context
- **Event Pipeline** - Event-driven architecture with subscribers for persistence (StoreSubscriber), dashboard (DashboardEventSubscriber), telemetry (OTelSubscriber), and transcript projection (TranscriptSubscriber)
- **Tool Gateway** - Central execution entry point with middleware pipeline: Exposure → Validation → Authorization → Workspace Admission → Scheduling → Execution → Normalization
- **Configuration System** - Three-layer precedence (global, session, call) with Pydantic schemas
- **Repository Context Service** - Builds a lightweight code index using AST parsing, metadata extraction, repo-map generation, and symbol search
- **Sandbox Manager** - Secure execution environment with resource constraints and isolation
- **Dashboard System** - Web-based observability interface with REST API and WebSocket for real-time monitoring
- **Execution Store** - SQLite-backed durable storage for sessions, turns, frames, messages, and events

### Repository Context Service

The repository layer is implemented as a first-class service, not just a prompt helper. It indexes tracked files, extracts project metadata, builds a repo map, and supports symbol search using SQLite + FTS-based lookups.

Key capabilities:
- AST-based parsing for Python, JavaScript/TypeScript, Rust, Go, and other supported languages
- Project metadata extraction from `pyproject.toml`, Cargo manifests, TS config, and workspace config files
- Repository map generation with ranked symbols and language distribution snapshots
- Incremental reindexing and file-change monitoring for git-tracked files
- Persistent, temporary, and test database modes via `StorageManager`
- Search APIs for symbol names, types, and file filtering through `RepositoryContextService.search_symbols()`

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

See [Configuration Files](config/README.md) for detailed documentation of all configuration sections including `tools_platform`, `execution_store`, `telemetry`, `continuation_policy`, and the new `runtime` settings.

### Path Management

The `path_manager` utility provides a unified way to handle file paths throughout the application. Valid runtime aliases include:

- `@project`: Project root directory
- `@config`: Configuration directory (from `LOCCA_CONFIG_DIR`)
- `@data`: Data directory (from `LOCCA_DATA_DIR`)
- `@cache`: Cache directory (from `LOCCA_CACHE_DIR`)
- `@log`: Log directory (from `LOCCA_LOG_DIR`)
- `@module`: Module or package root
- `@tools`: Tools directory
- `@templates`: Template directory
- `@target`: Target project root (user project being worked on)

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
.env.local             # Local overrides (gitignored)
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
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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
│   ├── auto_model_policies.yaml  # Auto-model configuration
│   ├── defaults.yaml         # Default configuration
│   ├── providers.default.yaml # Default provider configurations
│   └── providers.local.yaml  # Local provider overrides (gitignored)
├── data/                     # Persistent data storage (created at runtime)
├── logs/                     # Application logs (created at runtime)
├── src/                      # Source code
│   └── local_coding_assistant/
│       ├── agent/            # Agent and LLM management
│       │   ├── frame_agent.py # Frame-based agent implementation
│       │   ├── langgraph_agent.py # Deprecated - do not extend
│       │   ├── agent_loop.py # Deprecated - do not extend
│       │   └── llm/          # LLM service components
│       │       ├── service.py # Main LLM service interface
│       │       ├── models.py # LLM models and schemas
│       │       ├── pipeline.py # Request processing pipeline
│       │       ├── routing.py # Provider routing logic
│       │       ├── fallback.py # Fallback mechanisms
│       │       └── telemetry.py # LLM telemetry and monitoring
│       ├── cli/              # CLI commands and interface
│       │   ├── rendering/    # CLI rendering components
│       │   │   ├── frame_renderer.py
│       │   │   ├── plain.py
│       │   │   ├── json_renderer.py
│       │   │   ├── registry.py
│       │   │   └── summary.py
│       │   └── commands/     # Command implementations
│       │       ├── run.py    # Single query execution
│       │       ├── chat.py   # Interactive chat session
│       │       ├── sessions.py # Session management
│       │       ├── replay.py # Session/turn replay
│       │       ├── serve.py  # Runtime server
│       │       ├── tool.py   # Tool management
│       │       ├── config.py # Configuration management
│       │       ├── provider.py # Provider management
│       │       ├── sandbox.py # Sandbox management
│       │       └── dashboard.py # Dashboard commands
│       ├── config/           # Configuration management
│       │   ├── config_manager.py # Multi-source configuration management
│       │   ├── schemas.py    # Pydantic models for configuration
│       │   ├── path_manager.py # Path resolution utility
│       │   ├── env_manager.py # Environment management
│       │   └── field.py      # Configuration fields
│       ├── core/             # Core infrastructure
│       │   ├── bootstrap.py  # Application bootstrap
│       │   ├── error_handler.py # Safe entrypoints
│       │   ├── exceptions.py # Custom exceptions
│       │   ├── protocols.py  # Protocol interfaces
│       │   ├── dependencies.py # Dependency container
│       │   └── app_context.py # Application context
│       ├── providers/        # LLM provider implementations
│       ├── runtime/          # Runtime and session management
│       │   ├── manager.py           # RuntimeManager - orchestrates sessions/turns
│       │   ├── types.py             # Core type definitions
│       │   ├── options.py           # RunOptions for CLI/frontend
│       │   ├── session/             # Session and turn management
│       │   │   ├── session_manager.py
│       │   │   ├── turn_manager.py
│       │   │   ├── session.py
│       │   │   └── turn.py
│       │   ├── context/             # Context resolution
│       │   │   ├── manager.py
│       │   │   ├── provider.py
│       │   │   ├── resolution.py
│       │   │   └── profile.py
│       │   ├── engine/              # Frame execution engine
│       │   │   ├── engine.py        # RuntimeEngine - state machine
│       │   │   ├── frame_agent.py   # FrameAgent - continuation loop
│       │   │   ├── frame.py         # RuntimeFrame and actions
│       │   │   ├── policy.py        # ContinuationPolicy
│       │   │   ├── continuation.py  # FrameContinuation directives
│       │   │   ├── config.py        # Engine limits config
│       │   │   ├── approval.py      # Approval handling
│       │   │   └── subscribers/     # Event subscribers
│       │   ├── events/              # Event pipeline and contracts
│       │   │   ├── pipeline.py
│       │   │   ├── contracts.py
│       │   │   ├── projector.py
│       │   │   └── reporting.py
│       │   ├── resume/              # Resume/branch logic
│       │   │   ├── planner.py
│       │   │   ├── validator.py
│       │   │   └── exceptions.py
│       │   └── integrations/        # Dashboard integration
│       ├── repository/       # Repository-aware context layer
│       │   ├── ast_parser/   # Tree-sitter symbol extraction
│       │   ├── metadata/     # Project metadata extraction
│       │   ├── call_graph.py # Repo Map ranking
│       │   ├── database.py   # SQLite-backed repository index
│       │   ├── repo_map.py   # Repo map builder
│       │   ├── service.py    # RepositoryContextService
│       │   ├── storage.py    # Persistent/temp/test storage
│       │   ├── symbol_search.py # Full-text symbol lookup
│       │   └── file_monitoring.py # Reindex callbacks
│       ├── sandbox/          # Secure code execution environment
│       │   ├── guest/        # Guest-side sandbox components
│       │   │   ├── agent.py
│       │   │   ├── resource_tracker.py
│       │   │   ├── session.py
│       │   │   └── tools_api.py
│       │   ├── docker_sandbox.py # Docker-based sandbox
│       │   ├── manager.py    # Sandbox lifecycle management
│       │   ├── sandbox_types.py # Type definitions
│       │   └── security.py   # Security policies
│       ├── tools_platform/   # New tool execution platform
│       │   ├── gateway.py    # ToolGateway - central execution
│       │   ├── registry.py   # ToolRegistry - definitions
│       │   ├── contracts.py  # Tool contracts and types
│       │   ├── definitions.py # Tool definitions
│       │   ├── exposure_manager.py # Tool exposure filtering
│       │   ├── approval.py   # Approval service
│       │   ├── scheduler.py  # Scheduler for parallel execution
│       │   ├── workspace/    # Workspace management
│       │   │   ├── manager.py
│       │   │   ├── content_store.py
│       │   │   ├── snapshots.py
│       │   │   ├── transactions.py
│       │   │   ├── conflicts.py
│       │   │   └── models.py
│       │   ├── backends/     # Execution backends
│       │   │   ├── base.py
│       │   │   ├── host.py
│       │   │   └── docker.py # Not yet implemented
│       │   ├── middleware/   # Middleware pipeline
│       │   │   ├── prepare/
│       │   │   │   ├── exposure.py
│       │   │   │   ├── validation.py
│       │   │   │   ├── authorization.py
│       │   │   │   └── workspace_admission.py
│       │   │   ├── execute/
│       │   │   │   ├── scheduling.py
│       │   │   │   └── execution.py
│       │   │   └── finalize/
│       │   │       └── normalization.py
│       │   └── tools/        # Built-in tool implementations
│       │       ├── read_file.py
│       │       ├── write_file.py
│       │       ├── replace_exact.py
│       │       ├── delete_file.py
│       │       ├── search_text.py
│       │       ├── search_symbols.py
│       │       ├── match_paths.py
│       │       ├── get_symbol.py
│       │       ├── run_command.py
│       │       ├── kill_process.py
│       │       └── check_process_status.py
│       ├── dashboard/         # Web dashboard for observability
│       │   ├── app.py        # FastAPI application
│       │   ├── run_dashboard.py # Dashboard runner
│       │   ├── config.py     # Dashboard configuration
│       │   ├── api/          # REST API endpoints
│       │   │   ├── sessions.py
│       │   │   ├── turns.py
│       │   │   ├── frames.py
│       │   │   ├── messages.py
│       │   │   ├── events.py
│       │   │   ├── live.py
│       │   │   ├── debug.py
│       │   │   └── deps.py
│       │   ├── routes/       # Web routes
│       │   │   ├── web.py
│       │   │   └── websocket.py
│       │   └── services/     # Business logic services
│       │       ├── session_service.py
│       │       ├── turn_service.py
│       │       ├── frame_service.py
│       │       ├── message_service.py
│       │       ├── event_service.py
│       │       ├── event_preview.py
│       │       ├── live_monitor.py
│       │       ├── snapshot_context_resolution.py
│       │       └── debug_service.py
│       └── utils/            # Utility functions
├── tests/                    # Test suite
│   ├── e2e/                  # End-to-end tests
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
# Run a single query with streaming runtime events
uv run locca run query "What is the weather like in New York?" --log-level DEBUG -v

# Run with specific model
uv run locca run query "Calculate 2 + 2" --model gpt-4 --temperature 0.7

# Route through a specific provider/model for the request only
uv run locca run query "Summarize RFC 9110" --provider openrouter --model "openrouter:google/gemini-pro"

# Continue an existing durable session (creates a new turn)
uv run locca run query "Continue the previous task" --session-id <session-id>

# Auto-approve all tool calls
uv run locca run query "Run the tests" --auto-approve
```

### Interactive Chat

```bash
# Start an interactive chat session (one turn per input)
uv run locca chat start

# Continue an existing session
uv run locca chat start --session-id <session-id>

# With model override
uv run locca chat start --model gpt-4 --verbose
```

### Session Management

```bash
# List recent sessions from durable store
uv run locca sessions list --limit 20

# List only running (interrupted) sessions
uv run locca sessions list --status running

# Resume a session where it stopped (crash recovery - no new message)
uv run locca sessions resume <session-id>

# Resume with a new message (creates new turn in same session)
uv run locca sessions resume <session-id> "Continue with this change"

# Branch from a past turn (creates new session forked at that turn)
uv run locca sessions resume <session-id> "New direction" --index 2
```

### Session Replay (Read-only)

```bash
# Replay a full session from durable storage
uv run locca replay show <session-id>

# Replay a specific turn
uv run locca replay show <session-id> --turn <turn-id>

# Replay a specific frame with event timeline
uv run locca replay show <session-id> --frame <frame-id> --events

# Output as JSON
uv run locca replay show <session-id> --format json
```

### Tool Management

```bash
# List available tools (exposed to LLM)
uv run locca tool list

# List all registered tools
uv run locca tool list --available-only false

# Execute a tool directly
uv run locca tool run read_file path=src/main.py

# Execute a tool with JSON input
uv run locca tool run search_text '{"query": "TODO", "max_results": 10}'

# Execute a tool in sandbox (when sandbox enabled)
uv run locca tool run --sandbox run_command command="echo hello"
```

### Configuration

```bash
# Get/set config (env-backed, prefix LOCCA_)
uv run locca config set LLM__TEMPERATURE 0.5
uv run locca config get LLM__TEMPERATURE

# Show current configuration
uv run locca config show
```

### Provider Management

```bash
# List available providers
uv run locca provider list

# List models for a specific provider
uv run locca provider list --provider openrouter

# Add a new provider
uv run locca provider add openrouter --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --models "openrouter:google/gemini-pro" "openrouter:anthropic/claude-3.5-sonnet"

# Validate provider configuration
uv run locca provider validate

# Remove a provider
uv run locca provider remove openrouter

# Reload providers from all sources
uv run locca provider reload
```

### Development Server

```bash
# Start the runtime server for out-of-process approvals and live monitoring
uv run locca serve start --host 127.0.0.1 --port 8001
```

The server provides:
- REST API for session management, approval responses, and interrupt
- Server-Sent Events (SSE) for streaming runtime events
- WebSocket for real-time event streaming to dashboard

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

### Frame Agent

LOCCA implements the Frame Agent as the main agent implementation:

- **Frame-based Execution** - Flexible execution patterns using configurable frames
- **Continuation Policy** - Configurable frame loop with max frames, retries, compaction, and approval handling
- **Tool Integration** - Seamless integration with the tool system for extended capabilities
- **Session Persistence** - Maintains context across multiple queries with resume and branching
- **Error Recovery** - Graceful handling of failures with retry mechanisms

### Provider System

- **Layered Configuration** – Built-in defaults merged with user YAML and environment variables.
- **Automatic Reloads** – CLI commands trigger reloads through `ProviderManager.reload()`.
- **Health Monitoring** – Providers expose health checks cached by `LLMService`.
- **Fallback Routing** – `ProviderRouter` promotes healthy providers following agent policies.

### Streaming Support

- **Real-time Responses** - Streaming LLM responses via Server-Sent Events
- **Tool Call Streaming** - Stream tool calls as they're generated
- **Progress Updates** - Live progress reporting during long operations

### Tools Platform

The new tools platform provides a unified execution layer:

- **Middleware Pipeline** - Exposure → Validation → Authorization → Workspace Admission → Scheduling → Execution → Normalization
- **Backends** - Host backend (local process execution); Docker backend not yet implemented
- **Workspace Management** - Snapshots, baselines, checkpoints, and conflict detection
- **Approval System** - Policy-driven approval with session/project/global overrides
- **Scheduler** - Parallel execution with resource locking and conflict resolution

**Note**: Currently only the host backend and classic tool calling mode are implemented. PTC (Programmatic Tool Calling), sandbox mode, and Docker backend are not yet implemented.

### Durable Execution Store

SQLite-backed event store providing:
- Session, turn, frame, and message persistence
- Event sourcing for complete execution history
- Configurable retention policies per table type
- Read-only replay API for debugging and analysis

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

# Run a single test file
uv run pytest tests/unit/<pkg>/test_x.py -vx
```

**Note**: Plain `pytest` (no args) collects `tests/` AND `benchmark/` (per `testpaths` in pyproject), which is slow. Always pass an explicit path.

## Linting and Type Checking

```bash
# Lint with auto-fix
ruff check --fix .

# Format code
ruff format .

# Type check (excludes tests/ and sandbox/guest/)
pyrefly check src/
```

Pre-commit hooks (`.pre-commit-config.yaml`): lint/format/pyrefly + unit tests run on commit, integration + e2e on push. Install with `pre-commit install`.

## Mock Model for Testing

Use the `mock-model` with `mock` driver to test without real LLM calls. Look in `data/mock_responses` to setup mocked responses for mock-model.

---

**Note**: Legacy agent implementations (`agent_loop.py`, `langgraph_agent.py`) are deprecated in favor of Frame Agent. Do not extend them.
