# Local Coding Assistant (LOCCA)

An AI-powered coding assistant that runs locally with support for LLMs, tools, and advanced agent patterns including observe-plan-act-reflect loops and LangGraph compatibility.

## Features

- **Local-first architecture** with optional cloud backend support
- **Advanced Agent Patterns** - Observe-Plan-Act-Reflect loop implementation
- **LangGraph Compatibility** - Integration with LangGraph for graph-based orchestration
- **Streaming LLM Support** - Real-time streaming responses and tool calling
- **Plugin System** - Extensible tools and integrations
- **Session Management** - Persistent sessions with context awareness
- **CLI-first interface** with comprehensive commands
- **Configuration System** - YAML and environment variable configuration
- **Error Handling** - Robust error handling and logging throughout
- **Tool System** - Extensible tool registry with JSON schema validation

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

### Core Systems
- **LLM Manager** - Unified interface for local and remote LLMs with streaming support
- **Tool Registry** - Dynamic tool loading and management with schema validation
- **Runtime Manager** - Session and context management with persistence
- **Configuration System** - Multi-source configuration with precedence handling

## Configuration

The application supports multiple configuration sources with the following priority order (highest to lowest):

1. **Environment variables** (including .env files)
2. **YAML configuration files**
3. **Default values**

### Environment Variables

Environment variables use the `LOCCA_` prefix. For example:

```bash
export LOCCA_LLM__MODEL_NAME=gpt-4
export LOCCA_LLM__TEMPERATURE=0.8
export LOCCA_RUNTIME__PERSISTENT_SESSIONS=true
export LOCCA_RUNTIME__MAX_SESSION_HISTORY=100
```

### .env Files

Create a `.env` file in the project root for local development:

```bash
# Copy from .env and modify as needed
cp .env .env.local
```

The application automatically loads:
- `.env` - Base environment variables
- `.env.local` - Local overrides (loaded after .env, takes precedence)

### YAML Configuration Files

You can also provide YAML configuration files that will be merged with environment variables and defaults.

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
├── .github/           # GitHub Actions workflows
├── .journal/          # Development journal and decisions
├── src/               # Source code
│   └── local_coding_assistant/  # Main package
│       ├── agent/              # Agent loop and LLM management
│       │   ├── agent_loop.py   # Observe-Plan-Act-Reflect implementation
│       │   └── llm_manager.py  # LLM interface with streaming
│       ├── tools/              # Tool system
│       │   ├── base/           # Base tool classes and schemas
│       │   └── tool_registry.py # Tool registration and management
│       ├── runtime/             # Runtime and session management
│       ├── core/                # Core infrastructure
│       ├── config/              # Configuration management
│       └── cli/                 # CLI commands and interface
├── tests/             # Comprehensive test suite
│   ├── unit/          # Unit tests for individual components
│   ├── integration/   # Integration tests with mocks
│   └── e2e/           # End-to-end CLI tests
├── .gitignore
├── pyproject.toml
└── README.md
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
```

### Tool Management

```bash
# List available tools (text and JSON)
uv run locca list-tools list
uv run locca list-tools list --json

# Register a new tool
uv run locca tools register --name calculator --path ./my_tools/calc.py
```

### Configuration

```bash
# Get/set config (env-backed, prefix LOCCA_)
uv run locca config set LLM__MODEL_NAME gpt-4
uv run locca config get LLM__MODEL_NAME

# Show current configuration
uv run locca config show
```

### Development Server

```bash
# Start the development server
uv run locca serve start --host 0.0.0.0 --port 8080 --reload
```

## Advanced Features

### Agent Loop Patterns

LOCCA implements sophisticated agent patterns:

- **Observe-Plan-Act-Reflect** - Multi-step reasoning with tool integration
- **Session Persistence** - Maintains context across multiple queries
- **Tool Calling** - Automatic tool discovery and execution
- **Error Recovery** - Graceful handling of failures with retry logic

### LangGraph Integration

The system is designed for LangGraph compatibility:

- **Graph Orchestration** - AgentLoop can serve as execution nodes in graphs
- **State Management** - Proper state passing between graph nodes
- **Node Isolation** - Clean separation of concerns in graph execution

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

## Recent Additions

- **LangGraph Compatibility Tests** - Integration tests for graph-based orchestration
- **Streaming LLM Responses** - Real-time streaming for better user experience
- **Advanced Agent Loop** - Observe-Plan-Act-Reflect pattern implementation
- **Session & Context Awareness** - Persistent sessions with context management
- **Enhanced Tool System** - JSON schema validation and better error handling
- **CLI Tools & Commands** - Comprehensive command-line interface
- **Centralized Logging** - Structured logging throughout the application
- **Error Handling** - Robust error handling with safe entry points

This project continues to evolve with a focus on reliability, extensibility, and advanced AI patterns.