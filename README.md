# Local Coding Assistant (LOCCA)

An AI-powered coding assistant that runs locally with support for LLMs and a plugin system.

## Features

- Local-first architecture with optional cloud backend support
- Plugin system for extending functionality
- CLI-first interface
- Support for local and remote LLMs

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
├── tests/             # Test files
├── .gitignore
├── pyproject.toml
└── README.md
```

## CLI Usage

Use the installed console script `locca`.

Examples:

```bash
# Run a single query
uv run locca run query "Hello" --log-level DEBUG -v

# List tools (text and JSON)
uv run locca list-tools list
uv run locca list-tools list --json

# Get/set config (env-backed, prefix LOCCA_)
uv run locca config set API_KEY secret
uv run locca config get API_KEY

# Start the (placeholder) server
uv run locca serve start --host 0.0.0.0 --port 8080 --reload
```

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
```