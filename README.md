# Local Coding Assistant

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
│   └── local_assistant/  # Main package
├── tests/             # Test files
├── .gitignore
├── pyproject.toml
└── README.md
```

## Testing

Run tests using pytest:
```bash
pytest
```

## Linting

```bash
ruff check .
```

## Code Formatting

```bash
ruff format .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request