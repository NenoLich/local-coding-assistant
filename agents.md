# Agents in LOCCA

This document provides coding instructions and conventions for developing agents within the Local Coding Assistant (LOCCA) project.

## General Guidelines

- Always use type hints for all function parameters and return types
- Include comments where necessary to explain complex logic or non-obvious decisions.
-  Ensure that the code is well-structured and follows PEP 8 guidelines.

## Coding Conventions

### Project coding style
- For data structures always use either Pydantic models or dataclasses.
- For data flowing between modules prefer to use data contracts instead of dicts.
- Use f-strings for string formatting throughout the code instead of the % operator or .format() method.
- Use the | operator for type unions and optional types (PEP 604) instead of Union or Optional; for example, use int | str instead of Union[int, str] and str | None instead of Optional[str].

### Error Handling
- Use custom exceptions from `core/exceptions` or other 'exceptions' files across project with inherited from core exception types.
- Provide meaningful error messages for debugging
- Use centralized error handling from `core/error_handler.py` for consistent error processing
- Apply the `safe_entrypoint` decorator for CLI command functions to ensure proper error handling and logging

### Async Patterns
- All I/O operations (LLM calls, tool executions) must be async
- Use `asyncio.gather()` for concurrent operations when appropriate
- Implement proper cancellation handling for long-running tasks

### Logging
- Use centralized logging via `utils/logging`
- When logging complex structures like dicts or lists with structlog, please use keyword arguments (e.g., logger.info("message", data=some_data)) to ensure the data is properly serialized as structured context.

### Tool Integration
- Register tools via `config/tools.default.yaml` or `config/sandbox_tools.yaml` for sandbox tools
- Implement tool classes with `run()` and optional `stream()` methods
- Use JSON schema validation for regular tools and docstring args description for sandbox tools
- Handle tool failures gracefully with fallback mechanisms

## Configuration

- Agent configurations should use the three-layer system (global/session/call)
- Define agent-specific settings in Pydantic models
- Use `@` path aliases in configuration files (e.g., `@data`, `@logs`)

### Environment Variables
- Environment variables are loaded from multiple `.env` files in order of precedence (later files override earlier ones):
  - `.env` - Base configuration (always loaded first)
  - `.env.${LOCCA_ENV}` - Environment-specific settings (e.g., `.env.development`)
  - `.env.local` - Local environment overrides (gitignored)
- Environment variables can reference each other using `${VARIABLE_NAME}` syntax
- Use `LOCCA_` prefixed variables for project-specific settings

### Path Management
- Use the `path_manager` utility for unified file path handling across the codebase
- Path aliases are resolved according to the current environment:
  - `@root`: Project root directory
  - `@config`: Configuration directory (from `LOCCA_CONFIG_DIR`)
  - `@data`: Data directory (from `LOCCA_DATA_DIR`)
  - `@cache`: Cache directory (from `LOCCA_CACHE_DIR`)
  - `@logs`: Logs directory (from `LOCCA_LOGS_DIR`)
- Always use path manager for path resolution instead of hardcoded paths

## Testing

### Tests
- Place unit tests in `tests/unit/`
- Use `tests/integration/` for modules interaction testing
- Use `tests/e2e/` for end-to-end testing
- Use centralize fixtures in hierarchical conftest files.
- For golden tests create or use dedicated golden/ dir inside specific test sub-dir like `tests/unit/cli/golden`.
- Use `pytest.mark.asyncio` for async tests
- Assert on complete objects rather than individual fields
- Mock external dependencies to isolate agent logic


## Development Workflow

### Code Changes
- Type check code with `uvx ty check --ignore unresolved-import <path/to/file>`
- Format code with `ruff format <path/to/file>`
- Lint with `ruff check --fix <path/to/file>`
- Run relevant tests for changed components

Follow these guidelines to maintain consistency and reliability across LOCCA project files.
