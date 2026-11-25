# Python CLI Tool - Test Fixture

A simple CLI tool for testing LOCCA's agent capabilities.

## Project Structure

```
python_cli_tool/
├── pyproject.toml          # Project configuration
├── src/python_cli_tool/    # Source code
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration management (has bug to fix)
│   ├── main.py             # CLI entry point with Click
│   └── commands/           # Command implementations
│       └── __init__.py
└── tests/                  # Test suite
    └── test_cli.py         # Unit and integration tests
```

## Features

- **Configuration management** - YAML config files + environment variables
- **Click-based CLI** - Commands: version, greet, config-show, config-set, process, fetch
- **File processing** - Reads input files, outputs JSON
- **Simulated API fetching** - Mock HTTP calls

## Known Issues (for testing)

1. **Config loading bug** - `Config._find_config_file()` doesn't handle missing files gracefully when no config exists
2. **No export command** - Missing `export` command for data export
3. **No validation** - Config values aren't validated on set
4. **Error handling** - Some commands lack proper error handling

## Running Tests

```bash
cd tests/fixtures/projects/python_cli_tool
pip install -e ".[dev]"
pytest tests/ -v
```

## Running the CLI

```bash
pip install -e .
python-cli --help
python-cli greet --name "Test"
python-cli config-show
python-cli process input.txt
```