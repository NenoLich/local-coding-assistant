# CLI Usage

LOCCA provides a comprehensive CLI interface for all operations.

## Basic Commands

### Running Queries

```bash
# Run a simple query
locca run query "Explain Python decorators"

# Run with streaming
locca run query "Write a function to calculate fibonacci" --streaming

# Specify model and provider
locca run query "Debug this code" --model gpt-4 --provider openai
```

### Tool Management

```bash
# List available tools
locca tool list

# Add a custom tool
locca tool add --name mytool --path ./tools/mytool.py

# Run a tool directly
locca tool run calculator --input '{"expression": "2 + 2"}'

# Reload tools after changes
locca tool reload
```

### Provider Management

```bash
# List configured providers
locca provider list

# Add a new provider
locca provider add anthropic --api-key-env ANTHROPIC_API_KEY

# Validate provider configurations
locca provider validate

# Test provider connectivity
locca provider test openai
```

### Configuration

```bash
# Show current configuration
locca config show

# Set configuration values
locca config set LLM__TEMPERATURE 0.7

# Get specific values
locca config get LLM__MODEL_NAME
```

## Advanced Usage

### Session Management

```bash
# Start a persistent session
locca session start my-session

# Run queries in session context
locca run query "Continue our discussion" --session my-session

# List active sessions
locca session list

# End a session
locca session end my-session
```

### Development Server

```bash
# Start development server
locca serve start --host 0.0.0.0 --port 8080 --reload
```

## Command Reference

For complete command reference, run:

```bash
locca --help
```

Or get help for specific commands:

```bash
locca run --help
locca tool --help
locca provider --help
```

## Next Steps

- [Agent Patterns](agent-patterns.md)
- [Tool System](tool-system.md)
- [Provider Management](provider-management.md)
