# Configuration

LOCCA uses a three-layer configuration system for maximum flexibility.

## Configuration Layers

1. **Global Layer**: Base configuration from YAML files and environment variables
2. **Session Layer**: Runtime overrides for session duration  
3. **Call Layer**: Per-request overrides supplied by CLI flags

## Environment Variables

Environment variables are loaded from multiple `.env` files in precedence order:

1. `.env` - Base configuration
2. `.env.${LOCCA_ENV}` - Environment-specific settings
3. `.env.local` - Local overrides (gitignored)

## YAML Configuration Files

Configuration is loaded from YAML files with layered precedence:

- `config/defaults.yaml` - Default settings
- `config/${LOCCA_ENV}.yaml` - Environment overrides
- `config/local.yaml` - Local overrides (gitignored)

## CLI Configuration

Use the CLI to manage configuration:

```bash
# View current configuration
locca config show

# Set a configuration value
locca config set LLM__MODEL_NAME gpt-4

# Get a specific value
locca config get LLM__MODEL_NAME
```

## Provider Configuration

Configure LLM providers using the CLI:

```bash
# Add a new provider
locca provider add openai --api-key-env OPENAI_API_KEY

# List available providers
locca provider list

# Validate provider configurations
locca provider validate
```

## Next Steps

- [CLI Usage](../user-guide/cli-usage.md)
- [Provider Management](../user-guide/provider-management.md)
