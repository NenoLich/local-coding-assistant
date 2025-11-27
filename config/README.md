# Configuration Files

This directory contains configuration files for the Local Coding Assistant application.

## File Structure

- `defaults.yaml` - Base application configuration
- `defaults.local.yaml` - Local application configuration overrides
- `tools.default.yaml` - Default tool configurations
- `tools.local.yaml` - Local tool configuration overrides
- `providers.yaml` - Default LLM provider configurations
- `providers.local.yaml` - Local provider overrides
- `auto_model_policies.yaml` - Model selection policies for the provider router

## Getting Started

1. Copy the default config files to create your local overrides:
   ```bash
   cp defaults.yaml defaults.local.yaml
   cp tools.default.yaml tools.local.yaml
   cp providers.yaml providers.local.yaml  # If you have a providers.yaml
   ```

2. Edit the `providers.local.yaml` file with your settings.

## Environment-Based Paths

Configuration paths are resolved based on the environment:

- **Development**: `{project_root}/config/`
- **Production**: System config directory (e.g., `~/.config/local-coding-assistant/` on Linux)
- **Testing**: `{project_root}/tests/test_data/`

## Security

- Never commit sensitive data to version control
- Use environment variables for secrets
- Keep local overrides in `*.local.yaml` files

## Available Configuration Files

### Core Configuration
- `defaults.yaml` - Base application configuration
- `auto_model_policies.yaml` - Model selection policies for the provider router

### Tool Configurations
- `tools.default.yaml` - Default tool configurations
- `tools.local.yaml` - Local tool configuration overrides (create as needed)

### Provider Configurations
- `providers.yaml` - Default LLM provider configurations
- `providers.local.yaml` - Local provider overrides (create as needed)

## Best Practices

1. Always copy from example/template files
2. Never commit local overrides
3. Document custom configurations
4. Use environment variables for sensitive data
