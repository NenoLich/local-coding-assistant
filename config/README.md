# Configuration Files

This directory contains configuration files for the Local Coding Assistant application.

## File Structure

- `defaults.yaml` - Base application configuration (all settings including tools_platform, execution_store, telemetry, runtime, etc.)
- `local.yaml` - Local application configuration overrides (gitignored)
- `providers.default.yaml` - Default LLM provider configurations
- `providers.local.yaml` - Local provider overrides (gitignored)
- `auto_model_policies.yaml` - Model selection policies for the provider router

## Getting Started

1. Copy the default config files to create your local overrides:
   ```bash
   cp defaults.yaml local.yaml
   cp providers.default.yaml providers.local.yaml
   ```

2. Edit the `providers.local.yaml` file with your provider settings (API keys, base URLs, models).

3. Edit `local.yaml` for any application-level overrides (log level, sandbox settings, etc.).

## Environment-Based Paths

Configuration paths are resolved based on the environment:

- **Development**: `{project_root}/config/`
- **Production**: System config directory (e.g., `~/.config/local-coding-assistant/` on Linux)
- **Testing**: `{project_root}/tests/test_data/`

Use the `LOCCA_ENV` environment variable to select environment-specific configs (e.g., `LOCCA_ENV=production`).

## Security

- Never commit sensitive data to version control
- Use environment variables for secrets (referenced via `api_key_env` in providers)
- Keep local overrides in `*.local.yaml` files (both are gitignored)

## Configuration Layers

LOCCA uses a three-layer configuration system managed by `ConfigManager`:

1. **Global layer** – Defaults from `defaults.yaml`, `providers.default.yaml`, and environment variables (prefix `LOCCA_`)
2. **Session layer** – Runtime overrides applied for the duration of a session (via CLI `--session-id` or API)
3. **Call layer** – Per-call overrides supplied by CLI flags or API parameters

Higher layers override lower layers.

## Key Configuration Sections (in `defaults.yaml`)

### `llm` - LLM Generation Settings
- `temperature`, `max_tokens`, `timeout`, `max_retries`
- `capture_reasoning`, `reasoning_max_chars`
- `auto_register_litellm_models` - Register unmapped models with LiteLLM
- `providers` - Dynamic provider list (managed at runtime)

### `runtime` - Runtime Behavior
- `persistent_sessions` - Enable durable session storage
- `session_retention_enabled` / `session_retention_max_turns` - In-memory LRU eviction
- `enable_logging`, `log_level`
- `agent_mode` - `"frame"` (Frame Agent, the only supported mode)
- `stream` - Enable streaming (always on via SSE)
- `tool_call_mode` - `"classic"` (default), `"ptc"` (not yet implemented), `"reasoning_only"`
- `session_context_resolution` - `"static"` (same config for all turns) or `"dynamic"`
- `continuation_policy` - Frame loop control:
  - `max_frames_per_turn`, `max_generation_retries`, `max_compaction_retries`
  - `transient_retry_base_delay`, `transient_retry_max_delay`
  - `tool_retry_max_retries`, `tool_retry_base_delay`, `tool_retry_max_delay`
  - `approval_timeout_seconds`
  - `loop_detection`, `token_budget`, `deadline`

### `dashboard` - Dashboard Configuration
- `enabled`, `host`, `port`, `auto_start`

### `sandbox` - Sandbox Environment
- `enabled`, `image`, `timeout`, `memory_limit`, `cpu_limit`
- `network_enabled`, `allowed_directories`, `working_directory`
- Security settings: `allowed_imports`, `blocked_patterns`, `blocked_shell_commands`
- `logging` configuration for container logs

### `repository` - Repository Context Service
- `enabled`
- `storage` - `mode` ("temporary" or "persistent"), paths, size limits
- `file_filter` - Supported extensions, max file size
- `file_monitoring` - Git-tracked file change detection
- `ast_parser` - Tree-sitter language pack
- `repo_map` - Symbol ranking and inclusion settings
- `call_graph` - Per-language relationship weights for PageRank
- `indexing` - Startup and agent/user edit indexing settings

### `tools_platform` - Tool Execution Platform
- `backend` - `"host"` (only implemented backend)
- `exposed_tools` - `"auto"` (all backend-compatible) or explicit list
- `default_policy` - `"allow"`, `"deny"`, `"ask"` (approval default)
- `allow_network`, `non_interactive`, `auto_approve`
- `workspace_root` - Path alias (e.g., `@target`)
- `env_allowlist` - Environment variables passed to commands
- `max_output_bytes`, `max_read_lines`, `max_search_matches`
- `normalize_to_lf_line_endings`
- `path_policy` - Allowed/denied patterns, symlinks, max depth
- `ignore_policy` - Ignore files (`.gitignore`, `.aiignore`)
- `snapshots` - Workspace snapshots configuration
- `timeout` - Soft/hard/idle timeouts, heartbeat, cancellation grace
- `output_limits` - Stdout/stderr/line limits
- `concurrency` - Parallel reads/processes, write conflict policy
- `workspace_state` - Baselines/checkpoints directories and retention

**Note**: Currently only the host backend and classic tool calling mode are implemented. PTC (Programmatic Tool Calling), sandbox mode, and Docker backend are not yet implemented.

### `execution_store` - Durable Execution Store (SQLite)
- `enabled`
- `db_path` - Database path (supports `@` aliases)
- `flush_interval_ms`, `batch_size` - Writer queue settings
- `retention_days` - Per-table retention (default, sessions, frames, messages, events, tools, templates)

### `telemetry` - OpenTelemetry Tracing
- `enabled` - Set true or `OTEL_EXPORTER_OTLP_ENDPOINT` to export spans
- `service_name`, `otlp_endpoint`, `console_export`

### `agent` - Agent Profiles and Policies
- `profiles` - Named agent profiles (default, planner, executor)
- `policies` - Model routing policies (planning_mode, coding_task, general, budget_mode)

### `prompt` - Prompt Templates
- `template_root` - Root directory for templates (null = package defaults)
- `templates` - Template paths for identity, personality, agents, blocks, modes
- `presets` - Agent preset registries
- `agents_md` - AGENTS.md discovery configuration
- `config_instructions` - Additional config instructions

## Best Practices

1. Always copy from default files to create local overrides
2. Never commit local overrides (`*.local.yaml` are gitignored)
3. Document custom configurations
4. Use environment variables for sensitive data (API keys)
5. Use `@` path aliases in configuration (e.g., `@data/models`, `@config/templates`)

## Valid Path Aliases

- `@project` - Project root directory
- `@config` - Configuration directory
- `@data` - Data directory
- `@cache` - Cache directory
- `@log` - Log directory
- `@module` - Module/package root
- `@tools` - Tools directory
- `@templates` - Template directory
- `@target` - Target project root (user project being worked on)

Note: `@root` and `@logs` are NOT valid aliases; use `@project` and `@log` instead.