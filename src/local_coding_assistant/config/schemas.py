"""Configuration schemas for the Local Coding Assistant."""

from __future__ import annotations

import time
from typing import Annotated, Any

from pydantic import field_validator

from local_coding_assistant.config.dependencies import SettingDependency
from local_coding_assistant.config.field import ConfigModel, config_field


class ProviderStatus(ConfigModel, section="provider_status"):
    """Status information for a provider."""

    name: str = config_field(description="Provider name")
    is_alive: bool = config_field(
        default=True,
        description="Whether the provider is currently available and healthy",
    )
    last_health_check: float = config_field(
        default_factory=lambda: 0.0, description="Unix timestamp of last health check"
    )
    error_count: int = config_field(
        default=0, description="Number of consecutive errors"
    )


class LLMConfig(ConfigModel, section="llm"):
    """Configuration for LLM provider and model settings."""

    model_name: str | None = config_field(
        default=None, description="Name of the LLM model to use"
    )
    temperature: float = config_field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = config_field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    timeout: int | float = config_field(
        default=120.0,
        gt=0,
        description="Timeout for llm requests",
    )
    max_retries: int = config_field(
        default=3,
        gt=0,
        description="Maximum number of retry attempts for failed requests",
    )
    capture_reasoning: bool = config_field(
        default=False,
        description="Whether to capture internal reasoning text for diagnostics",
    )
    reasoning_max_chars: int = config_field(
        default=1000,
        gt=0,
        description="Maximum number of reasoning characters to store when captured",
    )
    retry_delay: float = config_field(
        default=1.0, gt=0.0, description="Base delay in seconds between retry attempts"
    )
    providers: list[ProviderStatus] = config_field(
        default_factory=list,
        description="List of available providers with their status",
    )
    auto_register_litellm_models: bool = config_field(
        default=False,
        description="Automatically register provider models with LiteLLM to bypass cost mapping for unmapped models",
    )

    def with_overrides(self, **overrides: Any) -> LLMConfig:
        """Create a new LLMConfig with specified overrides.

        Args:
            **overrides: Configuration values to override

        Returns:
            New LLMConfig instance with overrides applied

        Raises:
            ValueError: If any override values are invalid
        """
        # Validate overrides before applying
        try:
            # Create a temporary config with overrides to validate
            override_dict = {k: v for k, v in overrides.items() if v is not None}
            if override_dict:
                # This will raise ValidationError if invalid
                LLMConfig(**{**self.model_dump(), **override_dict})
        except Exception as e:
            raise ValueError(f"Invalid LLM configuration override: {e}") from e

        # Create new config with overrides
        return LLMConfig(**{**self.model_dump(), **override_dict})

    def get_healthy_providers(self) -> list[ProviderStatus]:
        """Get list of providers that are currently healthy."""
        return [p for p in self.providers if p.is_alive]

    def mark_provider_unhealthy(self, provider_name: str, error_count: int = 1) -> None:
        """Mark a provider as unhealthy and update error count."""
        for provider in self.providers:
            if provider.name == provider_name:
                provider.is_alive = False
                provider.error_count += error_count
                provider.last_health_check = time.time()
                break

    def mark_provider_healthy(self, provider_name: str) -> None:
        """Mark a provider as healthy and reset error count."""
        for provider in self.providers:
            if provider.name == provider_name:
                provider.is_alive = True
                provider.error_count = 0
                provider.last_health_check = time.time()
                break

    def add_provider(self, provider_name: str) -> None:
        """Add a new provider to the list."""
        if not any(p.name == provider_name for p in self.providers):
            self.providers.append(ProviderStatus(name=provider_name))

    def remove_provider(self, provider_name: str) -> None:
        """Remove a provider from the list."""
        self.providers = [p for p in self.providers if p.name != provider_name]


class LoopDetectionConfig(ConfigModel, section="loop_detection"):
    """Configuration for loop detection."""

    max_repetitions: int = config_field(
        default=3, ge=1, description="Max times same tool signature can repeat"
    )
    window_size: int = config_field(
        default=5, ge=1, description="Sliding window size for signature comparison"
    )
    enabled: bool = config_field(default=True, description="Enable loop detection")
    check_consecutive: bool = config_field(
        default=True, description="Check for identical consecutive calls"
    )
    check_window_repetitions: bool = config_field(
        default=True, description="Check for repetitions within window"
    )


class TokenBudgetConfig(ConfigModel, section="token_budget"):
    """Configuration for token budget."""

    input_max_tokens: int = config_field(
        default=100_000, ge=1, description="Maximum input tokens per turn"
    )
    output_max_tokens: int = config_field(
        default=100_000, ge=1, description="Maximum output tokens per turn"
    )
    warn_threshold: float = config_field(
        default=0.8, ge=0.0, le=1.0, description="Warning threshold as fraction of max"
    )


class DeadlineTrackerConfig(ConfigModel, section="deadline_tracker"):
    """Configuration for deadline tracking."""

    timeout_seconds: int = config_field(
        default=300, ge=1, description="Turn timeout in seconds"
    )
    warn_threshold: float = config_field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Warning threshold as fraction of timeout",
    )


class ContinuationPolicyConfig(ConfigModel, section="continuation_policy"):
    """Configuration for continuation policy - fully serializable for resume/replay."""

    max_frames_per_turn: int = config_field(
        default=10, ge=1, description="Maximum frames per turn"
    )
    max_generation_retries: int = config_field(
        default=2, ge=0, description="Maximum LLM generation retries"
    )
    max_compaction_retries: int = config_field(
        default=1, ge=0, description="Maximum context compaction retries"
    )

    respect_retry_after: bool = config_field(
        default=True,
        description="Honor provider Retry-After hints before retrying a transient error",
    )

    transient_retry_base_delay: float = config_field(
        default=1.0,
        ge=0.0,
        description="Base delay (seconds) for exponential backoff between transient-error retries",
    )

    transient_retry_max_delay: float = config_field(
        default=60.0,
        ge=0.0,
        description="Maximum backoff delay (seconds) between transient-error retries",
    )
    approval_timeout_seconds: int = config_field(
        default=300, ge=1, description="Approval timeout in seconds"
    )

    # Tool retry configuration for retryable infrastructure failures
    # (timeout, backend failure, connection error)
    tool_retry_max_retries: int = config_field(
        default=2, ge=0, description="Maximum retries for retryable tool failures"
    )
    tool_retry_base_delay: float = config_field(
        default=1.0,
        ge=0.0,
        description="Base delay (seconds) for exponential backoff between tool retries",
    )
    tool_retry_max_delay: float = config_field(
        default=30.0,
        ge=0.0,
        description="Maximum backoff delay (seconds) between tool retries",
    )

    loop_detection: LoopDetectionConfig = config_field(
        default_factory=LoopDetectionConfig, description="Loop detection config"
    )
    token_budget: TokenBudgetConfig = config_field(
        default_factory=TokenBudgetConfig, description="Token budget config"
    )
    deadline: DeadlineTrackerConfig = config_field(
        default_factory=DeadlineTrackerConfig, description="Deadline config"
    )


class RuntimeConfig(ConfigModel, section="runtime"):
    """Configuration for runtime behavior and settings."""

    persistent_sessions: bool = config_field(
        default=False, description="Whether to maintain persistent sessions"
    )
    max_session_history: int = config_field(
        default=100,
        gt=0,
        description="Maximum number of messages to keep in session history",
    )
    session_retention_enabled: bool = config_field(
        default=True,
        description="Whether in-memory session file-version retention/eviction is enabled",
    )
    session_retention_max_turns: int = config_field(
        default=50,
        gt=0,
        description="Maximum cumulative turns before evicting the LRU session's in-memory file-version data",
    )
    enable_logging: bool = config_field(
        default=True, description="Whether to enable runtime logging"
    )
    log_level: str | None = config_field(
        default="INFO", description="Logging level for runtime operations"
    )
    agent_mode: str = config_field(
        default="no_agent",
        description="Agent execution mode: 'default_loop' (legacy AgentLoop), 'graph' (LangGraph-based agent), 'frame' (new ExecutionFrame workflow), or 'no_agent'",
        dependencies=SettingDependency(
            value_requirements={
                "frame": ["llm_service", "tool_gateway"],
                "no_agent": [],
            },
            fallback_order=["frame", "no_agent"],
        ),
    )
    stream: bool = config_field(
        default=True, description="Default streaming mode for agent responses"
    )
    tool_call_mode: str = config_field(
        default="classic",
        description="Tool calling mode: 'classic' (default) uses function calling, 'ptc' uses programmatic tool calling, 'reasoning_only' uses no tool calls",
        dependencies=SettingDependency(
            value_requirements={
                "ptc": [
                    "tools_available",
                    "sandbox_manager",
                    "python_execution",
                    "sandbox_available",
                ],
                "classic": ["tools_available"],
                "reasoning_only": [],
            },
            fallback_order=["ptc", "classic", "reasoning_only"],
        ),
    )
    session_context_resolution: str = config_field(
        default="dynamic",
        description="Context resolution mode: 'static' uses session-level config that doesn't change turn-to-turn, 'dynamic' uses config_manager to build live RuntimeContextConfig per turn",
        dependencies=SettingDependency(
            value_requirements={
                "static": [],
                "dynamic": [],
            },
            fallback_order=["dynamic", "static"],
        ),
    )
    continuation_policy: ContinuationPolicyConfig = config_field(
        default_factory=ContinuationPolicyConfig,
        description="Continuation policy configuration for frame execution",
    )

    @field_validator("tool_call_mode")
    @classmethod
    def validate_tool_call_mode(cls, v: str) -> str:
        """Validate and normalize tool_call_mode.

        Accepts 'reasoning' as an alias for 'reasoning_only' for better UX.
        """
        # Map 'reasoning' to 'reasoning_only' for backward compatibility
        if v == "reasoning":
            return "reasoning_only"

        if v not in ("classic", "ptc", "reasoning_only"):
            raise ValueError(
                "tool_call_mode must be one of: 'classic', 'ptc', 'reasoning' (or 'reasoning_only')"
            )
        return v


class ModelConfig(ConfigModel, section="model"):
    """Configuration for a specific model within a provider."""

    name: str = config_field(description="Model identifier")
    supported_parameters: list[str] = config_field(
        default_factory=list,
        description="List of supported parameter names for this model",
    )
    extra_body: dict[str, Any] = config_field(
        default_factory=dict,
        description="Provider-specific parameters passed via extra_body to litellm",
    )

    @classmethod
    def from_dict(cls, name: str, config: dict) -> ModelConfig:
        """Create ModelConfig from a model name and its configuration."""
        return cls(
            name=name,
            supported_parameters=config.get("supported_parameters", []),
            extra_body=config.get("extra_body", {}),
        )


class ProviderConfig(ConfigModel, section="provider"):
    """Configuration for LLM providers."""

    name: str = config_field(description="Provider name")
    driver: str = config_field(
        description="Driver type (openai_chat, openai_responses, local)"
    )
    base_url: str = config_field(description="Base URL for the provider API")
    api_key_env: str | None = config_field(
        default=None, description="Environment variable name for API key"
    )
    models: list[ModelConfig] = config_field(
        default_factory=list, description="List of available model configurations"
    )
    health_check_endpoint: str | None = config_field(
        default=None, description="Endpoint to check provider health"
    )
    health_check_method: str = config_field(
        default="GET", description="HTTP method to use for health checks"
    )
    health_check_timeout: float = config_field(
        default=5.0, description="Timeout in seconds for health checks"
    )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ProviderConfig:
        """Create ProviderConfig from dictionary."""
        # Make a copy to avoid modifying the input
        config = config_dict.copy()

        # Handle models configuration
        models_config = config.pop("models", {})
        models = []

        # If models is a list, convert it to the expected dict format
        if isinstance(models_config, list):
            for model in models_config:
                if isinstance(model, dict) and "name" in model:
                    models.append(ModelConfig(**model))
        # If models is a dict, process as before
        elif isinstance(models_config, dict):
            models = [
                ModelConfig(name=name, **model_config)
                for name, model_config in models_config.items()
            ]

        return cls(models=models, **config)

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model by name."""
        for model in self.models:
            if model.name == model_name:
                return model
        return None

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return [model.name for model in self.models]

    def is_parameter_supported(self, model_name: str, param: str) -> bool:
        """Check if a parameter is supported by the specified model."""
        model = self.get_model_config(model_name)
        if not model:
            return False
        return param in model.supported_parameters


class AgentProfileConfig(ConfigModel, section="agent_profile"):
    """Configuration for an agent profile."""

    name: Annotated[
        str, config_field(..., description="Unique identifier for the profile")
    ]
    kind: str = config_field("default", description="Type/category of the profile")
    description: Annotated[
        str,
        config_field(..., description="Description of the agent's role and behavior"),
    ]
    goals: list[str] = config_field(
        default_factory=list,
        description="List of primary objectives for this agent profile",
    )
    tone: str | None = config_field(
        None, description="Communication style and tone for this profile"
    )
    constraints: list[str] = config_field(
        default_factory=list,
        description="List of constraints or guidelines for this profile",
    )
    model_policy: str | None = config_field(
        None,
        description=(
            "Name of the model policy to use for this profile. "
            "If not specified, falls back to the profile name or 'general'"
        ),
    )

    @classmethod
    def default(cls) -> AgentProfileConfig:
        """Create a default agent profile configuration."""
        return cls(
            name="default",
            kind="default",
            description=(
                "Primary coding assistant focused on safe, step-by-step reasoning "
                "with practical guidance."
            ),
            goals=[
                "Deliver concise answers grounded in repository state",
                "Surface trade-offs and assumptions explicitly",
            ],
            tone="Confident, pragmatic, collaborative",
            model_policy="general",
            constraints=[
                "Never fabricate file paths or code",
                "Prefer actionable steps over vague suggestions",
            ],
        )

    @classmethod
    def planner(cls) -> AgentProfileConfig:
        """Create a planner agent profile configuration."""
        return cls(
            name="planner",
            kind="planner",
            description="Decomposes the request into executable steps and highlights risks.",
            goals=[
                "Summarize objectives",
                "Outline numbered plan with verification points",
            ],
            tone="Analytical and structured",
            model_policy="planner",
        )

    @classmethod
    def executor(cls) -> AgentProfileConfig:
        """Create an executor agent profile configuration."""
        return cls(
            name="executor",
            kind="executor",
            description="Executes the plan, writes code, and validates results.",
            goals=[
                "Apply plan precisely",
                "Capture diffs and side-effects",
                "Report blockers or verifications needed",
            ],
            tone="Hands-on and detail oriented",
            model_policy="general",
        )


class ModelPolicyConfig(ConfigModel, section="model_policy"):
    """Typed representation of a routing/fallback policy."""

    models: list[str] = config_field(
        default_factory=list,
        description="Ordered list of provider:model routes plus fallback directives",
    )
    max_failovers: int | None = config_field(
        default=None,
        ge=1,
        description="Maximum number of failovers allowed for this policy",
    )
    fallback_strategy: str = config_field(
        default="sequential",
        description="Fallback strategy name (sequential, exponential, etc.)",
    )


class AgentConfig(ConfigModel, section="agent"):
    """Configuration for agent behavior and model policies."""

    # Agent profiles define different behaviors and configurations
    profiles: dict[str, AgentProfileConfig] = config_field(
        default_factory=dict,
        description=(
            "Available agent profiles. Each profile defines a different "
            "personality, goals, and behavior for the agent."
        ),
    )

    # Model policies for different agent roles
    policies: dict[str, ModelPolicyConfig] = config_field(
        default_factory=dict,
        description="Agent role to model fallback policies.",
    )

    # Minimal fallback policies for safety (used only if policies dict is empty)
    general: dict[str, list[str]] = config_field(
        default_factory=lambda: {"models": ["fallback:any"]},
        description="Ultimate fallback policy",
    )

    def get_profile(self, name: str) -> AgentProfileConfig:
        """Get an agent profile by name, falling back to defaults if not found.

        Args:
            name: Name of the profile to retrieve

        Returns:
            AgentProfileConfig: The requested profile or default if not found
        """
        # Return the profile if it exists
        if name in self.profiles:
            return self.profiles[name]

        # Try to create a default profile based on the name
        if name == "planning_mode":
            return AgentProfileConfig.planner()
        elif name == "executor":
            return AgentProfileConfig.executor()

        # Fall back to default profile
        return AgentProfileConfig.default()

    def get_model_policy_for_profile(self, profile_name: str) -> list[str]:
        """Get the model policy for a given profile.

        Args:
            profile_name: Name of the agent profile

        Returns:
            List of model names in order of preference

        Raises:
            ValueError: If no model policy is found for the profile
        """
        profile = self.get_profile(profile_name)

        # First try the profile's model_policy
        if profile.model_policy and profile.model_policy in self.policies:
            return self.policies[profile.model_policy].models

        # Then try the profile name as a policy
        if profile_name in self.policies:
            return self.policies[profile_name].models

        # Fall back to the 'general' policy
        if "general" in self.policies:
            return self.policies["general"].models

        # As a last resort, use the hardcoded defaults
        if hasattr(self, profile_name) and isinstance(
            getattr(self, profile_name), dict
        ):
            return getattr(self, profile_name).get("models", [])

        # If we still don't have a policy, try the general default
        if hasattr(self, "general"):
            return self.general.get("models", [])

        raise ValueError(f"No model policy found for profile: {profile_name}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AgentConfig:
        """Create AgentConfig from dictionary."""
        return cls(**config_dict)

    def get_policy_for_role(self, role: str) -> list[str]:
        """Get model fallback policy for a specific role."""
        if role in self.policies:
            return self.policies[role].models
        elif hasattr(self, role):
            return getattr(self, role)["models"]
        else:
            return self.general["models"]


class AgentsMdConfig(ConfigModel, section="prompt_agents_md"):
    """Configuration for AGENTS.md / CLAUDE.md / LOCCA.md discovery."""

    enabled: bool = config_field(
        True, description="Whether AGENTS.md discovery is enabled"
    )
    walk_up: bool = config_field(
        True, description="Whether to walk up from cwd to workspace root"
    )
    filenames: list[str] = config_field(
        default_factory=lambda: ["AGENTS.md", "CLAUDE.md", "LOCCA.md"],
        description="Filenames to discover, closest to cwd first",
    )
    global_paths: list[str] = config_field(
        default_factory=lambda: [
            "~/.config/locca/AGENTS.md",
            "~/.claude/CLAUDE.md",
        ],
        description="Global AGENTS.md paths",
    )


class PromptTemplateConfig(ConfigModel, section="prompt"):
    """Configuration for prompt templates and rendering."""

    template_root: str | None = config_field(
        None,
        description=(
            "Root directory for prompt templates. "
            "If not provided, will use package defaults."
        ),
    )
    templates: dict[str, str] = config_field(
        default_factory=lambda: {
            "identity_base": "base/identity.jinja2",
            "personality": "base/personality.jinja2",
            "model_default": "base/model_default.jinja2",
            "model_gemini": "base/model_gemini.jinja2",
            "static_env": "blocks/static_env.jinja2",
            "agents_md": "blocks/agents_md.jinja2",
            "global_rules": "blocks/global_rules.jinja2",
            "config_overrides": "blocks/config_rules.jinja2",
            "skills": "blocks/skills.jinja2",
            "memories": "blocks/memories.jinja2",
            "examples": "blocks/examples.jinja2",
            "repo_metadata": "blocks/repo_metadata.jinja2",
            "repo_map": "blocks/repo_map.jinja2",
            "constraints": "blocks/constraints.jinja2",
            "safety": "blocks/safety.jinja2",
            "output_format": "blocks/output_format.jinja2",
            "classic_tools_mode": "modes/classic_tools.jinja2",
            "sandbox_python_mode": "modes/sandbox_python.jinja2",
            "sandbox_shell_mode": "modes/sandbox_shell.jinja2",
            "reasoning_only_mode": "modes/reasoning_only.jinja2",
            "default_agent": "agents/default.jinja2",
            "planner_agent": "agents/planner.jinja2",
            "executor_agent": "agents/executor.jinja2",
            "compaction_agent": "agents/compaction.jinja2",
            "subagent_agent": "agents/subagent.jinja2",
            "planner_extra": "agents/planner_extra.jinja2",
            "executor_extra": "agents/executor_extra.jinja2",
        },
        description="Template paths for different prompt sections",
    )
    presets: dict[str, str] = config_field(
        default_factory=lambda: {
            "default": "default",
            "compaction": "compaction",
            "subagent": "subagent",
            "planner": "planner",
            "executor": "executor",
        },
        description="Agent preset registry names",
    )
    agents_md: AgentsMdConfig = config_field(
        default_factory=AgentsMdConfig,
        description="AGENTS.md discovery configuration",
    )
    config_instructions: list[str] = config_field(
        default_factory=list,
        description="Array of instruction strings or URLs",
    )
    enable_jinja_autoescape: bool = config_field(
        False,
        description="Whether to enable Jinja2 autoescaping (usually not needed for LLM prompts)",
    )
    trim_blocks: bool = config_field(
        True, description="Whether to trim whitespace around template blocks"
    )
    lstrip_blocks: bool = config_field(
        True, description="Whether to strip whitespace from the start of lines"
    )


class SandboxLoggingConfig(ConfigModel, section="sandbox_logging"):
    """Configuration for sandbox logging."""

    level: str = config_field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    console: bool = config_field(
        default=True, description="Whether to enable console logging"
    )
    file: bool = config_field(
        default=True, description="Whether to enable file logging"
    )
    directory: str = config_field(
        default="/workspace/logs",
        description="Directory to store log files (inside container)",
    )
    file_name: str = config_field(
        default="container_{session_id}.log",
        description="Log file name pattern. {session_id} will be replaced with the actual session ID.",
    )
    max_size: int = config_field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes before rotation",
        gt=0,
    )
    backup_count: int = config_field(
        default=5, description="Number of backup log files to keep", ge=0
    )


class SandboxConfig(ConfigModel, section="sandbox"):
    """Configuration for the Docker sandbox environment."""

    enabled: bool = config_field(
        default=False,
        description="Whether to enable sandbox execution",
        dependencies=SettingDependency(
            value_requirements={
                "True": ["sandbox_manager", "sandbox_available"],
                "False": [],
            },
            fallback_order=["True", "False"],
        ),
    )

    image: str = config_field(
        default="locca-sandbox:latest",
        description="Docker image to use for the sandbox",
    )
    timeout: int = config_field(default=30, description="Execution timeout in seconds")
    memory_limit: str = config_field(
        default="512m", description="Memory limit for the container (e.g., 512m, 1g)"
    )
    cpu_limit: float = config_field(
        default=0.5, description="CPU limit (fraction of one CPU)"
    )
    network_enabled: bool = config_field(
        default=False, description="Whether to allow network access in the sandbox"
    )
    persistence: bool = config_field(
        default=False, description="Whether to persist the sandbox container"
    )
    session_id: str = config_field(
        default="default", description="Session ID for the sandbox"
    )
    allowed_imports: list[str] = config_field(
        default_factory=list, description="List of allowed Python top-level imports"
    )
    blocked_patterns: list[str] = config_field(
        default_factory=list, description="List of regex patterns to block in code"
    )
    blocked_shell_commands: list[str] = config_field(
        default_factory=list, description="List of blocked shell commands"
    )
    max_code_length: int = config_field(
        default=10000, description="Maximum allowed code length in characters"
    )
    max_file_size: int = config_field(
        default=1024 * 1024, description="Maximum allowed file size for created files"
    )
    allowed_directories: list[str] = config_field(
        default_factory=list, description="List of allowed directories for file access"
    )
    working_directory: str = config_field(
        default="/workspace", description="Working directory inside the container"
    )
    session_timeout: int = config_field(
        default=300, description="Session timeout in seconds (default: 5 minutes)"
    )
    max_sessions: int = config_field(
        default=5, description="Maximum number of concurrent persistent sessions"
    )
    logging: SandboxLoggingConfig = config_field(
        default_factory=SandboxLoggingConfig,
        description="Logging configuration for the sandbox",
    )


class DashboardConfig(ConfigModel, section="dashboard"):
    """Configuration for the dashboard."""

    enabled: bool = config_field(
        default=True, description="Whether the dashboard integration is enabled"
    )
    port: int = config_field(
        default=8080, gt=0, le=65535, description="Port for the dashboard server"
    )
    host: str = config_field(
        default="localhost", description="Host for the dashboard server"
    )
    auto_start: bool = config_field(
        default=False, description="Whether to auto-start dashboard with LOCCA"
    )


class TelemetryConfig(ConfigModel, section="telemetry"):
    """Configuration for OpenTelemetry tracing (OTLP/gRPC export)."""

    enabled: bool = config_field(
        default=False,
        description="Whether OTel tracing is enabled (OTEL_EXPORTER_OTLP_ENDPOINT also enables it)",
    )
    service_name: str = config_field(
        default="locca-runtime",
        description="OTel service.name resource attribute",
    )
    otlp_endpoint: str | None = config_field(
        default=None,
        description="OTLP/gRPC collector endpoint (e.g. http://localhost:4317 for Jaeger)",
    )
    console_export: bool = config_field(
        default=False,
        description="Also export spans to the console (debugging)",
    )


class ToolsPlatformTimeoutConfig(ConfigModel, section="tools_platform_timeout"):
    """Timeout policy for the tools platform (seconds)."""

    soft_seconds: float = config_field(
        default=10.0,
        gt=0.0,
        description="Soft timeout before progress reporting / escalation",
    )
    hard_seconds: float = config_field(
        default=300.0,
        gt=0.0,
        description="Maximum execution deadline",
    )
    idle_seconds: float = config_field(
        default=30.0,
        gt=0.0,
        description="Timeout when no output is produced",
    )
    heartbeat_seconds: float = config_field(
        default=5.0,
        gt=0.0,
        description="Interval for progress checks",
    )
    cancellation_grace_seconds: float = config_field(
        default=5.0,
        gt=0.0,
        description="Grace period before forceful termination",
    )


class ToolsPlatformOutputLimitsConfig(
    ConfigModel, section="tools_platform_output_limits"
):
    """Output limits for command execution."""

    max_stdout_bytes: int = config_field(
        default=262144, gt=0, description="Maximum stdout bytes before truncation"
    )
    max_stderr_bytes: int = config_field(
        default=262144, gt=0, description="Maximum stderr bytes before truncation"
    )
    max_lines: int = config_field(
        default=400, gt=0, description="Maximum output lines before truncation"
    )


class ToolsPlatformConcurrencyConfig(ConfigModel, section="tools_platform_concurrency"):
    """Concurrency policy for the tools platform scheduler."""

    max_parallel_reads: int = config_field(
        default=8, gt=0, description="Maximum concurrent read operations"
    )
    max_parallel_processes: int = config_field(
        default=2, gt=0, description="Maximum concurrent process operations"
    )
    write_conflict_policy: str = config_field(
        default="deny",
        description="Behavior on write conflicts: deny or overwrite",
    )


class ToolsPlatformSnapshotsConfig(ConfigModel, section="tools_platform_snapshots"):
    """Snapshot storage configuration for workspace transactions."""

    enabled: bool = config_field(
        default=True, description="Whether workspace snapshots are enabled"
    )
    directory: str = config_field(
        default="@data/workspace_snapshots",
        description="Snapshot storage directory (supports @ aliases)",
    )
    max_snapshots: int = config_field(
        default=10, gt=0, description="Maximum snapshots kept per workspace"
    )
    max_snapshot_age_days: int = config_field(
        default=30, gt=0, description="Maximum snapshot age in days"
    )


class ToolsPlatformIgnorePolicyConfig(
    ConfigModel, section="tools_platform_ignore_policy"
):
    """Ignore file policy for file traversal tools."""

    files: list[str] = config_field(
        default_factory=lambda: [".gitignore", ".aiignore"],
        description="List of ignore filenames in order of precedence (higher index = higher priority)",
    )


class ToolsPlatformPathPolicyConfig(ConfigModel, section="tools_platform_path_policy"):
    """Workspace path policy for the tools platform."""

    allowed_patterns: list[str] = config_field(
        default_factory=lambda: ["**"],
        description="Glob patterns for allowed paths (relative to workspace root)",
    )
    denied_patterns: list[str] = config_field(
        default_factory=list,
        description="Glob patterns for denied paths (relative to workspace root)",
    )
    allow_symlinks: bool = config_field(
        default=False, description="Whether symlink traversal is allowed"
    )
    max_depth: int | None = config_field(
        default=None, description="Maximum directory depth (None for unlimited)"
    )


class ArgumentMatcherConfig(ConfigModel, section="tools_platform_argument_matcher"):
    """Argument matcher for approval rules.

    Matches tool arguments using glob patterns on specific argument values
    or command string patterns for run_command.
    """

    command_pattern: str | None = config_field(
        default=None, description="Glob pattern for command string (run_command only)"
    )
    arg_patterns: dict[str, str] = config_field(
        default_factory=dict,
        description="Glob patterns for specific argument values",
    )
    condition: str | None = config_field(
        default=None, description="Optional Python expression for complex matching"
    )


class MatcherConfig(ConfigModel, section="tools_platform_matcher"):
    """Matcher configuration for approval rules."""

    type: str = config_field(
        description="Matcher type: any_args, command_pattern, path_outside_workspace, arg_equals, arg_pattern"
    )
    pattern: str | None = config_field(
        default=None, description="Glob pattern (for command_pattern, arg_pattern)"
    )
    arg: str | None = config_field(
        default=None, description="Argument name (for arg_equals, arg_pattern)"
    )
    value: Any | None = config_field(
        default=None, description="Expected value (for arg_equals)"
    )


class PolicyRuleConfig(ConfigModel, section="tools_platform_policy_rule"):
    """A single approval policy rule."""

    id: str = config_field(description="Unique rule identifier")
    tool_id: str | list[str] = config_field(description="Tool this rule applies to")
    matcher: MatcherConfig = config_field(description="Argument matching criteria")
    outcome: str = config_field(description="Decision outcome: allow, deny, or ask")
    description: str = config_field(
        default="", description="Human-readable description"
    )
    source: str = config_field(
        default="config", description="Rule source: config or user_approval"
    )
    priority: int = config_field(
        default=100, description="Rule priority (higher = evaluated first)"
    )


class ProjectOverrideConfig(ConfigModel, section="tools_platform_project_override"):
    """Project-level approval override (persisted to .local.yaml)."""

    base_rule_id: str = config_field(description="ID of base rule to override")
    outcome: str = config_field(description="Override outcome: allow, deny, or ask")
    description: str = config_field(
        default="", description="Human-readable description"
    )


class ToolsPlatformConfig(ConfigModel, section="tools_platform"):
    """Configuration for the tools_platform tool gateway."""

    backend: str = config_field(
        default="host",
        description="Tool execution backend: 'host' or 'docker'",
    )
    exposed_tools: str | list[str] = config_field(
        default="auto",
        description="Exposure policy: 'auto' for all backend-compatible tools, or an explicit list of tool IDs",
    )
    default_policy: str = config_field(
        default="ask",
        description="Approval default when no rule matches: allow, deny, or ask",
    )
    # Risk classes are advisory only (for UI display), not used for policy decisions
    risk_defaults: dict[str, str] = config_field(
        default_factory=lambda: {
            "safe": "allow",
            "low": "ask",
            "medium": "ask",
            "high": "ask",
            "critical": "deny",
        },
        description="Advisory risk-based default policies (not enforced)",
    )
    allow_network: bool = config_field(
        default=False, description="Default network access policy for commands"
    )
    non_interactive: bool = config_field(
        default=False,
        description="If true, 'ask' approval outcomes become typed blocked results",
    )
    auto_approve: bool = config_field(
        default=False,
        description="If true, automatically approve tools that would require approval (fallback rule becomes allow)",
    )
    workspace_root: str = config_field(
        default="@target",
        description="Workspace root (supports @ aliases, resolved via PathManager)",
    )
    # Deprecated: use tools.<tool_id>.command_approval instead
    command_approval_path: str | None = config_field(
        default="@config/command_approval.toml",
        description="Path to the command approval TOML policy file (deprecated)",
    )
    env_allowlist: list[str] | None = config_field(
        default=None,
        description="Environment keys passed to commands (None = pass through)",
    )
    max_output_bytes: int = config_field(
        default=262144, gt=0, description="Maximum bytes returned by read/search tools"
    )
    max_read_lines: int = config_field(
        default=400, gt=0, description="Maximum lines returned by read tools"
    )
    max_search_matches: int = config_field(
        default=200, gt=0, description="Maximum matches returned by search tools"
    )
    normalize_to_lf_line_endings: bool = config_field(
        default=False,
        description="When true, normalizes CRLF/CR line endings to LF for LLM-facing reads and edit matching. Baseline hashes and snapshots remain raw disk bytes.",
    )
    path_policy: ToolsPlatformPathPolicyConfig = config_field(
        default_factory=ToolsPlatformPathPolicyConfig,
        description="Workspace path access policy",
    )
    ignore_policy: ToolsPlatformIgnorePolicyConfig = config_field(
        default_factory=ToolsPlatformIgnorePolicyConfig,
        description="Ignore file policy for file traversal tools",
    )
    snapshots: ToolsPlatformSnapshotsConfig = config_field(
        default_factory=ToolsPlatformSnapshotsConfig,
        description="Workspace snapshot storage configuration",
    )
    timeout: ToolsPlatformTimeoutConfig = config_field(
        default_factory=ToolsPlatformTimeoutConfig,
        description="Timeout policy for tool execution",
    )
    output_limits: ToolsPlatformOutputLimitsConfig = config_field(
        default_factory=ToolsPlatformOutputLimitsConfig,
        description="Output limits for command execution",
    )
    concurrency: ToolsPlatformConcurrencyConfig = config_field(
        default_factory=ToolsPlatformConcurrencyConfig,
        description="Concurrency policy for the scheduler",
    )
    # All approval rules with explicit priorities
    rules: list[PolicyRuleConfig] = config_field(
        default_factory=list,
        description="Complete rule set with priorities",
    )
    # Project overrides (auto-populated by YES_ALWAYS_PROJECT)
    project_overrides: list[ProjectOverrideConfig] = config_field(
        default_factory=list,
        description="Project-level approval overrides (ASK -> ALLOW)",
    )
    # Configurable paths for tool policies files
    tool_policies_path: str = config_field(
        default="@config/tool_policies.yaml",
        description="Path to base tool policies file (supports @ aliases)",
    )
    tool_policies_local_path: str = config_field(
        default="@config/tool_policies.local.yaml",
        description="Path to project overrides file (supports @ aliases)",
    )


class ExecutionStoreConfig(ConfigModel, section="execution_store"):
    """Configuration for the durable execution store (SQLite)."""

    enabled: bool = config_field(
        default=True,
        description="Whether the durable execution store is enabled",
    )
    db_path: str = config_field(
        default="@data/execution_store.db",
        description="Path to the SQLite database (supports @ aliases)",
    )
    flush_interval_ms: int = config_field(
        default=1000,
        gt=0,
        description="Writer queue flush interval in milliseconds",
    )
    batch_size: int = config_field(
        default=100,
        gt=0,
        description="Maximum number of operations per write batch",
    )
    retention_days: dict[str, int] = config_field(
        default_factory=lambda: {
            "default": 30,
            "sessions": 30,
            "frames": 30,
            "messages": 30,
            "events": 30,
            "tools": 365,
            "templates": 365,
        },
        description="Retention policy in days per table type",
    )


class StorageConfig(ConfigModel, section="storage"):
    """Configuration for repository database storage."""

    mode: str = config_field(
        default="temporary",
        description="Storage mode: 'temporary' or 'persistent'",
    )
    persistent_db_path: str = config_field(
        default=".locca/repo_context.db",
        description="Path for persistent database (relative to repo root)",
    )
    temp_db_dir: str = config_field(
        default="@data/temp",
        description="Directory for temporary databases (supports @ aliases)",
    )
    temp_db_naming_strategy: str = config_field(
        default="path_hash",
        description="Naming strategy for temporary databases: 'path_hash' (folder_name_hash) or 'uuid'",
    )
    max_cum_file_size: int = config_field(
        default=25000000,
        gt=0,
        description="Maximum cumulative file size in bytes to index in temporary storage (25MB)",
    )


class FileFilterConfig(ConfigModel, section="file_filter"):
    """Configuration for file filtering during re-indexing."""

    supported_extensions: list[str] = config_field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".rs",
            ".go",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
        ],
        description="File extensions to include in re-indexing",
    )
    max_file_size_kb: int = config_field(
        default=500,
        gt=0,
        description="Maximum file size in kilobytes for re-indexing",
    )


class FileMonitoringConfig(ConfigModel, section="file_monitoring"):
    """Configuration for background file monitoring."""

    enabled: bool = config_field(
        default=True,
        description="Whether to enable background file monitoring",
    )
    use_git_tracking: bool = config_field(
        default=True,
        description="Use git for tracking files",
    )
    notify_on_changes: bool = config_field(
        default=True,
        description="Emit system notifications for file changes",
    )
    notification_types: list[str] = config_field(
        default_factory=lambda: ["modified", "created", "deleted"],
        description="Types of file changes to notify about",
    )


class ASTParserConfig(ConfigModel, section="ast_parser"):
    """Configuration for AST parsing."""

    language_pack: str = config_field(
        default="tree-sitter-language-pack",
        description="Language pack to use for AST parsing",
    )


class RepoMapConfig(ConfigModel, section="repo_map"):
    """Configuration for repository map generation."""

    max_symbols: int = config_field(
        default=100,
        gt=0,
        description="Maximum number of symbols to include in repo map (pageranked)",
    )
    include_docstrings: bool = config_field(
        default=True,
        description="Whether to include docstrings in repo map",
    )
    include_imports: bool = config_field(
        default=False,
        description="Whether to include imports in repo map",
    )
    included_symbol_types: list[str] = config_field(
        default_factory=lambda: [
            "function",
            "method",
            "class",
            "interface",
            "type_alias",
        ],
        description="Symbol types to include in repo map (for pageranking)",
    )


class CallGraphConfig(ConfigModel, section="call_graph"):
    """Configuration for call graph analysis."""

    relationship_weights: dict[str, dict[str, float]] = config_field(
        default_factory=lambda: {
            "python": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "javascript": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "typescript": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "rust": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "go": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
            "default": {
                "call": 1.5,
                "type_reference": 1.2,
                "inherits": 1.0,
                "belongs_to": 0.2,
                "contains": 0.0,
            },
        },
        description="Per-language relationship weights for PageRank computation",
    )


class IndexingConfig(ConfigModel, section="indexing"):
    """Configuration for indexing operations."""

    startup_enabled: bool = config_field(
        default=True,
        description="Whether to enable startup re-indexing",
    )
    startup_parallel_workers: int = config_field(
        default=4,
        gt=0,
        description="Number of parallel workers for startup re-indexing",
    )
    agent_edit_enabled: bool = config_field(
        default=True,
        description="Whether to enable agent edit re-indexing",
    )
    user_edit_enabled: bool = config_field(
        default=True,
        description="Whether to enable user edit re-indexing",
    )
    debounce_window: float = config_field(
        default=2.0,
        gt=0.0,
        description="Debounce window in seconds for user edit re-indexing",
    )
    max_cumulative_file_size_kb: int = config_field(
        default=10000,
        gt=0,
        description="Max cumulative file size for batch indexing (default 10MB)",
    )


class RepositoryConfig(ConfigModel, section="repository"):
    """Configuration for repository context service."""

    enabled: bool = config_field(
        default=True,
        description="Whether the repository context service is enabled",
    )
    storage: StorageConfig = config_field(
        default_factory=StorageConfig,
        description="Storage configuration for repository database",
    )
    file_filter: FileFilterConfig = config_field(
        default_factory=FileFilterConfig,
        description="File filtering configuration for re-indexing",
    )
    file_monitoring: FileMonitoringConfig = config_field(
        default_factory=FileMonitoringConfig,
        description="File monitoring configuration",
    )
    ast_parser: ASTParserConfig = config_field(
        default_factory=ASTParserConfig,
        description="AST parser configuration",
    )
    repo_map: RepoMapConfig = config_field(
        default_factory=RepoMapConfig,
        description="Repository map configuration",
    )
    call_graph: CallGraphConfig = config_field(
        default_factory=CallGraphConfig,
        description="Call graph configuration",
    )
    indexing: IndexingConfig = config_field(
        default_factory=IndexingConfig,
        description="Indexing configuration",
    )


class AppConfig(ConfigModel, section="app"):
    """Top-level application configuration."""

    llm: LLMConfig = config_field(
        default_factory=LLMConfig, description="LLM configuration"
    )
    runtime: RuntimeConfig = config_field(
        default_factory=RuntimeConfig, description="Runtime configuration"
    )
    providers: dict[str, ProviderConfig] = config_field(
        default_factory=dict, description="Available LLM providers"
    )
    agent: AgentConfig = config_field(
        default_factory=AgentConfig,
        description="Agent configuration and model policies",
    )
    prompt: PromptTemplateConfig = config_field(
        default_factory=PromptTemplateConfig,
        description="Prompt template configuration",
    )
    sandbox: SandboxConfig = config_field(
        default_factory=SandboxConfig,
        description="Sandbox configuration",
    )
    dashboard: DashboardConfig = config_field(
        default_factory=DashboardConfig,
        description="Dashboard configuration",
    )
    telemetry: TelemetryConfig = config_field(
        default_factory=TelemetryConfig,
        description="OpenTelemetry tracing configuration",
    )
    tools_platform: ToolsPlatformConfig = config_field(
        default_factory=ToolsPlatformConfig,
        description="Tools platform (tool gateway) configuration",
    )
    execution_store: ExecutionStoreConfig = config_field(
        default_factory=ExecutionStoreConfig,
        description="Durable execution store configuration",
    )
    repository: RepositoryConfig = config_field(
        default_factory=RepositoryConfig,
        description="Repository context service configuration",
    )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AppConfig:
        """Create AppConfig from a dictionary, handling nested models."""
        llm_config = LLMConfig(**config_dict.get("llm", {}))
        runtime_config = RuntimeConfig(**config_dict.get("runtime", {}))

        providers = {}
        for name, provider_data in config_dict.get("providers", {}).items():
            providers[name] = ProviderConfig.from_dict(provider_data)

        agent_config = AgentConfig.from_dict(config_dict.get("agent", {}))
        prompt_config = PromptTemplateConfig(**config_dict.get("prompt", {}))
        sandbox_config = SandboxConfig(**config_dict.get("sandbox", {}))
        dashboard_config = DashboardConfig(**config_dict.get("dashboard", {}))
        telemetry_config = TelemetryConfig(**config_dict.get("telemetry", {}))
        tools_platform_config = ToolsPlatformConfig(
            **config_dict.get("tools_platform", {})
        )
        execution_store_config = ExecutionStoreConfig(
            **config_dict.get("execution_store", {})
        )
        repository_config = RepositoryConfig(**config_dict.get("repository", {}))

        return cls(
            llm=llm_config,
            runtime=runtime_config,
            providers=providers,
            agent=agent_config,
            prompt=prompt_config,
            sandbox=sandbox_config,
            dashboard=dashboard_config,
            telemetry=telemetry_config,
            tools_platform=tools_platform_config,
            execution_store=execution_store_config,
            repository=repository_config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            "llm": self.llm.model_dump(exclude_unset=True),
            "runtime": self.runtime.model_dump(exclude_unset=True),
            "providers": {
                name: provider.model_dump(exclude_unset=True)
                for name, provider in self.providers.items()
            },
            "agent": self.agent.model_dump(exclude_unset=True),
            "prompt": self.prompt.model_dump(exclude_unset=True),
            "sandbox": self.sandbox.model_dump(exclude_unset=True),
            "dashboard": self.dashboard.model_dump(exclude_unset=True),
            "telemetry": self.telemetry.model_dump(exclude_unset=True),
            "tools_platform": self.tools_platform.model_dump(exclude_unset=True),
            "execution_store": self.execution_store.model_dump(exclude_unset=True),
            "repository": self.repository.model_dump(exclude_unset=True),
        }
