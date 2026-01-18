"""Configuration schemas for the Local Coding Assistant."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from local_coding_assistant.tools.types import (
    ToolCategory,
    ToolInfo,
    ToolPermission,
    ToolSource,
    ToolTag,
)


class ProviderStatus(BaseModel):
    """Status information for a provider."""

    name: str = Field(description="Provider name")
    is_alive: bool = Field(
        default=True,
        description="Whether the provider is currently available and healthy",
    )
    last_health_check: float = Field(
        default_factory=lambda: 0.0, description="Unix timestamp of last health check"
    )
    error_count: int = Field(default=0, description="Number of consecutive errors")


class LLMConfig(BaseModel):
    """Configuration for LLM provider and model settings."""

    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    max_retries: int = Field(
        default=3,
        gt=0,
        description="Maximum number of retry attempts for failed requests",
    )
    retry_delay: float = Field(
        default=1.0, gt=0.0, description="Base delay in seconds between retry attempts"
    )
    providers: list[ProviderStatus] = Field(
        default_factory=list,
        description="List of available providers with their status",
    )

    def with_overrides(self, **overrides) -> LLMConfig:
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


class RuntimeConfig(BaseModel):
    """Configuration for runtime behavior and settings."""

    persistent_sessions: bool = Field(
        default=False, description="Whether to maintain persistent sessions"
    )
    max_session_history: int = Field(
        default=100,
        gt=0,
        description="Maximum number of messages to keep in session history",
    )
    enable_logging: bool = Field(
        default=True, description="Whether to enable runtime logging"
    )
    log_level: str | None = Field(
        default="INFO", description="Logging level for runtime operations"
    )
    use_graph_mode: bool = Field(
        default=False,
        description="Whether to use LangGraph-based agent instead of legacy AgentLoop",
    )
    stream: bool = Field(
        default=True, description="Default streaming mode for agent responses"
    )
    tool_call_mode: str = Field(
        default="classic",
        description="Tool calling mode: 'classic' (default) uses function calling, 'ptc' uses programmatic tool calling, 'reasoning_only' uses no tool calls",
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


class ModelConfig(BaseModel):
    """Configuration for a specific model within a provider."""

    name: str = Field(description="Model identifier")
    supported_parameters: list[str] = Field(
        default_factory=list,
        description="List of supported parameter names for this model",
    )

    @classmethod
    def from_dict(cls, name: str, config: dict) -> ModelConfig:
        """Create ModelConfig from a model name and its configuration."""
        return cls(
            name=name, supported_parameters=config.get("supported_parameters", [])
        )


class ProviderConfig(BaseModel):
    """Configuration for LLM providers."""

    name: str = Field(description="Provider name")
    driver: str = Field(
        description="Driver type (openai_chat, openai_responses, local)"
    )
    base_url: str = Field(description="Base URL for the provider API")
    api_key_env: str | None = Field(
        default=None, description="Environment variable name for API key"
    )
    models: list[ModelConfig] = Field(
        default_factory=list, description="List of available model configurations"
    )
    health_check_endpoint: str | None = Field(
        default=None, description="Endpoint to check provider health"
    )
    health_check_method: str = Field(
        default="GET", description="HTTP method to use for health checks"
    )
    health_check_timeout: float = Field(
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


class AgentProfileConfig(BaseModel):
    """Configuration for an agent profile."""

    name: str = Field(..., description="Unique identifier for the profile")
    kind: str = Field("default", description="Type/category of the profile")
    description: str = Field(
        ..., description="Description of the agent's role and behavior"
    )
    goals: list[str] = Field(
        default_factory=list,
        description="List of primary objectives for this agent profile",
    )
    tone: str | None = Field(
        None, description="Communication style and tone for this profile"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="List of constraints or guidelines for this profile",
    )
    model_policy: str | None = Field(
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


class AgentConfig(BaseModel):
    """Configuration for agent behavior and model policies."""

    # Agent profiles define different behaviors and configurations
    profiles: dict[str, AgentProfileConfig] = Field(
        default_factory=dict,
        description=(
            "Available agent profiles. Each profile defines a different "
            "personality, goals, and behavior for the agent."
        ),
    )

    # Model policies for different agent roles
    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Agent role to model fallback policies. "
            "Each key is a role name, and the value is a dict containing 'models' list."
        ),
    )

    # Default fallback policies if YAML file is missing
    planner: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "models": [
                "openrouter:qwen/qwen3-coder:free",
                "google_gemini:gemini-2.5-flash",
                "fallback:any",
            ]
        },
        description="Default planner model fallback policy",
    )

    researcher: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "models": [
                "google_gemini:gemini-2.5-flash",
                "openrouter:qwen/qwen3-235b-a22b:free",
                "fallback:any",
            ]
        },
        description="Default researcher model fallback policy",
    )

    analyzer: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "models": [
                "openrouter:moonshotai/kimi-dev-72b:free",
                "google_gemini:gemini-2.5-flash",
                "fallback:any",
            ]
        },
        description="Default analyzer model fallback policy",
    )

    general: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "models": ["google_gemini:gemini-2.5-flash", "fallback:any"]
        },
        description="Default general purpose model fallback policy",
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
        if name == "planner":
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
            return self.policies[profile.model_policy].get("models", [])

        # Then try the profile name as a policy
        if profile_name in self.policies:
            return self.policies[profile_name].get("models", [])

        # Fall back to the 'general' policy
        if "general" in self.policies:
            return self.policies["general"].get("models", [])

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
            return self.policies[role].get("models", [])
        elif hasattr(self, role):
            return getattr(self, role)["models"]
        else:
            return self.general["models"]


class ToolConfig(BaseModel):
    """Configuration for a tool (YAML/JSON representation).

    This is the configuration model used for loading tool definitions from
    configuration files. It can be converted to a ToolInfo object for runtime use.
    """

    id: str = Field(..., description="Unique identifier for the tool")
    name: str | None = Field(
        None, description="Display name of the tool (defaults to id if not provided)"
    )
    description: str = Field(..., description="Description of what the tool does")
    source: ToolSource | str = Field(
        ToolSource.EXTERNAL, description="Source of the tool implementation"
    )
    path: str | None = Field(
        None,
        description=(
            "Path to the Python module containing the tool implementation. "
            "Can be relative to the config directory or absolute."
        ),
    )
    module: str | None = Field(
        None,
        description=(
            "Python import path to the module containing the tool implementation. "
            "Alternative to path, used for installed packages."
        ),
    )
    tool_class: type | str | None = Field(
        None,
        description=(
            "The tool class type. This should be set by the ToolLoader "
            "when loading the tool module. Can be a string class name or actual type."
        ),
    )

    @field_validator("tool_class", mode="before")
    @classmethod
    def validate_tool_class(cls, v):
        if v is None or isinstance(v, (type, str)):
            return v
        return str(v)  # Convert any non-None, non-type value to string

    endpoint: str | None = Field(
        None,
        description="Endpoint URL for remote tools (MCP tools)",
    )
    provider: str | None = Field(
        None,
        description="Provider name for MCP or external tools",
    )
    category: ToolCategory | str | None = Field(
        None,
        description="Category for organizing tools (e.g., 'math', 'network')",
    )
    permissions: list[ToolPermission | str] = Field(
        default_factory=list, description="List of permissions required by this tool"
    )
    tags: list[ToolTag | str] = Field(
        default_factory=list, description="Tags for categorizing and filtering tools"
    )
    enabled: bool = Field(
        True, description="Whether this tool is enabled and should be loaded"
    )
    available: bool = Field(
        True,
        description="Whether this tool was successfully loaded and is ready to use",
    )
    is_async: bool = Field(False, description="Whether the tool's run method is async")
    supports_streaming: bool = Field(
        False, description="Whether the tool supports streaming output"
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific configuration options"
    )
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description=(
            "OpenAI-compatible parameter schema for the tool. "
            "Defines the expected input parameters in the format: "
            '{"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]}'
        ),
    )

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v):
        if isinstance(v, str):
            try:
                return ToolSource(v.lower())
            except ValueError:
                return v
        return v

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v):
        if v is None or isinstance(v, ToolCategory):
            return v
        try:
            return ToolCategory(v.lower())
        except ValueError:
            return v

    @field_validator("permissions", mode="before")
    @classmethod
    def validate_permissions(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v):
        if v is None:
            return {"type": "object", "properties": {}, "required": []}
        if not isinstance(v, dict):
            return {"type": "object", "properties": {}, "required": []}

        # Ensure required fields exist
        if "type" not in v:
            v["type"] = "object"
        if "properties" not in v:
            v["properties"] = {}
        if "required" not in v:
            v["required"] = []

        return v

    @field_validator("available", mode="before")
    @classmethod
    def validate_available(cls, v, info):
        # If tool is disabled, it cannot be available
        if not info.data.get("enabled", True):
            return False
        return v if v is not None else True

    def to_tool_info(self) -> ToolInfo:
        """Convert this configuration to a runtime ToolInfo object.

        Note: tool_class should be converted to a proper type by the ToolLoader before calling this method.
        """
        # Ensure available is False if tool is disabled
        available = self.available and self.enabled

        # Ensure tool_class is either a type or None
        tool_class = self.tool_class if isinstance(self.tool_class, type) else None

        # Convert category to ToolCategory enum if it's a string
        category = (
            ToolCategory(self.category.lower())
            if isinstance(self.category, str)
            else self.category
        )

        # Ensure parameters has the correct structure
        parameters = self.parameters or {}
        if not isinstance(parameters, dict):
            parameters = {}

        # Ensure required fields exist in parameters
        if "type" not in parameters:
            parameters["type"] = "object"
        if "properties" not in parameters:
            parameters["properties"] = {}
        if "required" not in parameters:
            parameters["required"] = []

        return ToolInfo(
            name=self.id,
            tool_class=tool_class,
            description=self.description,
            category=category,
            source=self.source,
            permissions=self.permissions,
            tags=self.tags,
            is_async=self.is_async,
            supports_streaming=self.supports_streaming,
            enabled=self.enabled,
            available=available,
            endpoint=self.endpoint,
            provider=self.provider,
            config=self.config,
            parameters=parameters,
        )

    def model_dump(self, *args, **kwargs):
        """Custom dump to handle enums properly."""
        data = super().model_dump(*args, **kwargs)

        # Convert enums to their values
        if "source" in data and isinstance(data["source"], ToolSource):
            data["source"] = data["source"].value

        if "category" in data and isinstance(data["category"], ToolCategory):
            data["category"] = data["category"].value

        if "permissions" in data:
            data["permissions"] = [
                p.value if isinstance(p, ToolPermission) else p
                for p in data["permissions"]
            ]

        if "tags" in data:
            data["tags"] = [
                t.value if isinstance(t, ToolTag) else t for t in data["tags"]
            ]

        return data


class ToolConfigList(BaseModel):
    """Wrapper for a list of tool configurations."""

    tools: list[ToolConfig] = Field(
        default_factory=list, description="List of tool configurations"
    )

    @classmethod
    def from_dict(cls, data: dict | list[Any]) -> ToolConfigList:
        """Create ToolConfigList from a dictionary or a list of ToolConfig objects.

        Args:
            data: Either a dict with a 'tools' key containing a list of tool configs,
                or a list of ToolConfig objects or dicts.
        """
        if isinstance(data, dict):
            if "tools" not in data:
                return cls(tools=[])
            tools_data = data["tools"]
        else:
            tools_data = data

        tools = []
        for tool in tools_data:
            if isinstance(tool, ToolConfig):
                tools.append(tool)
            elif isinstance(tool, dict):
                tools.append(ToolConfig(**tool))
            else:
                raise ValueError(
                    f"Expected ToolConfig or dict, got {type(tool).__name__}"
                )

        return cls(tools=tools)

    def to_dict(self) -> dict:
        """Convert ToolConfigList to dictionary."""
        return {"tools": [tool.model_dump(exclude_unset=True) for tool in self.tools]}


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates and rendering."""

    template_root: str | None = Field(
        None,
        description=(
            "Root directory for prompt templates. "
            "If not provided, will use package defaults."
        ),
    )
    templates: dict[str, str] = Field(
        default_factory=lambda: {
            "system": "base/system_core.jinja2",
            "execution_rules": "base/execution_rules.jinja2",
            "agent_identity": "base/agent_identity.jinja2",
            "skills": "blocks/skills.jinja2",
            "memories": "blocks/memories.jinja2",
            "tools": "blocks/tools.jinja2",
            "examples": "blocks/examples.jinja2",
            "constraints": "blocks/constraints.jinja2",
        },
        description="Template paths for different prompt sections",
    )
    enable_jinja_autoescape: bool = Field(
        False,
        description="Whether to enable Jinja2 autoescaping (usually not needed for LLM prompts)",
    )
    trim_blocks: bool = Field(
        True, description="Whether to trim whitespace around template blocks"
    )
    lstrip_blocks: bool = Field(
        True, description="Whether to strip whitespace from the start of lines"
    )


class SandboxLoggingConfig(BaseModel):
    """Configuration for sandbox logging."""

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    console: bool = Field(default=True, description="Whether to enable console logging")
    file: bool = Field(default=True, description="Whether to enable file logging")
    directory: str = Field(
        default="/workspace/logs",
        description="Directory to store log files (inside container)",
    )
    file_name: str = Field(
        default="container_{session_id}.log",
        description="Log file name pattern. {session_id} will be replaced with the actual session ID.",
    )
    max_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes before rotation",
        gt=0,
    )
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep", ge=0
    )


class SandboxConfig(BaseModel):
    """Configuration for the Docker sandbox environment."""

    enabled: bool = Field(
        default=False, description="Whether to enable sandbox execution"
    )
    image: str = Field(
        default="locca-sandbox:latest",
        description="Docker image to use for the sandbox",
    )
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    memory_limit: str = Field(
        default="512m", description="Memory limit for the container (e.g., 512m, 1g)"
    )
    cpu_limit: float = Field(default=0.5, description="CPU limit (fraction of one CPU)")
    network_enabled: bool = Field(
        default=False, description="Whether to allow network access in the sandbox"
    )
    persistence: bool = Field(
        default=False, description="Whether to persist the sandbox container"
    )
    session_id: str = Field(default="default", description="Session ID for the sandbox")
    allowed_imports: list[str] = Field(
        default_factory=list, description="List of allowed Python top-level imports"
    )
    blocked_patterns: list[str] = Field(
        default_factory=list, description="List of regex patterns to block in code"
    )
    blocked_shell_commands: list[str] = Field(
        default_factory=list, description="List of blocked shell commands"
    )
    max_code_length: int = Field(
        default=10000, description="Maximum allowed code length in characters"
    )
    max_file_size: int = Field(
        default=1024 * 1024, description="Maximum allowed file size for created files"
    )
    allowed_directories: list[str] = Field(
        default_factory=list, description="List of allowed directories for file access"
    )
    working_directory: str = Field(
        default="/workspace", description="Working directory inside the container"
    )
    session_timeout: int = Field(
        default=300, description="Session timeout in seconds (default: 5 minutes)"
    )
    max_sessions: int = Field(
        default=5, description="Maximum number of concurrent persistent sessions"
    )
    logging: SandboxLoggingConfig = Field(
        default_factory=SandboxLoggingConfig,
        description="Logging configuration for the sandbox",
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Runtime configuration"
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Available LLM providers"
    )
    agent: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration and model policies",
    )
    tools: ToolConfigList = Field(
        default_factory=ToolConfigList,
        description="Tool configurations",
    )
    prompt: PromptTemplateConfig = Field(
        default_factory=PromptTemplateConfig,
        description="Prompt template configuration",
    )
    sandbox: SandboxConfig = Field(
        default_factory=SandboxConfig,
        description="Sandbox configuration",
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
        tool_config = ToolConfigList.from_dict(config_dict.get("tools", {}))
        sandbox_config = SandboxConfig(**config_dict.get("sandbox", {}))

        return cls(
            llm=llm_config,
            runtime=runtime_config,
            providers=providers,
            agent=agent_config,
            tools=tool_config,
            sandbox=sandbox_config,
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
            "tools": self.tools.to_dict(),
            "sandbox": self.sandbox.model_dump(exclude_unset=True),
        }
