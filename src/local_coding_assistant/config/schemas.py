"""Configuration schemas for the Local Coding Assistant."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


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


class AgentConfig(BaseModel):
    """Configuration for agent behavior and model policies."""

    policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Agent role to model fallback policies"
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

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AppConfig:
        """Create AppConfig from a dictionary, handling nested models."""
        llm_data = config_dict.get("llm", {})
        runtime_data = config_dict.get("runtime", {})
        providers_data = config_dict.get("providers", {})
        agent_data = config_dict.get("agent", {})

        # Convert providers to ProviderConfig objects
        providers = {}
        for name, provider_data in providers_data.items():
            providers[name] = ProviderConfig.from_dict(provider_data)

        return cls(
            llm=LLMConfig(**llm_data),
            runtime=RuntimeConfig(**runtime_data),
            providers=providers,
            agent=AgentConfig.from_dict(agent_data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert AppConfig to dictionary."""
        providers_dict = {
            name: provider.model_dump() for name, provider in self.providers.items()
        }
        return {
            "llm": self.llm.model_dump(),
            "runtime": self.runtime.model_dump(),
            "providers": providers_dict,
            "agent": self.agent.model_dump(),
        }
