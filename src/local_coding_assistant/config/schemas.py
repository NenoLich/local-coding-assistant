"""Configuration schemas for the Local Coding Assistant."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM provider and model settings."""

    model_name: str = Field(
        default="gpt-5-mini", description="Name of the LLM model to use"
    )
    provider: str = Field(
        default="openai", description="LLM provider (openai, local, etc.)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    api_key: str | None = Field(default=None, description="API key for the provider")

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
        default=False, description="Whether to use streaming mode for graph execution"
    )


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Runtime configuration"
    )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AppConfig:
        """Create AppConfig from a dictionary, handling nested models."""
        llm_data = config_dict.get("llm", {})
        runtime_data = config_dict.get("runtime", {})

        return cls(llm=LLMConfig(**llm_data), runtime=RuntimeConfig(**runtime_data))

    def to_dict(self) -> dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {"llm": self.llm.model_dump(), "runtime": self.runtime.model_dump()}
