"""Data models for the LLM orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.schemas import LLMConfig
from local_coding_assistant.core.exceptions import LLMContentError

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Safely convert tool arguments to dictionaries."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            import json

            return json.loads(arguments)
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMContentError(f"Failed parsing tool arguments: {exc}") from exc
    if hasattr(arguments, "model_dump"):
        return arguments.model_dump()
    raise LLMContentError(f"Unsupported tool argument type: {type(arguments)}")


@dataclass(slots=True)
class LLMToolCall:
    """Normalized representation of a tool call emitted by the provider."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    type: str = "function"


@dataclass(slots=True)
class LLMTask:
    """Domain-level request for language model generation."""

    prompt: str
    context: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_provider_request(
        self,
        *,
        model: str,
        stream: bool,
        temperature: float,
    ) -> ProviderLLMRequest:
        """Convert task into ProviderLLMRequest for downstream providers."""
        from local_coding_assistant.providers.base import (
            OptionalParameters,
            ProviderLLMRequest,
        )

        # Simple structure - just system + prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add history
        if self.context:
            for message in self.context:
                if isinstance(message, dict):
                    # Filter out None values to handle optional fields
                    filtered_message = {
                        k: v for k, v in message.items() if v is not None
                    }
                    messages.append(filtered_message)

        if self.prompt:
            messages.append({"role": "user", "content": self.prompt})

        optional_params = OptionalParameters(stream=stream, include_usage=True)
        if self.tools:
            optional_params.tools = self.tools
            optional_params.tool_choice = "auto"

        return ProviderLLMRequest(
            messages=messages,
            model=model or "auto",
            temperature=temperature,
            parameters=optional_params,
        )


@dataclass(slots=True)
class LLMOptions:
    """User-supplied options for an LLM call."""

    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    retry_attempts: int | None = None
    retry_delay: float | None = None
    stream: bool | None = None
    response_format: str | None = None
    tool_choice: str | None = None
    policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved(
        self,
        *,
        defaults: LLMConfig,
        runtime_stream: bool,
        max_failovers: int | None = None,
    ) -> ResolvedLLMOptions:
        """Merge explicit overrides with config defaults."""
        retry_attempts = (
            self.retry_attempts
            if self.retry_attempts is not None
            else defaults.max_retries
        )
        retry_delay = (
            self.retry_delay if self.retry_delay is not None else defaults.retry_delay
        )
        temperature = (
            self.temperature if self.temperature is not None else defaults.temperature
        )
        max_tokens = (
            self.max_tokens if self.max_tokens is not None else defaults.max_tokens
        )
        if self.model and self.model.lower() in ["any", "default", "auto", ""]:
            model = defaults.model_name or "auto"
        else:
            model = self.model or defaults.model_name or "auto"

        return ResolvedLLMOptions(
            provider_hint=self.provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            stream=(self.stream if self.stream is not None else runtime_stream),
            response_format=self.response_format,
            tool_choice=self.tool_choice,
            policy_name=self.policy,
            metadata=self.metadata,
            max_failovers=max_failovers or max(retry_attempts, 1),
        )


@dataclass(slots=True)
class ResolvedLLMOptions:
    """Concrete options derived from config defaults and overrides."""

    provider_hint: str | None
    model: str | None
    temperature: float
    max_tokens: int | None
    retry_attempts: int
    retry_delay: float
    stream: bool
    response_format: str | None
    tool_choice: str | None
    policy_name: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_failovers: int = 1


@dataclass(slots=True)
class LLMPolicy:
    """Represents a routing policy defined in configuration."""

    name: str
    routes: list[str] | None = None
    max_failovers: int | None = None
    fallback_strategy: str = "sequential"

    def preferred_route(self) -> str | None:
        """Return the first viable route entry."""
        return self.routes[0] if self.routes else None

    def allowed_failovers(self, default: int) -> int:
        """Return the maximum number of failovers permitted for this policy."""
        if self.max_failovers is not None:
            return max(1, self.max_failovers)
        return max(1, default)


@dataclass(slots=True)
class LLMResult:
    """Normalized response returned to the rest of the application."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    reasoning: str | None = None
    reasoning_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMStreamChunk:
    """Represents a chunk of streaming content."""

    content: str
    provider: str
    model: str
    finish_reason: str | None = None
    reasoning: str | None = None
    is_final: bool = False
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_coding_assistant.providers.base import ProviderLLMRequest
