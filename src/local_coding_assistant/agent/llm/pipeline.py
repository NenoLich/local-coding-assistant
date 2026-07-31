"""Pipeline utilities for LLM request preparation and response normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_coding_assistant.agent.llm.models import (
    LLMResult,
    LLMStreamChunk,
    LLMTask,
    LLMToolCall,
    ResolvedLLMOptions,
    _normalize_tool_arguments,
)
from local_coding_assistant.core.exceptions import LLMContentError
from local_coding_assistant.providers.base import (
    OptionalParameters,
    ProviderLLMRequest,
    ProviderLLMResponse,
    ProviderLLMResponseDelta,
)


def _extract_usage_metrics(
    usage: dict[str, Any] | None, tokens_used: int | None
) -> tuple[int | None, int | None, int | None]:
    if not usage:
        return None, None, tokens_used

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens
    if total_tokens is None:
        total_tokens = tokens_used
    return prompt_tokens, completion_tokens, total_tokens


def _extract_reasoning_tokens(usage: dict[str, Any] | None) -> int | None:
    if not usage:
        return None

    # Check both possible keys for token details
    details_keys = ["completion_tokens_details", "output_tokens_details"]
    for details_key in details_keys:
        details = usage.get(details_key)
        if isinstance(details, dict):
            tokens = details.get("reasoning_tokens")
            if tokens is not None:
                try:
                    return int(tokens)
                except (TypeError, ValueError):
                    continue
    return None


def _normalize_tool_calls(raw_calls: list[dict[str, Any]] | None) -> list[LLMToolCall]:
    tool_calls: list[LLMToolCall] = []
    if not raw_calls:
        return tool_calls

    for raw_call in raw_calls:
        function_payload = raw_call.get("function") or raw_call
        name = function_payload.get("name", "unknown")
        args = _normalize_tool_arguments(function_payload.get("arguments"))
        extra_content = raw_call.get("extra_content") or {}
        tool_calls.append(
            LLMToolCall(
                id=raw_call.get("id"),
                name=name,
                arguments=args,
                type=raw_call.get("type", "function"),
                extra_content=extra_content,
            )
        )
    return tool_calls


@dataclass(slots=True)
class GenerationContext:
    """Carries data shared across pipeline stages."""

    task: LLMTask
    resolved_options: ResolvedLLMOptions
    provider_request: ProviderLLMRequest
    streaming: bool


class ProviderRequestBuilder:
    """Transforms an :class:`LLMTask` into a provider-ready request."""

    def build(
        self,
        task: LLMTask,
        *,
        options: ResolvedLLMOptions,
        stream: bool,
    ) -> ProviderLLMRequest:
        """Generate a ``ProviderLLMRequest`` with common overrides applied."""

        request = task.build_provider_request(
            model=options.model or "default",
            stream=stream,
            temperature=options.temperature if options.temperature is not None else 0.7,
        )

        params: OptionalParameters = request.parameters or OptionalParameters()
        params.stream = stream

        if options.max_tokens is not None:
            params.max_tokens = options.max_tokens

        if options.response_format is not None:
            params.response_format = {"type": options.response_format}

        if options.tool_choice is not None:
            params.tool_choice = options.tool_choice

        request.parameters = params

        if options.model:
            request.model = options.model

        return request


class ResponseNormalizer:
    """Converts provider responses into :class:`LLMResult` objects."""

    def to_result(
        self,
        response: ProviderLLMResponse,
        *,
        provider_name: str,
        policy_name: str | None,
    ) -> LLMResult:
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_metrics(
            response.usage, response.tokens_used
        )

        metadata: dict[str, Any] = {
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "provider_metadata": response.metadata or {},
        }
        if policy_name:
            metadata["policy"] = policy_name
        reasoning_tokens = _extract_reasoning_tokens(response.usage)
        if reasoning_tokens is not None:
            metadata["reasoning_tokens"] = reasoning_tokens

        result = LLMResult(
            content=response.content,
            model=response.model,
            provider=provider_name,
            finish_reason=response.finish_reason,
            reasoning=(response.metadata or {}).get("reasoning"),
            reasoning_tokens=reasoning_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )
        try:
            tool_calls = _normalize_tool_calls(response.tool_calls)
            result.tool_calls = tool_calls
        except LLMContentError as exc:
            # Create partial LLMResult with content error info
            # This preserves raw response data for handler context
            result.metadata.update(
                {
                    "content_error": {
                        "error_type": "tool_call_parsing_error",
                        "message": str(exc),
                        "raw_response": response.content,  # Full raw response
                        "tool_calls_attempted": response.tool_calls,  # What we tried to parse
                    }
                }
            )

        return result


class StreamingNormalizer:
    """Converts streaming deltas into :class:`LLMStreamChunk` entries."""

    def to_chunk(
        self,
        delta: ProviderLLMResponseDelta,
        *,
        provider_name: str,
        model_name: str,
    ) -> LLMStreamChunk:
        stream_chunk = LLMStreamChunk(
            content=delta.content or "",
            provider=provider_name,
            model=model_name,
            finish_reason=delta.finish_reason,
            reasoning=delta.reasoning,
            is_final=delta.finish_reason is not None,
            usage=(delta.metadata or {}).get("usage"),
            metadata=delta.metadata or {},
        )
        try:
            tool_calls = _normalize_tool_calls(delta.tool_calls)
            stream_chunk.tool_calls = tool_calls
        except LLMContentError as exc:
            # Create stream chunk with content error info
            stream_chunk.metadata.update(
                {
                    "content_error": {
                        "error_type": "tool_call_parsing_error",
                        "message": str(exc),
                        "tool_calls_attempted": delta.tool_calls,  # What we tried to parse
                    }
                }
            )

        return stream_chunk
