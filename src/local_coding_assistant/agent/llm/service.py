"""High-level LLM service coordinating routing, retries, and normalization."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from local_coding_assistant.core.exceptions import (
    LLMError,
)
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.providers import ProviderManager
from local_coding_assistant.providers.health import ProviderHealthManager
from local_coding_assistant.utils.logging import get_logger

from .fallback import FallbackStrategy, get_strategy
from .models import LLMOptions, LLMResult, LLMStreamChunk, LLMTask, ResolvedLLMOptions
from .pipeline import (
    GenerationContext,
    ProviderRequestBuilder,
    ResponseNormalizer,
    StreamingNormalizer,
    _extract_usage_metrics,
)
from .routing import PolicyResolver, ProviderSelector
from .telemetry import TelemetryEmitter


class LLMService:
    """Primary entry point for orchestrating LLM generations."""

    def __init__(
        self,
        config_manager: IConfigManager,
        provider_manager: ProviderManager | None = None,
        *,
        health_manager: ProviderHealthManager | None = None,
    ) -> None:
        self._logger = get_logger("agent.llm.service")
        self._config_manager = config_manager
        self._provider_manager = provider_manager or ProviderManager()
        self._provider_manager.reload(config_manager)

        self._health_manager = health_manager or ProviderHealthManager(config_manager)
        self._policy_resolver = PolicyResolver(config_manager)
        self._provider_selector = ProviderSelector(
            self._provider_manager, self._health_manager
        )

        self._request_builder = ProviderRequestBuilder()
        self._response_normalizer = ResponseNormalizer()
        self._streaming_normalizer = StreamingNormalizer()
        self._telemetry = TelemetryEmitter()

        # Provider status cache
        self._provider_status_cache: dict[str, dict[str, Any]] = {}
        self._last_health_check: float = 0
        self._cache_ttl: float = 30.0 * 60.0  # 30 minutes cache TTL
        self._background_tasks = []

    @property
    def provider_manager(self) -> ProviderManager:
        """Access to provider manager"""
        return self._provider_manager

    async def generate(
        self,
        task: LLMTask,
        *,
        policy: str | None = None,
        options: LLMOptions | None = None,
    ) -> LLMResult:
        """Generate a full response without streaming."""

        resolved_options = self._resolve_options(options, policy, stream=False)
        base_request = self._request_builder.build(
            task,
            options=resolved_options,
            stream=False,
        )
        context = GenerationContext(
            task=task,
            resolved_options=resolved_options,
            provider_request=base_request,
            streaming=False,
        )
        return await self._run_generation(context)

    async def stream(
        self,
        task: LLMTask,
        *,
        policy: str | None = None,
        options: LLMOptions | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Yield streaming chunks for a task."""

        resolved_options = self._resolve_options(options, policy, stream=True)
        base_request = self._request_builder.build(
            task,
            options=resolved_options,
            stream=True,
        )
        context = GenerationContext(
            task=task,
            resolved_options=resolved_options,
            provider_request=base_request,
            streaming=True,
        )
        async for chunk in self._run_stream(context):
            yield chunk

    async def _run_generation(self, context: GenerationContext) -> LLMResult:
        attempts = 0
        last_error: Exception | None = None

        if (
            context.resolved_options.model
            and context.resolved_options.model.lower() == "auto"
        ):
            policy = self._policy_resolver.resolve(context.resolved_options.policy_name)
            unchecked_routes = policy.routes
            max_attempts = policy.allowed_failovers(
                context.resolved_options.max_failovers
            )
        else:
            policy = None
            unchecked_routes = []
            max_attempts = max(1, context.resolved_options.max_failovers)

        strategy: FallbackStrategy = get_strategy(policy)

        while strategy.should_continue(attempts, max_attempts=max_attempts):
            attempts += 1
            cloned_request = self._clone_request(context.provider_request)
            try:
                decision, unchecked_routes = await self._provider_selector.select(
                    cloned_request,
                    options=context.resolved_options,
                    routes=unchecked_routes,
                )
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                break

            if policy:
                policy_name = policy.name
            else:
                policy_name = "no policy"
            provider_name = getattr(decision.provider, "name", "unknown")
            cloned_request.model = getattr(decision, "model", cloned_request.model)
            self._logger.debug(
                "Invoking provider", provider=provider_name, model=cloned_request.model
            )
            self._telemetry.attempt_start(
                mode="generate",
                attempt=attempts,
                provider=provider_name,
                model=cloned_request.model,
                policy=policy_name,
                task_metadata=context.task.metadata,
                options_metadata=context.resolved_options.metadata,
            )

            try:
                response = await decision.provider.generate_with_retry(
                    cloned_request,
                    max_retries=max(0, context.resolved_options.retry_attempts - 1),
                    retry_delay=context.resolved_options.retry_delay,
                )
                self._health_manager.mark_provider_success(provider_name)
                _, _, total_tokens = _extract_usage_metrics(
                    response.usage, response.tokens_used
                )

                self._telemetry.attempt_success(
                    mode="generate",
                    attempt=attempts,
                    provider=provider_name,
                    model=response.model,
                    tokens_used=total_tokens,
                )
                return self._response_normalizer.to_result(
                    response,
                    provider_name=provider_name,
                    policy_name=policy_name,
                )

            except Exception as exc:
                last_error = exc
                self._telemetry.attempt_failure(
                    mode="generate",
                    attempt=attempts,
                    provider=provider_name,
                    model=cloned_request.model,
                    error=exc,
                )
                # Infrastructure errors SHOULD penalize provider health
                self._health_manager.mark_provider_failure(provider_name, exc)

                delay = (
                    strategy.next_delay(
                        attempts, base_delay=context.resolved_options.retry_delay
                    )
                    if strategy
                    else context.resolved_options.retry_delay
                )
                await asyncio.sleep(delay)

        self._telemetry.failover_exhausted(
            mode="generate",
            attempts=attempts,
            error=last_error,
        )
        raise LLMError(
            "LLM generation failed after exhausting failover attempts"
        ) from last_error

    async def _run_stream(
        self, context: GenerationContext
    ) -> AsyncIterator[LLMStreamChunk]:
        attempts = 0
        last_error: Exception | None = None

        if (
            context.resolved_options.model
            and context.resolved_options.model.lower() == "auto"
        ):
            policy = self._policy_resolver.resolve(context.resolved_options.policy_name)
            unchecked_routes = policy.routes
            max_attempts = policy.allowed_failovers(
                context.resolved_options.max_failovers
            )
        else:
            policy = None
            unchecked_routes = []
            max_attempts = max(1, context.resolved_options.max_failovers)

        strategy: FallbackStrategy = get_strategy(policy)

        while strategy.should_continue(attempts, max_attempts=max_attempts):
            attempts += 1
            cloned_request = self._clone_request(context.provider_request)
            try:
                decision, unchecked_routes = await self._provider_selector.select(
                    cloned_request,
                    options=context.resolved_options,
                    routes=unchecked_routes,
                )
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                break

            if policy:
                policy_name = policy.name
            else:
                policy_name = "no policy"
            provider_name = getattr(decision.provider, "name", "unknown")
            cloned_request.model = getattr(decision, "model", cloned_request.model)
            self._logger.debug(
                "Invoking provider", provider=provider_name, model=cloned_request.model
            )
            self._telemetry.attempt_start(
                mode="stream",
                attempt=attempts,
                provider=provider_name,
                model=cloned_request.model,
                policy=policy_name,
                task_metadata=context.task.metadata,
                options_metadata=context.resolved_options.metadata,
            )

            last_usage: dict[str, Any] | None = None
            try:
                async for delta in decision.provider.stream_with_retry(
                    cloned_request,
                    max_retries=max(0, context.resolved_options.retry_attempts - 1),
                    retry_delay=context.resolved_options.retry_delay,
                ):
                    if delta.metadata and "usage" in delta.metadata:
                        last_usage = delta.metadata["usage"]
                    yield self._streaming_normalizer.to_chunk(
                        delta,
                        provider_name=provider_name,
                        model_name=cloned_request.model,
                    )
                    # self._telemetry.stream_chunk(
                    #     provider=provider_name,
                    #     model=cloned_request.model,
                    #     is_final=delta.finish_reason is not None,
                    # )
                self._health_manager.mark_provider_success(provider_name)
                _, _, total_tokens = _extract_usage_metrics(last_usage, None)
                self._telemetry.attempt_success(
                    mode="stream",
                    attempt=attempts,
                    provider=provider_name,
                    model=cloned_request.model,
                    tokens_used=total_tokens,
                )
                return

            except Exception as exc:
                last_error = exc
                self._telemetry.attempt_failure(
                    mode="stream",
                    attempt=attempts,
                    provider=provider_name,
                    model=cloned_request.model,
                    error=exc,
                )
                # Infrastructure errors SHOULD penalize provider health
                self._health_manager.mark_provider_failure(provider_name, exc)

                delay = (
                    strategy.next_delay(
                        attempts, base_delay=context.resolved_options.retry_delay
                    )
                    if strategy
                    else context.resolved_options.retry_delay
                )
                await asyncio.sleep(delay)

        self._telemetry.failover_exhausted(
            mode="stream",
            attempts=attempts,
            error=last_error,
        )
        raise LLMError(
            "LLM streaming failed after exhausting failover attempts"
        ) from last_error

    def _resolve_options(
        self,
        options: LLMOptions | None,
        policy: str | None,
        *,
        stream: bool,
    ) -> ResolvedLLMOptions:
        candidate = options or LLMOptions()
        if policy is not None:
            candidate = replace(candidate, policy=policy)

        llm_defaults = self._config_manager.global_config.llm
        return candidate.resolved(defaults=llm_defaults, runtime_stream=stream)

    @staticmethod
    def _clone_request(base_request):
        """Create a deep copy of the provider request for isolated attempts."""
        return base_request.model_copy(deep=True)

    def get_provider_status_list(self) -> list[dict[str, Any]]:
        """Get cached provider status list without triggering health checks.

        Returns:
            List of provider status dictionaries with name, source, status, and models
        """
        current_time = time.time()

        # Check if cache is still valid
        if current_time - self._last_health_check > self._cache_ttl:
            # Cache expired, refresh it synchronously if we're in an async context
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # We're in an async context, schedule the refresh
                task = asyncio.create_task(self._refresh_provider_status_cache())
                self._background_tasks.append(task)
                task.add_done_callback(self._background_tasks.remove)
            except RuntimeError:
                # No async context, refresh synchronously
                asyncio.run(self._refresh_provider_status_cache())

        # Get the current list of providers from the provider manager
        provider_names = self._provider_manager.list_providers()

        # Convert cached status to CLI format
        status_list = []
        for provider_name in provider_names:
            # Always get the current source directly from the provider manager
            source = (
                self._provider_manager.get_provider_source(provider_name) or "unknown"
            )

            # Get the cached status if it exists
            cached_status = self._provider_status_cache.get(provider_name, {})

            # Determine provider status
            if cached_status.get("status") == "not_implemented":
                status = "config"
                error = "Configured but no implementation"
                models = 0
            else:
                status = (
                    "available"
                    if cached_status.get("healthy", False)
                    else "unavailable"
                )
                error = cached_status.get("error")
                models = len(cached_status.get("models", []))

            # Add provider to status list
            status_list.append(
                {
                    "name": provider_name,
                    "source": source,
                    "status": status,
                    "models": models,
                    "error": error,
                }
            )

        return status_list

    def reload_providers(self):
        """Reload providers from configuration."""
        self._logger.info("Reloading providers through LLM service")
        self._provider_manager.reload(self._config_manager)
        # Clear cache to force refresh on next access
        self._provider_status_cache.clear()
        self._last_health_check = 0

    async def get_provider_status(
        self, provider_name: str | None = None
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """Get status information for all providers, or a specific provider if name is provided."""
        status = {}

        for p_name in self._provider_manager.list_providers():
            provider = self._provider_manager.get_provider(p_name)
            if provider:
                try:
                    health_status = await provider.health_check()

                    # Handle the case where health check is not available
                    if health_status == "unavailable":
                        status[p_name] = {
                            "name": p_name,
                            "healthy": False,
                            "models": provider.get_available_models(),
                            "in_unhealthy_set": False,  # Not unhealthy, just not checkable
                            "status": "health_check_not_configured",
                        }
                    else:
                        is_healthy = bool(health_status)
                        status[p_name] = {
                            "name": p_name,
                            "healthy": is_healthy,
                            "models": provider.get_available_models(),
                            "in_unhealthy_set": p_name
                            in self._health_manager.get_unhealthy_providers(),
                            "status": "healthy" if is_healthy else "unhealthy",
                        }
                except Exception as e:
                    self._logger.warning(
                        f"Error checking health for {p_name}",
                        error=str(e),
                        exc_info=True,
                    )
                    status[p_name] = {
                        "name": p_name,
                        "healthy": False,
                        "models": provider.get_available_models() if provider else [],
                        "error": str(e),
                        "in_unhealthy_set": p_name
                        in self._health_manager.get_unhealthy_providers(),
                        "status": "error",
                    }
            else:
                # Config-only provider
                status[p_name] = {
                    "name": p_name,
                    "healthy": False,
                    "models": [],
                    "error": "Configured but no implementation",
                    "in_unhealthy_set": True,
                    "status": "not_implemented",
                }

        # Update cache
        self._provider_status_cache = status
        self._last_health_check = time.time()

        if provider_name is not None:
            return status.get(provider_name, {})
        return status

    async def _refresh_provider_status_cache(self):
        """Refresh the provider status cache."""
        self._logger.debug("Refreshing provider status cache")
        self._provider_status_cache = await self.get_provider_status()
        self._last_health_check = time.time()
