"""Routing helpers built on top of :class:`ProviderRouter`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_coding_assistant.agent.llm.models import LLMPolicy, ResolvedLLMOptions
from local_coding_assistant.providers import ProviderResolver
from local_coding_assistant.providers.base import BaseProvider, ProviderLLMRequest
from local_coding_assistant.providers.health import ProviderHealthManager


@dataclass(slots=True)
class RoutingDecision:
    """Provider/model pairing chosen for a request."""

    provider: BaseProvider
    model: str


class PolicyResolver:
    """Resolve policy definitions from configuration."""

    def __init__(self, config_manager):
        self._config_manager = config_manager

    def _normalize_policy_data(self, policy_data: Any) -> dict[str, Any]:
        """Normalize policy data to ensure correct types."""
        if not isinstance(policy_data, dict):
            policy_data = {}

        models = policy_data.get("models")
        if not isinstance(models, list):
            models = ["fallback:any"]
        else:
            models = [str(m) for m in models]

        max_failovers = policy_data.get("max_failovers")
        if not isinstance(max_failovers, int):
            max_failovers = None

        fallback_strategy = policy_data.get("fallback_strategy")
        if not isinstance(fallback_strategy, str):
            fallback_strategy = "sequential"

        return {
            "models": models,
            "max_failovers": max_failovers,
            "fallback_strategy": fallback_strategy,
        }

    def resolve(self, policy_name: str | None) -> LLMPolicy:
        agent_config = self._config_manager.global_config.agent
        fallback_name = policy_name or "general"

        # Try to get the policy from policies dict first
        policy_obj = agent_config.policies.get(fallback_name)

        # Convert ModelPolicyConfig to dict if needed
        if policy_obj and hasattr(policy_obj, "model_dump"):
            policy_data = policy_obj.model_dump()
        elif policy_obj and hasattr(policy_obj, "__dict__"):
            policy_data = {
                "models": getattr(policy_obj, "models", []),
                "max_failovers": getattr(policy_obj, "max_failovers", None),
                "fallback_strategy": getattr(
                    policy_obj, "fallback_strategy", "sequential"
                ),
            }
        else:
            policy_data = None

        # If not found, try to get from hardcoded defaults (for backward compatibility)
        if policy_data is None and hasattr(agent_config, fallback_name):
            policy_data = getattr(agent_config, fallback_name)

        # If still not found, use general policy as ultimate fallback
        if policy_data is None:
            general_policy = agent_config.policies.get("general")
            if general_policy and hasattr(general_policy, "model_dump"):
                policy_data = general_policy.model_dump()
            else:
                policy_data = {"models": ["fallback:any"]}

        # Normalize the policy data
        normalized = self._normalize_policy_data(policy_data)

        routes = normalized["models"]
        max_failovers_value = normalized["max_failovers"]
        if isinstance(max_failovers_value, int):
            max_failovers = min(max_failovers_value, len(routes))
        else:
            max_failovers = len(routes)
        fallback_strategy = normalized["fallback_strategy"]

        return LLMPolicy(
            name=fallback_name,
            routes=routes,
            max_failovers=max_failovers,
            fallback_strategy=fallback_strategy,
        )


class ProviderSelector:
    """Encapsulates provider selection with policies."""

    def __init__(self, provider_manager, health_manager: ProviderHealthManager):
        self._resolver = ProviderResolver(provider_manager, health_manager)

    async def select(
        self,
        request: ProviderLLMRequest,
        *,
        options: ResolvedLLMOptions,
        routes: list[str] | None,
    ) -> tuple[RoutingDecision, list[str]]:
        """Select provider and model based on options or policy routes.

        Args:
            request: The provider request
            options: Resolved options containing model/provider hints
            routes: Unchecked routes for a using policy to find model from

        Returns:
            RoutingDecision with selected provider and model and unchecked routes for future selection
        """
        checked_routes: list[str] = []
        # Use policy routes with attempts as index
        if routes:
            from local_coding_assistant.providers import ProviderNotFoundError

            for route in routes:
                # Parse the route to extract provider and model
                if route == "fallback:any":
                    # Use resolver's fallback logic
                    provider, model = await self._resolver.find_any_available_provider(
                        request
                    )
                else:
                    # Route is just a model name, use provider hint if available
                    request_with_model = request.model_copy(update={"model": route})

                    try:
                        if options.provider_hint:
                            (
                                provider,
                                model,
                            ) = await self._resolver.resolve_provider_and_model(
                                options.provider_hint,
                                request_with_model.model,
                                request_with_model,
                            )
                        else:
                            provider, model = await self._resolver.resolve_model_only(
                                request_with_model.model, request_with_model
                            )

                    except ProviderNotFoundError:
                        continue
                    finally:
                        checked_routes.append(route)

                unchecked_routes = [
                    route for route in routes if route not in checked_routes
                ]
                return RoutingDecision(provider=provider, model=model), unchecked_routes

        # No policy or routes - use resolver's default logic based on options
        if options.model and options.provider_hint:
            provider, model = await self._resolver.resolve_provider_and_model(
                options.provider_hint, options.model, request
            )
        elif options.provider_hint:
            provider, model = await self._resolver.resolve_provider_only(
                options.provider_hint, request
            )
        elif options.model:
            provider, model = await self._resolver.resolve_model_only(
                options.model, request
            )
        else:
            provider, model = await self._resolver.find_any_available_provider(request)
        return RoutingDecision(provider=provider, model=model), []
