"""PromptComposer - renders PromptContext into concrete LLM-ready payloads.

Implements the Phase 3 prompt redesign:
- Block registry presets (default, compaction, subagent, planner, executor)
- Dual placement: blocks render as system (static mode) or user (dynamic mode)
- Static mode system-prompt caching keyed on a hash of static context fields
- Template paths may contain jinja2 variables (e.g. ``modes/{{execution_mode}}.jinja2``)

Uses static PromptConfig instead of ConfigManager for deterministic replay.
Supports both FileSystemLoader (live) and DictLoader (replay).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import (
    DictLoader,
    Environment,
    FileSystemLoader,
    TemplateError,
    TemplateNotFound,
    UndefinedError,
    select_autoescape,
)

from local_coding_assistant.config.schemas import PromptTemplateConfig as PromptConfig
from local_coding_assistant.prompt.registry import (
    Preset,
    PromptBlock,
    get_preset,
)
from local_coding_assistant.runtime.types import (
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
)
from local_coding_assistant.utils.logging import get_logger

log = get_logger("prompt.composer")


class PromptComposer:
    """Render PromptContext objects into concrete LLM-ready payloads.

    Uses static PromptConfig for template resolution, enabling deterministic
    replay without live ConfigManager dependency.

    For live execution: provide template_root (Path) for FileSystemLoader
    For replay: provide templates_dict (dict[str, str]) for DictLoader
    """

    def __init__(
        self,
        *,
        prompt_config: PromptConfig,
        template_root: Path | None = None,
        templates_dict: dict[str, str] | None = None,
    ) -> None:
        if template_root is None and templates_dict is None:
            raise ValueError("Either template_root or templates_dict must be provided")
        if template_root is not None and templates_dict is not None:
            raise ValueError("Provide either template_root or templates_dict, not both")

        self.template_root = template_root
        self._templates_dict = templates_dict
        self._prompt_config = prompt_config
        self._env: Environment | None = None

        # Static-mode system prompt cache
        self._system_prompt_cache: str | None = None
        self._system_prompt_hash: str | None = None

    @property
    def config(self) -> PromptConfig:
        """Get the static prompt configuration."""
        return self._prompt_config

    @property
    def env(self) -> Environment:
        """Get or create the Jinja2 environment."""
        if self._env is None:
            self._env = self._create_environment()
        return self._env

    def _create_environment(self) -> Environment:
        """Create and configure the Jinja2 environment."""
        if self._templates_dict is not None:
            # Replay mode: use DictLoader with stored template content
            loader = DictLoader(self._templates_dict)
        else:
            # Live mode: use FileSystemLoader
            if self.template_root is None or not self.template_root.exists():
                raise ValueError(f"Template root does not exist: {self.template_root}")
            loader = FileSystemLoader(self.template_root)

        return Environment(
            loader=loader,
            autoescape=select_autoescape(["html", "xml", "htm"]),
            trim_blocks=self._prompt_config.trim_blocks,
            lstrip_blocks=self._prompt_config.lstrip_blocks,
        )

    def get_template_path(self, template_name: str) -> str:
        """Get the path for a template from config."""
        template_path = self._prompt_config.templates.get(template_name, template_name)
        if not isinstance(template_path, str):
            log.warning(
                "Template path for %s is not a string (got %s), using fallback",
                template_name,
                type(template_path).__name__,
            )
            return template_name
        return template_path

    def render(
        self,
        context: PromptContext,
        history: list[dict[str, Any]] | None = None,
        user_input: str = "",
        *,
        preset: str | Preset = Preset.DEFAULT,
        session_context_resolution: str = "static",
    ) -> RenderedPrompt:
        """Render the given context into discrete system/user messages.

        Args:
            context: PromptContext without history/user_input
            history: Conversation history to include in prompt
            user_input: Current user input (if any)
            preset: Block registry preset name (default, compaction, subagent, planner, executor)
            session_context_resolution: "static" (cache system prompt) or "dynamic" (fresh each turn)
        """
        registry = get_preset(preset)
        payload = self._build_payload(context)

        # 1. Render system blocks (cached in static mode)
        system_prompt = self._get_system_prompt(
            registry, context, payload, session_context_resolution
        )

        # 2. Render user blocks (dynamic per turn)
        user_parts = self._render_placement(
            registry, context, payload, "user", session_context_resolution
        )

        # 3. Build user messages: context blocks + actual user input
        user_messages = user_parts + ([user_input] if user_input.strip() else [])

        # 4. Tool schemas
        tool_schemas = [tool.to_openai_function() for tool in context.tools]

        system_reminder = self.build_system_reminder(context=context)
        if system_reminder:
            context.metadata.update({"system_reminder": system_reminder})

        log.debug(
            "Rendered prompt for session=%s preset=%s mode=%s system=%d history=%d user=%d",
            context.session_id,
            preset,
            session_context_resolution,
            1 if system_prompt else 0,
            len(history or []),
            len(user_messages),
        )

        return RenderedPrompt(
            system_messages=[system_prompt] if system_prompt else [],
            user_messages=user_messages,
            tool_schemas=tool_schemas,
            history=history or [],
            metadata=context.metadata,
        )

    def _get_system_prompt(
        self,
        registry: list[PromptBlock],
        context: PromptContext,
        payload: dict[str, Any],
        mode: str,
    ) -> str:
        """Return the system prompt, using the static cache when applicable."""
        if mode == "static":
            static_hash = self._compute_static_hash(context)
            if self._system_prompt_cache and self._system_prompt_hash == static_hash:
                return self._system_prompt_cache

            system_parts = self._render_placement(
                registry, context, payload, "system", "static"
            )
            system_prompt = "\n\n".join(system_parts)
            self._system_prompt_cache = system_prompt
            self._system_prompt_hash = static_hash
            return system_prompt

        # Dynamic mode: system blocks only, rendered fresh each turn
        system_parts = self._render_placement(
            registry, context, payload, "system", "dynamic"
        )
        return "\n\n".join(system_parts)

    def _render_placement(
        self,
        registry: list[PromptBlock],
        context: PromptContext,
        payload: dict[str, Any],
        placement: str,
        mode: str,
    ) -> list[str]:
        """Render blocks for given placement (system/user) and mode (static/dynamic)."""
        placement_attr = f"placement_{mode}"
        blocks = [
            b
            for b in registry
            if getattr(b, placement_attr) == placement and b.condition(context)
        ]
        blocks.sort(key=lambda b: b.priority)

        rendered = []
        for block in blocks:
            resolved_path = self._resolve_template_path(block.template, payload)
            content = self._render_template(resolved_path, payload)
            if content:
                rendered.append(content)
        return rendered

    def _build_payload(self, context: PromptContext) -> dict[str, Any]:
        """Build the jinja2 payload, exposing context fields for template path resolution."""
        execution_mode_value = (
            context.execution_mode.value
            if isinstance(context.execution_mode, ExecutionMode)
            else str(context.execution_mode)
        )
        agent_kind = context.agent_profile.kind if context.agent_profile else "default"
        return {
            "context": context,
            "ctx": context,
            "model_id": context.model_id,
            "model_provider": context.model_provider,
            "execution_mode": execution_mode_value,
            "agent_kind": agent_kind,
            "personality": context.personality,
        }

    def _resolve_template_path(
        self, template_name: str, payload: dict[str, Any]
    ) -> str:
        """Resolve a template path that may contain jinja2 variables.

        Block templates such as ``modes/{{execution_mode}}.jinja2`` or
        ``base/model_{{model_provider}}.jinja2`` are rendered as jinja2
        strings first to produce the concrete path, then loaded and rendered.
        """
        try:
            return self.env.from_string(template_name).render(**payload)
        except Exception:
            log.warning(
                "Failed to resolve template path %r, using as-is", template_name
            )
            return template_name

    def _render_templates(
        self, template_names: list[str], payload: dict[str, Any]
    ) -> list[str]:
        """Render multiple templates and return non-empty results."""
        return [
            rendered
            for template_name in template_names
            if (rendered := self._render_template(template_name, payload))
        ]

    def _render_template(self, template_name: str, payload: dict[str, Any]) -> str:
        """Render a single template with the given payload."""
        try:
            template = self.env.get_template(template_name)
            return template.render(**payload).strip()
        except TemplateNotFound:
            log.warning("Prompt template %s not found", template_name)
            return ""
        except UndefinedError as e:
            log.warning("Undefined variable in template %s: %s", template_name, str(e))
            return ""
        except TemplateError as e:
            log.error("Template error in %s: %s", template_name, str(e), exc_info=True)
            return ""
        except Exception as e:
            log.error(
                "Unexpected error rendering template %s: %s",
                template_name,
                str(e),
                exc_info=True,
            )
            return ""

    def _resolve_mode_template(self, execution_mode: ExecutionMode | str) -> str:
        """Resolve the template path for the given execution mode."""
        if isinstance(execution_mode, ExecutionMode):
            return self.get_template_path(execution_mode.template_name)

        try:
            mode = ExecutionMode(execution_mode)
        except ValueError:
            mode = ExecutionMode.REASONING_ONLY
        return self.get_template_path(mode.template_name)

    def _compute_static_hash(self, context: PromptContext) -> str:
        """Compute a hash over static (session-level) context fields for cache invalidation."""
        static_fields = {
            "session_id": context.session_id,
            "model_id": context.model_id,
            "model_provider": context.model_provider,
            "personality": context.personality,
            "workspace_root": context.workspace_root,
            "platform": context.platform,
            "agents_md_content": context.agents_md_content,
            "global_rules": context.global_rules,
            "config_instructions": context.config_instructions,
            "project_info": context.project_info,
            "repo_map_data": context.repo_map_data,
        }
        payload = json.dumps(static_fields, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build_system_reminder(self, context: PromptContext) -> str:
        """Build the ephemeral system-reminder block.

        Contains:
        - mode reminder when in planning (reasoning_only) mode
        - sandbox notice when sandbox is enabled
        - current_date and cwd when cwd differs from workspace_root

        The result is not part of canonical history; it is injected as a tail
        of the last user/tool message for the current API call only.
        """
        parts: list[str] = []
        if context.execution_mode == ExecutionMode.REASONING_ONLY:
            parts.append(
                "You are in PLANNING MODE. Analyze and plan only; do not execute tools."
            )
        if context.sandbox_enabled:
            parts.append("Sandbox is enabled. File/network operations are restricted.")
        if context.cwd and context.cwd != context.workspace_root:
            parts.append(f"Working directory: {context.cwd}")

        if not parts:
            return ""
        else:
            parts.append(f"Current date: {context.current_date}")

        return f"<system-reminder>{' '.join(parts)}</system-reminder>"


__all__ = ["PromptComposer"]
