from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.runtime.runtime_types import (
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
)
from local_coding_assistant.utils.logging import get_logger

log = get_logger("prompt.composer")


class PromptComposer:
    """Render PromptContext objects into concrete LLM-ready payloads."""

    def __init__(
        self,
        config_manager: IConfigManager,
        *,
        template_root: Path | None = None,
    ) -> None:
        if config_manager is None:
            raise ValueError("config_manager is required for PromptComposer")

        self._config_manager = config_manager
        self.template_root = self._resolve_template_root(template_root)
        self._env = None

    @property
    def config(self) -> Any:
        """Get the current prompt configuration with session overrides applied."""
        return self._config_manager.global_config.prompt

    @property
    def env(self) -> Environment:
        """Get or create the Jinja2 environment."""
        if self._env is None:
            self._env = self._create_environment()
        return self._env

    def _create_environment(self) -> Environment:
        """Create and configure the Jinja2 environment."""
        return Environment(
            loader=FileSystemLoader(self.template_root),
            autoescape=select_autoescape(["html", "xml", "htm"]),
            trim_blocks=self.config.trim_blocks,
            lstrip_blocks=self.config.lstrip_blocks,
        )

    def get_template_path(self, template_name: str) -> str:
        """Get the path for a template from config."""
        return self.config.templates.get(template_name, template_name)

    def render(self, context: PromptContext) -> RenderedPrompt:
        """Render the given context into discrete system/user messages."""
        payload = {"context": context, "ctx": context}

        system_messages = self._render_system_messages(payload)
        user_messages = self._render_user_messages(payload)

        log.debug(
            "Rendered prompt for session=%s system=%d user=%d",
            context.session_id,
            len(system_messages),
            len(user_messages),
        )

        return RenderedPrompt(
            system_messages=system_messages,
            user_messages=user_messages,
            tool_schemas=list(context.tools),
            metadata=context.metadata,
        )

    def _render_system_messages(self, payload: dict[str, Any]) -> list[str]:
        """Render all system message templates including context blocks."""
        context: PromptContext = payload["context"]
        mode_template = self._resolve_mode_template(context.execution_mode)

        # Core system templates
        system_templates = [
            self.get_template_path("system"),
            self.get_template_path("execution_rules"),
            self.get_template_path("agent_identity"),
            mode_template,  # Add execution mode template to system
        ]

        # Add optional blocks to system context
        blocks = [
            (self.get_template_path("skills"), bool(context.active_skills)),
            (self.get_template_path("memories"), bool(context.memories)),
            (self.get_template_path("tools"), bool(context.tools)),
            (self.get_template_path("examples"), bool(context.examples)),
            (self.get_template_path("constraints"), True),  # Always include constraints
        ]

        # Add block templates that should be rendered
        for template_name, should_render in blocks:
            if should_render:
                system_templates.append(template_name)

        return self._render_templates(system_templates, payload)

    def _render_user_messages(self, payload: dict[str, Any]) -> list[str]:
        """Render just the user's input as the user message."""
        context: PromptContext = payload["context"]
        return [context.user_input] if context.user_input.strip() else []

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

    def _resolve_template_root(self, override: Path | None) -> Path:
        if override is not None:
            return Path(override)

        path_manager = getattr(self._config_manager, "path_manager", None)
        if path_manager is not None:
            return path_manager.get_template_dir()

        # Fallback to package-relative resolution (should be rare)
        return Path(__file__).resolve().parent / "templates"

    def _resolve_mode_template(self, execution_mode: ExecutionMode | str) -> str:
        """Resolve the template path for the given execution mode."""
        if isinstance(execution_mode, ExecutionMode):
            return self.get_template_path(execution_mode.template_name)

        try:
            mode = ExecutionMode(execution_mode)
        except ValueError:
            mode = ExecutionMode.REASONING_ONLY
        return self.get_template_path(mode.template_name)


__all__ = ["PromptComposer"]
