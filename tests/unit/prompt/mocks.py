"""Mock implementations for prompt testing."""

from typing import Any


class MockPromptComposer:
    """Mock implementation of PromptComposer for testing."""

    def __init__(self, template_env=None):
        self.template_env = template_env
        self.last_compose_args = {}

    def compose(
        self,
        system_core: str | None = None,
        agent_identity: str | None = None,
        execution_rules: str | None = None,
        constraints: list[str] | None = None,
        skills: list[str] | None = None,
        tools: list[str] | None = None,
        examples: list[str] | None = None,
        memories: list[str] | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Mock compose method that stores arguments and returns a simple string."""
        self.last_compose_args = {
            "system_core": system_core,
            "agent_identity": agent_identity,
            "execution_rules": execution_rules,
            "constraints": constraints or [],
            "skills": skills or [],
            "tools": tools or [],
            "examples": examples or [],
            "memories": memories or [],
            "context": context,
            **kwargs,
        }

        if context is not None:
            if self.template_env is None:
                raise ValueError("template_env is required when composing repo context")
            template = self.template_env.get_template("blocks/repo_context.jinja2")
            return template.render(context=context)

        # Simple string representation of the prompt
        parts = [
            f"# System Core\n{system_core}\n",
            f"# Agent Identity\n{agent_identity}\n",
        ]

        if execution_rules:
            parts.append(f"# Execution Rules\n{execution_rules}\n")

        if constraints:
            parts.append(
                "# Constraints\n" + "\n".join(f"- {c}" for c in constraints) + "\n"
            )

        if skills:
            parts.append("# Skills\n" + "\n".join(f"- {s}" for s in skills) + "\n")

        if tools:
            parts.append("# Tools\n" + "\n".join(f"- {t}" for t in tools) + "\n")

        if examples:
            parts.append("# Examples\n" + "\n".join(f"- {e}" for e in examples) + "\n")

        if memories:
            parts.append("# Memories\n" + "\n".join(f"- {m}" for m in memories) + "\n")

        return "\n".join(parts)
