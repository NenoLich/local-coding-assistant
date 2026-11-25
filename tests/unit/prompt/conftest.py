"""Fixtures for rendering real PromptComposer output against golden files.

The prompt module is tested with the *real* ``PromptComposer`` (not a mock):
inbound is a ``PromptContext`` plus render settings, outbound is the
``RenderedPrompt`` compared against golden files.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_coding_assistant.config.path_manager import PathManager
from local_coding_assistant.config.schemas import PromptTemplateConfig as PromptConfig
from local_coding_assistant.prompt.composer import PromptComposer
from local_coding_assistant.runtime.types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
    ToolSpec,
)


@pytest.fixture(scope="session")
def templates_dir() -> Path:
    """The package prompt templates directory (FileSystemLoader source)."""
    return PathManager(is_testing=True).get_template_dir()


@pytest.fixture
def prompt_config(templates_dir: Path) -> PromptConfig:
    """Static prompt config pointing at the real templates."""
    return PromptConfig(template_root=str(templates_dir))


@pytest.fixture
def composer(prompt_config: PromptConfig, templates_dir: Path) -> PromptComposer:
    """Real PromptComposer wired to the on-disk templates."""
    return PromptComposer(prompt_config=prompt_config, template_root=templates_dir)


@pytest.fixture
def make_context() -> Callable[[dict[str, Any]], PromptContext]:
    """Build a PromptContext from a plain dict (JSON-friendly test data)."""

    def build(data: dict[str, Any]) -> PromptContext:
        agent_profile = None
        if data.get("agent_profile"):
            agent_profile = AgentProfile(**data["agent_profile"])
        tools = [ToolSpec(**t) for t in data.get("tools", [])]
        return PromptContext(
            session_id=data["session_id"],
            execution_mode=ExecutionMode(data.get("execution_mode", "classic_tools")),
            tool_call_mode=data.get("tool_call_mode", "classic"),
            agent_profile=agent_profile,
            active_skills=list(data.get("active_skills", [])),
            memories=list(data.get("memories", [])),
            tools=tools,
            metadata=dict(data.get("metadata", {})),
            model_id=data.get("model_id", ""),
            model_provider=data.get("model_provider", ""),
            personality=data.get("personality"),
            project_info=data.get("project_info"),
            repo_map_data=data.get("repo_map_data"),
            examples=list(data.get("examples", [])),
            workspace_root=data.get("workspace_root", ""),
            platform=data.get("platform", ""),
            agents_md_content=[tuple(x) for x in data.get("agents_md_content", [])],
            global_rules=data.get("global_rules", ""),
            config_instructions=list(data.get("config_instructions", [])),
            cwd=data.get("cwd", ""),
            current_date=int(data.get("current_date", 0)),
            is_git_repo=bool(data.get("is_git_repo", False)),
            sandbox_enabled=bool(data.get("sandbox_enabled", False)),
        )

    return build


@pytest.fixture
def render_prompt(
    composer: PromptComposer, make_context
) -> Callable[..., RenderedPrompt]:
    """Render a context dict into a RenderedPrompt."""

    def render(context_data: dict[str, Any], **render_kwargs: Any) -> RenderedPrompt:
        context = make_context(context_data)
        return composer.render(context, **render_kwargs)

    return render


def serialize_rendered(rendered: RenderedPrompt) -> str:
    """Deterministic text rendering of a RenderedPrompt for golden comparison."""
    parts: list[str] = ["=== SYSTEM MESSAGES ==="]
    for i, message in enumerate(rendered.system_messages, 1):
        parts.append(f"--- system {i} ---")
        parts.append(message)
    parts.append("=== USER MESSAGES ===")
    for i, message in enumerate(rendered.user_messages, 1):
        parts.append(f"--- user {i} ---")
        parts.append(message)
    parts.append("=== TOOL SCHEMAS ===")
    parts.append(json.dumps(rendered.tool_schemas, indent=2, sort_keys=True))
    parts.append("=== METADATA ===")
    parts.append(json.dumps(rendered.metadata, indent=2, sort_keys=True, default=str))
    return "\n".join(parts) + "\n"


def pytest_addoption(parser):
    """Add custom command line options for tests."""
    # Only add the option if it doesn't already exist (avoid conflicts with
    # tests/unit/utils/conftest.py which registers the same option).
    if not any(opt.dest == "update_goldens" for opt in parser._anonymous.options):
        parser.addoption(
            "--update-goldens",
            action="store_true",
            default=False,
            help="Update golden files instead of testing against them",
        )
