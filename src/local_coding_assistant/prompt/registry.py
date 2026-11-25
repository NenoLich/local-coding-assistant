"""Prompt block registry system.

Declarative registry of prompt blocks with override/append/prepend/merge/skip
policies, priority ordering, and per-mode (static/dynamic) placement.
Preset registries group blocks per agent kind (default, compaction, subagent,
planner, executor).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_coding_assistant.runtime.types import PromptContext


class BlockPolicy(str, Enum):
    """Policy for combining blocks that share the same key."""

    OVERRIDE = "override"  # Replace previous block with same key
    APPEND = "append"  # Concatenate after previous (with separator)
    PREPEND = "prepend"  # Concatenate before previous
    MERGE = "merge"  # Deep merge for structured data
    SKIP_IF_EXISTS = "skip"  # Only render if no prior block with same key


class Preset(str, Enum):
    """Named preset block registries used by PromptComposer."""

    DEFAULT = "default"
    COMPACTION = "compaction"
    SUBAGENT = "subagent"
    PLANNER = "planner"
    EXECUTOR = "executor"


@dataclass
class PromptBlock:
    """A single renderable prompt block."""

    key: str
    template: str
    policy: BlockPolicy = BlockPolicy.APPEND
    condition: Callable[[PromptContext], bool] = field(
        default_factory=lambda: lambda ctx: True
    )
    priority: int = 0
    placement_static: str = "system"  # "system" | "user"
    placement_dynamic: str = "user"  # "system" | "user"


def _true(_ctx: PromptContext) -> bool:
    return True


def _has_provider(ctx: PromptContext) -> bool:
    return bool(ctx.model_provider)


def _has_personality(ctx: PromptContext) -> bool:
    return bool(ctx.personality)


def _has_sandbox(ctx: PromptContext) -> bool:
    return ctx.sandbox_enabled


def _has_agent_profile(ctx: PromptContext) -> bool:
    return ctx.agent_profile is not None


def _has_agents_md(ctx: PromptContext) -> bool:
    return bool(ctx.agents_md_content)


def _has_global_rules(ctx: PromptContext) -> bool:
    return bool(ctx.global_rules)


def _has_config_instructions(ctx: PromptContext) -> bool:
    return bool(ctx.config_instructions)


def _has_project_info(ctx: PromptContext) -> bool:
    return ctx.project_info is not None


def _has_repo_map(ctx: PromptContext) -> bool:
    return ctx.repo_map_data is not None


def _has_memories(ctx: PromptContext) -> bool:
    return bool(ctx.memories)


def _has_examples(ctx: PromptContext) -> bool:
    return bool(ctx.examples)


# DEFAULT — Main coding agent
DEFAULT_BLOCK_REGISTRY: list[PromptBlock] = [
    PromptBlock(
        "identity_base",
        "base/identity.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=10,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "model_specific",
        "base/model_{{model_provider}}.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=20,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_provider,
    ),
    PromptBlock(
        "personality",
        "base/personality.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=30,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_personality,
    ),
    PromptBlock(
        "static_env",
        "blocks/static_env.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=40,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "execution_mode",
        "modes/{{execution_mode}}.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=50,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "agent_profile",
        "agents/{{agent_kind}}.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=60,
        policy=BlockPolicy.APPEND,
        condition=_has_agent_profile,
    ),
    PromptBlock(
        "constraints",
        "blocks/constraints.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=100,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "safety",
        "blocks/safety.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=110,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_sandbox,
    ),
    PromptBlock(
        "output_format",
        "blocks/output_format.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=120,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    # Context blocks: system in static, user in dynamic
    PromptBlock(
        "agents_md",
        "blocks/agents_md.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=20,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_agents_md,
    ),
    PromptBlock(
        "global_rules",
        "blocks/global_rules.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=30,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_global_rules,
    ),
    PromptBlock(
        "config_overrides",
        "blocks/config_rules.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=40,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_config_instructions,
    ),
    PromptBlock(
        "repo_metadata",
        "blocks/repo_metadata.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=60,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_project_info,
    ),
    PromptBlock(
        "repo_map",
        "blocks/repo_map.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=70,
        policy=BlockPolicy.OVERRIDE,
        condition=_has_repo_map,
    ),
    PromptBlock(
        "memories",
        "blocks/memories.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=80,
        policy=BlockPolicy.APPEND,
        condition=_has_memories,
    ),
    PromptBlock(
        "examples",
        "blocks/examples.jinja2",
        placement_static="system",
        placement_dynamic="user",
        priority=90,
        policy=BlockPolicy.APPEND,
        condition=_has_examples,
    ),
]

# COMPACTION — Minimal prompt for conversation summarization
COMPACTION_BLOCK_REGISTRY: list[PromptBlock] = [
    PromptBlock(
        "identity_base",
        "base/identity.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=10,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "compaction_instructions",
        "agents/compaction.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=20,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "constraints",
        "blocks/constraints.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=100,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    # NO repo_context, NO skills, NO agent_profile, NO execution_mode
]

# SUBAGENT — Task-specific subagent
SUBAGENT_BLOCK_REGISTRY: list[PromptBlock] = [
    PromptBlock(
        "identity_base",
        "base/identity.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=10,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "subagent_instructions",
        "agents/subagent.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=20,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    PromptBlock(
        "constraints",
        "blocks/constraints.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=100,
        policy=BlockPolicy.OVERRIDE,
        condition=_true,
    ),
    # Relevant context passed via PromptContext fields (project_info subset, repo_map subset)
]

# PLANNER / EXECUTOR — Specialized variants
PLANNER_BLOCK_REGISTRY: list[PromptBlock] = [
    *DEFAULT_BLOCK_REGISTRY,
    PromptBlock(
        "planner_extra",
        "agents/planner_extra.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=65,
        policy=BlockPolicy.APPEND,
        condition=_true,
    ),
]

EXECUTOR_BLOCK_REGISTRY: list[PromptBlock] = [
    *DEFAULT_BLOCK_REGISTRY,
    PromptBlock(
        "executor_extra",
        "agents/executor_extra.jinja2",
        placement_static="system",
        placement_dynamic="system",
        priority=65,
        policy=BlockPolicy.APPEND,
        condition=_true,
    ),
]

PRESETS: dict[str, list[PromptBlock]] = {
    "default": DEFAULT_BLOCK_REGISTRY,
    "compaction": COMPACTION_BLOCK_REGISTRY,
    "subagent": SUBAGENT_BLOCK_REGISTRY,
    "planner": PLANNER_BLOCK_REGISTRY,
    "executor": EXECUTOR_BLOCK_REGISTRY,
}


def get_preset(name: str | Preset) -> list[PromptBlock]:
    """Return the block registry for the given preset name."""
    key = name.value if isinstance(name, Preset) else name
    return PRESETS.get(key, PRESETS["default"])


__all__ = [
    "COMPACTION_BLOCK_REGISTRY",
    "DEFAULT_BLOCK_REGISTRY",
    "EXECUTOR_BLOCK_REGISTRY",
    "PLANNER_BLOCK_REGISTRY",
    "PRESETS",
    "SUBAGENT_BLOCK_REGISTRY",
    "BlockPolicy",
    "Preset",
    "PromptBlock",
    "get_preset",
]
