"""Lightweight utilities to resolve and render context blocks from stored snapshots.

Intended for read-only dashboard/historical use. Deliberately avoids importing
ContextManager (which pulls ToolGateway, IConfigManager, RepositoryContextService,
and session bundle caches) into the dashboard plane. Only the store reader,
profile models, PromptComposer, and runtime types are used.

project_info and repo_map_data are deserialized as plain dicts (PromptContext
accepts Any) to avoid importing heavy repository models (RepoMapData imports
CallGraph/RepositoryDatabase).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.schemas import PromptTemplateConfig
from local_coding_assistant.prompt import PromptComposer
from local_coding_assistant.runtime.context.profile import (
    ContextSnapshot,
    RuntimeContextConfig,
)
from local_coding_assistant.runtime.types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
    ToolSpec,
)

if TYPE_CHECKING:
    from local_coding_assistant.storage.execution_store_reader import (
        IExecutionStoreReader,
    )


@dataclass(slots=True)
class SnapshotRefs:
    """Lightweight resolved refs from a stored context snapshot."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    templates: dict[str, str] = field(default_factory=dict)
    memories: list[str] = field(default_factory=list)
    target_project_root: str = ""
    project_info: Any = None
    repo_map_data: Any = None
    tool_set_version: str = ""
    template_set_version: str = ""
    repo_snapshot_hash: str = ""
    memory_snapshot_version: str = ""


@dataclass(slots=True)
class SnapshotContextResult:
    """Rendered context blocks for display."""

    system_messages: list[str] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    system_reminder: str | None = None
    warnings: list[str] = field(default_factory=list)
    resolved: bool = True


async def _resolve_tools(
    store_reader: IExecutionStoreReader,
    tool_ids: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    tools: list[dict[str, Any]] = []
    if tool_ids:
        try:
            rows = await store_reader.get_tools_by_ids(tool_ids) or []
            tools = [dict(r) for r in rows]
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"tools: {exc}")
    return tools, warnings


async def _resolve_templates(
    store_reader: IExecutionStoreReader,
    template_ids: list[int],
) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    templates: dict[str, str] = {}
    if template_ids:
        try:
            rows = await store_reader.get_templates_by_ids(template_ids) or []
            for row in rows:
                templates[row.get("path", str(row.get("id", "")))] = row.get(
                    "content", ""
                )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"templates: {exc}")
    return templates, warnings


async def _resolve_repo(
    store_reader: IExecutionStoreReader,
    repo_snapshot_hash: str | None,
) -> tuple[str, Any, Any, list[str]]:
    warnings: list[str] = []
    target_project_root = ""
    project_info: Any = None
    repo_map_data: Any = None
    if repo_snapshot_hash:
        try:
            repo_data = await store_reader.get_repo_snapshot_by_hash(repo_snapshot_hash)
            if repo_data is not None:
                target_project_root = repo_data.target_project_root
                if repo_data.project_info_json:
                    try:
                        project_info = json.loads(repo_data.project_info_json)
                    except Exception:
                        project_info = None
                if repo_data.repo_map_json:
                    try:
                        repo_map_data = json.loads(repo_data.repo_map_json)
                    except Exception:
                        repo_map_data = None
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"repo: {exc}")
    return target_project_root, project_info, repo_map_data, warnings


async def _resolve_memories(
    store_reader: IExecutionStoreReader,
    memory_ids: list[str],
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    memories: list[str] = []
    if memory_ids:
        try:
            memories = await store_reader.get_memories_by_ids(memory_ids) or []
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"memories: {exc}")
    return memories, warnings


async def resolve_snapshot_refs(
    store_reader: IExecutionStoreReader,
    snapshot: ContextSnapshot,
) -> tuple[SnapshotRefs, list[str]]:
    """Resolve snapshot refs from the store, tolerantly.

    Missing refs produce warnings rather than raising, so the dashboard can still
    render partial context.
    """
    warnings: list[str] = []

    tools, w = await _resolve_tools(
        store_reader, snapshot.tool_set_ref.tool_ids if snapshot.tool_set_ref else []
    )
    warnings.extend(w)

    template_ids = (
        snapshot.template_set_ref.template_ids if snapshot.template_set_ref else []
    )
    templates, w = await _resolve_templates(store_reader, template_ids)
    warnings.extend(w)

    repo_hash = (
        snapshot.repo_snapshot_ref.repo_snapshot_hash
        if snapshot.repo_snapshot_ref
        else None
    )
    target_project_root, project_info, repo_map_data, w = await _resolve_repo(
        store_reader, repo_hash
    )
    warnings.extend(w)

    memory_ids = (
        snapshot.memory_snapshot_ref.memory_ids if snapshot.memory_snapshot_ref else []
    )
    memories, w = await _resolve_memories(store_reader, memory_ids)
    warnings.extend(w)

    refs = SnapshotRefs(
        tools=tools,
        templates=templates,
        memories=memories,
        target_project_root=target_project_root,
        project_info=project_info,
        repo_map_data=repo_map_data,
        tool_set_version=snapshot.tool_set_ref.version if snapshot.tool_set_ref else "",
        template_set_version=snapshot.template_set_ref.version
        if snapshot.template_set_ref
        else "",
        repo_snapshot_hash=snapshot.repo_snapshot_ref.repo_snapshot_hash
        if snapshot.repo_snapshot_ref
        else "",
        memory_snapshot_version=snapshot.memory_snapshot_ref.version
        if snapshot.memory_snapshot_ref
        else "",
    )
    return refs, warnings


def build_prompt_context(
    refs: SnapshotRefs,
    config_snapshot: RuntimeContextConfig,
    session_id: str,
    last_updated_at: int | None,
) -> PromptContext:
    """Build a prompt context from resolved refs + config (pure transform)."""

    execution_mode_raw = config_snapshot.execution_mode or "classic_tools"
    if execution_mode_raw == "classic":
        execution_mode_raw = "classic_tools"
    try:
        execution_mode = ExecutionMode(execution_mode_raw)
    except ValueError:
        execution_mode = ExecutionMode.REASONING_ONLY

    workspace_root = refs.target_project_root or ""
    cwd = config_snapshot.cwd or os.getcwd()
    platform = config_snapshot.platform or sys.platform
    is_git_repo = config_snapshot.is_git_repo
    current_date = last_updated_at or int(datetime.now(tz=UTC).timestamp() * 1000)

    tools = []
    for t in refs.tools:
        parameters: Any = {}
        schema_json = t.get("schema_json")
        if schema_json:
            try:
                parameters = (
                    json.loads(schema_json)
                    if isinstance(schema_json, str)
                    else schema_json
                )
            except Exception:
                parameters = {}
        tools.append(
            ToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=parameters,
            )
        )

    model_id = config_snapshot.llm_model or ""
    model_provider = config_snapshot.llm_provider or ""
    personality = config_snapshot.personality

    # AGENTS.md discovery driven by stored config
    agents_md_content: list[tuple[str, str]] = []
    if getattr(config_snapshot, "agents_md_enabled", False):
        agents_md_content = _discover_agents_md(
            cwd,
            workspace_root,
            list(getattr(config_snapshot, "agents_md_filenames", []) or []),
            bool(getattr(config_snapshot, "agents_md_walk_up", True)),
        )

    # Global rules from stored global AGENTS.md paths
    global_rules = ""
    if getattr(config_snapshot, "agents_md_enabled", False):
        global_rules = _load_global_rules(
            list(getattr(config_snapshot, "agents_md_global_paths", []) or [])
        )

    config_instructions = list(
        getattr(config_snapshot, "config_instructions", []) or []
    )

    metadata = {
        "session_id": session_id,
        "context_snapshot_id": refs.repo_snapshot_hash,
        "tool_set_version": refs.tool_set_version,
        "template_set_version": refs.template_set_version,
        "workspace_root": workspace_root,
        "platform": platform,
        "current_date": current_date,
    }

    return PromptContext(
        session_id=session_id,
        execution_mode=execution_mode,
        tool_call_mode=config_snapshot.tool_call_mode or "classic",
        agent_profile=AgentProfile.default(),
        active_skills=[],
        memories=refs.memories,
        tools=tools,
        metadata=metadata,
        model_id=model_id,
        model_provider=model_provider,
        personality=personality,
        project_info=refs.project_info,
        repo_map_data=refs.repo_map_data,
        examples=[],
        sandbox_enabled=bool(config_snapshot.sandbox_enabled),
        workspace_root=workspace_root,
        platform=platform,
        agents_md_content=agents_md_content,
        global_rules=global_rules,
        config_instructions=config_instructions,
        cwd=cwd,
        current_date=current_date,
        is_git_repo=is_git_repo,
    )


def _discover_agents_md(
    cwd: str,
    workspace_root: str,
    filenames: list[str],
    walk_up: bool,
) -> list[tuple[str, str]]:
    """Walk from cwd to workspace_root, return [(path, content)] closest first."""
    results: list[tuple[str, str]] = []
    current = Path(cwd).resolve()
    root = Path(workspace_root).resolve()

    while True:
        for fname in filenames:
            fpath = current / fname
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    results.append((str(fpath), content))
                except OSError:
                    continue
        if not walk_up or current == root or current == current.parent:
            break
        current = current.parent

    return results


def _load_global_rules(global_paths: list[str]) -> str:
    """Load and concatenate global AGENTS.md/CLAUDE.md files."""
    parts: list[str] = []
    for path in global_paths:
        expanded = os.path.expanduser(path)
        if not expanded or not os.path.exists(expanded):
            continue
        try:
            parts.append(Path(expanded).read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n\n---\n\n".join(parts)


def render_context_blocks(
    prompt_context: PromptContext,
    refs: SnapshotRefs,
) -> tuple[list[str], list[dict[str, Any]], str | None]:
    """Render system_messages and tool_schemas via PromptComposer.

    Returns (system_messages, tool_schemas, system_reminder).
    """
    render_config = PromptTemplateConfig(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    composer = PromptComposer(
        prompt_config=render_config,
        templates_dict=dict(refs.templates),
    )
    rendered = composer.render(
        prompt_context,
        history=[],
        user_input="",
        preset="default",
        session_context_resolution="static",
    )
    return (
        list(rendered.system_messages),
        list(rendered.tool_schemas),
        rendered.metadata.get("system_reminder"),
    )


async def render_snapshot_context(
    store_reader: IExecutionStoreReader,
    snapshot: ContextSnapshot,
    config_snapshot: RuntimeContextConfig,
    session_id: str,
    last_updated_at: int | None,
) -> SnapshotContextResult:
    """Resolve + render context blocks from a stored snapshot.

    Tolerant: missing refs produce warnings rather than raising.
    """
    refs, warnings = await resolve_snapshot_refs(store_reader, snapshot)
    prompt_context = build_prompt_context(
        refs, config_snapshot, session_id, last_updated_at
    )
    system_messages, tool_schemas, system_reminder = render_context_blocks(
        prompt_context, refs
    )
    return SnapshotContextResult(
        system_messages=system_messages,
        tool_schemas=tool_schemas,
        system_reminder=system_reminder,
        warnings=warnings,
        resolved=len(warnings) == 0,
    )
