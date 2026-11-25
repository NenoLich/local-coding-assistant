"""Unified ContextManager for all execution regimes.

Single context manager that works for live, resume, replay, and branch execution.
No mode differentiation - behavior determined by available services and config.

Session Context Policy (from config_manager.global_config.runtime.session_context_resolution):
- static: reuse resolved context bundle across turns in the session
- dynamic: resolve fresh bundle for each turn

Invalidation:
- Config changes invalidate session config cache (via clear_session_config)
- Source changes (tools/templates/repo/memory) invalidate session context bundle cache
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_coding_assistant.config.schemas import PromptTemplateConfig
from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.prompt import PromptComposer
from local_coding_assistant.repository import RepositoryContextService
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.repository.repo_map import RepoMapData
from local_coding_assistant.runtime.context.profile import (
    ContextProfile,
    ContextResolutionMode,
    ContextSnapshot,
    MemorySnapshotRef,
    MessageBoundary,
    RepoSnapshotRef,
    RuntimeContextConfig,
    TemplateSetRef,
    ToolSetRef,
)
from local_coding_assistant.runtime.context.provider import MemoryProvider
from local_coding_assistant.runtime.context.resolution import (
    ResolvedRefs,
    SessionContextBundle,
)
from local_coding_assistant.runtime.resume.exceptions import (
    MissingContextReferencesError,
)

if TYPE_CHECKING:
    from local_coding_assistant.runtime.session import Message, SessionState
    from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.runtime.types import (
    AgentProfile,
    ExecutionMode,
    PromptContext,
    RenderedPrompt,
    ToolSpec,
)
from local_coding_assistant.storage.execution_store import (
    compute_tool_set_version as _compute_stored_tool_set_version,
)
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.context_manager")


class ContextManager:
    """Unified context resolution for live, resume, replay, and branch execution.

    Responsibilities:
    - Build/maintain ContextProfile per session (static/dynamic policy)
    - Create ContextSnapshot for new turns (captures profile + message_boundary)
    - Load snapshot + resolve refs for resume/replay
    - Resolve PromptContext + RenderedPrompt per frame
    - Cache management (session profile, session bundle, turn composer)

    Caches:
    - _session_bundles: session_id -> SessionContextBundle (static mode only)
    - _composer_cache: turn_id -> PromptComposer
    """

    def __init__(
        self,
        config_manager: IConfigManager,
        tool_gateway: ToolGateway | None,
        repo_service: RepositoryContextService | None,
        memory_provider: MemoryProvider | None = None,
        store: IExecutionStore | None = None,
    ):
        self._config_mgr = config_manager
        self._gateway = tool_gateway
        self._repo = repo_service
        self._memory = memory_provider or MemoryProvider()
        self._template_root = config_manager.path_manager.get_template_dir()
        self._store = store

        # Caches
        self._session_bundles: dict[str, SessionContextBundle] = {}
        self._composer_cache: dict[str, PromptComposer] = {}

        # Phase 4: dynamic context caches
        self._agents_md_cache: dict[str, list[tuple[str, str]]] = {}
        self._global_rules_cache: str = ""
        self._last_agents_md_hash: str = ""
        self._last_repo_map_hash: str = ""

    # ========== Session Context Profile & Bundle Management ==========

    async def get_session_bundle(self, session_id: str) -> SessionContextBundle:
        """Get cached session context bundle (static mode only)."""
        bundle = self._session_bundles.get(session_id)
        if not bundle:
            bundle = await self.update_bundle_cache(session_id)
            self._session_bundles[session_id] = bundle

        return bundle

    def get_resolution_mode(self) -> ContextResolutionMode:
        """Current session context resolution mode (static/dynamic)."""
        return ContextResolutionMode(
            self._config_mgr.global_config.runtime.session_context_resolution
        )

    async def resolve_profile_refs(self, profile: ContextProfile) -> ResolvedRefs:
        """Resolve a stored profile's refs from the store, no cache side effects.

        Used by resume validation to recompute version hashes over the rows
        the profile pins. Raises MissingContextReferencesError if a pinned
        ref no longer resolves.
        """
        return await self._resolve_all_refs_from_profile(profile)

    async def get_context_config(self, session_id: str) -> RuntimeContextConfig:
        """Get session's context config from SessionState."""
        bundle = self._session_bundles.get(session_id)
        if not bundle:
            bundle = await self.update_bundle_cache(session_id)
            self._session_bundles[session_id] = bundle

        return bundle.profile.runtime_context_config

    def get_context_profile(self, session: SessionState) -> ContextProfile | None:
        """Get session's context profile from SessionState."""
        return session.context_profile

    async def update_bundle_cache(
        self,
        session_id: str,
        profile: ContextProfile | None = None,
        config: RuntimeContextConfig | None = None,
    ) -> SessionContextBundle:
        """Update or create session context bundle.

        This is the unified function for bundle management:
        - Always updates from scratch in dynamic mode
        - In static mode:
        - Using cached if it exists and config unchanged
        - Builds from provided profile
        - Builds from scratch

        Args:
            session_id: Session ID
            profile: Optional ContextProfile (ignored in dynamic mode, built from scratch if not provided in static mode)
            config: Optional RuntimeContextConfig (uses live config if not provided)

        Returns:
            SessionContextBundle with profile and resolved refs
        """
        mode = self._config_mgr.global_config.runtime.session_context_resolution
        config = config or self._build_context_config()

        if mode == ContextResolutionMode.STATIC and session_id in self._session_bundles:
            cached = self._session_bundles[session_id]
            if cached.profile.runtime_context_config == config:
                logger.debug(
                    "Reusing static session context bundle", session_id=session_id
                )
                return cached
            logger.debug(
                "Config changed, rebuilding static bundle", session_id=session_id
            )

        if mode == ContextResolutionMode.STATIC and profile:
            refs = await self._resolve_all_refs_from_profile(profile)
            bundle = SessionContextBundle(profile=profile, refs=refs)
            self._session_bundles[session_id] = bundle
            return bundle

        refs = await self.build_and_persist_refs(config)
        profile = await self._build_context_profile(config, refs)
        bundle = SessionContextBundle(profile=profile, refs=refs)
        self._session_bundles[session_id] = bundle

        return bundle

    async def _build_context_profile(
        self,
        config: RuntimeContextConfig,
        refs: ResolvedRefs,
    ) -> ContextProfile:
        """Build initial ContextProfile from live config + services."""
        return ContextProfile(
            runtime_context_config=config,
            tool_set_ref=ToolSetRef(
                tool_ids=[t["id"] for t in refs.tools if t.get("id")],
                tool_names=[t["name"] for t in refs.tools if t.get("name")],
                version=refs.tool_set_version,
            ),
            template_set_ref=TemplateSetRef(
                template_ids=refs.template_int_ids,
                template_paths=list(refs.templates.keys()),
                version=refs.template_set_version,
            ),
            repo_snapshot_ref=RepoSnapshotRef(
                target_project_root=refs.target_project_root,
                repo_snapshot_hash=refs.repo_snapshot_hash,
            )
            if refs.repo_map_data
            else None,
            memory_snapshot_ref=MemorySnapshotRef(
                memory_ids=[],
                version=refs.memory_snapshot_version,
            )
            if refs.memories
            else None,
            version=1,
            updated_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        )

    # ========== Snapshot Creation (New Turn) ==========

    async def create_snapshot(
        self,
        session: SessionState,
        config_snapshot: RuntimeContextConfig,
        turn_id: str,
    ) -> ContextSnapshot:
        """Create ContextSnapshot for a NEW turn.

        Uses session's ContextProfile as baseline, captures message_boundary at turn start.
        Persists repo snapshot if needed.
        """

        # Get session bundle (builds/updates profile per policy)
        bundle = await self.get_session_bundle(session.id)
        profile = bundle.profile
        refs = bundle.refs

        # Message boundary = current history (before user message added)
        message_boundary = None
        if session.history:
            seqs = [m.seq for m in session.history if m.seq is not None]
            if seqs:
                message_boundary = MessageBoundary(
                    session_id=session.id,
                    start_seq=min(seqs),
                    end_seq=max(seqs),
                )

        # Build prompt context & render for content hash
        prompt_ctx = self._build_base_prompt_context(
            session_id=session.id,
            tool_specs=refs.tools,
            memories=refs.memories,
            repo_map=refs.repo_map_data,
            project_info=refs.project_info,
            config_snapshot=config_snapshot,
            workspace_root=refs.target_project_root,
        )
        history_dicts = [msg.model_dump() for msg in session.history]
        history_dicts = [msg for msg in history_dicts if msg.get("content") is not None]
        composer = self._get_or_create_composer(
            turn_id,
            config_snapshot,
            template_version=refs.template_set_version,
            templates_dict=refs.templates,
        )
        rendered = composer.render(
            prompt_ctx,
            history=history_dicts,
            user_input="",
            preset="default",
            session_context_resolution=self.get_resolution_mode().value,
        )
        content_hash = self._compute_hash(rendered)

        # Create snapshot with profile's refs + message_boundary
        snapshot = ContextSnapshot(
            id=f"ctx_{turn_id}",
            turn_id=turn_id,
            tool_set_ref=profile.tool_set_ref,
            template_set_ref=profile.template_set_ref,
            repo_snapshot_ref=profile.repo_snapshot_ref,
            memory_snapshot_ref=profile.memory_snapshot_ref,
            message_boundary=message_boundary,
            content_hash=content_hash,
            created_at_ms=int(datetime.now(UTC).timestamp() * 1000),
        )

        return snapshot

    # ========== Snapshot Loading (Resume/Replay) ==========

    async def update_bundle_cache_from_snapshot(
        self,
        session_id: str,
        snapshot: ContextSnapshot,
        config_snapshot: RuntimeContextConfig,
    ) -> SessionContextBundle:
        """Load snapshot from store, resolve refs, build/cache session bundle.

        Called when resuming a turn from storage.
        """
        # Resolve all refs from store using snapshot's stable identities
        refs = await self._resolve_all_refs_from_snapshot(snapshot)

        # Build profile from snapshot (for static mode caching)
        profile = ContextProfile(
            runtime_context_config=config_snapshot,
            tool_set_ref=snapshot.tool_set_ref,
            template_set_ref=snapshot.template_set_ref,
            repo_snapshot_ref=snapshot.repo_snapshot_ref,
            memory_snapshot_ref=snapshot.memory_snapshot_ref,
            version=1,  # Resumed profile starts at v1
            updated_at_ms=snapshot.created_at_ms,
        )

        bundle = SessionContextBundle(profile=profile, refs=refs)
        self._session_bundles[session_id] = bundle

        return bundle

    # ========== Context Resolution (Per Frame) ==========

    async def resolve_context(
        self,
        session: SessionState,
        snapshot: ContextSnapshot,
        continuation: Any,  # FrameContinuation
        config_snapshot: RuntimeContextConfig,
        history: list[Message] | None = None,
        history_through_seq: int | None = None,
    ) -> tuple[PromptContext, RenderedPrompt]:
        """Resolve prompt context and rendered prompt for a frame.

        Uses session bundle (cached per policy) + snapshot's message_boundary + frame's history_through_seq.
        """
        turn_id = snapshot.turn_id

        # Get session bundle (handles static/dynamic policy)
        bundle = await self.get_session_bundle(session.id)
        refs = bundle.refs

        # Build the fully-populated prompt context (Phase 4: dynamic/static env,
        # AGENTS.md, global rules, config instructions).
        prompt_ctx = await self.build_prompt_context(
            session=session,
            config_snapshot=config_snapshot,
            refs=refs,
        )

        # Determine history for this frame
        frame_history = history if history is not None else session.history
        if history_through_seq is not None:
            frame_history = [
                m for m in frame_history if m.seq and m.seq <= history_through_seq
            ]

        history_dicts = [msg.model_dump() for msg in frame_history]

        # Merge snapshot-specific metadata on top of the base context metadata.
        prompt_ctx.metadata.update(
            {
                "context_snapshot_id": snapshot.id,
                "content_hash": snapshot.content_hash,
                "schema_version": snapshot.schema_version,
                "tool_set_version": refs.tool_set_version,
                "template_set_version": refs.template_set_version,
                "repo_snapshot_hash": refs.repo_snapshot_hash,
                "memory_snapshot_version": refs.memory_snapshot_version,
                "context_profile_version": bundle.profile.version,
            }
        )

        # Apply continuation
        if hasattr(continuation, "to_prompt_context_updates"):
            updates = continuation.to_prompt_context_updates()
            if updates.get("metadata"):
                prompt_ctx.metadata.update(updates["metadata"])
            if getattr(continuation, "system_injections", None):
                prompt_ctx.metadata["system_injections"] = (
                    continuation.system_injections
                )

        # Get composer with pinned templates from snapshot
        composer = self._get_or_create_composer(
            turn_id,
            config_snapshot,
            template_version=refs.template_set_version,
            templates_dict=refs.templates,
        )
        rendered = composer.render(
            prompt_ctx,
            history=history_dicts,
            preset="default",
            session_context_resolution=self.get_resolution_mode().value,
        )
        return prompt_ctx, rendered

    # ========== Prompt Context Building ==========

    async def build_prompt_context(
        self,
        session: SessionState,
        config_snapshot: RuntimeContextConfig,
        refs: ResolvedRefs,
    ) -> PromptContext:
        """Build a fully-populated PromptContext for prompt composition.

        Populates static context (session-level, cached per policy) and dynamic
        context (per-turn): cwd, current_date, is_git_repo, platform,
        workspace_root, AGENTS.md content, global rules, config instructions,
        and model-aware fields.

        Args:
            session: SessionState for the current session
            config_snapshot: Resolved runtime context config for this turn
            refs: Resolved context refs (tools, templates, repo, memories)

        Returns:
            PromptContext with all dynamic/static fields populated.
        """
        execution_mode_raw = config_snapshot.execution_mode or "classic_tools"
        if execution_mode_raw == "classic":
            execution_mode_raw = "classic_tools"
        try:
            execution_mode = ExecutionMode(execution_mode_raw)
        except ValueError:
            execution_mode = ExecutionMode.REASONING_ONLY

        tool_call_mode = config_snapshot.tool_call_mode or "classic"
        sandbox_enabled = bool(config_snapshot.sandbox_enabled)

        current_date = int(datetime.now(UTC).timestamp())

        # Tools
        tools = [
            ToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
            for t in refs.tools
        ]

        target_project_root = self._config_mgr.path_manager.get_target_project_root()

        # AGENTS.md discovery (closest-first precedence)
        agents_md_content: list[tuple[str, str]] = []
        if config_snapshot.agents_md_enabled:
            agents_md_content = await self._discover_agents_md(
                config_snapshot.cwd,
                target_project_root,
                config_snapshot.agents_md_filenames,
                config_snapshot.agents_md_walk_up,
            )

        # Global rules from configured global AGENTS.md paths
        global_rules = await self._load_global_rules(
            config_snapshot.agents_md_global_paths
        )

        # Config instructions
        config_instructions = config_snapshot.config_instructions

        metadata = {
            "session_id": session.id,
            "execution_mode": execution_mode.value,
            "tool_call_mode": tool_call_mode,
            "sandbox_enabled": sandbox_enabled,
            "workspace_root": target_project_root,
            "current_date": current_date,
        }

        return PromptContext(
            session_id=session.id,
            execution_mode=execution_mode,
            tool_call_mode=tool_call_mode,
            agent_profile=AgentProfile.default(),
            active_skills=[],
            memories=refs.memories,
            tools=tools,
            metadata=metadata,
            model_id=config_snapshot.llm_model or "",
            model_provider=config_snapshot.llm_provider or "",
            personality=config_snapshot.personality,
            project_info=refs.project_info,
            repo_map_data=refs.repo_map_data,
            examples=[],
            workspace_root=str(target_project_root),
            platform=config_snapshot.platform,
            agents_md_content=agents_md_content,
            global_rules=global_rules,
            config_instructions=config_instructions,
            cwd=config_snapshot.cwd,
            current_date=current_date,
            is_git_repo=config_snapshot.is_git_repo,
            sandbox_enabled=sandbox_enabled,
        )

    async def _discover_agents_md(
        self,
        cwd: str,
        workspace_root: Path | str,
        filenames: list[str],
        walk_up: bool,
    ) -> list[tuple[str, str]]:
        """Walk from cwd to workspace_root, return [(path, content)] closest first.

        Results are cached per (cwd, workspace_root, filenames, walk_up) key.
        Closest (cwd) directory first = highest precedence.
        """
        cache_key = f"{cwd}|{workspace_root!s}|{','.join(filenames)}|{walk_up}"
        if cache_key in self._agents_md_cache:
            return self._agents_md_cache[cache_key]

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

        self._agents_md_cache[cache_key] = results
        self._last_agents_md_hash = self._hash_agents_md(results)
        return results

    async def _load_global_rules(self, global_paths: list[str]) -> str:
        """Load and concatenate global AGENTS.md/CLAUDE.md files."""
        if self._global_rules_cache:
            return self._global_rules_cache
        parts: list[str] = []
        for path in global_paths:
            expanded = os.path.expanduser(path)
            if not expanded or not os.path.exists(expanded):
                continue
            try:
                parts.append(Path(expanded).read_text(encoding="utf-8"))
            except OSError:
                continue
        self._global_rules_cache = "\n\n---\n\n".join(parts)
        return self._global_rules_cache

    @staticmethod
    def _hash_agents_md(agents_md_content: list[tuple[str, str]]) -> str:
        import hashlib as _hashlib

        payload = json.dumps(agents_md_content, sort_keys=True, default=str)
        return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ========== Cache Management ==========

    def clear_turn_cache(self, turn_id: str) -> None:
        self._composer_cache.pop(turn_id, None)

    def clear_session_bundle(self, session_id: str) -> None:
        self._session_bundles.pop(session_id, None)

    def clear_session_all(self, session_id: str) -> None:
        self.clear_session_bundle(session_id)

    # ========== Internal Helpers ==========

    def _build_context_config(self) -> RuntimeContextConfig:
        global_config = self._config_mgr.global_config
        runtime_config = global_config.runtime
        llm_config = global_config.llm
        sandbox_config = global_config.sandbox
        execution_mode = (
            ExecutionMode.CLASSIC_TOOLS
            if runtime_config.tool_call_mode == "classic"
            else ExecutionMode.REASONING_ONLY
        )
        target_project_root = self._config_mgr.path_manager.get_target_project_root()
        cwd = os.getcwd()
        if target_project_root:
            try:
                cwd_resolved = Path(cwd).resolve()
                root_resolved = Path(target_project_root).resolve()
                cwd_resolved.relative_to(root_resolved)
            except ValueError:
                cwd = target_project_root

        return RuntimeContextConfig(
            llm_model=llm_config.model_name,
            llm_temperature=llm_config.temperature,
            llm_max_tokens=llm_config.max_tokens,
            llm_provider=self._config_mgr.get_provider_by_model(
                llm_config.model_name or ""
            ),
            personality=None,
            platform=sys.platform,
            is_git_repo=Path(target_project_root, ".git").exists(),
            cwd=str(cwd),
            agents_md_enabled=getattr(global_config.prompt.agents_md, "enabled", True),
            agents_md_walk_up=getattr(global_config.prompt.agents_md, "walk_up", True),
            agents_md_filenames=list(
                getattr(global_config.prompt.agents_md, "filenames", []) or []
            ),
            agents_md_global_paths=list(
                getattr(global_config.prompt.agents_md, "global_paths", []) or []
            ),
            config_instructions=list(
                getattr(global_config.prompt, "config_instructions", []) or []
            ),
            tool_call_mode=runtime_config.tool_call_mode,
            execution_mode=str(execution_mode.value),
            agent_mode=runtime_config.agent_mode != "no_agent",
            graph_mode=runtime_config.agent_mode == "graph",
            streaming=runtime_config.stream,
            sandbox_enabled=sandbox_config.enabled,
            prompt_templates=global_config.prompt.templates,
            trim_blocks=global_config.prompt.trim_blocks,
            lstrip_blocks=global_config.prompt.lstrip_blocks,
        )

    def _compute_config_hash(self, config: RuntimeContextConfig) -> str:
        config_json = config.model_dump_json(exclude_none=True, by_alias=True)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]

    async def _resolve_all_refs_from_profile(
        self, profile: ContextProfile
    ) -> ResolvedRefs:
        """Resolve all refs from a context profile."""
        tools = await self._resolve_tools(profile.tool_set_ref)
        templates = await self._resolve_templates(profile.template_set_ref)
        repo_map, project_info, target_project_root = await self._resolve_repo(
            profile.repo_snapshot_ref
        )
        memories = await self._resolve_memories(profile.memory_snapshot_ref)

        return ResolvedRefs(
            tools=tools,
            templates=templates,
            repo_map_data=repo_map,
            project_info=project_info,
            target_project_root=target_project_root,
            memories=memories,
            template_int_ids=profile.template_set_ref.template_ids,
            tool_set_version=profile.tool_set_ref.version,
            template_set_version=profile.template_set_ref.version,
            repo_snapshot_hash=profile.repo_snapshot_ref.repo_snapshot_hash
            if profile.repo_snapshot_ref
            else "",
            memory_snapshot_version=profile.memory_snapshot_ref.version
            if profile.memory_snapshot_ref
            else "",
        )

    async def _resolve_all_refs_from_snapshot(
        self, snapshot: ContextSnapshot
    ) -> ResolvedRefs:
        tools = await self._resolve_tools(snapshot.tool_set_ref)
        templates = await self._resolve_templates(snapshot.template_set_ref)
        repo_map, project_info, target_project_root = await self._resolve_repo(
            snapshot.repo_snapshot_ref
        )
        memories = await self._resolve_memories(snapshot.memory_snapshot_ref)

        await self._assert_snapshot_refs_resolved(
            snapshot, tools, templates, target_project_root
        )

        return ResolvedRefs(
            tools=tools,
            templates=templates,
            repo_map_data=repo_map,
            project_info=project_info,
            target_project_root=target_project_root,
            memories=memories,
            template_int_ids=snapshot.template_set_ref.template_ids,
            tool_set_version=snapshot.tool_set_ref.version
            or self.compute_tool_set_version(tools),
            template_set_version=snapshot.template_set_ref.version
            or self.compute_template_set_version(list(templates.keys())),
            repo_snapshot_hash=snapshot.repo_snapshot_ref.repo_snapshot_hash
            if snapshot.repo_snapshot_ref
            else "no_repo",
            memory_snapshot_version=snapshot.memory_snapshot_ref.version
            if snapshot.memory_snapshot_ref
            else "empty",
        )

    async def _assert_snapshot_refs_resolved(
        self,
        snapshot: ContextSnapshot,
        tools: list[dict],
        templates: dict[str, str],
        repo_root: str,
    ) -> None:
        """Fail loudly when pinned snapshot refs no longer resolve.

        Silent degradation (a tool/template/repo disappearing from the
        resumed prompt) is worse than a hard error here: the resumed turn
        would run with a context the turn was never created with.
        """
        tool_ids = snapshot.tool_set_ref.tool_ids
        if tool_ids and len(tools) != len(tool_ids):
            resolved_ids = {t.get("id") for t in tools}
            missing = [tid for tid in tool_ids if tid not in resolved_ids]
            raise MissingContextReferencesError(
                f"Snapshot {snapshot.id} references tools that no longer exist in store",
                missing_refs=[f"tool:{tid}" for tid in missing],
                context_id=snapshot.id,
            )

        template_ids = snapshot.template_set_ref.template_ids
        if template_ids and len(templates) != len(template_ids):
            rows = (
                await self._store.get_templates_by_ids(template_ids)
                if self._store
                else []
            )
            resolved_ids = {row.get("id") for row in rows}
            missing = [tid for tid in template_ids if tid not in resolved_ids]
            if missing:
                raise MissingContextReferencesError(
                    f"Snapshot {snapshot.id} references templates that no longer exist in store",
                    missing_refs=[f"template:{tid}" for tid in missing],
                    context_id=snapshot.id,
                )

        if snapshot.repo_snapshot_ref and not repo_root:
            raise MissingContextReferencesError(
                f"Snapshot {snapshot.id} references a repo snapshot that no longer exists in store",
                missing_refs=[
                    f"repo:{snapshot.repo_snapshot_ref.target_project_root}@{snapshot.repo_snapshot_ref.repo_snapshot_hash}"
                ],
                context_id=snapshot.id,
            )

    async def build_and_persist_refs(
        self, config: RuntimeContextConfig
    ) -> ResolvedRefs:
        """Build and persist all context artifacts in a single transaction.

        Gets live tool specs, repo context, memories, templates from config.
        Calls store.update_context_artifacts() to register everything in DB
        and get back unique identifiers (tool_ids, template_ids, repo_snapshot_hash).
        Returns ResolvedRefs with DB-generated IDs cached for the session.

        This is the single source of truth for creating fresh context artifacts.
        """
        if not self._store:
            raise RuntimeError(
                "Store not initialized - cannot persist context artifacts"
            )

        # 1. Get live tool specs from gateway
        tool_specs = await self._get_tool_specs()

        # 2. Get repo context
        project_info, repo_map, target_project_root = await self._get_repo_context()

        # 3. Get memories
        memories = self._memory.fetch()

        # 4. Get templates from config with actual content
        template_specs = await self._get_template_specs(config)

        # 5. Persist all artifacts in single transaction
        result = await self._store.update_context_artifacts(
            tool_specs=tool_specs,
            template_specs=template_specs,
            repo_map_data=repo_map,
            project_info=project_info,
            target_project_root=target_project_root,
            memories=memories,
        )

        # 6. Build ResolvedRefs with DB-generated IDs
        # Templates already have content from _get_template_specs()
        templates = {}
        for spec in template_specs:
            for path, content in spec.items():
                templates[path] = content

        return ResolvedRefs(
            tools=result.get("tools_with_ids", []),
            templates=templates,
            repo_map_data=repo_map,
            project_info=project_info,
            target_project_root=target_project_root,
            memories=memories,
            template_int_ids=result.get("template_ids", []),
            tool_set_version=result.get("tool_set_version", ""),
            template_set_version=result.get("template_set_version", ""),
            repo_snapshot_hash=result.get("repo_snapshot_hash", "no_repo"),
            memory_snapshot_version=result.get("memory_snapshot_version", "empty"),
        )

    async def _resolve_tools(self, ref: ToolSetRef | None) -> list[dict]:
        if not ref or not ref.tool_ids or not self._store:
            return []
        return await self._store.get_tools_by_ids(ref.tool_ids)

    async def _resolve_templates(self, ref: TemplateSetRef | None) -> dict[str, str]:
        if not ref or not ref.template_ids or not self._store:
            return {}
        templates = {}
        template_list = await self._store.get_templates_by_ids(ref.template_ids)
        for template in template_list:
            templates[template.get("path", template.get("id", ""))] = template.get(
                "content", ""
            )
        return templates

    async def _resolve_repo(
        self, ref: RepoSnapshotRef | None
    ) -> tuple[RepoMapData | None, ProjectInfo | None, str]:
        if not ref or not self._store:
            return None, None, ""
        repo_data = await self._store.get_repo_snapshot_by_hash(ref.repo_snapshot_hash)
        if repo_data:
            repo_map = None
            if repo_data.repo_map_json:
                try:
                    repo_map_dict = json.loads(repo_data.repo_map_json)
                    repo_map = RepoMapData(**repo_map_dict)
                except Exception as e:
                    logger.warning("Failed to deserialize repo_map_json", error=str(e))
            project_info = None
            if repo_data.project_info_json:
                try:
                    project_info = ProjectInfo.model_validate_json(
                        repo_data.project_info_json
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to deserialize project_info_json", error=str(e)
                    )
            return repo_map, project_info, repo_data.target_project_root
        return None, None, ""

    async def _resolve_memories(self, ref: MemorySnapshotRef | None) -> list[str]:
        if not ref or not ref.memory_ids or not self._store:
            return []
        return await self._store.get_memories_by_ids(ref.memory_ids)

    async def _get_tool_specs(self) -> list[dict[str, Any]]:
        if not self._gateway:
            return []
        try:
            schemas = self._gateway.get_exposed_tool_schemas()
            specs = []
            for schema in schemas:
                specs.append(
                    {
                        "name": schema["id"],
                        "description": schema.get("description", ""),
                        "parameters": schema.get("input_schema", {}),
                    }
                )
            return specs
        except Exception:
            return []

    async def _get_template_specs(
        self, config: RuntimeContextConfig
    ) -> list[dict[str, str]]:
        """Get template specs with actual content from disk."""
        from pathlib import Path

        template_specs = []
        if config.prompt_templates:
            for _name, path in config.prompt_templates.items():
                try:
                    template_path = Path(self._template_root) / path
                    if template_path.exists():
                        content = template_path.read_text()
                        template_specs.append({path: content})
                    else:
                        logger.warning("Template file not found", path=path)
                except Exception as e:
                    logger.warning(
                        "Failed to read template content", path=path, error=str(e)
                    )
        return template_specs

    async def _get_repo_context(
        self,
    ) -> tuple[ProjectInfo | None, RepoMapData | None, str]:
        if not self._repo:
            return None, None, ""
        try:
            project_info, repo_map = self._repo.get_repo_context()
            target_project_root = str(self._repo.target_project_root)
            return project_info, repo_map, target_project_root
        except Exception:
            return None, None, ""

    def compute_tool_set_version(self, tool_specs: list[dict[str, Any]]) -> str:
        """Version hash over (tool id, name, argument-schema) of a tool set.

        Public because resume validation recomputes it over stored tool rows
        to detect schema drift against a pinned version.

        Delegates to the canonical implementation in the execution store so the
        pinned version (update_context_artifacts) and the recomputed version
        (resume/branch validation) always agree for the same tools.
        """
        return _compute_stored_tool_set_version(tool_specs)

    def compute_template_set_version(self, template_ids: list[str]) -> str:
        """Version hash over template ids (schema drift is caught by ref existence)."""
        version_str = json.dumps(sorted(template_ids), sort_keys=True)
        return hashlib.sha256(version_str.encode()).hexdigest()[:16]

    def _compute_repo_hash(
        self, repo_map: RepoMapData | None, project_info: ProjectInfo | None
    ) -> str:
        if repo_map is None:
            return "no_repo"
        import dataclasses

        repo_map_json = json.dumps(
            dataclasses.asdict(repo_map), default=str, sort_keys=True
        )
        project_info_json = project_info.model_dump_json() if project_info else "{}"
        repo_data = {"repo_map": repo_map_json, "project_info": project_info_json}
        return hashlib.sha256(
            json.dumps(repo_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def _compute_memory_version(self, memories: list[str]) -> str:
        if not memories:
            return "empty"
        return hashlib.sha256(
            json.dumps(sorted(memories), sort_keys=True).encode()
        ).hexdigest()[:16]

    def _build_base_prompt_context(
        self,
        session_id: str,
        tool_specs: list[dict[str, Any]],
        memories: list[str],
        repo_map: RepoMapData | None,
        project_info: ProjectInfo | None,
        config_snapshot: RuntimeContextConfig,
        workspace_root: str = "",
    ) -> PromptContext:
        execution_mode_raw = config_snapshot.execution_mode or "classic_tools"
        # Map legacy "classic" to "classic_tools" for ExecutionMode enum
        if execution_mode_raw == "classic":
            execution_mode_raw = "classic_tools"
        execution_mode = ExecutionMode(execution_mode_raw)
        tool_call_mode = config_snapshot.tool_call_mode or "classic"
        agent_mode = config_snapshot.agent_mode or False
        graph_mode = config_snapshot.graph_mode or False
        streaming = (
            config_snapshot.streaming if config_snapshot.streaming is not None else True
        )
        sandbox_enabled = config_snapshot.sandbox_enabled or False
        tools = [
            ToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("parameters", {}),
            )
            for t in tool_specs
        ]

        model_id = config_snapshot.llm_model or ""
        model_provider = self._config_mgr.get_provider_by_model(model_id)

        cwd = os.getcwd()
        if workspace_root:
            try:
                cwd_resolved = Path(cwd).resolve()
                root_resolved = Path(workspace_root).resolve()
                cwd_resolved.relative_to(root_resolved)
            except ValueError:
                cwd = workspace_root
        platform = sys.platform
        is_git_repo = Path(workspace_root, ".git").exists() if workspace_root else False
        current_date = int(datetime.now(UTC).timestamp())

        metadata = {
            "session_id": session_id,
            "execution_mode": execution_mode.value,
            "tool_call_mode": tool_call_mode,
            "graph_mode": graph_mode,
            "agent_mode": agent_mode,
            "streaming": streaming,
            "sandbox_enabled": sandbox_enabled,
            "workspace_root": workspace_root,
            "platform": platform,
            "current_date": current_date,
        }

        return PromptContext(
            session_id=session_id,
            execution_mode=execution_mode,
            tool_call_mode=tool_call_mode,
            agent_profile=AgentProfile.default(),
            active_skills=[],
            memories=memories,
            tools=tools,
            metadata=metadata,
            model_id=model_id,
            model_provider=model_provider,
            personality=None,
            project_info=project_info,
            repo_map_data=repo_map,
            examples=[],
            workspace_root=workspace_root,
            platform=platform,
            agents_md_content=[],
            global_rules="",
            config_instructions=[],
            cwd=cwd,
            current_date=current_date,
            is_git_repo=is_git_repo,
            sandbox_enabled=sandbox_enabled,
        )

    def _compute_hash(self, rendered: RenderedPrompt) -> str:
        messages = rendered.to_full_prompt()
        prompt_text = "\n".join(msg.get("content", "") for msg in messages)
        return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

    def _get_or_create_composer(
        self,
        turn_id: str,
        config: RuntimeContextConfig,
        template_version: str,
        templates_dict: dict[str, str] | None = None,
    ) -> PromptComposer:
        cache_key = f"{turn_id}:{template_version}"
        if cache_key in self._composer_cache:
            return self._composer_cache[cache_key]

        prompt_config = self._config_to_prompt_config(config)
        if templates_dict is not None:
            composer = PromptComposer(
                prompt_config=prompt_config, templates_dict=templates_dict
            )
        else:
            composer = PromptComposer(
                prompt_config=prompt_config, template_root=self._template_root
            )
        self._composer_cache[cache_key] = composer
        return composer

    def _config_to_prompt_config(
        self, config: RuntimeContextConfig
    ) -> PromptTemplateConfig:
        return PromptTemplateConfig(
            templates=config.prompt_templates,
            trim_blocks=config.trim_blocks,
            lstrip_blocks=config.lstrip_blocks,
        )
