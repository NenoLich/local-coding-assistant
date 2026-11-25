"""RuntimeManager - orchestrates end-to-end query handling using new architecture.

Three main entrypoints:
- start_session(): create new session, first turn, run it
- resume_session(): resume interrupted turn in 'running' session
- resume_session_with_message(): continue session with new message (optionally branch)

Uses SessionManager, TurnManager, ContextManager, and FrameAgent.
All persistence via EventPipeline -> StoreSubscriber -> store.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from local_coding_assistant.core.protocols import IConfigManager
from local_coding_assistant.repository import RepositoryContextService
from local_coding_assistant.runtime.context import ContextManager
from local_coding_assistant.runtime.engine.engine import RuntimeEngine
from local_coding_assistant.runtime.engine.frame_agent import (
    FrameAgent,
)
from local_coding_assistant.runtime.events.contracts import (
    ApprovalResponseForm,
    RuntimeEvent,
    SessionBranchEvent,
    SessionInterruptEvent,
    SessionResumeEvent,
    SessionStartEvent,
)
from local_coding_assistant.runtime.events.pipeline import EventPipeline
from local_coding_assistant.runtime.options import RunOptions
from local_coding_assistant.runtime.resume import (
    ResumeValidationError,
    build_approval_resume_plan,
    build_resume_plan,
    validate_session_profile,
)
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.session.session_manager import SessionManager
from local_coding_assistant.runtime.session.turn import TurnStatus
from local_coding_assistant.runtime.session.turn_manager import TurnManager
from local_coding_assistant.tools_platform.gateway import ToolGateway
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.agent.llm.service import LLMService
    from local_coding_assistant.storage import IExecutionStore

logger = get_logger("runtime.manager")


class RuntimeManager:
    """Orchestrates end-to-end query handling using new architecture."""

    def __init__(
        self,
        config_manager: IConfigManager,
        llm_service: LLMService | None = None,
        tool_gateway: ToolGateway | None = None,
        repository_context_service: RepositoryContextService | None = None,
        event_pipeline: EventPipeline | None = None,
        store: IExecutionStore | None = None,
    ) -> None:
        if config_manager is None:
            raise ValueError("config_manager is required")

        self.config_manager = config_manager
        self._llm_service = llm_service
        self._tool_gateway = tool_gateway
        self._repository_context_service = repository_context_service
        self._event_pipeline = event_pipeline
        self._store = store

        # Initialize core components
        self._context_manager = self._init_context_manager()
        self._session_manager = (
            SessionManager(
                store,
                config_manager=config_manager,
                workspace_manager=(
                    self._tool_gateway.workspace_manager if self._tool_gateway else None
                ),
            )
            if store
            else None
        )
        # Transcript projection mirrors events into the live SessionState.
        # Registered here, downstream of SessionManager (not in bootstrap).
        if self._event_pipeline and self._session_manager:
            from local_coding_assistant.runtime.engine.subscribers.transcript_subscriber import (
                TranscriptSubscriber,
            )

            self._event_pipeline.subscribe(
                TranscriptSubscriber(self._session_manager.get_cached)
            )
        self._turn_manager = (
            TurnManager(
                context_manager=self._context_manager,
                config_manager=config_manager,
            )
            if event_pipeline
            else None
        )
        self._frame_agent = self._init_frame_agent()

        # Ensure config manager has global configuration loaded
        if (
            not hasattr(config_manager, "global_config")
            or config_manager.global_config is None
        ):
            config_manager.load_global_config()

    def _init_context_manager(self) -> ContextManager:
        from local_coding_assistant.runtime.context.provider import MemoryProvider

        return ContextManager(
            tool_gateway=self._tool_gateway,
            repo_service=self._repository_context_service,
            memory_provider=MemoryProvider(),
            store=self._store,
            config_manager=self.config_manager,
        )

    def _init_frame_agent(self) -> FrameAgent | None:
        if not self._llm_service or not self._tool_gateway or not self._event_pipeline:
            return None

        # Create execution engine
        engine = RuntimeEngine(
            llm_service=self._llm_service,
            tool_gateway=self._tool_gateway,
            event_pipeline=self._event_pipeline,
        )

        return FrameAgent(
            engine=engine,
            context_manager=self._context_manager,
            event_pipeline=self._event_pipeline,
        )

    def _record_turn_and_maybe_evict(self, session: SessionState) -> None:
        """Record a turn for the session and evict LRU in-memory session data if the watermark is reached.

        Session retention policy: cumulative turns are tracked in SessionManager.
        When the threshold is hit, the least-recently-accessed session (excluding the
        active one) has its in-memory file-version overrides cleared. Stored baselines
        are unaffected (ON DELETE CASCADE handles them when the session row is deleted).

        Args:
            session: The session that just started a turn
        """
        if not self._session_manager:
            return
        self._session_manager.record_turn(session.id)
        self._session_manager.maybe_evict_retention(exclude_session_id=session.id)

    async def _reload_workspace_baselines(self, session: SessionState) -> None:
        """Invalidate and repopulate workspace file-version baselines for a resumed session.

        Clears any stale in-memory overrides for the session and reloads the
        persisted baselines so the workspace mutation baseline matches the
        resumed session rather than a merely cached one.
        """
        if not (
            self._tool_gateway
            and hasattr(self._tool_gateway, "workspace_manager")
            and self._store
        ):
            return
        workspace_manager = self._tool_gateway.workspace_manager
        workspace_id = workspace_manager.workspace_id
        baselines = await self._store.load_file_baselines(workspace_id, session.id)
        from local_coding_assistant.tools_platform.contracts import (
            FileObservation,
            ObservationSource,
        )

        file_versions = {}
        for obs_dict in baselines:
            source_str = (
                obs_dict.get("source", {}).get("value", "read")
                if isinstance(obs_dict.get("source"), dict)
                else obs_dict.get("source", "read")
            )
            obs = FileObservation(
                path=obs_dict["path"],
                hash=obs_dict["hash"],
                size=obs_dict["size"],
                mtime=obs_dict["mtime"],
                inode=obs_dict.get("inode"),
                source=ObservationSource(source_str),
                observed_at=obs_dict["observed_at"],
            )
            file_versions[obs.path] = obs
        workspace_manager.clear_session_overrides(session.id)
        workspace_manager.override_file_versions(file_versions, session_id=session.id)
        logger.info(
            "Reloaded workspace baselines for resume",
            workspace_id=workspace_id,
            session_id=session.id,
            count=len(baselines),
        )

    # ========== ENTRYPOINT 1: Start Session ==========

    async def start_session(
        self,
        user_input: str,
        *,
        trace_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        """Create new session, first turn, and run it.

        Flow:
        1. session_manager.create_session()
        2. turn_manager.create_turn() -> creates turn + snapshot
        3. frame_agent.run_turn()
        4. Session becomes idle after turn completes (handled by TurnCompleteEvent)

        Args:
            user_input: Initial user message
            trace_id: Optional trace ID for correlation
            session_metadata: Optional metadata for session

        Returns:
            SessionState after turn completes
        """
        if not self._session_manager or not self._turn_manager or not self._frame_agent:
            raise RuntimeError(
                "RuntimeManager not fully initialized (missing store/event_pipeline/frame_agent)"
            )

        trace_id = trace_id or f"trace_{uuid.uuid4()}"

        # 1. Create session
        session = await self._session_manager.create_session(self._context_manager)
        if session_metadata:
            session.metadata.update(session_metadata)
        logger.debug("Created session", session_id=session.id)

        # 2. Create first turn (index 0)
        turn = await self._turn_manager.create_turn(
            session=session,
            user_input=user_input,
            trace_id=trace_id,
            turn_index=0,
        )

        # Add turn to cache
        session.add_turn(turn)

        # 3. Persist the session row (identity fields + initial context
        #    profile) before the frame events reference it. Shares the
        #    turn's bucket with TurnStartEvent so create+update merge into
        #    a single upsert.
        await self._try_emit_runtime_event(
            SessionStartEvent(
                trace_id=trace_id,
                turn_id=turn.turn_id,
                session_id=session.id,
                user_input=user_input,
                agent_mode="frame_agent",
                model_name=turn.context_config.llm_model
                if turn.context_config
                else None,
                metadata_json=json.dumps(session.metadata)
                if session.metadata
                else None,
            )
        )

        # 4. Run turn (FrameAgent emits TurnStartEvent, TurnCompleteEvent for persistence)
        self._record_turn_and_maybe_evict(session)
        await self._frame_agent.run_turn(turn, session)

        return session

    # ========== ENTRYPOINT 2: Resume Session ==========
    async def _supersede_prior_turn(self, session: SessionState) -> None:
        """Finalize the session's prior running turn as superseded.

        ``resume_session_with_message`` starts a new turn in the same
        session, which makes any still-running (interrupted) turn
        obsolete. Finalize it so the store does not accumulate
        orphaned ``running`` turns (and their running frames).

        Only touches turns that are still ``running``; terminal turns
        (completed/failed/partial/blocked) are left as-is.
        """
        if not self._store:
            return

        prior = session.get_last_turn()
        if prior is None or prior.status != TurnStatus.RUNNING:
            return

        finished_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        prior.status = TurnStatus.SUPERSEDED
        prior.finished_at = datetime.now(UTC)
        await self._store.update_turn_status(
            prior.turn_id,
            status="superseded",
            finished_at_ms=finished_at_ms,
        )
        logger.debug(
            "Superseded prior running turn",
            session_id=session.id,
            turn_id=prior.turn_id,
        )

    async def resume_session(
        self,
        session_id: str,
        *,
        turn_id: str | None = None,
        turn_index: int | None = None,
        frame_id: str | None = None,
    ) -> SessionState:
        """Resume session from a specific turn (or last turn).

        If turn_id/turn_index points to a non-last turn, automatically branches
        to a new session and resumes that.

        Only works for 'running' sessions (interrupted turn).
        If session is 'idle', use resume_session_with_message() instead.

        Flow:
        1. Handle branching via _handle_branching()
        2. build_resume_plan() -> loads + validates (snapshot refs, frame
           state, tool support) and selects the frame to resume
        3. frame_agent.resume_turn() -> continues from frame.status

        Args:
            session_id: Session to resume
            turn_id: Specific turn to resume (must exist)
            turn_index: Alternative - resume from this turn index
            frame_id: Optional frame ID to resume from

        Returns:
            SessionState with history loaded (branched session if non-last turn)

        Raises:
            ResumeValidationError: If the restored context is stale (missing
                tool/template/repo refs, unsupported snapshot schema, ...).
        """
        if (
            not self._session_manager
            or not self._context_manager
            or not self._frame_agent
            or not self._store
        ):
            raise RuntimeError("RuntimeManager not fully initialized")

        # 1. Handle branching (loads session with turns)
        # Check if branching needed
        (
            should_branch,
            branch_from_turn_id,
        ) = await self._session_manager.check_branch_needed(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
        )

        if should_branch and branch_from_turn_id:
            # The branch copies turns verbatim (turn_index preserved) but
            # regenerates turn AND frame IDs, so parent-relative selectors
            # must be remapped onto the forked session before the resume
            # plan is built below (turn_index is the only turn selector
            # that survives the fork).
            branch_turn_index = await self._resolve_branch_turn_index(
                session_id,
                turn_id=turn_id,
                turn_index=turn_index,
            )

            # Auto-branch: create new session forked from parent at specified turn
            session = await self._session_manager.branch_session(
                parent_session_id=session_id,
                branch_from_turn_id=branch_from_turn_id,
            )
            is_loaded_from_store = True

            turn_id = None
            turn_index = branch_turn_index
            frame_id = await self._remap_branch_frame_id(
                branch_from_turn_id,
                session,
                branch_turn_index,
                frame_id,
            )

            # Emit session branch event for persistence (store already has data, but event logs the action)
            branch_event = SessionBranchEvent(
                session_id=session.id,
                parent_session_id=session_id,
                branch_from_turn_id=branch_from_turn_id,
                new_session_id=session.id,
                turn_id=branch_from_turn_id,
                trace_id=f"trace_{uuid.uuid4()}",
            )
            await self._try_emit_runtime_event(branch_event)

        else:
            # No branching - load original session with turns
            session = self._session_manager.get_cached(session_id)
            if session is not None:
                is_loaded_from_store = False
            else:
                session = await self._session_manager.resume_session(
                    session_id=session_id,
                    load_turns=True,
                )
                is_loaded_from_store = True

        if is_loaded_from_store:
            # The new turn re-resolves the stored session profile's refs
            # (same reference fields a turn snapshot pins), so validate them.
            profile_result = await validate_session_profile(
                context_manager=self._context_manager,
                store=self._store,
                profile=session.context_profile,
            )
            for warning in profile_result.warnings:
                logger.warning("Session profile validation warning", warning=warning)

            # (Re)populate workspace baselines from persistence
            await self._reload_workspace_baselines(session)

        # 2. Load + validate everything needed for the resume
        # that is the branched session (keying by the parent id would both
        # leak the fork's bundle into the parent's cache and let static
        # mode hand the fork the parent's cached profile).
        updated_bundle = await self._context_manager.update_bundle_cache(
            session.id, session.context_profile
        )
        session.context_profile = updated_bundle.profile

        # 2. Load + validate everything needed for the resume
        plan = await build_resume_plan(
            store=self._store,
            context_manager=self._context_manager,
            session_manager=self._session_manager,
            session_id=session.id,
            turn_id=turn_id,
            turn_index=turn_index,
            frame_id=frame_id,
            exposed_tool_names=self._exposed_tool_names(),
        )

        for warning in plan.warnings:
            logger.warning("Resume validation warning", warning=warning)

        resume_event = SessionResumeEvent(
            trace_id=plan.turn.trace_id,
            turn_id=plan.turn.turn_id,
            session_id=session.id,
            resumed_from_turn_id=plan.turn.turn_id,
            context_snapshot_id=plan.turn.context_snapshot_id,
            frame_id=plan.frame.id if plan.frame else None,
            validation_warnings=plan.warnings,
        )
        await self._try_emit_runtime_event(resume_event)

        # 3. Resume turn execution from frame.state
        await self._frame_agent.resume_turn(plan.turn, session, plan.frame)

        return session

    # ========== Branch Selector Remapping ==========

    async def _resolve_branch_turn_index(
        self,
        session_id: str,
        *,
        turn_id: str | None,
        turn_index: int | None,
    ) -> int | None:
        """Resolve the branch-point turn's index in the parent session.

        The store's cascade copy regenerates turn IDs but preserves
        ``turn_index``, so an index is the only turn selector that survives
        the fork.

        Args:
            session_id: Parent session ID
            turn_id: Turn selector (takes precedence, like the store lookups)
            turn_index: Alternative turn selector

        Returns:
            The parent turn's index, or None when no selector was given

        Raises:
            ResumeValidationError: If the selected turn does not exist
        """
        if turn_id is None and turn_index is None:
            return None
        if turn_id is None:
            return turn_index

        cached = self._session_manager.get_cached(session_id)  # type: ignore[union-attr]
        if cached is not None:
            cached_turn = cached.get_turn(turn_id)
            if cached_turn is not None:
                return cached_turn.turn_index

        turns = await self._store.list_turns_by_session(session_id)  # type: ignore[union-attr]
        target = next((t for t in turns if t.id == turn_id), None)
        if target is None:
            raise ResumeValidationError(
                f"Turn {turn_id} not found in session {session_id}",
                details={"turn_id": turn_id, "session_id": session_id},
            )
        return target.turn_index

    async def _remap_branch_frame_id(
        self,
        parent_turn_id: str,
        branched_session: SessionState,
        branch_turn_index: int | None,
        frame_id: str | None,
    ) -> str | None:
        """Remap a parent frame selector onto the branched session.

        Frame IDs are regenerated in the branch; iterations within a turn
        are preserved, so the fork's frame is matched by
        (branch turn, iteration).

        Args:
            parent_turn_id: Turn the fork was created from (parent session)
            branched_session: Session returned by branch_session
            branch_turn_index: Branch-point turn index (identical in both)
            frame_id: Parent frame selector (None when unset)

        Returns:
            The branched session's frame ID, or None when no frame was
            selected

        Raises:
            ResumeValidationError: If the frame cannot be remapped
        """
        if frame_id is None or branch_turn_index is None:
            return None

        parent_replay = await self._store.load_turn_replay(parent_turn_id)  # type: ignore[union-attr]
        parent_frame = next(
            (
                f
                for f in (parent_replay.frames if parent_replay else [])
                if f.id == frame_id
            ),
            None,
        )
        if parent_frame is None:
            raise ResumeValidationError(
                f"Frame {frame_id} not found in turn {parent_turn_id}",
                details={"frame_id": frame_id, "turn_id": parent_turn_id},
            )

        branched_turn = branched_session.get_turn_by_index(branch_turn_index)
        if branched_turn is None:
            raise ResumeValidationError(
                f"Branch turn index {branch_turn_index} not found in branched "
                f"session {branched_session.id}",
                details={
                    "turn_index": branch_turn_index,
                    "session_id": branched_session.id,
                },
            )

        branched_replay = await self._store.load_turn_replay(branched_turn.turn_id)  # type: ignore[union-attr]
        branched_frame = next(
            (
                f
                for f in (branched_replay.frames if branched_replay else [])
                if f.iteration == parent_frame.iteration
            ),
            None,
        )
        if branched_frame is None:
            raise ResumeValidationError(
                f"Frame iteration {parent_frame.iteration} not found in branched "
                f"turn {branched_turn.turn_id}",
                details={
                    "iteration": parent_frame.iteration,
                    "turn_id": branched_turn.turn_id,
                },
            )
        return branched_frame.id

    # ========== ENTRYPOINT 3: Resume Session With Message ==========

    async def resume_session_with_message(
        self,
        session_id: str,
        user_input: str,
        *,
        turn_id: str | None = None,
        turn_index: int | None = None,
    ) -> SessionState:
        """Resume session and create a NEW turn with user message.

        Automatically branches if turn_id/turn_index points to non-last turn.
        If turn is last turn (or not specified): continues naturally (new turn in same session).

        Flow:
        1. Handle branching via _handle_branching()
        2. Validate the stored session context profile (static mode only)
        3. Determine next turn index from cache (fallback to store)
        4. turn_manager.create_turn() -> creates new turn with snapshot
        5. frame_agent.run_turn()

        Args:
            session_id: Parent session
            user_input: New user message for the new turn
            turn_id: Turn to branch from (defaults to last turn)
            turn_index: Alternative turn selector

        Returns:
            SessionState (new branched session or continued parent session)
        """
        if (
            not self._session_manager
            or not self._turn_manager
            or not self._frame_agent
            or not self._store
        ):
            raise RuntimeError("RuntimeManager not fully initialized")

        # 1. Handle branching (loads session with turns)
        # Check if branching needed
        (
            should_branch,
            branch_from_turn_id,
        ) = await self._session_manager.check_branch_needed(
            session_id=session_id,
            turn_id=turn_id,
            turn_index=turn_index,
        )

        if should_branch and branch_from_turn_id:
            # Auto-branch: create new session forked from parent at specified turn
            session = await self._session_manager.branch_session(
                parent_session_id=session_id,
                branch_from_turn_id=branch_from_turn_id,
            )
            is_loaded_from_store = True

            # Emit session branch event for persistence (store already has data, but event logs the action)
            branch_event = SessionBranchEvent(
                session_id=session.id,
                parent_session_id=session_id,
                branch_from_turn_id=branch_from_turn_id,
                new_session_id=session.id,
                turn_id=branch_from_turn_id,
                trace_id=f"trace_{uuid.uuid4()}",
            )
            await self._try_emit_runtime_event(branch_event)

        else:
            # No branching - load original session with turns
            session = self._session_manager.get_cached(session_id)
            if session is not None:
                is_loaded_from_store = False
            else:
                session = await self._session_manager.resume_session(
                    session_id=session_id,
                    load_turns=True,
                )
                is_loaded_from_store = True

        if is_loaded_from_store:
            # The new turn re-resolves the stored session profile's refs
            # (same reference fields a turn snapshot pins), so validate them.
            profile_result = await validate_session_profile(
                context_manager=self._context_manager,
                store=self._store,
                profile=session.context_profile,
            )
            for warning in profile_result.warnings:
                logger.warning("Session profile validation warning", warning=warning)

            # (Re)populate workspace baselines from persistence
            await self._reload_workspace_baselines(session)

        # Re-resolve the bundle under the RESUMED session's id (for a fork
        # that is the branched session, not the parent).
        updated_bundle = await self._context_manager.update_bundle_cache(
            session.id, session.context_profile
        )
        session.context_profile = updated_bundle.profile

        # 2. Finalize any prior running turn - it is superseded by
        #     the new turn we are about to create.
        await self._supersede_prior_turn(session)

        # 3. Create new turn (next index) - use cache first, fallback to store
        cached_last = session.get_last_turn()
        if cached_last:
            next_index = cached_last.turn_index + 1
        else:
            last_turn = await self._store.get_last_turn(session.id)
            next_index = (last_turn.turn_index + 1) if last_turn else 0

        trace_id = f"trace_{uuid.uuid4()}"

        turn = await self._turn_manager.create_turn(
            session=session,
            user_input=user_input,
            trace_id=trace_id,
            turn_index=next_index,
        )

        # Add turn to cache
        session.add_turn(turn)

        # 4. Run new turn
        self._record_turn_and_maybe_evict(session)
        await self._frame_agent.run_turn(turn, session)

        return session

    # ========== Entrypoint: Apply Run Options ==========

    def apply_run_options(self, options: RunOptions) -> None:
        """Apply typed front-end run options as config session overrides.

        Single translation point between front-end options and configuration:
        must run before entrypoints that create a turn (``start_session``,
        ``resume_session_with_message``), because the turn pins the resolved
        config into its context snapshot at creation time.

        Args:
            options: Typed run options from any front-end
        """
        overrides = options.to_session_overrides()
        if overrides:
            self.config_manager.set_session_overrides(overrides)
            logger.debug(
                "Applied run options as session overrides", overrides=overrides
            )

    # ========== Entrypoint: Respond to Approval ==========

    async def respond_approval(
        self,
        session_id: str,
        turn_id: str,
        approval_responses: list[ApprovalResponseForm],
        frame_id: str | None = None,
    ) -> SessionState:
        """Deliver approval response(s) and resume execution.

        Args:
            session_id: Session identifier
            turn_id: Turn identifier
            approval_responses: List of approval response forms containing
                approval_token, tool_call_id, and response (YES_ONCE,
                YES_ALWAYS_SESSION, YES_ALWAYS_PROJECT, NO)
            frame_id: Optional frame ID to resume (defaults to finding paused frame)

        Returns:
            SessionState after turn completes or pauses again

        Raises:
            ResumeValidationError: If the restored context is stale.
        """
        if not self._session_manager or not self._frame_agent or not self._store:
            raise RuntimeError("RuntimeManager not fully initialized")

        # 1. Load session + turn (with pending_approvals) - cache only
        session = self._session_manager.get_cached(session_id)
        if not session:
            raise ValueError(
                f"Session {session_id} not available for approval response"
            )

        turn = session.get_turn(turn_id)
        if not turn:
            raise ValueError(f"Turn {turn_id} not found in session {session_id}")

        # 2. Validate and apply approval responses (token/expiry checks live here)
        for response_form in approval_responses:
            turn.apply_approval_response(response_form)

        # 3. Mark turn as resumed
        turn.resumed_at = datetime.now(UTC)
        turn.pause_reason = None  # Clear pause reason

        # 4. Resume turn (continues same frame) - the paused frame is rebuilt
        #    from stored events and the restored context is validated first.
        frame_id = frame_id or turn.get_last_frame_id()
        if not frame_id:
            raise ValueError(f"No frame_id provided and turn {turn_id} has no frames")

        plan = await build_approval_resume_plan(
            store=self._store,
            session=session,
            turn=turn,
            frame_id=frame_id,
            exposed_tool_names=self._exposed_tool_names(),
        )

        for warning in plan.warnings:
            logger.warning("Resume validation warning", warning=warning)

        resume_event = SessionResumeEvent(
            trace_id=turn.trace_id,
            turn_id=turn.turn_id,
            session_id=session.id,
            resumed_from_turn_id=turn.turn_id,
            context_snapshot_id=turn.context_snapshot_id,
            frame_id=plan.frame.id if plan.frame else None,
            validation_warnings=plan.warnings,
        )
        await self._try_emit_runtime_event(resume_event)

        await self._frame_agent.resume_turn(plan.turn, session, plan.frame)

        return session

    @property
    def event_pipeline(self) -> EventPipeline | None:
        """Read-only access to the runtime event pipeline.

        Front-ends (CLI, server) subscribe a transient subscriber here for
        the duration of a single entrypoint call.
        """
        return self._event_pipeline

    # ========== Entrypoint: Interrupt Session ==========
    async def interrupt_session(
        self,
        session_id: str,
        turn_id: str | None = None,
    ) -> SessionState:
        """Interrupt a running turn (e.g., user pressed Ctrl+C or stop button).

        Args:
            session_id: Session identifier
            turn_id: Specific turn to interrupt (defaults to last running turn)

        Returns:
            SessionState with turn paused
        """
        if not self._session_manager or not self._store:
            raise RuntimeError("RuntimeManager not fully initialized")

        # 1. Load session with turns
        session = self._session_manager.get_cached(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not running")

        if turn_id:
            turn = session.get_turn(turn_id)
        else:
            turn = session.get_last_turn()

        if not turn or turn.status != TurnStatus.RUNNING:
            raise ValueError(f"No running turn found in session {session_id}")

        # 3. Signal interrupt + record pause metadata
        turn.interrupt_event.set()
        turn.pause_reason = "interrupt"
        turn.paused_at = datetime.now(UTC)

        # 4. Persist the interrupt for observability / cross-process signalling
        interrupt_event = SessionInterruptEvent(
            trace_id=turn.trace_id,
            turn_id=turn.turn_id,
            session_id=session.id,
            reason="interrupt",
            paused_at=turn.paused_at,
        )
        await self._try_emit_runtime_event(interrupt_event)

        logger.info(
            "Interrupted session turn",
            session_id=session.id,
            turn_id=turn.turn_id,
            reason="interrupt",
        )

        return session

    async def _try_emit_runtime_event(self, event: RuntimeEvent) -> None:
        if self._event_pipeline:
            await self._event_pipeline.emit(event)

    def _exposed_tool_names(self) -> set[str] | None:
        """Names of tools the gateway currently exposes, for resume validation.

        Returns None when unavailable (validation then skips the
        still-supported check instead of failing the resume).
        """
        if not self._tool_gateway:
            return None
        try:
            return {
                schema["id"] for schema in self._tool_gateway.get_exposed_tool_schemas()
            }
        except Exception:
            logger.warning(
                "Failed to collect exposed tool schemas for resume validation"
            )
            return None

    async def _get_next_turn_index(self, session_id: str) -> int:
        if not self._store:
            return 0
        turns = await self._store.list_turns_by_session(session_id)
        return len(turns)

    def start(self) -> None:
        logger.debug("RuntimeManager.start() called")

    def stop(self) -> None:
        logger.debug("RuntimeManager.stop() called")


__all__ = ["RuntimeManager"]
