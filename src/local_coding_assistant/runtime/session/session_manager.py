"""SessionManager - manages session identity and transcript reconstruction.

In-memory SessionState is a working cache. Durable data lives in store.
Session reconstruction loads full transcript.
Supports branching sessions from a specific turn.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from local_coding_assistant.core.protocols import IConfigManager

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context import ContextManager
    from local_coding_assistant.storage import IExecutionStore
from local_coding_assistant.runtime.context.profile import ContextProfile
from local_coding_assistant.runtime.session import Message, SessionState
from local_coding_assistant.tools_platform.workspace.manager import WorkspaceManager
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.session_manager")


def _message_from_row(msg_data: dict) -> Message:
    """Build a Message from a stored message row.

    Restores assistant tool_calls from the metadata payload so resumed
    prompts keep the tool-call context providers require.
    """
    metadata_json = msg_data.get("metadata_json")
    tool_calls = None
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
            tool_calls = metadata.get("tool_calls") or None
        except (TypeError, ValueError):
            tool_calls = None
    return Message(
        role=msg_data["role"],
        content=msg_data.get("content"),
        tool_call_id=msg_data.get("tool_call_id"),
        tool_calls=tool_calls,
        seq=msg_data.get("seq"),
    )


class SessionManager:
    """Manages session identity and transcript reconstruction.

    In-memory SessionState is a working cache. Durable data lives in store.
    Session reconstruction loads transcript - full or truncated by turn.
    """

    def __init__(
        self,
        store: IExecutionStore,
        config_manager: IConfigManager | None = None,
        workspace_manager: WorkspaceManager | None = None,
    ) -> None:
        self._store = store
        self._config_manager = config_manager
        self._workspace_manager = workspace_manager
        self._session_cache: dict[str, SessionState] = {}
        # Session retention state (runtime-layer concern, not tools_platform).
        self._session_last_accessed: dict[str, datetime] = {}
        self._total_turns: int = 0

    def record_turn(self, session_id: str) -> None:
        """Record a turn for the session, updating last_accessed and the cumulative turn watermark.

        Args:
            session_id: Session that just started a turn
        """
        self._total_turns += 1
        self._session_last_accessed[session_id] = datetime.now(UTC)

    def maybe_evict_retention(
        self, exclude_session_id: str | None = None
    ) -> str | None:
        """Evict the LRU session's in-memory file-version data if the turn watermark is reached.

        Session retention policy: once cumulative turns reach
        ``runtime.session_retention_max_turns``, evict the least-recently-accessed
        session (excluding the active one) so in-memory file-version overrides stay
        bounded. Stored baselines are untouched (ON DELETE CASCADE handles them when
        the session row itself is deleted).

        Args:
            exclude_session_id: Session ID to protect from eviction (the active one)

        Returns:
            The evicted session ID, or None if nothing was evicted / retention disabled
        """
        if not self._config_manager or not self._workspace_manager:
            return None
        runtime_config = self._config_manager.global_config.runtime
        if not runtime_config.session_retention_enabled:
            return None
        if self._total_turns < runtime_config.session_retention_max_turns:
            return None
        candidates = {
            sid: ts
            for sid, ts in self._session_last_accessed.items()
            if sid != exclude_session_id
        }
        if not candidates:
            return None
        lru_session_id = min(candidates, key=lambda sid: candidates[sid])
        self._session_last_accessed.pop(lru_session_id, None)
        self._session_cache.pop(lru_session_id, None)
        if self._workspace_manager:
            self._workspace_manager.clear_session_overrides(lru_session_id)
        logger.info(
            "Evicted LRU session file-version overrides",
            session_id=lru_session_id,
            total_turns=self._total_turns,
        )
        return lru_session_id

    async def create_session(self, context_manager: ContextManager) -> SessionState:
        """Create a new session and cache it.

        Returns:
            New SessionState instance
        """
        session_id = str(uuid.uuid4())
        context_bundle = await context_manager.update_bundle_cache(session_id)
        session = SessionState(id=session_id, context_profile=context_bundle.profile)
        self._session_cache[session.id] = session

        return session

    async def load_stored_session(
        self,
        session_id: str,
        *,
        load_turns: bool = False,
    ) -> SessionState:
        """Load a session from store with messages, metadata, context_profile, and optionally turns.

        This is a helper that encapsulates the common session reconstruction logic
        used by resume_session and branch_session.

        Args:
            session_id: Session to load
            load_turns: If True, also load all turns into the session cache

        Returns:
            SessionState with history, metadata, context_profile loaded (and turns if requested)
        """
        # Load session metadata
        stored_session = await self._store.get_session(session_id)
        if not stored_session:
            raise ValueError(f"Session not found: {session_id}")

        # Create SessionState with metadata and context_profile
        context_profile_json = stored_session.context_profile_json
        if context_profile_json:
            context_profile = ContextProfile.model_validate_json(context_profile_json)
        else:
            raise ValueError(f"Context profile not found for session: {session_id}")

        session = SessionState(id=session_id, context_profile=context_profile)
        if stored_session.metadata_json:
            session.metadata = json.loads(stored_session.metadata_json)

        # Load full message history
        await self._load_full_history(session)

        # Optionally load turns into cache
        if load_turns:
            from local_coding_assistant.runtime.session.turn import RuntimeTurn

            turns = await self._store.list_turns_by_session(session_id)
            for turn in turns:
                runtime_turn = await RuntimeTurn.from_store(turn.id, self._store)
                session.add_turn(runtime_turn)

        return session

    async def resume_session(
        self,
        session_id: str,
        *,
        load_turns: bool = False,
    ) -> SessionState:
        """Reconstruct session with full transcript.

        Args:
            session_id: Session to reconstruct
            load_turns: If True, also load all turns into the session cache

        Returns:
            SessionState with full history loaded and context_profile
        """
        # Check cache first
        if session_id in self._session_cache:
            session = self._session_cache[session_id]
            return session

        # Use the common helper
        session = await self.load_stored_session(session_id, load_turns=load_turns)
        self._session_cache[session_id] = session

        return session

    async def _load_history_range(
        self, session: SessionState, session_id: str, start_seq: int, end_seq: int
    ):
        messages = await self._store.get_messages_by_range(
            session_id, start_seq, end_seq
        )
        for msg_data in messages:
            session.history.append(_message_from_row(msg_data))
        session.history_sequence = max((m.seq or 0 for m in session.history), default=0)

    async def _load_full_history(self, session: SessionState):
        # Load all messages for session
        messages = await self._store.get_all_messages(session.id)
        for msg_data in messages:
            session.history.append(_message_from_row(msg_data))
        session.history_sequence = max((m.seq or 0 for m in session.history), default=0)

    def _get_target_turn_from_cache(
        self,
        cached_session,
        turn_id: str | None,
        turn_index: int | None,
    ) -> tuple | tuple[None, None]:
        """Get target turn ID from cached session.

        Returns:
            (last_turn, target_turn_id) or (None, None) if not found
        """
        last_turn = cached_session.get_last_turn()
        target_turn_id = None

        if turn_id:
            cached_target = cached_session.get_turn(turn_id)
            if cached_target:
                target_turn_id = cached_target.turn_id
        elif turn_index is not None:
            cached_target = cached_session.get_turn_by_index(turn_index)
            if cached_target:
                target_turn_id = cached_target.turn_id

        return last_turn, target_turn_id

    async def _get_target_turn_from_store(
        self,
        session_id: str,
        turn_id: str | None,
        turn_index: int | None,
    ) -> tuple | tuple[None, None]:
        """Get target turn ID from store.

        Returns:
            (last_turn, target_turn_id) or (None, None) if not found
        """
        last_turn = await self._store.get_last_turn(session_id)
        if not last_turn:
            return None, None

        target_turn_id = None
        if turn_id:
            target_turn_id = turn_id
        elif turn_index is not None:
            turns = await self._store.list_turns_by_session(session_id)
            target = next((t for t in turns if t.turn_index == turn_index), None)
            if target:
                target_turn_id = target.id

        return last_turn, target_turn_id

    async def check_branch_needed(
        self,
        session_id: str,
        turn_id: str | None = None,
        turn_index: int | None = None,
    ) -> tuple[bool, str | None]:
        """Check if branching is needed for the given turn selector.

        Returns:
            (should_branch, branch_from_turn_id)
            - should_branch: True if turn_id/turn_index points to non-last turn
            - branch_from_turn_id: The turn ID to branch from (if should_branch)
        """
        if turn_id is None and turn_index is None:
            return False, None

        cached_session = self._session_cache.get(session_id)
        last_turn = None
        target_turn_id = None

        if cached_session:
            last_turn, target_turn_id = self._get_target_turn_from_cache(
                cached_session, turn_id, turn_index
            )

        if not last_turn or not target_turn_id:
            last_turn, target_turn_id = await self._get_target_turn_from_store(
                session_id, turn_id, turn_index
            )

        if not target_turn_id:
            return False, None

        is_last_turn = (
            target_turn_id == last_turn.turn_id
            if hasattr(last_turn, "turn_id")
            else target_turn_id == last_turn.id
            if hasattr(last_turn, "id")
            else False
        )
        return not is_last_turn, target_turn_id if not is_last_turn else None

    async def branch_session(
        self,
        parent_session_id: str,
        branch_from_turn_id: str,
        *,
        new_session_id: str | None = None,
    ) -> SessionState:
        """Create a new session branched from parent at specific turn.

        Delegates to store for atomic cascade copy of session data.

        Args:
            parent_session_id: Session to branch from
            branch_from_turn_id: Turn ID to branch from (uses its message_boundary)
            new_session_id: Optional specific ID for new session

        Returns:
            New SessionState forked from parent
        """
        if not new_session_id:
            import uuid

            new_session_id = str(uuid.uuid4())

        # Delegate to store for atomic cascade copy
        stored_session = await self._store.branch_session(
            new_session_id=new_session_id,
            parent_session_id=parent_session_id,
            branch_from_turn_id=branch_from_turn_id,
        )

        if not stored_session:
            raise RuntimeError("Failed to create branched session in store")

        # Use the common helper to reconstruct SessionState
        session = await self.load_stored_session(stored_session.id, load_turns=True)

        logger.info(
            "Successfully completed branch session",
            new_session_id=new_session_id,
            branched_from_session_id=parent_session_id,
            new_turn_index=stored_session.total_turns,
        )

        self._session_cache[session.id] = session
        return session

    def get_cached(self, session_id: str) -> SessionState | None:
        return self._session_cache.get(session_id)

    def cache_session(self, session: SessionState) -> None:
        self._session_cache[session.id] = session

    def invalidate(self, session_id: str) -> None:
        self._session_cache.pop(session_id, None)
