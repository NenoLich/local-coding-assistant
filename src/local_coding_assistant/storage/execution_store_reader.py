"""Read-only execution store for dashboard and observability consumers.

Opens the SQLite database in read-only mode (``mode=ro``) and exposes only
the query surface of ``IExecutionStoreReader``. No write lock, no event
adapter, no flush machinery, and no write-capable connection — safe to run
alongside a writer process using WAL mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

from local_coding_assistant.storage.contracts import (
    MissingRefs,
    ResumeBatch,
    SessionReplay,
    StoredContextSnapshot,
    StoredFrame,
    StoredSession,
    StoredTurn,
    TurnReplay,
)
from local_coding_assistant.storage.execution_store import IExecutionStoreReader
from local_coding_assistant.storage.repositories import (
    ContextSnapshotRepo,
    EventRepo,
    FrameRepo,
    MessageRepo,
    RepoSnapshotData,
    RepoSnapshotRepo,
    SessionRepo,
    TemplateRepo,
    ToolRepo,
    TurnRepo,
    WorkspaceRecord,
    WorkspaceRepo,
)
from local_coding_assistant.utils.logging import get_logger

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context.profile import ContextSnapshot

logger = get_logger("storage.execution_store_reader")


def _build_readonly_uri(db_path: Path) -> str:
    """Build a read-only SQLite URI for the given path."""
    posix = db_path.as_posix()
    if db_path.is_absolute():
        return f"file:///{posix}?mode=ro"
    return f"file:{posix}?mode=ro"


class ExecutionStoreReader(IExecutionStoreReader):
    """Read-only SQLite execution store with repository layer."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

        # Repositories (initialized in init())
        self._event_repo: EventRepo | None = None
        self._frame_repo: FrameRepo | None = None
        self._session_repo: SessionRepo | None = None
        self._message_repo: MessageRepo | None = None
        self._tool_repo: ToolRepo | None = None
        self._template_repo: TemplateRepo | None = None
        self._turn_repo: TurnRepo | None = None
        self._context_snapshot_repo: ContextSnapshotRepo | None = None
        self._repo_snapshot_repo: RepoSnapshotRepo | None = None
        self._workspace_repo: WorkspaceRepo | None = None

    @property
    def event_repo(self) -> EventRepo:
        assert self._event_repo is not None, "Store not initialized"
        return self._event_repo

    @property
    def frame_repo(self) -> FrameRepo:
        assert self._frame_repo is not None, "Store not initialized"
        return self._frame_repo

    @property
    def session_repo(self) -> SessionRepo:
        assert self._session_repo is not None, "Store not initialized"
        return self._session_repo

    @property
    def message_repo(self) -> MessageRepo:
        assert self._message_repo is not None, "Store not initialized"
        return self._message_repo

    @property
    def tool_repo(self) -> ToolRepo:
        assert self._tool_repo is not None, "Store not initialized"
        return self._tool_repo

    @property
    def template_repo(self) -> TemplateRepo:
        assert self._template_repo is not None, "Store not initialized"
        return self._template_repo

    @property
    def turn_repo(self) -> TurnRepo:
        assert self._turn_repo is not None, "Store not initialized"
        return self._turn_repo

    @property
    def context_snapshot_repo(self) -> ContextSnapshotRepo:
        assert self._context_snapshot_repo is not None, "Store not initialized"
        return self._context_snapshot_repo

    @property
    def repo_snapshot_repo(self) -> RepoSnapshotRepo:
        assert self._repo_snapshot_repo is not None, "Store not initialized"
        return self._repo_snapshot_repo

    @property
    def workspace_repo(self) -> WorkspaceRepo:
        assert self._workspace_repo is not None, "Store not initialized"
        return self._workspace_repo

    async def init(self) -> None:
        """Open the database read-only and initialize repositories."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        uri = _build_readonly_uri(self.db_path)
        self._conn = await aiosqlite.connect(uri, uri=True)
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        self._event_repo = EventRepo(self._conn)
        self._frame_repo = FrameRepo(self._conn)
        self._session_repo = SessionRepo(self._conn)
        self._message_repo = MessageRepo(self._conn)
        self._tool_repo = ToolRepo(self._conn)
        self._template_repo = TemplateRepo(self._conn)
        self._turn_repo = TurnRepo(self._conn)
        self._context_snapshot_repo = ContextSnapshotRepo(self._conn)
        self._repo_snapshot_repo = RepoSnapshotRepo(self._conn)
        self._workspace_repo = WorkspaceRepo(self._conn)

        logger.info(
            "ExecutionStoreReader initialized", db_path=str(self.db_path), readonly=True
        )

    async def close(self) -> None:
        """Close the read-only connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        logger.info("ExecutionStoreReader closed")

    # === Turn Management ===

    async def get_turn(self, turn_id: str) -> StoredTurn | None:
        """Get turn metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.turn_repo.get(turn_id)

    async def list_turns_by_session(self, session_id: str) -> list[StoredTurn]:
        """List all turns for a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.turn_repo.list_by_session(session_id)

    async def get_last_turn(self, session_id: str) -> StoredTurn | None:
        """Get the last turn for a session (highest turn_index)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        turns = await self.turn_repo.list_by_session(session_id)
        if not turns:
            return None
        return max(turns, key=lambda t: t.turn_index)

    async def get_frame_ids_by_turn(self, turn_id: str) -> list[str]:
        """Get all frame IDs for a turn."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.frame_repo.list_ids_by_turn(turn_id)

    async def get_frame(self, frame_id: str) -> StoredFrame | None:
        """Get frame by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.frame_repo.get(frame_id)

    # === Context Snapshot Management ===

    async def get_context_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Get context snapshot by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.context_snapshot_repo.get(snapshot_id)

    async def get_stored_context_snapshot(
        self, snapshot_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot by ID (for debug/replay)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.context_snapshot_repo.get_stored(snapshot_id)

    async def get_stored_context_snapshot_by_turn(
        self, turn_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot for a turn (for debug/replay)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.context_snapshot_repo.get_stored_by_turn(turn_id)

    async def get_context_snapshot_by_turn(
        self, turn_id: str
    ) -> ContextSnapshot | None:
        """Get context snapshot for a turn."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.context_snapshot_repo.get_by_turn(turn_id)

    # === Turn Replay ===

    async def load_resume_batch(
        self,
        session_id: str,
        turn_id: str | None = None,
        turn_index: int | None = None,
        frame_id: str | None = None,
    ) -> ResumeBatch | None:
        """Load everything the resume flow needs in a constant number of queries."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        session = await self.session_repo.get(session_id)
        if not session:
            return None

        turns = await self.turn_repo.list_by_session(session_id)
        if not turns:
            return None

        turn = None
        if turn_id:
            turn = next((t for t in turns if t.id == turn_id), None)
        elif turn_index is not None:
            turn = next((t for t in turns if t.turn_index == turn_index), None)
        else:
            turn = max(turns, key=lambda t: t.turn_index)
        if not turn:
            return None

        snapshot = await self.context_snapshot_repo.get_by_turn(turn.id)

        frames = await self.frame_repo.list_by_turn(turn.id)
        frame_to_load = frame_id
        if frame_to_load is None and frames:
            frame_to_load = frames[-1].id

        frame_events: dict[str, list[dict[str, Any]]] = {}
        if frame_to_load:
            events = await self.event_repo.get_by_frames([frame_to_load])
            frame_events[frame_to_load] = events

        missing_refs = MissingRefs()
        if snapshot:
            missing_refs = await self.check_context_refs(
                tool_ids=snapshot.tool_set_ref.tool_ids,
                template_ids=snapshot.template_set_ref.template_ids,
                repo_hashes=[snapshot.repo_snapshot_ref.repo_snapshot_hash]
                if snapshot.repo_snapshot_ref
                else None,
                memory_ids=snapshot.memory_snapshot_ref.memory_ids
                if snapshot.memory_snapshot_ref
                else None,
            )

        return ResumeBatch(
            session=session,
            turn=turn,
            snapshot=snapshot,
            turns=turns,
            frames=frames,
            frame_events=frame_events,
            missing_refs=missing_refs,
        )

    async def load_turn_replay(self, turn_id: str) -> TurnReplay | None:
        """Load one turn's durable graph for replay/debug."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        turn = await self.turn_repo.get(turn_id)
        if not turn:
            return None

        return await self._build_turn_replay(turn)

    async def load_replay_batch(self, session_id: str) -> SessionReplay | None:
        """Load the full durable session graph for replay/debug (read-only)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        session = await self.session_repo.get(session_id)
        if not session:
            return None

        turns = await self.turn_repo.list_by_session(session_id)
        messages = await self.message_repo.get_by_session(session_id)
        events = await self.event_repo.get_by_session(session_id)

        turns_replay: list[TurnReplay] = []
        for turn in turns:
            frames = await self.frame_repo.list_by_turn(turn.id)
            snapshot = await self.context_snapshot_repo.get_by_turn(turn.id)
            turns_replay.append(
                TurnReplay(
                    turn=turn,
                    snapshot=snapshot,
                    frames=frames,
                    frame_events=self._group_events_by_frame(events, frames),
                    messages=[m for m in messages if m.get("turn_id") == turn.id],
                    missing_refs=await self._missing_refs_for(snapshot),
                )
            )

        return SessionReplay(session=session, turns=turns_replay)

    async def _build_turn_replay(self, turn: StoredTurn) -> TurnReplay:
        """Build a TurnReplay for an already-loaded turn."""
        frames = await self.frame_repo.list_by_turn(turn.id)
        snapshot = await self.context_snapshot_repo.get_by_turn(turn.id)

        events: list[dict[str, Any]] = []
        for frame in frames:
            events.extend(await self.event_repo.get_by_frames([frame.id]))
        messages = await self.message_repo.get_by_session(turn.session_id)

        return TurnReplay(
            turn=turn,
            snapshot=snapshot,
            frames=frames,
            frame_events=self._group_events_by_frame(events, frames),
            messages=[m for m in messages if m.get("turn_id") == turn.id],
            missing_refs=await self._missing_refs_for(snapshot),
        )

    @staticmethod
    def _group_events_by_frame(
        events: list[dict[str, Any]], frames: list[StoredFrame]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group flat event rows by frame id, ordered by sequence."""
        by_frame: dict[str, list[dict[str, Any]]] = {frame.id: [] for frame in frames}
        for event in events:
            frame_id = event.get("frame_id")
            if frame_id in by_frame:
                by_frame[frame_id].append(event)
        for frame_events in by_frame.values():
            frame_events.sort(key=lambda e: e.get("seq", 0))
        return by_frame

    async def _missing_refs_for(self, snapshot: ContextSnapshot | None) -> MissingRefs:
        """Check which context refs of a snapshot still exist in the store."""
        if snapshot is None:
            return MissingRefs()

        return await self.check_context_refs(
            tool_ids=snapshot.tool_set_ref.tool_ids,
            template_ids=snapshot.template_set_ref.template_ids,
            repo_hashes=[snapshot.repo_snapshot_ref.repo_snapshot_hash]
            if snapshot.repo_snapshot_ref
            else None,
            memory_ids=snapshot.memory_snapshot_ref.memory_ids
            if snapshot.memory_snapshot_ref
            else None,
        )

    # === Sessions ===

    async def list_sessions(
        self, limit: int = 50, status: str | None = None
    ) -> list[StoredSession]:
        """List recent sessions, optionally filtered by status."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        sessions = await self.session_repo.list_recent(limit)
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    async def get_session(self, session_id: str) -> StoredSession | None:
        """Get session metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.session_repo.get(session_id)

    # === Events ===

    async def get_frame_events(self, frame_id: str) -> list[dict[str, Any]]:
        """Get all events for a frame."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.event_repo.get_by_frame(frame_id)

    async def get_events_by_turn(self, turn_id: str) -> list[dict[str, Any]]:
        """Get all events for a turn."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.event_repo.get_by_turn(turn_id)

    # === Context artifact lookups ===

    async def get_tool_by_id(self, tool_id: int) -> dict[str, Any] | None:
        """Get tool specification by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.tool_repo.get(tool_id)

    async def get_tools_by_ids(self, tool_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple tool specifications by IDs in a single query."""
        return await self.tool_repo.get_many(tool_ids)

    async def get_template_by_id(self, template_id: int) -> dict[str, Any] | None:
        """Get template by ID with metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.template_repo.get(template_id)

    async def get_templates_by_ids(
        self, template_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Get multiple templates by IDs in a single query."""
        return await self.template_repo.get_many(template_ids)

    async def get_repo_snapshot_by_hash(
        self, repo_snapshot_hash: str
    ) -> RepoSnapshotData | None:
        """Get repository snapshot by content hash."""
        return await self.repo_snapshot_repo.get_by_hash(repo_snapshot_hash)

    async def check_context_refs(
        self,
        *,
        tool_ids: list[int] | None = None,
        template_ids: list[int] | None = None,
        repo_hashes: list[str] | None = None,
        memory_ids: list[str] | None = None,
    ) -> MissingRefs:
        """Check which context references still exist in the store.

        One query per reference kind with an IN clause.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        missing_tools: list[int] = []
        missing_templates: list[int] = []
        missing_repo_hashes: list[str] = []
        missing_memory_ids: list[str] = []

        if tool_ids:
            existing = await self.tool_repo.get_existing_ids(tool_ids)
            missing_tools = [t for t in tool_ids if t not in existing]

        if template_ids:
            existing = await self.template_repo.get_existing_ids(template_ids)
            missing_templates = [t for t in template_ids if t not in existing]

        if repo_hashes:
            existing = await self.repo_snapshot_repo.get_existing_hashes(repo_hashes)
            missing_repo_hashes = [h for h in repo_hashes if h not in existing]

        return MissingRefs(
            tool_ids=missing_tools,
            template_ids=missing_templates,
            repo_hashes=missing_repo_hashes,
            memory_ids=missing_memory_ids,
        )

    async def get_memory_by_id(self, memory_id: str) -> str | None:
        """Get memory content by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        cursor = await self._conn.execute(
            "SELECT content FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_memories_by_ids(self, memory_ids: list[str]) -> list[str]:
        """Get multiple memory contents by IDs in a single query."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        query = "SELECT content FROM memories WHERE id IN (" + placeholders + ")"  # noqa: S608 - parameterized ? placeholders, IDs bound as query args
        cursor = await self._conn.execute(query, memory_ids)

        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_messages_by_range(
        self,
        session_id: str,
        start_seq: int,
        end_seq: int,
    ) -> list[dict[str, Any]]:
        """Get messages in sequence range."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.message_repo.get_by_range(session_id, start_seq, end_seq)

    async def get_all_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.message_repo.get_by_session(session_id)

    # === Workspace Identity ===

    async def get_workspace(self, workspace_root: str) -> WorkspaceRecord | None:
        """Find a persisted workspace record matching the given root.

        Args:
            workspace_root: Absolute workspace root path

        Returns:
            The matching WorkspaceRecord, or None if not present
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.workspace_repo.get_by_root(workspace_root)
