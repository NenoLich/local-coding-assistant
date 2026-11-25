"""Execution Store implementation with async SQLite backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context.profile import ContextSnapshot
    from local_coding_assistant.runtime.engine.frame import ExecutionStatus
    from local_coding_assistant.runtime.events.contracts import RuntimeEvent

from local_coding_assistant.runtime.events.contracts import EventType
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
from local_coding_assistant.storage.repositories import (
    ContextSnapshotRepo,
    EventRepo,
    FrameRepo,
    MessageRepo,
    RepoSnapshotData,
    RepoSnapshotRepo,
    RepoSnapshotResult,
    SessionRepo,
    TemplateRepo,
    ToolRepo,
    TurnRepo,
    WorkspaceFileBaselineRepo,
    WorkspaceRecord,
    WorkspaceRepo,
)
from local_coding_assistant.storage.store_event_adapter import StoreEventAdapter
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("storage.execution_store")


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a tool's argument schema as a dict, tolerant of row shape.

    ``update_context_artifacts`` passes specs keyed ``parameters`` (dict),
    while ``get_tools_by_ids`` returns DB rows keyed ``schema_json`` (JSON
    string). Both must normalize to the same schema dict so the pinned and
    recomputed tool set versions agree for identical tools.
    """
    schema = tool.get("parameters")
    if schema is None:
        schema_json = tool.get("schema_json", "{}")
        if isinstance(schema_json, str):
            try:
                schema = json.loads(schema_json)
            except (TypeError, ValueError):
                schema = {}
        elif schema_json:
            schema = schema_json
        else:
            schema = {}
    return schema if isinstance(schema, dict) else {}


def compute_tool_set_version(tool_specs: list[dict[str, Any]]) -> str:
    """Canonical version hash over (tool id, name, argument-schema).

    Single source of truth used both when pinning context artifacts
    (update_context_artifacts) and when recomputing on resume/branch
    (ContextManager.compute_tool_set_version), so the two always agree.
    """
    tool_identities = sorted(
        (
            t.get("name", ""),
            hashlib.sha256(
                json.dumps(_tool_schema(t), sort_keys=True).encode()
            ).hexdigest()[:8],
        )
        for t in tool_specs
    )
    version_str = json.dumps(tool_identities, sort_keys=True)
    return hashlib.sha256(version_str.encode()).hexdigest()[:16]


class SessionStatus(str, Enum):
    """Session execution status for storage."""

    RUNNING = "running"
    IDLE = "idle"


class IExecutionStoreReader(ABC):
    """Read-only interface for execution store queries.

    Dashboard and observability consumers depend on this surface so they
    cannot mutate execution state.
    """

    @abstractmethod
    async def get_turn(self, turn_id: str) -> StoredTurn | None:
        """Get turn metadata."""
        ...

    @abstractmethod
    async def get_last_turn(self, session_id: str) -> StoredTurn | None:
        """Get the last turn for a session (highest turn_index)."""
        ...

    @abstractmethod
    async def list_turns_by_session(self, session_id: str) -> list[StoredTurn]:
        """List all turns for a session."""
        ...

    @abstractmethod
    async def get_frame_ids_by_turn(self, turn_id: str) -> list[str]:
        """Get all frame IDs for a turn."""
        ...

    @abstractmethod
    async def get_frame(self, frame_id: str) -> StoredFrame | None:
        """Get frame by ID."""
        ...

    @abstractmethod
    async def get_context_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Get context snapshot by ID."""
        ...

    @abstractmethod
    async def get_stored_context_snapshot(
        self, snapshot_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot by ID (for debug/replay)."""
        ...

    @abstractmethod
    async def get_stored_context_snapshot_by_turn(
        self, turn_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot for a turn (for debug/replay)."""
        ...

    @abstractmethod
    async def get_context_snapshot_by_turn(
        self, turn_id: str
    ) -> ContextSnapshot | None:
        """Get context snapshot for a turn."""
        ...

    @abstractmethod
    async def load_resume_batch(
        self,
        session_id: str,
        turn_id: str | None = None,
        turn_index: int | None = None,
        frame_id: str | None = None,
    ) -> ResumeBatch | None:
        """Load everything the resume flow needs in a constant number of queries."""
        ...

    @abstractmethod
    async def load_turn_replay(self, turn_id: str) -> TurnReplay | None:
        """Load one turn's durable graph (snapshot, frames, events, messages)."""
        ...

    @abstractmethod
    async def load_replay_batch(self, session_id: str) -> SessionReplay | None:
        """Load the full durable session graph for replay/debug (read-only)."""
        ...

    @abstractmethod
    async def list_sessions(
        self, limit: int = 50, status: str | None = None
    ) -> list[StoredSession]:
        """List recent sessions, optionally filtered by status."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> StoredSession | None:
        """Get session metadata."""
        ...

    @abstractmethod
    async def get_frame_events(self, frame_id: str) -> list[dict[str, Any]]:
        """Get all events for a frame."""
        ...

    @abstractmethod
    async def get_events_by_turn(self, turn_id: str) -> list[dict[str, Any]]:
        """Get all events for a turn."""
        ...

    @abstractmethod
    async def get_tool_by_id(self, tool_id: int) -> dict[str, Any] | None:
        """Get tool specification by ID."""
        ...

    @abstractmethod
    async def get_tools_by_ids(self, tool_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple tool specifications by IDs in a single query."""
        ...

    @abstractmethod
    async def get_template_by_id(self, template_id: int) -> dict[str, Any] | None:
        """Get template by ID with metadata."""
        ...

    @abstractmethod
    async def get_templates_by_ids(
        self, template_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Get multiple templates by IDs in a single query."""
        ...

    @abstractmethod
    async def get_repo_snapshot_by_hash(
        self, repo_snapshot_hash: str
    ) -> RepoSnapshotData | None:
        """Get repository snapshot by content hash."""
        ...

    @abstractmethod
    async def get_memory_by_id(self, memory_id: str) -> str | None:
        """Get memory content by ID."""
        ...

    @abstractmethod
    async def get_memories_by_ids(self, memory_ids: list[str]) -> list[str]:
        """Get multiple memory contents by IDs in a single query."""
        ...

    @abstractmethod
    async def get_messages_by_range(
        self,
        session_id: str,
        start_seq: int,
        end_seq: int,
    ) -> list[dict[str, Any]]:
        """Get messages in sequence range."""
        ...

    @abstractmethod
    async def get_all_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        ...

    @abstractmethod
    async def check_context_refs(
        self,
        *,
        tool_ids: list[int] | None = None,
        template_ids: list[int] | None = None,
        repo_hashes: list[str] | None = None,
        memory_ids: list[str] | None = None,
    ) -> MissingRefs:
        """Check which context references still exist in the store."""
        ...

    @abstractmethod
    async def get_workspace(self, workspace_root: str) -> WorkspaceRecord | None:
        """Find a persisted workspace record matching the given root."""
        ...


class IExecutionStore(ABC):
    """Interface for execution store implementations."""

    @abstractmethod
    async def init(self) -> None:
        """Initialize the store and create tables."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the store and flush pending writes."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Flush all pending writes immediately."""
        ...

    @abstractmethod
    async def register_tool(
        self,
        name: str,
        schema_json: str,
        risk_class: str,
        description: str | None = None,
        version: str = "1.0.0",
    ) -> int:
        """Register a global tool. Returns tool_id."""
        ...

    @abstractmethod
    async def register_template(
        self,
        path: str,
        content: str,
        content_hash: str,
        version: str = "1.0.0",
    ) -> int:
        """Register a global template. Returns template_id."""
        ...

    @abstractmethod
    async def create_session(
        self,
        session_id: str,
        user_input: str,
        agent_mode: str,
        model_name: str | None,
        metadata_json: str | None = None,
    ) -> None:
        """Create a new session."""
        ...

    @abstractmethod
    async def associate_session_tools(
        self, session_id: str, tool_ids: list[int]
    ) -> None:
        """Associate tools with a session."""
        ...

    @abstractmethod
    async def associate_session_templates(
        self, session_id: str, template_ids: list[int]
    ) -> None:
        """Associate templates with a session."""
        ...

    # === Turn Management ===
    @abstractmethod
    async def create_turn(
        self,
        turn_id: str,
        session_id: str,
        trace_id: str,
        turn_index: int,
        user_input: str,
        status: str = "running",
        started_at_ms: int | None = None,
        context_snapshot_id: str | None = None,
        config_snapshot_json: str | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Create a new turn."""
        ...

    @abstractmethod
    async def update_turn_status(
        self,
        turn_id: str,
        status: str,
        finished_at_ms: int | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_frames: int | None = None,
        paused_at_ms: int | None = None,
        pause_reason: str | None = None,
        resumed_at_ms: int | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Update turn status and metadata."""
        ...

    @abstractmethod
    async def get_turn(self, turn_id: str) -> StoredTurn | None:
        """Get turn metadata."""
        ...

    @abstractmethod
    async def get_last_turn(self, session_id: str) -> StoredTurn | None:
        """Get the last turn for a session."""
        ...

    @abstractmethod
    async def list_turns_by_session(self, session_id: str) -> list[StoredTurn]:
        """List all turns for a session."""
        ...

    @abstractmethod
    async def get_frame_ids_by_turn(self, turn_id: str) -> list[str]:
        """Get all frame IDs for a turn."""
        ...

    @abstractmethod
    async def get_frame(self, frame_id: str) -> StoredFrame | None:
        """Get frame by ID."""
        ...

    # === Context Snapshot Management ===
    @abstractmethod
    async def create_context_snapshot(
        self,
        snapshot_id: str,
        turn_id: str,
        schema_version: int = 1,
        tool_set_ref: str | None = None,
        template_set_ref: str | None = None,
        repo_snapshot_ref: str | None = None,
        memory_snapshot_ref: str | None = None,
        message_boundary_json: str | None = None,
        content_hash: str = "",
        payload_json: str | None = None,
        created_at_ms: int | None = None,
    ) -> None:
        """Create a new context snapshot."""
        ...

    @abstractmethod
    async def get_context_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Get context snapshot by ID."""
        ...

    @abstractmethod
    async def get_stored_context_snapshot(
        self, snapshot_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot by ID (for debug/replay)."""
        ...

    @abstractmethod
    async def get_stored_context_snapshot_by_turn(
        self, turn_id: str
    ) -> StoredContextSnapshot | None:
        """Get stored context snapshot for a turn (for debug/replay)."""
        ...

    @abstractmethod
    async def get_context_snapshot_by_turn(
        self, turn_id: str
    ) -> ContextSnapshot | None:
        """Get context snapshot for a turn."""
        ...

    @abstractmethod
    async def create_frame(
        self,
        frame_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
        iteration: int,
        model_name: str | None = None,
        context_snapshot_id: str | None = None,
        history_through_seq: int = 0,
        prompt_hash: str | None = None,
    ) -> None:
        """Create a new frame."""
        ...

    @abstractmethod
    async def update_frame_status(
        self,
        frame_id: str,
        status: ExecutionStatus,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        error_message: str | None = None,
        pending_call_id: str | None = None,
    ) -> None:
        """Update frame status and metadata."""
        ...

    @abstractmethod
    async def append_message(
        self,
        session_id: str,
        turn_id: str,
        frame_id: str,
        seq: int,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        """Append a message to the turn."""
        ...

    @abstractmethod
    async def append_event(
        self,
        trace_id: str,
        session_id: str,
        turn_id: str | None,
        frame_id: str | None,
        seq: int,
        event_type: EventType,
        data_json: str,
        timestamp: datetime | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append an event to the store."""
        ...

    @abstractmethod
    async def load_resume_batch(
        self,
        session_id: str,
        turn_id: str | None = None,
        turn_index: int | None = None,
        frame_id: str | None = None,
    ) -> ResumeBatch | None:
        """Load everything the resume flow needs in a constant number of queries.

        Returns the session, its turns (for selection/branch decisions), the
        selected turn's snapshot, that turn's frame rows, the events of the
        single frame being resumed, and the snapshot's missing refs. Does NOT
        reconstruct every turn's frames and events - that full session graph
        shape belongs to the future replay flow.
        """
        ...

    @abstractmethod
    async def load_turn_replay(self, turn_id: str) -> TurnReplay | None:
        """Load one turn's durable graph (snapshot, frames, events, messages)."""
        ...

    @abstractmethod
    async def load_replay_batch(self, session_id: str) -> SessionReplay | None:
        """Load the full durable session graph for replay/debug (read-only)."""
        ...

    @abstractmethod
    async def list_sessions(
        self, limit: int = 50, status: str | None = None
    ) -> list[StoredSession]:
        """List recent sessions, optionally filtered by status."""
        ...

    @abstractmethod
    async def update_session_status(self, session_id: str, status: str) -> None:
        """Update session status."""
        ...

    @abstractmethod
    async def check_context_refs(
        self,
        *,
        tool_ids: list[int] | None = None,
        template_ids: list[int] | None = None,
        repo_hashes: list[str] | None = None,
        memory_ids: list[str] | None = None,
    ) -> MissingRefs:
        """Check which context references still exist in the store."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> StoredSession | None:
        """Get session metadata."""
        ...

    @abstractmethod
    async def get_frame_events(self, frame_id: str) -> list[dict[str, Any]]:
        """Get all events for a frame."""
        ...

    @abstractmethod
    async def get_events_by_turn(self, turn_id: str) -> list[dict[str, Any]]:
        """Get all events for a turn."""
        ...

    # === Methods for context factory (snapshot resolution) ===
    @abstractmethod
    async def get_tool_by_id(self, tool_id: int) -> dict[str, Any] | None:
        """Get tool specification by ID."""
        ...

    @abstractmethod
    async def get_tools_by_ids(self, tool_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple tool specifications by IDs in a single query."""
        ...

    @abstractmethod
    async def get_template_by_id(self, template_id: int) -> dict[str, Any] | None:
        """Get template by ID with metadata."""
        ...

    @abstractmethod
    async def get_templates_by_ids(
        self, template_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Get multiple templates by IDs in a single query."""
        ...

    @abstractmethod
    async def get_repo_snapshot_by_hash(
        self, repo_snapshot_hash: str
    ) -> RepoSnapshotData | None:
        """Get repository snapshot by content hash."""
        ...

    @abstractmethod
    async def register_repo_snapshot(
        self,
        target_project_root: str,
        repo_map_json: str,
        project_info_json: str | None,
        repo_snapshot_hash: str,
    ) -> RepoSnapshotResult:
        """Register a repository snapshot (insert or ignore by hash)."""
        ...

    @abstractmethod
    async def get_memory_by_id(self, memory_id: str) -> str | None:
        """Get memory content by ID."""
        ...

    @abstractmethod
    async def get_memories_by_ids(self, memory_ids: list[str]) -> list[str]:
        """Get multiple memory contents by IDs in a single query."""
        ...

    @abstractmethod
    async def update_context_artifacts(
        self,
        tool_specs: list[dict[str, Any]],
        template_specs: list[dict[str, str]],
        repo_map_data: Any | None,
        project_info: Any | None,
        target_project_root: str,
        memories: list[str],
    ) -> dict[str, Any]:
        """Update context artifacts in a single transaction.

        Registers tools, templates, and repo snapshot if needed.
        Returns dict with generated IDs and versions:
        {
            "tool_ids": list[int],
            "template_ids": list[int],
            "tool_set_version": str,
            "template_set_version": str,
            "repo_snapshot_hash": str,
            "memory_snapshot_version": str,
        }
        """
        ...

    @abstractmethod
    async def get_messages_by_range(
        self,
        session_id: str,
        start_seq: int,
        end_seq: int,
    ) -> list[dict[str, Any]]:
        """Get messages in sequence range."""
        ...

    @abstractmethod
    async def get_all_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        ...

    @abstractmethod
    async def branch_session(
        self,
        new_session_id: str,
        parent_session_id: str,
        branch_from_turn_id: str,
    ) -> StoredSession:
        """Create a new session branched from parent at specific turn.

        Atomically copies session, turns, frames, messages, and snapshots
        up to the branch point in a single transaction.

        Args:
            new_session_id: ID for the new branched session
            parent_session_id: Session to branch from
            branch_from_turn_id: Turn ID to branch from (uses its message_boundary)

        Returns:
            New StoredSession for the branched session
        """
        ...

    @abstractmethod
    async def cleanup_retention(self, retention_days: dict[str, int]) -> None:
        """Clean up expired data based on retention policy."""
        ...

    # === Workspace File Baselines ===
    @abstractmethod
    async def upsert_file_baseline(
        self,
        workspace_id: str,
        observation: dict[str, Any],
        turn_id: str | None,
        frame_id: str | None,
        session_id: str | None = None,
    ) -> None:
        """Store or update a baseline file observation."""
        ...

    @abstractmethod
    async def load_file_baselines(
        self, workspace_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load all baseline observations for a workspace and optionally a session."""
        ...

    @abstractmethod
    async def delete_file_baseline(self, workspace_id: str, file_path: str) -> None:
        """Delete a baseline observation."""
        ...

    # === Workspace Identity ===
    @abstractmethod
    async def get_workspace(self, workspace_root: str) -> WorkspaceRecord | None:
        """Find a persisted workspace record matching the given root.

        Args:
            workspace_root: Absolute workspace root path

        Returns:
            The matching WorkspaceRecord, or None if not present
        """
        ...

    @abstractmethod
    async def upsert_workspace(
        self,
        workspace_id: str,
        workspace_root: str,
        config_hash: str,
    ) -> WorkspaceRecord:
        """Persist a new workspace identity record.

        Args:
            workspace_id: Stable UUID for the workspace
            workspace_root: Absolute workspace root path
            config_hash: Hash of the effective config at creation time

        Returns:
            The persisted WorkspaceRecord
        """
        ...

    @abstractmethod
    async def refresh_workspace_config_hash(
        self, workspace_id: str, config_hash: str
    ) -> None:
        """Update the recorded config hash for an existing workspace.

        Lets callers detect drift against the stored value instead of
        silently invalidating history when a known workspace is reopened
        with a changed effective config.

        Args:
            workspace_id: Workspace to update
            config_hash: New effective config hash
        """
        ...

    # === Event processing (delegated to internal adapter) ===
    @abstractmethod
    async def process_event(self, event: RuntimeEvent) -> None:
        """Process a runtime event through the internal adapter."""
        ...

    @abstractmethod
    async def get_max_seq(self, session_id: str) -> int:
        """Get the maximum event sequence number for a session."""
        ...


class ExecutionStore(IExecutionStore, IExecutionStoreReader):
    """SQLite-based execution store with repository layer and event adapter."""

    def __init__(
        self, db_path: Path, flush_interval_ms: int = 1000, batch_size: int = 100
    ):
        self.db_path = db_path
        self.flush_interval_ms = flush_interval_ms
        self.batch_size = batch_size
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

        # Internal event adapter for in-memory rebase
        self._event_adapter = StoreEventAdapter()

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
        self._workspace_file_baseline_repo: WorkspaceFileBaselineRepo | None = None
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
    def workspace_file_baseline_repo(self) -> WorkspaceFileBaselineRepo:
        assert self._workspace_file_baseline_repo is not None, "Store not initialized"
        return self._workspace_file_baseline_repo

    @property
    def workspace_repo(self) -> WorkspaceRepo:
        assert self._workspace_repo is not None, "Store not initialized"
        return self._workspace_repo

    async def init(self) -> None:
        """Initialize the store and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        # Load and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()
        await self._conn.executescript(schema_sql)
        await self._conn.commit()

        # Initialize repositories
        self._event_repo = EventRepo(self._conn)
        self._frame_repo = FrameRepo(self._conn)
        self._session_repo = SessionRepo(self._conn)
        self._message_repo = MessageRepo(self._conn)
        self._tool_repo = ToolRepo(self._conn)
        self._template_repo = TemplateRepo(self._conn)
        self._turn_repo = TurnRepo(self._conn)
        self._context_snapshot_repo = ContextSnapshotRepo(self._conn)
        self._repo_snapshot_repo = RepoSnapshotRepo(self._conn)
        self._workspace_file_baseline_repo = WorkspaceFileBaselineRepo(self._conn)
        self._workspace_repo = WorkspaceRepo(self._conn)

        # Start periodic background flush so buffered (non-terminal) events,
        # e.g. ApprovalRequiredEvent / ContinuationDecisionEvent emitted while
        # a turn is paused for approval, are durably persisted within
        # flush_interval_ms rather than waiting for the next flush-triggering
        # event or close().
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info("ExecutionStore initialized", db_path=str(self.db_path))

    async def close(self) -> None:
        """Close the store and flush pending writes."""
        # Stop the periodic background flush first so no writes race the close
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        final_batch = self._event_adapter.close()
        if final_batch and self._conn is not None:
            try:
                await self._write_batch(final_batch)
            except Exception:
                logger.exception("Failed to flush final batch on store close")
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        logger.info("ExecutionStore closed")

    async def _flush_loop(self) -> None:
        """Periodically flush buffered (non-terminal) events to the store.

        Runs for the lifetime of the store. Events that do not trigger an
        immediate flush via ``_should_flush`` (e.g. ``ApprovalRequiredEvent``
        emitted while a turn is paused for approval) are still written to
        SQLite within ``flush_interval_ms``, so they become visible in the
        dashboard timeline without waiting for a flush-triggering event or
        ``close()``.
        """
        while True:
            try:
                await asyncio.sleep(self.flush_interval_ms / 1000.0)
                await self.flush()
            except asyncio.CancelledError:
                return
            except Exception as e:
                # Keep the loop alive so a transient write error does not
                # disable flushing for the rest of the process lifetime.
                logger.warning("Periodic flush failed", error=str(e))

    async def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        """Write a batch of operations in a single transaction."""
        if not self._conn:
            logger.error("Cannot write batch: connection not initialized")
            return
        async with self._write_lock:
            await self._write_batch_locked(batch)

    async def _write_batch_locked(self, batch: list[dict[str, Any]]) -> None:
        """Write a batch of operations in a single transaction.

        The caller must hold ``self._write_lock``.
        """
        # Bounded retries for "database is locked" errors.
        # PRAGMA busy_timeout=5000 limits each attempt to ~5s; we retry
        # a few times to handle transient contention without holding the
        # asyncio lock indefinitely.
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._execute_batch_locked(batch)
                logger.debug("Wrote batch", size=len(batch))
                return
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(
                        "Database locked during write, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    if self._conn:
                        await self._conn.rollback()
                    # Brief async yield to let other tasks run before retry
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                logger.error("Failed to write batch", error=str(e))
                if self._conn:
                    await self._conn.rollback()
                raise
            except Exception as e:
                logger.error("Failed to write batch", error=str(e))
                if self._conn:
                    await self._conn.rollback()
                raise

    async def _execute_batch_locked(self, batch: list[dict[str, Any]]) -> None:
        """Execute a batch of operations in a single transaction.

        The caller must hold ``self._write_lock`` and handle retries/errors.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")
        started_tx = not self._conn.in_transaction
        if started_tx:
            await self._conn.execute("BEGIN IMMEDIATE")
        try:
            for item in batch:
                await self._execute_batch_item(item)
            if started_tx:
                await self._conn.commit()
        except Exception:
            if started_tx and self._conn.in_transaction:
                await self._conn.rollback()
            raise

    async def _execute_batch_item(self, item: dict[str, Any]) -> None:
        """Execute a single batch item operation using dispatch table."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        op_type = item["op"]
        data = item["data"]

        handlers = {
            "event": lambda: self.event_repo.append(**data),
            "message": lambda: self.message_repo.append(**data),
            "associate_tool": lambda: self._conn.execute(
                "INSERT OR IGNORE INTO session_tools (session_id, tool_id) VALUES (?, ?)",
                (data["session_id"], data["tool_id"]),
            ),
            "associate_template": lambda: self._conn.execute(
                "INSERT OR IGNORE INTO session_templates (session_id, template_id) VALUES (?, ?)",
                (data["session_id"], data["template_id"]),
            ),
            "upsert_session": lambda: self.session_repo.upsert(**data),
            "upsert_turn": lambda: self.turn_repo.upsert(**data),
            "upsert_frame": lambda: self.frame_repo.upsert(**data),
            "upsert_context_snapshot": lambda: self.context_snapshot_repo.upsert(
                **data
            ),
            "increment_session_totals": lambda: self.session_repo.increment_totals(
                **data
            ),
            "increment_turn_totals": lambda: self.turn_repo.increment_totals(**data),
            "increment_frame_tokens": lambda: self.frame_repo.increment_tokens(**data),
            "upsert_file_baseline": lambda: self.upsert_file_baseline(**data),
        }

        handler = handlers.get(op_type)
        if handler is None:
            raise ValueError(f"Unknown batch operation type: {op_type}")
        await handler()

    def _should_flush(self, event_type: EventType) -> bool:
        """Determine if an event type should trigger immediate flush.

        Tool events are included so tool intent (start) and outcome (result)
        are durable before the side effect or before reporting completion -
        this is what makes tool actions reconstructable and skippable after a
        crash.
        """
        return event_type in (
            EventType.ERROR,
            EventType.FRAME_COMPLETE,
            EventType.LLM_COMPLETE,
            EventType.TOOL_RESULT,
            EventType.TURN_COMPLETE,
        )

    async def process_event(self, event: RuntimeEvent) -> None:
        """Process a runtime event through the internal adapter.

        The whole accumulate->prepare->write sequence runs under ``_write_lock``
        so concurrent callers cannot interleave reads/writes of the shared
        adapter state (seq counters, pending events, deltas), which would
        otherwise produce duplicate event rows and sequence collisions.
        """
        async with self._write_lock:
            if not self._event_adapter.has_seq_counter(event.session_id):
                last_seq = await self.event_repo.get_max_seq(event.session_id)
                self._event_adapter.initialize_seq_counter(event.session_id, last_seq)
            self._event_adapter.process(event)
            # Trigger immediate flush for terminal events
            if self._should_flush(event.type):
                batch = self._event_adapter.prepare_batch()
                if batch:
                    await self._write_batch_locked(batch)
                    # Clear events that are now durable so close() does not
                    # re-append them for incomplete (non-terminal) keys. Entity
                    # mutation data is kept so subsequent frame/turn updates can
                    # still merge against their create data.
                    self._event_adapter.clear_flushed_events()

    async def flush(self) -> None:
        """Flush all pending writes immediately."""
        async with self._write_lock:
            batch = self._event_adapter.prepare_batch()
            if batch:
                await self._write_batch_locked(batch)
                self._event_adapter.clear_flushed_events()

    async def get_max_seq(self, session_id: str) -> int:
        """Get the maximum event sequence number for a session."""
        return await self.event_repo.get_max_seq(session_id)

    async def register_tool(
        self,
        name: str,
        schema_json: str,
        risk_class: str,
        description: str | None = None,
        version: str = "1.0.0",
    ) -> int:
        """Register a global tool. Returns tool_id."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        result = await self.tool_repo.register(
            name, schema_json, risk_class, description, version
        )
        await self._conn.commit()
        return result

    async def register_template(
        self,
        path: str,
        content: str,
        content_hash: str,
        version: str = "1.0.0",
    ) -> int:
        """Register a global template. Returns template_id."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        result = await self.template_repo.register(path, content, content_hash, version)
        await self._conn.commit()
        return result

    async def create_session(
        self,
        session_id: str,
        user_input: str,
        agent_mode: str,
        model_name: str | None,
        metadata_json: str | None = None,
    ) -> None:
        """Create a new session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        status = "running"
        started_at_ms = int(datetime.now(UTC).timestamp() * 1000)

        await self.session_repo.create(
            session_id,
            user_input,
            status,
            started_at_ms,
            agent_mode,
            model_name,
            metadata_json,
        )
        await self._conn.commit()

    async def associate_session_tools(
        self, session_id: str, tool_ids: list[int]
    ) -> None:
        """Associate tools with a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        for tool_id in tool_ids:
            await self._conn.execute(
                "INSERT OR IGNORE INTO session_tools (session_id, tool_id) VALUES (?, ?)",
                (session_id, tool_id),
            )
        await self._conn.commit()

    async def associate_session_templates(
        self, session_id: str, template_ids: list[int]
    ) -> None:
        """Associate templates with a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        for template_id in template_ids:
            await self._conn.execute(
                "INSERT OR IGNORE INTO session_templates (session_id, template_id) VALUES (?, ?)",
                (session_id, template_id),
            )
        await self._conn.commit()

    async def create_frame(
        self,
        frame_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
        iteration: int,
        model_name: str | None = None,
        context_snapshot_id: str | None = None,
        history_through_seq: int = 0,
        prompt_hash: str | None = None,
    ) -> None:
        """Create a new frame (new signature with optional turn_id)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        status = "running_llm"
        started_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self.frame_repo.create(
            frame_id,
            session_id,
            trace_id,
            iteration,
            status,
            started_at_ms,
            model_name,
            turn_id,
            context_snapshot_id,
            history_through_seq,
            prompt_hash,
        )
        await self._conn.commit()

    # === Turn Management ===
    async def create_turn(
        self,
        turn_id: str,
        session_id: str,
        trace_id: str,
        turn_index: int,
        user_input: str,
        status: str = "running",
        started_at_ms: int | None = None,
        context_snapshot_id: str | None = None,
        config_snapshot_json: str | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Create a new turn."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        if started_at_ms is None:
            started_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self.turn_repo.create(
            turn_id,
            session_id,
            trace_id,
            turn_index,
            user_input,
            status,
            started_at_ms,
            context_snapshot_id,
            config_snapshot_json,
            final_answer,
        )
        await self._conn.commit()

    async def update_turn_status(
        self,
        turn_id: str,
        status: str,
        finished_at_ms: int | None = None,
        total_input_tokens: int | None = None,
        total_output_tokens: int | None = None,
        total_frames: int | None = None,
        paused_at_ms: int | None = None,
        pause_reason: str | None = None,
        resumed_at_ms: int | None = None,
        final_answer: str | None = None,
    ) -> None:
        """Update turn status and metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        await self.turn_repo.update_status(
            turn_id,
            status,
            finished_at_ms,
            total_input_tokens,
            total_output_tokens,
            total_frames,
            paused_at_ms,
            pause_reason,
            resumed_at_ms,
            final_answer,
        )
        await self._conn.commit()

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

    async def get_last_turn(self, session_id: str) -> StoredTurn | None:
        """Get the last turn for a session (highest turn_index)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        turns = await self.turn_repo.list_by_session(session_id)
        if not turns:
            return None
        return max(turns, key=lambda t: t.turn_index)

    # === Context Snapshot Management ===
    async def create_context_snapshot(
        self,
        snapshot_id: str,
        turn_id: str,
        schema_version: int = 1,
        tool_set_ref: str | None = None,
        template_set_ref: str | None = None,
        repo_snapshot_ref: str | None = None,
        memory_snapshot_ref: str | None = None,
        message_boundary_json: str | None = None,
        content_hash: str = "",
        payload_json: str | None = None,
        created_at_ms: int | None = None,
    ) -> None:
        """Create a new context snapshot."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        if created_at_ms is None:
            created_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self.context_snapshot_repo.create(
            snapshot_id,
            turn_id,
            schema_version,
            tool_set_ref,
            template_set_ref,
            repo_snapshot_ref,
            memory_snapshot_ref,
            message_boundary_json,
            content_hash,
            payload_json,
            created_at_ms,
        )
        await self._conn.commit()

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

    async def update_frame_status(
        self,
        frame_id: str,
        status: ExecutionStatus,
        finished_at: datetime | None = None,
        duration_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        error_message: str | None = None,
        pending_call_id: str | None = None,
    ) -> None:
        """Update frame status and metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        finished_at_ms = int(finished_at.timestamp() * 1000) if finished_at else None
        await self.frame_repo.update_status(
            frame_id,
            status,
            finished_at_ms,
            duration_ms,
            input_tokens,
            output_tokens,
            finish_reason,
            error_message,
            pending_call_id,
        )
        await self._conn.commit()

    async def append_message(
        self,
        session_id: str,
        turn_id: str,
        frame_id: str,
        seq: int | None,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        metadata_json: str | None = None,
        dedupe_key: str | None = None,
    ) -> None:
        """Append a message to the turn (new signature with optional turn_id)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        created_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        await self.message_repo.append(
            session_id,
            turn_id,
            frame_id,
            seq,
            role,
            content,
            tool_call_id,
            tool_name,
            metadata_json,
            created_at_ms,
            dedupe_key,
        )
        await self._conn.commit()

    async def append_event(
        self,
        trace_id: str,
        session_id: str,
        turn_id: str | None,
        frame_id: str | None,
        seq: int,
        event_type: EventType,
        data_json: str,
        timestamp: datetime | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        """Append an event to the store (new signature with optional turn_id)."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        if timestamp_ms is None:
            if timestamp is None:
                timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
            else:
                timestamp_ms = int(timestamp.timestamp() * 1000)
        await self.event_repo.append(
            trace_id,
            session_id,
            turn_id,
            frame_id,
            seq,
            event_type.value,
            data_json,
            timestamp_ms,
        )
        await self._conn.commit()

    async def load_resume_batch(
        self,
        session_id: str,
        turn_id: str | None = None,
        turn_index: int | None = None,
        frame_id: str | None = None,
    ) -> ResumeBatch | None:
        """Load everything the resume flow needs in a constant number of queries.

        Flow-shaped: session + its turns (selection/branch) + the selected
        turn + its snapshot + that turn's frame rows + events of the single
        frame being resumed + the snapshot's missing refs. The full session
        graph (all turns' frames and events) is intentionally not loaded -
        that shape belongs to the future replay flow.

        Args:
            session_id: Session to resume
            turn_id: Specific turn (defaults to highest turn_index)
            turn_index: Alternative turn selector (ignored when turn_id given)
            frame_id: Frame to resume (defaults to the turn's last frame)
        """
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
        """Load one turn's durable graph for replay/debug.

        Args:
            turn_id: Turn to load

        Returns:
            TurnReplay with snapshot, frames, per-frame events and messages,
            or None if the turn does not exist
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        turn = await self.turn_repo.get(turn_id)
        if not turn:
            return None

        return await self._build_turn_replay(turn)

    async def load_replay_batch(self, session_id: str) -> SessionReplay | None:
        """Load the full durable session graph for replay/debug (read-only).

        Loads session + turns + all frames + per-frame events + messages in a
        constant number of queries, grouped per turn.

        Args:
            session_id: Session to load

        Returns:
            SessionReplay, or None if the session does not exist
        """
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

    async def list_sessions(
        self, limit: int = 50, status: str | None = None
    ) -> list[StoredSession]:
        """List recent sessions, optionally filtered by status.

        Args:
            limit: Maximum number of sessions to return
            status: Optional status filter (e.g. 'completed', 'failed')

        Returns:
            Sessions ordered by recency
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        sessions = await self.session_repo.list_recent(limit)
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    async def update_session_status(self, session_id: str, status: str) -> None:
        """Update session status."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        await self.session_repo.update_status(session_id, status)
        await self._conn.commit()

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

        # NOTE: memory refs are version-only constructs (MemorySnapshotRef
        # carries a version hash, never per-id rows), so there is nothing to
        # existence-check for them here.

        return MissingRefs(
            tool_ids=missing_tools,
            template_ids=missing_templates,
            repo_hashes=missing_repo_hashes,
            memory_ids=missing_memory_ids,
        )

    async def get_tool_by_id(self, tool_id: int) -> dict[str, Any] | None:
        """Get tool specification by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        result = await self.tool_repo.get(tool_id)
        await self._conn.commit()
        return result

    async def get_tools_by_ids(self, tool_ids: list[int]) -> list[dict[str, Any]]:
        """Get multiple tool specifications by IDs in a single query."""
        return await self.tool_repo.get_many(tool_ids)

    async def get_template_by_id(self, template_id: int) -> dict[str, Any] | None:
        """Get template by ID with metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        result = await self.template_repo.get(template_id)
        await self._conn.commit()
        return result

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

    async def register_repo_snapshot(
        self,
        target_project_root: str,
        repo_map_json: str,
        project_info_json: str | None,
        repo_snapshot_hash: str,
    ) -> RepoSnapshotResult:
        """Register a repository snapshot (insert or ignore by hash)."""
        return await self.repo_snapshot_repo.register(
            target_project_root, repo_map_json, project_info_json, repo_snapshot_hash
        )

    async def _register_tools(
        self, tool_specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Register tools and return list with IDs."""
        tools_with_ids = []
        for spec in tool_specs:
            name = spec["name"]
            parameters = spec.get("parameters", {})
            schema_json = json.dumps(parameters, sort_keys=True)
            risk_class = "low"
            description = spec.get("description", "")
            version = "1.0.0"

            tool_id = await self.tool_repo.register(
                name=name,
                schema_json=schema_json,
                risk_class=risk_class,
                description=description,
                version=version,
            )
            tools_with_ids.append(
                {
                    "id": tool_id,
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
        return tools_with_ids

    async def _register_templates(
        self, template_specs: list[dict[str, str]]
    ) -> tuple[list[int], list[str], list[str]]:
        """Register templates and return (ids, paths, hashes)."""
        template_ids = []
        template_paths = []
        template_hashes = []
        for spec in template_specs:
            for path, content in spec.items():
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                template_id = await self.template_repo.register(
                    path=path,
                    content=content,
                    content_hash=content_hash,
                    version="1.0.0",
                )
                template_ids.append(template_id)
                template_paths.append(path)
                template_hashes.append(content_hash)
        return template_ids, template_paths, template_hashes

    async def _register_repo_snapshot(
        self,
        repo_map_data: Any | None,
        project_info: Any | None,
        target_project_root: str,
    ) -> str:
        """Register repo snapshot and return hash."""
        import dataclasses

        repo_snapshot_hash = "no_repo"
        if repo_map_data is not None:
            repo_map_json = json.dumps(dataclasses.asdict(repo_map_data), default=str)
            project_info_json = project_info.model_dump_json() if project_info else None
            repo_snapshot_hash = hashlib.sha256(
                json.dumps(
                    {
                        "repo_map": repo_map_json,
                        "project_info": project_info_json,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]

            await self.repo_snapshot_repo.register(
                target_project_root=target_project_root,
                repo_map_json=repo_map_json,
                project_info_json=project_info_json,
                repo_snapshot_hash=repo_snapshot_hash,
            )
        return repo_snapshot_hash

    def _compute_versions(
        self,
        tools_with_ids: list[dict[str, Any]],
        template_hashes: list[str],
        memories: list[str],
    ) -> tuple[str, str, str]:
        """Compute version strings for tools, templates, and memories."""
        tool_set_version = compute_tool_set_version(tools_with_ids)
        template_set_version = hashlib.sha256(
            json.dumps(sorted(template_hashes), sort_keys=True).encode()
        ).hexdigest()[:16]
        memory_snapshot_version = "empty"
        if memories:
            memory_snapshot_version = hashlib.sha256(
                json.dumps(sorted(memories), sort_keys=True).encode()
            ).hexdigest()[:16]
        return tool_set_version, template_set_version, memory_snapshot_version

    async def update_context_artifacts(
        self,
        tool_specs: list[dict[str, Any]],
        template_specs: list[dict[str, str]],
        repo_map_data: Any | None,
        project_info: Any | None,
        target_project_root: str,
        memories: list[str],
    ) -> dict[str, Any]:
        """Update context artifacts in a single transaction.

        Registers tools, templates, and repo snapshot if needed.
        Uses repos for deduplication by content_hash.
        Returns dict with generated IDs and versions.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        async with self._write_lock:
            started_tx = not self._conn.in_transaction
            try:
                if started_tx:
                    await self._conn.execute("BEGIN IMMEDIATE")

                tools_with_ids = await self._register_tools(tool_specs)
                (
                    template_ids,
                    _template_paths,
                    template_hashes,
                ) = await self._register_templates(template_specs)
                repo_snapshot_hash = await self._register_repo_snapshot(
                    repo_map_data, project_info, target_project_root
                )
                tool_set_version, template_set_version, memory_snapshot_version = (
                    self._compute_versions(tools_with_ids, template_hashes, memories)
                )

                if started_tx:
                    await self._conn.commit()

                return {
                    "tools_with_ids": tools_with_ids,
                    "template_ids": template_ids,
                    "tool_set_version": tool_set_version,
                    "template_set_version": template_set_version,
                    "repo_snapshot_hash": repo_snapshot_hash,
                    "memory_snapshot_version": memory_snapshot_version,
                }

            except Exception as e:
                logger.error("Failed to update context artifacts", error=str(e))
                if started_tx and self._conn.in_transaction:
                    await self._conn.rollback()
                raise

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

        result = await self.message_repo.get_by_range(session_id, start_seq, end_seq)
        await self._conn.commit()

        return result

    async def get_all_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get all messages for a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.message_repo.get_by_session(session_id)

    async def branch_session(
        self,
        new_session_id: str,
        parent_session_id: str,
        branch_from_turn_id: str,
    ) -> StoredSession:
        """Create a new session branched from parent at specific turn.

        Atomically copies the session, turns, frames, messages and snapshots
        up to the branch point in a single transaction.

        The fork is a self-consistent mini-session: every foreign key is
        re-pointed at the forked rows (turns by turn_index, context
        snapshots, frames, events, messages, file baselines), so resuming
        inside the fork never touches the parent. A fork from an
        interrupted turn inherits the interrupted state and is itself
        resumable.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        # Get parent session
        parent_session = await self.session_repo.get(parent_session_id)
        if not parent_session:
            raise ValueError(f"Parent session not found: {parent_session_id}")

        # Get branch turn to determine message boundary
        branch_turn = await self.turn_repo.get(branch_from_turn_id)
        if not branch_turn:
            raise ValueError(f"Branch turn not found: {branch_from_turn_id}")

        # Get message boundary from branch turn
        message_boundary = None
        if branch_turn.message_boundary_json:
            message_boundary = json.loads(branch_turn.message_boundary_json)

        # Get all turns up to and including branch turn
        parent_turns = await self.turn_repo.list_by_session(parent_session_id)
        branch_turn_index = branch_turn.turn_index
        turns_to_copy = [t for t in parent_turns if t.turn_index <= branch_turn_index]
        parent_turn_ids = {t.id for t in turns_to_copy}

        # Frames of the copied turns (loaded up-front so the forked
        # session's usage counters can report them)
        parent_frames = await self.frame_repo.list_by_session(parent_session_id)
        frames_to_copy = [f for f in parent_frames if f.turn_id in parent_turn_ids]

        async with self._write_lock:
            started_tx = not self._conn.in_transaction
            try:
                if started_tx:
                    await self._conn.execute("BEGIN IMMEDIATE")

                await self._copy_branch_session_row(
                    new_session_id=new_session_id,
                    parent_session_id=parent_session_id,
                    parent_session=parent_session,
                    branch_from_turn_id=branch_from_turn_id,
                    branch_turn=branch_turn,
                    turns_to_copy=turns_to_copy,
                    frames_to_copy=frames_to_copy,
                )

                (
                    turn_index_to_new_id,
                    turn_id_to_new_id,
                    snapshot_id_to_new_id,
                ) = await self._copy_branch_turns_and_snapshots(
                    turns_to_copy, new_session_id
                )

                old_frame_id_to_new = await self._copy_branch_frames(
                    frames_to_copy,
                    turn_id_to_new_id,
                    snapshot_id_to_new_id,
                    new_session_id,
                )

                await self._copy_branch_events(
                    frames_to_copy,
                    old_frame_id_to_new,
                    turn_id_to_new_id,
                    new_session_id,
                )

                await self._copy_branch_messages(
                    parent_session_id,
                    turns_to_copy,
                    turn_index_to_new_id,
                    old_frame_id_to_new,
                    new_session_id,
                    message_boundary,
                )

                await self._copy_branch_associations(parent_session_id, new_session_id)
                await self._copy_branch_baselines(
                    parent_session_id,
                    parent_turn_ids,
                    turn_id_to_new_id,
                    old_frame_id_to_new,
                    new_session_id,
                )

                await self._conn.commit()

                # Return new session
                new_session = await self.session_repo.get(new_session_id)
                if not new_session:
                    raise RuntimeError(
                        f"Failed to retrieve branched session {new_session_id} after creation"
                    )
                return new_session

            except Exception as e:
                logger.error("Failed to branch session", error=str(e))
                if self._conn.in_transaction:
                    await self._conn.rollback()
                raise

    # === Branch cascade copy helpers ===
    # Each helper writes rows directly on the connection and must run
    # inside the branch_session transaction (no commit/rollback inside).

    async def _copy_branch_session_row(
        self,
        *,
        new_session_id: str,
        parent_session_id: str,
        parent_session: StoredSession,
        branch_from_turn_id: str,
        branch_turn: StoredTurn,
        turns_to_copy: list[StoredTurn],
        frames_to_copy: list[StoredFrame],
    ) -> None:
        """Create the forked session row, inheriting the interrupted state."""
        started_at_ms = int(datetime.now(UTC).timestamp() * 1000)
        metadata = (
            dict(json.loads(parent_session.metadata_json))
            if parent_session.metadata_json
            else {}
        )
        metadata["branched_from_session"] = parent_session_id
        metadata["branched_from_turn"] = branch_from_turn_id

        # A fork from an interrupted turn inherits the interrupted state: the
        # branch-point turn is copied with its 'running' status, so the
        # branched session must be 'running' too - otherwise the resume
        # planner's idle-session guard rejects resuming the fork (no-message
        # resume from a past turn).
        branch_status = "running" if branch_turn.status == "running" else "idle"

        await self.session_repo.upsert(
            session_id=new_session_id,
            user_input=parent_session.user_input,
            status=branch_status,
            started_at_ms=started_at_ms,
            agent_mode=parent_session.agent_mode,
            model_name=parent_session.model_name,
            total_input_tokens=0,
            total_output_tokens=0,
            total_turns=len(turns_to_copy),
            total_frames=len(frames_to_copy),
            metadata_json=json.dumps(metadata),
            context_profile_json=parent_session.context_profile_json,
        )

    async def _copy_branch_turns_and_snapshots(
        self, turns_to_copy: list[StoredTurn], new_session_id: str
    ) -> tuple[dict[int, str], dict[str, str], dict[str, str]]:
        """Copy turns (and their snapshots) under new IDs.

        Turn IDs and snapshot IDs are regenerated; ``turn_index`` is
        preserved, which is what makes index-based selectors survive the
        fork.

        Returns:
            (turn_index -> new turn id, old turn id -> new turn id,
            old snapshot id -> new snapshot id)
        """
        turn_index_to_new_id: dict[int, str] = {}
        turn_id_to_new_id: dict[str, str] = {}
        snapshot_id_to_new_id: dict[str, str] = {}

        for turn in turns_to_copy:
            new_turn_id = f"turn_{uuid.uuid4()}"
            turn_index_to_new_id[turn.turn_index] = new_turn_id
            turn_id_to_new_id[turn.id] = new_turn_id

            new_snapshot_id = turn.context_snapshot_id
            if turn.context_snapshot_id:
                new_snapshot_id = f"snap_{uuid.uuid4()}"
                snapshot_id_to_new_id[turn.context_snapshot_id] = new_snapshot_id

            await self.turn_repo.upsert(
                turn_id=new_turn_id,
                session_id=new_session_id,
                trace_id=turn.trace_id,
                turn_index=turn.turn_index,
                user_input=turn.user_input,
                status=turn.status,
                started_at_ms=turn.started_at_ms,
                finished_at_ms=turn.finished_at_ms,
                paused_at_ms=turn.paused_at_ms,
                pause_reason=turn.pause_reason,
                context_snapshot_id=new_snapshot_id,
                config_snapshot_json=turn.config_snapshot_json,
                total_input_tokens=turn.total_input_tokens,
                total_output_tokens=turn.total_output_tokens,
                total_frames=turn.total_frames,
                message_boundary_json=turn.message_boundary_json,
                final_answer=turn.final_answer,
            )

        for turn in turns_to_copy:
            if not turn.context_snapshot_id:
                continue
            snapshot = await self.context_snapshot_repo.get(turn.context_snapshot_id)
            if not snapshot:
                continue
            await self.context_snapshot_repo.upsert(
                snapshot_id=snapshot_id_to_new_id[turn.context_snapshot_id],
                turn_id=turn_index_to_new_id[turn.turn_index],
                schema_version=snapshot.schema_version,
                tool_set_ref=self._snapshot_ref_json(snapshot.tool_set_ref),
                template_set_ref=self._snapshot_ref_json(snapshot.template_set_ref),
                repo_snapshot_ref=self._snapshot_ref_json(snapshot.repo_snapshot_ref),
                memory_snapshot_ref=self._snapshot_ref_json(
                    snapshot.memory_snapshot_ref
                ),
                message_boundary_json=self._snapshot_ref_json(
                    snapshot.message_boundary
                ),
                content_hash=snapshot.content_hash,
                payload_json=snapshot.payload_json,
                created_at_ms=snapshot.created_at_ms,
            )

        return turn_index_to_new_id, turn_id_to_new_id, snapshot_id_to_new_id

    @staticmethod
    def _snapshot_ref_json(ref: Any) -> str | None:
        """Serialize a snapshot reference model, tolerating legacy NULLs."""
        return ref.model_dump_json() if ref else None

    async def _copy_branch_frames(
        self,
        frames_to_copy: list[StoredFrame],
        turn_id_to_new_id: dict[str, str],
        snapshot_id_to_new_id: dict[str, str],
        new_session_id: str,
    ) -> dict[str, str]:
        """Copy frames onto their forked turns under new IDs.

        Returns:
            Mapping of old frame id -> new frame id
        """
        old_frame_id_to_new: dict[str, str] = {}

        for frame in frames_to_copy:
            new_frame_id = f"frame_{uuid.uuid4()}"
            old_frame_id_to_new[frame.id] = new_frame_id

            # The forked frame must point at the fork's snapshot, not the
            # parent's, so a resumed frame stays consistent with its turn.
            snapshot_id = frame.context_snapshot_id
            if snapshot_id in snapshot_id_to_new_id:
                snapshot_id = snapshot_id_to_new_id[snapshot_id]

            await self.frame_repo.upsert(
                frame_id=new_frame_id,
                session_id=new_session_id,
                # Frames are re-attached by their owning turn ID (turn IDs
                # are regenerated in the branch); frame iterations are
                # per-turn and must never be used as turn selectors.
                turn_id=turn_id_to_new_id[frame.turn_id],
                trace_id=frame.trace_id,
                iteration=frame.iteration,
                status=frame.status,
                started_at_ms=frame.started_at_ms,
                finished_at_ms=frame.finished_at_ms,
                duration_ms=frame.duration_ms,
                input_tokens=frame.input_tokens,
                output_tokens=frame.output_tokens,
                model_name=frame.model_name,
                finish_reason=frame.finish_reason,
                error_message=frame.error_message,
                pending_call_id=frame.pending_call_id,
                context_snapshot_id=snapshot_id,
                history_through_seq=frame.history_through_seq,
                prompt_hash=frame.prompt_hash,
            )

        return old_frame_id_to_new

    async def _copy_branch_events(
        self,
        frames_to_copy: list[StoredFrame],
        old_frame_id_to_new: dict[str, str],
        turn_id_to_new_id: dict[str, str],
        new_session_id: str,
    ) -> None:
        """Copy the frames' events onto the forked rows.

        Resuming a forked turn rebuilds the frame from these events
        (recorded LLM turn, tool calls, results); without them the fork
        would silently restart the turn from an empty frame.
        """
        for frame in frames_to_copy:
            new_frame_id = old_frame_id_to_new[frame.id]
            new_turn_id = turn_id_to_new_id[frame.turn_id]
            for event in await self.event_repo.get_by_frame(frame.id):
                await self.event_repo.append(
                    trace_id=event["trace_id"],
                    session_id=new_session_id,
                    turn_id=new_turn_id,
                    frame_id=new_frame_id,
                    seq=event["seq"],
                    event_type=event["type"],
                    data_json=event["data_json"],
                    timestamp_ms=event["timestamp"],
                )

    async def _copy_branch_messages(
        self,
        parent_session_id: str,
        turns_to_copy: list[StoredTurn],
        turn_index_to_new_id: dict[int, str],
        old_frame_id_to_new: dict[str, str],
        new_session_id: str,
        message_boundary: dict[str, Any] | None,
    ) -> None:
        """Copy transcript messages for the copied turns."""
        if message_boundary:
            start_seq = message_boundary.get("start_seq", 0)
            end_seq = message_boundary.get("end_seq", 0)
            messages = await self.message_repo.get_by_range(
                parent_session_id, start_seq, end_seq
            )
        else:
            messages = await self.message_repo.get_by_session(parent_session_id)

        # Filter messages to only those in copied turns
        parent_turn_ids = {t.id for t in turns_to_copy}
        messages_to_copy = [m for m in messages if m.get("turn_id") in parent_turn_ids]
        turn_index_by_turn_id = {t.id: t.turn_index for t in turns_to_copy}

        for msg in messages_to_copy:
            new_turn_id = turn_index_to_new_id[turn_index_by_turn_id[msg["turn_id"]]]
            # frame_id is NOT NULL; unmapped (cross-turn) frames keep their
            # original reference.
            new_frame_id = old_frame_id_to_new.get(msg["frame_id"], msg["frame_id"])

            await self.message_repo.append(
                session_id=new_session_id,
                turn_id=new_turn_id,
                frame_id=new_frame_id,
                seq=msg["seq"],
                role=msg["role"],
                content=msg["content"],
                tool_call_id=msg["tool_call_id"],
                tool_name=msg["tool_name"],
                metadata_json=msg["metadata_json"],
                created_at_ms=msg["created_at"],
                dedupe_key=msg["dedupe_key"],
            )

    async def _copy_branch_associations(
        self, parent_session_id: str, new_session_id: str
    ) -> None:
        """Copy session_tools and session_templates associations."""
        assert self._conn is not None, "Store not initialized"
        await self._conn.execute(
            """
            INSERT INTO session_tools (session_id, tool_id)
            SELECT ?, tool_id FROM session_tools WHERE session_id = ?
            """,
            (new_session_id, parent_session_id),
        )
        await self._conn.execute(
            """
            INSERT INTO session_templates (session_id, template_id)
            SELECT ?, template_id FROM session_templates WHERE session_id = ?
            """,
            (new_session_id, parent_session_id),
        )

    async def _copy_branch_baselines(
        self,
        parent_session_id: str,
        parent_turn_ids: set[str],
        turn_id_to_new_id: dict[str, str],
        old_frame_id_to_new: dict[str, str],
        new_session_id: str,
    ) -> None:
        """Copy file baselines of the copied turns onto the forked rows."""
        assert self._conn is not None, "Store not initialized"

        rows = await self.workspace_file_baseline_repo.get_by_turn_id(
            parent_session_id, parent_turn_ids
        )

        for row in rows:
            (
                workspace_id,
                file_path,
                hash_val,
                size,
                mtime,
                inode,
                source,
                observed_at,
                old_turn_id,
                old_frame_id,
            ) = row
            new_turn_id = turn_id_to_new_id.get(old_turn_id)
            new_frame_id = (
                old_frame_id_to_new.get(old_frame_id) if old_frame_id else None
            )

            if new_turn_id:
                await self.workspace_file_baseline_repo.upsert(
                    workspace_id=workspace_id,
                    observation={
                        "path": file_path,
                        "hash": hash_val,
                        "size": size,
                        "mtime": mtime,
                        "inode": inode,
                        "source": {"value": source},
                        "observed_at": observed_at,
                    },
                    turn_id=new_turn_id,
                    frame_id=new_frame_id,
                    session_id=new_session_id,
                )

    async def get_session(self, session_id: str) -> StoredSession | None:
        """Get session metadata."""
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.session_repo.get(session_id)

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

    async def cleanup_retention(self, retention_days: dict[str, int]) -> None:
        """Clean up expired data based on retention policy."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        default_days = retention_days.get("default", 30)
        cutoff_ms = int((datetime.now(UTC).timestamp() - default_days * 86400) * 1000)

        # Delete expired repo snapshots
        await self._conn.execute(
            "DELETE FROM repo_snapshots WHERE created_at < ?", (cutoff_ms,)
        )

        # Delete expired tools
        await self._conn.execute("DELETE FROM tools WHERE created_at < ?", (cutoff_ms,))

        # Delete expired templates
        await self._conn.execute(
            "DELETE FROM templates WHERE created_at < ?", (cutoff_ms,)
        )

        # Delete expired workspaces
        await self._conn.execute(
            "DELETE FROM workspaces WHERE updated_at < ?", (cutoff_ms,)
        )

        # Delete expired sessions and cascading data (exclude running sessions)
        await self._conn.execute(
            "DELETE FROM sessions WHERE updated_at < ? AND status != 'running'",
            (cutoff_ms,),
        )
        # SQLite will cascade delete related rows due to foreign keys
        await self._conn.commit()

        logger.info("Retention cleanup completed", cutoff_ms=cutoff_ms)

    # === Workspace File Baselines ===

    async def upsert_file_baseline(
        self,
        workspace_id: str,
        observation: dict[str, Any],
        turn_id: str | None,
        frame_id: str | None,
        session_id: str | None = None,
    ) -> None:
        """Store or update a baseline file observation.

        No lock is acquired here; callers that need atomicity (e.g. the batch
        writer) already hold ``self._write_lock`` and manage the transaction
        boundary themselves.
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")
        await self.workspace_file_baseline_repo.upsert(
            workspace_id, observation, turn_id, frame_id, session_id
        )

    async def load_file_baselines(
        self, workspace_id: str, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Load all baseline observations for a workspace and optionally a session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        return await self.workspace_file_baseline_repo.load(workspace_id, session_id)

    async def delete_file_baseline(self, workspace_id: str, file_path: str) -> None:
        """Delete a baseline observation."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        async with self._write_lock:
            await self.workspace_file_baseline_repo.delete(workspace_id, file_path)
            await self._conn.commit()

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

    async def upsert_workspace(
        self,
        workspace_id: str,
        workspace_root: str,
        config_hash: str,
    ) -> WorkspaceRecord:
        """Persist a new workspace identity record.

        Args:
            workspace_id: Stable UUID for the workspace
            workspace_root: Absolute workspace root path
            config_hash: Hash of the effective config at creation time

        Returns:
            The persisted WorkspaceRecord
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")
        return await self.workspace_repo.create(
            workspace_id, workspace_root, config_hash
        )

    async def refresh_workspace_config_hash(
        self, workspace_id: str, config_hash: str
    ) -> None:
        """Update the recorded config hash for an existing workspace.

        Lets callers detect drift against the stored value instead of
        silently invalidating history when a known workspace is reopened
        with a changed effective config.

        Args:
            workspace_id: Workspace to update
            config_hash: New effective config hash
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")
        await self.workspace_repo.update_config_hash(workspace_id, config_hash)
