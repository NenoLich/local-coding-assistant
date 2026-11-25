"""TurnManager - creates and loads turns with context snapshots.

All persistence via EventPipeline -> StoreSubscriber -> store.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from local_coding_assistant.core.protocols import IConfigManager

if TYPE_CHECKING:
    from local_coding_assistant.runtime.context import ContextManager
    from local_coding_assistant.storage import IExecutionStore, StoredTurn
from local_coding_assistant.runtime.engine.config import RuntimeLimitsConfig
from local_coding_assistant.runtime.engine.limits import RuntimeLimits
from local_coding_assistant.runtime.session import SessionState
from local_coding_assistant.runtime.session.turn import RuntimeTurn, TurnStatus
from local_coding_assistant.utils.logging import get_logger

logger = get_logger("runtime.turn_manager")


class TurnManager:
    """Manages turn creation and loading via events.

    All persistence via EventPipeline -> StoreSubscriber -> store.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        config_manager: IConfigManager,
    ):
        self._context_manager = context_manager
        self._config_manager = config_manager

    async def create_turn(
        self,
        session: SessionState,
        user_input: str,
        trace_id: str,
        turn_index: int,
    ) -> RuntimeTurn:
        """Create a new turn with fresh snapshot.

        Flow:
        1. Get session config from ContextManager (cached or fresh)
        2. Create snapshot via ContextManager (builds & caches refs, validates)
        3. Sync SessionState's profile with the bundle used by the snapshot
        4. Build RuntimeTurn with attached snapshot (NOT persisted yet)
        5. Return turn - persistence happens via TurnStartEvent in FrameAgent
        """
        turn_id = f"turn_{uuid.uuid4()}"

        # 1. Get config from ContextManager (handles static/dynamic via cache)
        context_config = await self._context_manager.get_context_config(session.id)

        # 2. Create snapshot (builds refs, validates)
        snapshot = await self._context_manager.create_snapshot(
            session=session,
            config_snapshot=context_config,
            turn_id=turn_id,
        )

        # 3. Build limits config dynamically from config_manager
        runtime_config = self._config_manager.global_config.runtime
        limits_config = RuntimeLimitsConfig(
            continuation_policy=runtime_config.continuation_policy.model_dump()
        )

        # Create RuntimeLimits from config to use the correct token budget
        limits = RuntimeLimits.from_config(limits_config)

        # 4. Build RuntimeTurn (in-memory, with MANDATORY snapshot)
        # Persistence happens via TurnStartEvent emitted in FrameAgent.run_turn
        turn = RuntimeTurn(
            turn_id=turn_id,
            session_id=session.id,
            trace_id=trace_id,
            turn_index=turn_index,
            user_input=user_input,
            context_snapshot=snapshot,
            context_snapshot_id=snapshot.id,
            context_config=context_config,
            limits_config=limits_config,
            limits=limits,
            status=TurnStatus.RUNNING,
        )

        return turn

    @staticmethod
    async def from_store(
        turn_id: str,
        store: IExecutionStore,
        stored_turn: StoredTurn | None = None,
    ) -> RuntimeTurn:
        """Load turn AND snapshot from store for resume/replay.

        Uses STORED config (deterministic replay). ContextManager resolves refs.
        """
        return await RuntimeTurn.from_store(turn_id, store, stored_turn)
