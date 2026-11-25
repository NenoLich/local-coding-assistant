"""Unit tests for resume validation (pure checks)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from local_coding_assistant.agent.llm import LLMToolCall
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
from local_coding_assistant.runtime.engine.frame import (
    ActionKind,
    ActionRecord,
    ExecutionStatus,
    RuntimeFrame,
)
from local_coding_assistant.runtime.resume import (
    MissingContextReferencesError,
    ResumeValidationError,
    ResumeValidationInput,
    validate_config_support,
    validate_frame,
    validate_history_coverage,
    validate_missing_refs,
    validate_resume_context,
    validate_session_profile,
    validate_snapshot_schema,
    validate_tool_set_drift,
    validate_tool_support,
)
from local_coding_assistant.storage.contracts import MissingRefs


def make_snapshot(schema_version: int = 1) -> ContextSnapshot:
    return ContextSnapshot(
        id="snap_1",
        turn_id="turn_1",
        schema_version=schema_version,
        tool_set_ref=ToolSetRef(tool_ids=[1], tool_names=["read_file"]),
        template_set_ref=TemplateSetRef(template_ids=[2], template_paths=["base.j2"]),
        repo_snapshot_ref=RepoSnapshotRef(
            target_project_root="project_root", repo_snapshot_hash="abc"
        ),
        memory_snapshot_ref=MemorySnapshotRef(memory_ids=["m1"]),
        message_boundary=MessageBoundary(session_id="s", start_seq=1, end_seq=2),
        content_hash="hash",
        created_at_ms=0,
    )


def make_frame(
    tool_names: list[str] | None = None,
    history_through_seq: int = 0,
    with_actions: bool = True,
) -> RuntimeFrame:
    frame = RuntimeFrame(
        id="frame_1",
        session_id="s",
        turn_id="turn_1",
        trace_id="t",
        status=ExecutionStatus.RUNNING_TOOLS,
        history_through_seq=history_through_seq,
    )
    if with_actions:
        action = ActionRecord(kind=ActionKind.LLM_MESSAGE, name="llm")
        action.tool_calls = [
            LLMToolCall(id=f"call_{name}", name=name, arguments={})
            for name in (tool_names or [])
        ]
        frame.actions.append(action)
    return frame


class TestSnapshotSchema:
    def test_missing_snapshot_is_fatal(self) -> None:
        with pytest.raises(ResumeValidationError) as exc:
            validate_snapshot_schema(None)
        assert "no context snapshot" in str(exc.value)

    def test_supported_schema_passes(self) -> None:
        assert validate_snapshot_schema(make_snapshot(schema_version=1)) == []

    def test_newer_schema_is_fatal(self) -> None:
        with pytest.raises(ResumeValidationError) as exc:
            validate_snapshot_schema(make_snapshot(schema_version=2))
        assert "schema version 2" in str(exc.value)
        assert exc.value.details["schema_version"] == 2


class TestMissingRefs:
    def test_missing_tools_fatal(self) -> None:
        missing = MissingRefs(tool_ids=[7])
        with pytest.raises(ResumeValidationError) as exc:
            validate_missing_refs(missing)
        assert exc.value.details["missing_refs"]["tool_ids"] == [7]

    def test_missing_templates_fatal(self) -> None:
        missing = MissingRefs(template_ids=[3])
        with pytest.raises(ResumeValidationError):
            validate_missing_refs(missing)

    def test_missing_repo_fatal(self) -> None:
        missing = MissingRefs(repo_hashes=["abc"])
        with pytest.raises(ResumeValidationError):
            validate_missing_refs(missing)

    def test_missing_memory_warns_only(self) -> None:
        warnings = validate_missing_refs(MissingRefs(memory_ids=["m1"]))
        assert len(warnings) == 1
        assert "memory" in warnings[0]

    def test_no_missing_refs(self) -> None:
        assert validate_missing_refs(MissingRefs()) == []


class TestFrame:
    def test_terminal_frame_warns(self) -> None:
        frame = make_frame()
        frame.status = ExecutionStatus.COMPLETED
        warnings = validate_frame(frame)
        assert len(warnings) == 1
        assert "terminal status completed" in warnings[0]

    def test_frame_without_events_warns(self) -> None:
        frame = make_frame(with_actions=False)
        warnings = validate_frame(frame)
        assert len(warnings) == 1
        assert "no restored events" in warnings[0]

    def test_none_frame_no_warnings(self) -> None:
        assert validate_frame(None) == []


class TestToolSupport:
    def test_unsupported_tools_warn(self) -> None:
        frame = make_frame(tool_names=["read_file", "ghost_tool"])
        warnings = validate_tool_support(frame, {"read_file", "write_file"})
        assert len(warnings) == 1
        assert "ghost_tool" in warnings[0]

    def test_supported_tools_no_warnings(self) -> None:
        frame = make_frame(tool_names=["read_file"])
        assert validate_tool_support(frame, {"read_file"}) == []

    def test_skipped_without_exposed_names(self) -> None:
        frame = make_frame(tool_names=["ghost_tool"])
        assert validate_tool_support(frame, None) == []


class TestHistoryCoverage:
    def test_shorter_history_warns(self) -> None:
        frame = make_frame(history_through_seq=5)
        warnings = validate_history_coverage(frame, history_sequence=3)
        assert len(warnings) == 1
        assert "seq 3" in warnings[0]

    def test_sufficient_history_no_warnings(self) -> None:
        frame = make_frame(history_through_seq=3)
        assert validate_history_coverage(frame, history_sequence=5) == []

    def test_zero_history_warns_when_boundary_positive(self) -> None:
        frame = make_frame(history_through_seq=5)
        warnings = validate_history_coverage(frame, history_sequence=0)
        assert len(warnings) == 1
        assert "seq 0" in warnings[0]

    def test_no_frame_no_warnings(self) -> None:
        assert validate_history_coverage(None, history_sequence=0) == []


class TestToolSetDrift:
    def test_legacy_unpinned_version_skipped(self) -> None:
        assert validate_tool_set_drift("1.0.0", "v2") == []
        assert validate_tool_set_drift(None, "v2") == []
        assert validate_tool_set_drift("", "v2") == []

    def test_identical_version_no_warnings(self) -> None:
        assert validate_tool_set_drift("abc123", "abc123") == []

    def test_drift_warns(self) -> None:
        warnings = validate_tool_set_drift(
            "abc123", "def456", tool_names=["read_file", "write_file"]
        )
        assert len(warnings) == 1
        assert "abc123 -> def456" in warnings[0]
        assert "read_file, write_file" in warnings[0]

    def test_drift_without_names(self) -> None:
        warnings = validate_tool_set_drift("abc123", "def456")
        assert len(warnings) == 1
        assert "drifted" in warnings[0]


class TestSessionProfile:
    def _profile(
        self, tool_ids: list[int] | None = None, version: str = "v_profile"
    ) -> ContextProfile:
        return ContextProfile(
            runtime_context_config=RuntimeContextConfig(),
            tool_set_ref=ToolSetRef(
                tool_ids=tool_ids if tool_ids is not None else [1],
                tool_names=["read_file"],
                version=version,
            ),
            template_set_ref=TemplateSetRef(template_ids=[2], template_paths=["b.j2"]),
        )

    def _context_manager(
        self, mode: str = "static", resolved_version: str = "v_resolved"
    ) -> MagicMock:
        cm = MagicMock()
        cm.get_resolution_mode = MagicMock(return_value=ContextResolutionMode(mode))
        cm.resolve_profile_refs = AsyncMock(
            return_value=MagicMock(tools=[{"id": 1, "name": "read_file"}])
        )
        cm.compute_tool_set_version = MagicMock(return_value=resolved_version)
        return cm

    async def test_dynamic_mode_skips_checks(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(return_value=MissingRefs(tool_ids=[1]))

        result = await validate_session_profile(
            context_manager=self._context_manager(mode="dynamic"),
            store=store,
            profile=self._profile(),
        )
        assert result.warnings == []
        store.check_context_refs.assert_not_awaited()

    async def test_missing_profile_tools_fatal(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(return_value=MissingRefs(tool_ids=[1]))

        with pytest.raises(MissingContextReferencesError) as exc:
            await validate_session_profile(
                context_manager=self._context_manager(),
                store=store,
                profile=self._profile(),
            )
        assert exc.value.missing_refs == ["tool:1"]

    async def test_missing_profile_repo_fatal(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(
            return_value=MissingRefs(repo_hashes=["h"])
        )

        with pytest.raises(MissingContextReferencesError):
            await validate_session_profile(
                context_manager=self._context_manager(),
                store=store,
                profile=self._profile(),
            )

    async def test_clean_profile_no_warnings(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(return_value=MissingRefs())

        result = await validate_session_profile(
            context_manager=self._context_manager(resolved_version="v_profile"),
            store=store,
            profile=self._profile(),
        )
        assert result.warnings == []

    async def test_profile_tool_drift_warns(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(return_value=MissingRefs())

        result = await validate_session_profile(
            context_manager=self._context_manager(resolved_version="v_drifted"),
            store=store,
            profile=self._profile(),
        )
        assert len(result.warnings) == 1
        assert "v_profile -> v_drifted" in result.warnings[0]

    async def test_legacy_profile_version_skips_drift(self) -> None:
        store = MagicMock()
        store.check_context_refs = AsyncMock(return_value=MissingRefs())

        result = await validate_session_profile(
            context_manager=self._context_manager(resolved_version="v_drifted"),
            store=store,
            profile=self._profile(version="1.0.0"),
        )
        assert result.warnings == []


class TestValidateResumeContext:
    def test_clean_context_passes(self) -> None:
        result = validate_resume_context(
            ResumeValidationInput(
                snapshot=make_snapshot(),
                missing_refs=MissingRefs(),
                frame=make_frame(tool_names=["read_file"], history_through_seq=2),
                exposed_tool_names={"read_file"},
                history_sequence=2,
            )
        )
        assert result.warnings == []
        assert result.is_valid

    def test_stale_refs_raise_before_warnings(self) -> None:
        with pytest.raises(ResumeValidationError):
            validate_resume_context(
                ResumeValidationInput(
                    snapshot=make_snapshot(),
                    missing_refs=MissingRefs(tool_ids=[1]),
                    frame=None,
                )
            )

    def test_config_support_hook_is_extensible(self) -> None:
        # Extension point for future config-support checks: today no checks.
        assert validate_config_support(RuntimeContextConfig()) == []
        assert validate_config_support(None) == []
