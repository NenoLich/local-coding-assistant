"""Unit tests for the replace_exact tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_replace_exact_tool.py):

- exact match count=1 succeeds (success=true, match_count=1, changed=true, diff)
- expected_match_count > actual matches fails with match_count_mismatch error
- expected_match_count < actual matches fails with match_count_mismatch error
- baseline-driven expected_hash matching current file succeeds (observe first)
- baseline-driven expected_hash mismatched fails with conflict + ConflictReport
- no baseline (no observe) uses current hash as unguarded baseline
- snapshot created before write for rollback
- verifies written content matches proposed_hash after commit
- transaction commit persists changes
- transaction rollback restores original content from snapshot
- fails for path outside workspace (ToolValidationError)
- fails for non-existent file (ToolValidationError)
- preserves Unicode and line endings exactly
- returns changed=true/false and unified diff in LLM output
- LLM output excludes before/after hashes, sizes, line counts, conflict
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_coding_assistant.tools_platform.exceptions import ToolValidationError
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.replace_exact import (
    ReplaceExactInput,
    ReplaceExactOutput,
    register_replace_exact,
    replace_exact_impl,
)
from local_coding_assistant.tools_platform.workspace.transactions import (
    OperationType,
)


def _write(workspace: Path, name: str, content: str) -> Path:
    file_path = workspace / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, newline="")
    return file_path


def _hash_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class TestReplaceExactContent:
    @pytest.mark.asyncio
    async def test_exact_match_count_one_succeeds(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        assert result.match_count == 1
        assert result.changed is True
        assert result.diff is not None
        assert result.error is None
        assert "hi world" in (temp_workspace / "test.py").read_text()

    @pytest.mark.asyncio
    async def test_expected_match_count_greater_than_actual_fails(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="hello",
                new_text="hi",
                expected_match_count=5,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is False
        assert result.match_count == 1
        assert result.changed is False
        assert result.diff is None
        assert "match count mismatch" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_expected_match_count_less_than_actual_fails(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="hello",
                new_text="hi",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is False
        assert result.match_count == 2
        assert result.changed is False
        assert "ambiguous match" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_no_baseline_unguarded_write_succeeds(
        self, backend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        # No observe_file call: no baseline, unguarded write proceeds.
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        assert result.match_count == 1
        assert result.changed is True

    @pytest.mark.asyncio
    async def test_baseline_matching_current_succeeds(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        content = "hello world\n"
        _write(temp_workspace, "test.py", content)
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_baseline_mismatched_conflict_with_report(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        # Edit file externally after baseline was recorded.
        (temp_workspace / "test.py").write_text("hello there\n", newline="")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is False
        assert result.match_count == 0
        assert result.changed is False
        assert result.conflict is not None
        assert result.external_diff is not None
        assert result.proposed_diff is not None
        assert result.recommendation in ("re_read_file", "retry_or_adapt")
        conflict = result.conflict
        assert conflict["base_hash"] == _hash_of("hello world\n")
        assert conflict["current_hash"] == _hash_of("hello there\n")
        assert conflict["proposed_hash"] == _hash_of("hi there\n")
        assert conflict["conflict_type"] == "version_stale"
        assert conflict["conflict_detected"] is True

    @pytest.mark.asyncio
    async def test_no_match_count_zero_fails(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="goodbye",
                new_text="hi",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is False
        assert result.match_count == 0
        assert "no match found" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_unchanged_text_returns_changed_false(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        content = "hello world\n"
        _write(temp_workspace, "test.py", content)
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="hello",
                new_text="hello",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        assert result.match_count == 1
        assert result.changed is False
        assert result.diff is None


class TestReplaceExactTransaction:
    @pytest.mark.asyncio
    async def test_commit_persists_changes(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert result.success is True
        assert (temp_workspace / "test.py").read_text() == "hi world\n"

    @pytest.mark.asyncio
    async def test_snapshot_created_before_write(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        snapshot_dir = temp_workspace / ".locca" / "snapshots"
        await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert snapshot_dir.exists()
        snapshot_dirs = [p for p in snapshot_dir.iterdir() if p.is_dir()]
        assert len(snapshot_dirs) >= 1

    @pytest.mark.asyncio
    async def test_rollback_restores_original_content(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        tx_manager = workspace_manager.transaction_manager
        tx_manager.begin_transaction()
        tx_manager.observe_file(Path("test.py"))
        tx_manager.atomic_write(Path("test.py"), content="new content\n")
        tx_manager.rollback_transaction()
        assert (temp_workspace / "test.py").read_text() == "hello world\n"

    @pytest.mark.asyncio
    async def test_verifies_written_content_matches_proposed_hash(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        assert result.success is True
        written = (temp_workspace / "test.py").read_text()
        assert written == "hi world\n"
        assert result.after_hash == _hash_of("hi world\n")


class TestReplaceExactValidation:
    @pytest.mark.asyncio
    async def test_outside_workspace_raises_tool_validation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolValidationError):
            await replace_exact_impl(
                backend,
                ReplaceExactInput(
                    path="/outside/foo.py",
                    old_text="hello",
                    new_text="hi",
                    expected_match_count=1,
                ),
                session_id="test_session",
            )

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_tool_validation_error(
        self, backend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolValidationError):
            await replace_exact_impl(
                backend,
                ReplaceExactInput(
                    path="does_not_exist.py",
                    old_text="hello",
                    new_text="hi",
                    expected_match_count=1,
                ),
                session_id="test_session",
            )


class TestReplaceExactEncoding:
    @pytest.mark.asyncio
    async def test_preserves_unicode_exactly(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        content = "héllo wörld\nunicode: 日本語\n"
        _write(temp_workspace, "test.py", content)
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="wörld",
                new_text="there",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        assert result.changed is True
        written = (temp_workspace / "test.py").read_text()
        assert written == "héllo there\nunicode: 日本語\n"

    @pytest.mark.asyncio
    async def test_preserves_crlf_line_endings(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        content = "hello world\r\nsecond line\r\n"
        _write(temp_workspace, "test.py", content)
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="hello",
                new_text="hi",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert isinstance(result, ReplaceExactOutput)
        assert result.success is True
        # Normalize to check CRLF preserved by reading raw bytes.
        raw = (temp_workspace / "test.py").read_bytes().decode("utf-8")
        assert raw == "hi world\r\nsecond line\r\n"


class TestReplaceExactLLMOutput:
    @pytest.mark.asyncio
    async def test_llm_output_returns_changed_and_diff(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        llm = result.to_llm_dict()
        assert llm["success"] is True
        assert llm["match_count"] == 1
        assert llm["changed"] is True
        assert llm["diff"] is not None
        assert "-hello world" in llm["diff"]
        assert "+hi world" in llm["diff"]

    @pytest.mark.asyncio
    async def test_llm_output_excludes_internal_fields(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py", old_text="hello", new_text="hi", expected_match_count=1
            ),
            session_id="test_session",
        )
        llm = result.to_llm_dict()
        assert set(llm.keys()) == {"success", "match_count", "changed", "diff", "error"}
        assert "before_hash" not in llm
        assert "after_hash" not in llm
        assert "before_size" not in llm
        assert "after_size" not in llm
        assert "before_lines" not in llm
        assert "after_lines" not in llm
        assert "conflict" not in llm


class TestReplaceExactSchema:
    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_replace_exact(registry)
        definition = registry.get_tool("replace_exact")
        assert definition is not None
        schema = definition.output_schema
        assert schema == ReplaceExactOutput.model_llm_schema()
        schema_str = str(schema)
        assert "success" in schema_str
        assert "match_count" in schema_str
        assert "changed" in schema_str
        assert "diff" in schema_str
        assert "error" in schema_str
        assert "before_hash" not in schema_str
        assert "after_hash" not in schema_str
        assert "conflict" not in schema_str


class TestReplaceExactWorkspaceManager:
    def test_transaction_manager_records_replace_exact_operation(
        self, workspace_manager, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "test.py", "hello world\n")
        tx_manager = workspace_manager.transaction_manager
        tx_manager.begin_transaction()
        tx_manager.observe_file(Path("test.py"))
        tx_manager.atomic_write(Path("test.py"), content="hi world\n")
        tx_manager.commit_transaction()
        journal = tx_manager.get_transaction_history()
        assert len(journal) == 1
        tx_id = next(iter(journal))
        entries = journal[tx_id].get_completed_entries()
        assert len(entries) == 1
        assert entries[0].operation == OperationType.WRITE.value
        assert entries[0].path == "test.py"


class TestReplaceExactTrailingNewlineFallback:
    """Tests for Pass 2 trailing-newline fallback matching (Case A and Case B)."""

    @pytest.mark.asyncio
    async def test_case_a_old_missing_trailing_newline_but_disk_has_it(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """old_text lacks trailing \n but disk content has old_text+\n uniquely.

        content: "line2\nline2extra\n"
        old_text: "line2"  -> count("line2")=2 (fails expected=1)
        old_text+"\n": "line2\n" -> count=1 (Case A matches)
        """
        _write(temp_workspace, "test.py", "line2\nline2extra\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="line2",
                new_text="CHANGED",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is True
        assert result.match_count == 1
        written = (temp_workspace / "test.py").read_text()
        assert written == "CHANGED\nline2extra\n"

    @pytest.mark.asyncio
    async def test_case_b_old_extraneous_trailing_newline(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """old_text has extraneous trailing \n but disk matches without it.

        content: "line2\nline3\n"
        old_text: "line2\n\n"  -> count("line2\n\n")=0 (fails Pass 1)
        old_text[:-1]: "line2\n" -> count=1 (Case B matches)
        Note: Case B strips trailing \n from new_text per spec.
        """
        _write(temp_workspace, "test.py", "line2\nline3\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="line2\n\n",
                new_text="CHANGED",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is True
        assert result.match_count == 1
        # Case B strips trailing \n from old_text, consuming the line separator.
        written = (temp_workspace / "test.py").read_text()
        assert written == "CHANGEDline3\n"

    @pytest.mark.asyncio
    async def test_case_a_preserves_crlf_disk_ending(
        self, backend_with_config, workspace_manager, temp_workspace: Path
    ) -> None:
        """Case A works on CRLF disk content; disk preserves CRLF after write.

        Requires normalization enabled so CRLF disk content is normalized to LF
        before matching against LF old_text.
        """
        workspace_manager._line_normalizer.enabled = True
        _write(temp_workspace, "test.py", "line2\r\nline2extra\r\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend_with_config,
            ReplaceExactInput(
                path="test.py",
                old_text="line2",
                new_text="CHANGED",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is True
        raw = (temp_workspace / "test.py").read_bytes().decode("utf-8")
        assert raw == "CHANGED\r\nline2extra\r\n"

    @pytest.mark.asyncio
    async def test_crlf_content_with_lf_llm_text_base_logic(
        self, backend_with_config, workspace_manager, temp_workspace: Path
    ) -> None:
        """Base logic: CRLF content on disk, LLM sends LF-only old_text/new_text.

        Without normalization this would fail: LLF old_text won't match CRLF disk
        bytes (count == 0). With normalization enabled, disk content is converted
        to LF for matching, the edit succeeds, and CRLF is preserved on disk.
        """
        workspace_manager._line_normalizer.enabled = True
        _write(temp_workspace, "test.py", "line1\r\nline2\r\nline3\r\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend_with_config,
            ReplaceExactInput(
                path="test.py",
                old_text="line2\n",
                new_text="CHANGED\n",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is True
        assert result.match_count == 1
        raw = (temp_workspace / "test.py").read_bytes().decode("utf-8")
        assert raw == "line1\r\nCHANGED\r\nline3\r\n"

    @pytest.mark.asyncio
    async def test_case_b_preserves_crlf_disk_ending(
        self, backend_with_config, workspace_manager, temp_workspace: Path
    ) -> None:
        """Case B works on CRLF disk content; disk preserves CRLF after write.

        Requires normalization enabled. Note: Case B strips trailing \n from
        both old_text and new_text per spec, so the line separator from
        old_text is consumed.
        """
        workspace_manager._line_normalizer.enabled = True
        _write(temp_workspace, "test.py", "line2\r\nline3\r\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend_with_config,
            ReplaceExactInput(
                path="test.py",
                old_text="line2\r\n\r\n",
                new_text="CHANGED",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is True
        raw = (temp_workspace / "test.py").read_bytes().decode("utf-8")
        # Case B consumed the \n from old_text; result is "CHANGEDline3" + CRLF.
        assert raw == "CHANGEDline3\r\n"

    @pytest.mark.asyncio
    async def test_failure_hint_too_many_matches(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """Too many matches produces an ambiguity hint in the error."""
        _write(temp_workspace, "test.py", "line2\nline2\nline3\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="line2\n",
                new_text="X\n",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is False
        assert result.match_count == 2
        assert result.error is not None
        assert "Ambiguous match" in result.error

    @pytest.mark.asyncio
    async def test_failure_hint_indentation_mismatch(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """Indentation mismatch produces a whitespace hint.

        old_text is indented but content is not, so Case B's substring check
        (old_text[:-1] = "  line2") doesn't match content "line2\nline3\n".
        """
        _write(temp_workspace, "test.py", "line2\nline3\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="  line2\n",
                new_text="X\n",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is False
        assert result.error is not None
        assert "indentation/whitespace" in result.error.lower()

    @pytest.mark.asyncio
    async def test_failure_hint_fuzzy_similar(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """Near-match produces a fuzzy similarity hint with diff."""
        _write(temp_workspace, "test.py", "line2\nline3\n")
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="line2X\n",
                new_text="X\n",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is False
        assert result.error is not None
        assert "similar" in result.error.lower()
        assert "line2" in result.error

    @pytest.mark.asyncio
    async def test_failure_hint_no_match(
        self, backend, workspace_manager, temp_workspace: Path
    ) -> None:
        """Completely absent old_text produces a no-match hint.

        Uses content very different from old_text so the fuzzy-match fallback
        (which needs >75% similarity) does not trigger.
        """
        _write(
            temp_workspace, "test.py", "XYZ completely different content here\nABC\n"
        )
        workspace_manager.observe_file("test.py")
        result = await replace_exact_impl(
            backend,
            ReplaceExactInput(
                path="test.py",
                old_text="line2\n",
                new_text="X\n",
                expected_match_count=1,
            ),
            session_id="test_session",
        )
        assert result.success is False
        assert result.error is not None
        assert "No match found" in result.error
        assert "read_file" in result.error
