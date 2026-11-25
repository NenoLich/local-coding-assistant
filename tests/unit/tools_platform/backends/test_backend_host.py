"""Unit tests for HostBackend.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_backend_host.py):

File Operations
- read_file reads file content with line offset/limit and byte limit
- observe_file returns FileObservation via WorkspaceManager
- replace_exact delegates to WorkspaceManager.atomic_write with conflict detection
- replace_exact returns changed and unified diff in result
- create_file delegates to WorkspaceManager.atomic_create with overwrite support
- delete_file delegates to WorkspaceManager.atomic_delete with recursive/dry_run

Command Execution
- execute_command streams ToolOutputChunk (stdout, stderr, status channels)
- execute_command applies output_limits (max_stdout_bytes, max_stderr_bytes, max_lines)
- execute_command filters env through env_allowlist
- execute_command resolves cwd relative to workspace_root
- cancel_execution kills process tree by handle_id
- check_process_status returns process status by handle_id

Search Operations
- search_text uses ripgrep when available, falls back to Python
- match_paths uses match_paths implementation with gitignore support
- query_repository delegates to WorkspaceManager for search_symbols/get_symbol

Path Resolution
- resolve_path uses WorkspaceManager.resolve_path for canonical path validation
- get_workspace_root returns configured workspace_root
- get_backend_type returns "host"
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_coding_assistant.config.config_manager import ConfigManager
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import BackendType, ProcessState
from local_coding_assistant.tools_platform.exceptions import (
    ToolBackendError,
    ToolPathViolationError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.workspace import WorkspaceManager


def _write(workspace: Path, name: str, content: str) -> Path:
    file_path = workspace / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, newline="")
    return file_path


async def _collect(gen) -> list:
    return [chunk async for chunk in gen]


class TestHostBackendReadFile:
    @pytest.mark.asyncio
    async def test_read_file_returns_full_content(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "full.py", "line1\nline2\nline3\n")
        content = await host_backend.read_file(Path("full.py"))
        assert content == "line1\nline2\nline3\n"

    @pytest.mark.asyncio
    async def test_read_file_offset_and_limit(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(
            temp_workspace, "rangelines.py", "".join(f"line{i}\n" for i in range(1, 11))
        )
        content = await host_backend.read_file(
            Path("rangelines.py"), start_line=3, end_line=5
        )
        assert "line3" in content
        assert "line4" in content
        assert "line5" in content
        assert "line2" not in content
        assert "line6" not in content

    @pytest.mark.asyncio
    async def test_read_file_applies_byte_limit(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "long.txt", "x" * 200)
        content = await host_backend.read_file(Path("long.txt"), byte_limit=50)
        assert len(content.encode("utf-8")) <= 50

    @pytest.mark.asyncio
    async def test_read_file_nonexistent_raises(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolValidationError):
            await host_backend.read_file(Path("does_not_exist.py"))

    @pytest.mark.asyncio
    async def test_read_file_directory_raises(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        (temp_workspace / "adir").mkdir()
        with pytest.raises(ToolValidationError):
            await host_backend.read_file(Path("adir"))


class TestHostBackendObserveFile:
    @pytest.mark.asyncio
    async def test_observe_file_returns_file_observation(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "obs.txt", "observed content\n")
        observation = await host_backend.observe_file(Path("obs.txt"))
        assert observation.path == "obs.txt"
        assert observation.size == len(b"observed content\n")
        assert observation.source.value == "read"
        assert observation.hash != ""


class TestHostBackendReplaceExact:
    @pytest.mark.asyncio
    async def test_replace_exact_success_returns_changed_and_diff(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "replace.txt", "Hello world\nHello world\n")
        result = await host_backend.replace_exact(
            path="replace.txt",
            old_text="Hello world",
            new_text="Hi there",
            session_id="test_session",
            expected_match_count=2,
        )
        assert result["success"] is True
        assert result["match_count"] == 2
        assert result["changed"] is True
        assert result["diff"] is not None
        assert "-Hello world\n" in result["diff"]
        assert "+Hi there\n" in result["diff"]
        assert (temp_workspace / "replace.txt").read_text() == "Hi there\nHi there\n"

    @pytest.mark.asyncio
    async def test_replace_exact_match_count_mismatch(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "mismatch.txt", "Hello world\nHello world\n")
        result = await host_backend.replace_exact(
            path="mismatch.txt",
            old_text="Hello world",
            new_text="Hi there",
            session_id="test_session",
            expected_match_count=1,
        )
        assert result["success"] is False
        assert result["match_count"] == 2
        assert result["changed"] is False
        assert "Ambiguous match" in result["error"]
        assert result["diff"] is None

    @pytest.mark.asyncio
    async def test_replace_exact_nonexistent_raises(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await host_backend.replace_exact(
                path="nope.txt",
                old_text="a",
                new_text="b",
                session_id="test_session",
            )

    @pytest.mark.asyncio
    async def test_replace_exact_conflict_detection(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "conflict.txt", "v1\n")
        await host_backend.observe_file(Path("conflict.txt"))
        # Simulate an external edit on disk after the read baseline was captured.
        _write(temp_workspace, "conflict.txt", "v2\n")
        result = await host_backend.replace_exact(
            path="conflict.txt",
            old_text="v1",
            new_text="v3",
            session_id="test_session",
        )
        assert result["success"] is False
        assert result["changed"] is False
        assert "conflict" in result
        assert result["conflict"]["conflict_type"]

    @pytest.mark.asyncio
    async def test_replace_exact_unchanged_returns_changed_false(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "same.txt", "Hello world\n")
        result = await host_backend.replace_exact(
            path="same.txt",
            old_text="Hello world",
            new_text="Hello world",
            session_id="test_session",
        )
        assert result["success"] is True
        assert result["changed"] is False
        assert result["diff"] is None


class TestHostBackendCreateFile:
    @pytest.mark.asyncio
    async def test_create_file_writes_new_file(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        result = await host_backend.create_file(
            path="new.txt", content="created content\n"
        )
        assert result["success"] is True
        assert result["path"] == "new.txt"
        assert result["error"] is None
        assert (temp_workspace / "new.txt").read_text() == "created content\n"

    @pytest.mark.asyncio
    async def test_create_file_overwrite_replaces_content(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "ov.txt", "old\n")
        result = await host_backend.create_file(
            path="ov.txt", content="new\n", overwrite=True
        )
        assert result["success"] is True
        assert (temp_workspace / "ov.txt").read_text() == "new\n"

    @pytest.mark.asyncio
    async def test_create_file_exists_without_overwrite_raises(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "exists.txt", "old\n")
        with pytest.raises(ToolPathViolationError):
            await host_backend.create_file(
                path="exists.txt", content="new\n", overwrite=False
            )


class TestHostBackendDeleteFile:
    @pytest.mark.asyncio
    async def test_delete_file_removes_existing_file(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "del_me.txt", "delete me\n")
        result = await host_backend.delete_file(path="del_me.txt")
        assert result["success"] is True
        assert result["error"] is None
        assert not (temp_workspace / "del_me.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_file_nonexistent_raises(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            await host_backend.delete_file(path="nope.txt")

    @pytest.mark.asyncio
    async def test_delete_file_recursive_removes_directory(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "sub/nested.py", "nested\n")
        result = await host_backend.delete_file(path="sub", recursive=True)
        assert result["success"] is True
        assert not (temp_workspace / "sub").exists()

    @pytest.mark.asyncio
    async def test_delete_file_dry_run_previews_without_deleting(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "dry.txt", "dry\n")
        result = await host_backend.delete_file(path="dry.txt", dry_run=True)
        assert result["success"] is True
        assert result["would_delete"] == ["dry.txt"]
        assert (temp_workspace / "dry.txt").exists()

    @pytest.mark.asyncio
    async def test_delete_file_dry_run_directory_lists_contents(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "subd/nested.py", "nested\n")
        result = await host_backend.delete_file(
            path="subd", recursive=True, dry_run=True
        )
        assert result["success"] is True
        assert "subd/nested.py" in result["would_delete"]
        assert (temp_workspace / "subd").exists()


class TestHostBackendCommandExecution:
    @pytest.mark.asyncio
    async def test_execute_command_streams_stdout_and_status(
        self, host_backend: HostBackend
    ) -> None:
        chunks = await _collect(
            host_backend.execute_command(argv=["echo", "hello world"])
        )
        stdout_chunks = [c for c in chunks if c.channel == "stdout"]
        status_chunks = [c for c in chunks if c.channel == "status"]
        assert len(stdout_chunks) > 0
        assert "hello world" in stdout_chunks[0].payload
        assert len(status_chunks) == 1
        assert status_chunks[0].payload["return_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_command_resolves_cwd_within_workspace(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "sub/cwd.txt", "cwd content\n")
        chunks = await _collect(
            host_backend.execute_command(
                argv=["cmd", "/c", "type", "cwd.txt"], cwd=Path("sub")
            )
        )
        stdout = "".join(c.payload for c in chunks if c.channel == "stdout")
        assert "cwd content" in stdout

    @pytest.mark.asyncio
    async def test_execute_command_applies_output_limits(
        self, host_backend: HostBackend
    ) -> None:
        chunks = await _collect(
            host_backend.execute_command(
                argv=["python", "-c", "import sys; sys.stdout.write('a\\n' * 150000)"]
            )
        )
        stdout_bytes = sum(
            len(c.payload.encode("utf-8")) for c in chunks if c.channel == "stdout"
        )
        assert stdout_bytes <= 262144

    @pytest.mark.asyncio
    async def test_execute_command_filters_env_through_allowlist(
        self, filtered_backend: HostBackend
    ) -> None:
        chunks = await _collect(
            filtered_backend.execute_command(
                argv=["cmd", "/c", "echo", "%FOO%"], env={"FOO": "bar"}
            )
        )
        combined = "".join(c.payload for c in chunks if c.channel == "stdout")
        assert "bar" not in combined

    @pytest.mark.asyncio
    async def test_execute_command_filter_env_allows_allowlisted_keys(
        self, filtered_backend: HostBackend
    ) -> None:
        assert filtered_backend._filter_env({"PATH": "/x", "FOO": "bar"}) == {
            "PATH": "/x"
        }
        assert filtered_backend._filter_env({"FOO": "bar"}) == {}

    @pytest.mark.asyncio
    async def test_check_process_status_invalid_handle_raises(
        self, host_backend: HostBackend
    ) -> None:
        with pytest.raises(ToolValidationError):
            await host_backend.check_process_status("nonexistent-handle")

    @pytest.mark.asyncio
    async def test_check_process_status_valid_handle(
        self, host_backend: HostBackend
    ) -> None:
        async def run() -> None:
            async for _ in host_backend.execute_command(
                argv=["python", "-c", "import time; time.sleep(10)"]
            ):
                pass

        task = asyncio.create_task(run())
        handle_id = None
        for _ in range(200):
            await asyncio.sleep(0.05)
            if host_backend._supervisor._active_processes:
                handle_id = next(iter(host_backend._supervisor._active_processes))
                break
        assert handle_id is not None, "process was not registered by the supervisor"
        status = await host_backend.check_process_status(handle_id)
        assert status["handle_id"] == handle_id
        assert status["state"] == ProcessState.RUNNING.value
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_cancel_execution_invalid_handle_raises(
        self, host_backend: HostBackend
    ) -> None:
        with pytest.raises(ToolValidationError):
            await host_backend.cancel_execution("nonexistent-handle")

    @pytest.mark.asyncio
    async def test_cancel_execution_terminates_process(
        self, host_backend: HostBackend
    ) -> None:
        async def run() -> None:
            async for _ in host_backend.execute_command(
                argv=["python", "-c", "import time; time.sleep(30)"]
            ):
                pass

        task = asyncio.create_task(run())
        handle_id = None
        for _ in range(200):
            await asyncio.sleep(0.05)
            if host_backend._supervisor._active_processes:
                handle_id = next(iter(host_backend._supervisor._active_processes))
                break
        assert handle_id is not None, "process was not registered by the supervisor"
        await host_backend.cancel_execution(handle_id)
        for _ in range(200):
            await asyncio.sleep(0.05)
            if await host_backend._supervisor.get_handle(handle_id) is None:
                break
        assert await host_backend._supervisor.get_handle(handle_id) is None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestHostBackendSearchOperations:
    @pytest.mark.asyncio
    async def test_search_text_literal_returns_matches(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "search_a.txt", "alpha beta gamma\n")
        _write(temp_workspace, "search_b.txt", "delta alpha\n")
        result = await host_backend.search_text(query="alpha", path=Path("."))
        assert result.total_matches >= 2
        matched_files = {m.path for m in result.matches}
        assert "search_a.txt" in matched_files
        assert "search_b.txt" in matched_files

    @pytest.mark.asyncio
    async def test_search_text_regex_pattern(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "regex.txt", "item42 and item99\n")
        result = await host_backend.search_text(
            query=r"item\d+", path=Path("."), pattern="regex"
        )
        assert result.total_matches >= 2
        assert any("item42" in m.line_content for m in result.matches)

    @pytest.mark.asyncio
    async def test_search_text_case_sensitive(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "case.txt", "Alpha alpha\n")
        result = await host_backend.search_text(
            query="Alpha", path=Path("."), case_sensitive=True
        )
        assert result.total_matches == 1
        assert result.matches[0].line_content == "Alpha alpha"

    @pytest.mark.asyncio
    async def test_search_text_falls_back_to_python(
        self,
        host_backend: HostBackend,
        temp_workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write(temp_workspace, "fallback.txt", "fallback query here\n")

        async def _raise_ripgrep(*args, **kwargs):
            raise FileNotFoundError("ripgrep not found")

        monkeypatch.setattr(
            "local_coding_assistant.tools_platform.backends.host.search_with_ripgrep",
            _raise_ripgrep,
        )
        result = await host_backend.search_text(query="fallback", path=Path("."))
        assert result.total_matches >= 1
        assert any("fallback.txt" == m.path for m in result.matches)

    @pytest.mark.asyncio
    async def test_match_paths_glob_returns_sorted_paths(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "src/a.py", "a\n")
        _write(temp_workspace, "src/b.py", "b\n")
        _write(temp_workspace, "tests/c.py", "c\n")
        result = await host_backend.match_paths(pattern="src/**/*.py", path=Path("."))
        paths = [m.path.replace("\\", "/") for m in result.matches]
        assert paths == sorted(paths)
        assert "src/a.py" in paths
        assert "src/b.py" in paths
        assert "tests/c.py" not in paths

    @pytest.mark.asyncio
    async def test_match_paths_extension_filter(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "ext/a.py", "a\n")
        _write(temp_workspace, "ext/b.txt", "b\n")
        result = await host_backend.match_paths(
            pattern="**/*", path=Path("."), file_type="py"
        )
        paths = [m.path.replace("\\", "/") for m in result.matches]
        assert "ext/a.py" in paths
        assert "ext/b.txt" not in paths

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_query_repository_search_symbols_delegates(
        self,
        workspace_manager: WorkspaceManager,
        repo_service: MagicMock,
        config_manager: ConfigManager,
    ) -> None:
        workspace_manager._repository_context_service = repo_service
        repo_service.search_symbols.return_value = []
        backend = HostBackend(
            workspace_root=workspace_manager.workspace_root,
            workspace_manager=workspace_manager,
            config_manager=config_manager,
        )
        result = await backend.query_repository(
            query_type="search_symbols", params={"pattern": "foo", "limit": 5}
        )
        assert result == []
        repo_service.search_symbols.assert_called_once_with(
            query="foo", symbol_types=None, file_path=None, limit=5
        )

    @pytest.mark.asyncio
    async def test_query_repository_get_symbol_delegates(
        self,
        workspace_manager: WorkspaceManager,
        repo_service: MagicMock,
        config_manager: ConfigManager,
    ) -> None:
        from local_coding_assistant.repository.models import SymbolDetail, SymbolType

        workspace_manager._repository_context_service = repo_service
        repo_service.get_symbol.return_value = [
            SymbolDetail(
                symbol_id=7,
                file_id=1,
                name="foo",
                symbol_type=SymbolType.FUNCTION,
                file_path="src/foo.py",
                line_number=3,
            )
        ]
        backend = HostBackend(
            workspace_root=workspace_manager.workspace_root,
            workspace_manager=workspace_manager,
            config_manager=config_manager,
        )
        result = await backend.query_repository(
            query_type="get_symbol", params={"name": "foo", "path": "src/foo.py"}
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "foo"
        assert result[0]["symbol_id"] == 7
        repo_service.get_symbol.assert_called_once_with(
            name="foo",
            symbol_type=None,
            file_path="src/foo.py",
            limit=20,
        )

    @pytest.mark.asyncio
    async def test_query_repository_no_service_raises(
        self, host_backend: HostBackend
    ) -> None:
        with pytest.raises(ToolBackendError):
            await host_backend.query_repository(
                query_type="search_symbols", params={"pattern": "foo"}
            )

    @pytest.mark.asyncio
    async def test_query_repository_unknown_type_raises(
        self,
        workspace_manager: WorkspaceManager,
        repo_service: MagicMock,
        config_manager: ConfigManager,
    ) -> None:
        workspace_manager._repository_context_service = repo_service
        backend = HostBackend(
            workspace_root=workspace_manager.workspace_root,
            workspace_manager=workspace_manager,
            config_manager=config_manager,
        )
        with pytest.raises(ToolValidationError):
            await backend.query_repository(query_type="bogus", params={})


class TestHostBackendPathResolution:
    @pytest.mark.asyncio
    async def test_resolve_path_returns_absolute_within_workspace(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        resolved = host_backend.resolve_path(Path("sub/deep.py"))
        assert resolved.is_absolute()
        assert resolved == temp_workspace / "sub/deep.py"

    @pytest.mark.asyncio
    async def test_resolve_path_rejects_outside_workspace(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        with pytest.raises(ToolPathViolationError):
            host_backend.resolve_path(Path("/outside/foo.py"))

    @pytest.mark.asyncio
    async def test_get_workspace_root_returns_configured_root(
        self, host_backend: HostBackend, temp_workspace: Path
    ) -> None:
        assert host_backend.get_workspace_root() == temp_workspace

    def test_get_backend_type_returns_host(self, host_backend: HostBackend) -> None:
        assert host_backend.get_backend_type() == BackendType.HOST
        assert host_backend.get_backend_type() == "host"
