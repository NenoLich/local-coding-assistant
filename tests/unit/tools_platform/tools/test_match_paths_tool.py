"""Unit tests for the match_paths tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_match_paths_tool.py):

- pattern "src/**/*.py" returns matching file paths relative to workspace
- include_hidden flag for dotfiles
- include_ignored flag for git-ignored files
- extension filter (e.g., extension="py" -> *.py)
- max_results limit
- honors .gitignore rules (not just hardcoded hidden dirs)
- honors .aiignore rules
- returns paths sorted lexicographically
- supports offset for pagination
- LLM output excludes internal fields (via to_llm_dict)
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.match_paths import (
    MatchPathResult,
    MatchPathsInput,
    MatchPathsOutput,
    match_paths_impl,
    register_match_paths,
)


def _write(workspace: Path, name: str, content: str) -> Path:
    file_path = workspace / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, newline="")
    return file_path


class _MockBackend:
    """Mock backend returning a controllable MatchPathsOutput."""

    def __init__(self, output: MatchPathsOutput | None = None) -> None:
        self.output = output or MatchPathsOutput(
            matches=[], total_matches=0, truncated=False
        )
        self.match_paths_kwargs: list[dict[str, Any]] = []

    async def match_paths(
        self,
        pattern: str,
        path: Path,
        include_hidden: bool = False,
        include_ignored: bool = False,
        file_type: str | None = None,
        max_results: int = 200,
    ) -> MatchPathsOutput:
        self.match_paths_kwargs.append(
            {
                "pattern": pattern,
                "path": path,
                "include_hidden": include_hidden,
                "include_ignored": include_ignored,
                "file_type": file_type,
                "max_results": max_results,
            }
        )
        return self.output


class TestMatchPathsImpl:
    """Tests for match_paths_impl (argument pass-through, pagination, schema)."""

    @pytest.mark.asyncio
    async def test_passes_input_arguments_to_backend(self) -> None:
        backend = _MockBackend()
        input_model = MatchPathsInput(
            pattern="src/**/*.py",
            path="src",
            include_hidden=True,
            include_ignored=True,
            extension="py",
            max_results=50,
        )
        await match_paths_impl(backend, input_model, session_id="sess-1")

        assert len(backend.match_paths_kwargs) == 1
        kwargs = backend.match_paths_kwargs[0]
        assert kwargs["pattern"] == "src/**/*.py"
        assert kwargs["path"] == Path("src")
        assert kwargs["include_hidden"] is True
        assert kwargs["include_ignored"] is True
        assert kwargs["file_type"] == "py"
        assert kwargs["max_results"] == 50

    @pytest.mark.asyncio
    async def test_applies_offset_to_matches(self) -> None:
        backend = _MockBackend(
            output=MatchPathsOutput(
                matches=[
                    MatchPathResult(path="a.py", type="file", size=1),
                    MatchPathResult(path="b.py", type="file", size=2),
                    MatchPathResult(path="c.py", type="file", size=3),
                ],
                total_matches=3,
                truncated=False,
            )
        )
        result = await match_paths_impl(
            backend, MatchPathsInput(pattern="*.py", offset=1)
        )

        assert [m.path for m in result.matches] == ["b.py", "c.py"]

    @pytest.mark.asyncio
    async def test_no_offset_when_offset_zero(self) -> None:
        backend = _MockBackend(
            output=MatchPathsOutput(
                matches=[MatchPathResult(path="a.py", type="file", size=1)],
                total_matches=1,
                truncated=False,
            )
        )
        result = await match_paths_impl(
            backend, MatchPathsInput(pattern="*.py", offset=0)
        )
        assert [m.path for m in result.matches] == ["a.py"]

    @pytest.mark.asyncio
    async def test_total_matches_passthrough(self) -> None:
        backend = _MockBackend(
            output=MatchPathsOutput(
                matches=[MatchPathResult(path="a.py", type="file", size=1)],
                total_matches=42,
                truncated=False,
            )
        )
        result = await match_paths_impl(backend, MatchPathsInput(pattern="*.py"))
        assert result.total_matches == 42

    @pytest.mark.asyncio
    async def test_to_llm_dict_excludes_internal_fields(self) -> None:
        backend = _MockBackend(
            output=MatchPathsOutput(
                matches=[MatchPathResult(path="a.py", type="file", size=1)],
                total_matches=1,
                truncated=False,
            )
        )
        result = await match_paths_impl(backend, MatchPathsInput(pattern="*.py"))
        llm_dict = result.to_llm_dict()
        assert set(llm_dict.keys()) == {"matches", "total_matches", "truncated"}
        assert llm_dict["matches"] == [{"path": "a.py", "type": "file", "size": 1}]
        assert llm_dict["total_matches"] == 1
        assert llm_dict["truncated"] is False

    def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_match_paths(registry)
        definition = registry.get_tool("match_paths")
        assert definition is not None
        assert definition.output_schema == MatchPathsOutput.model_llm_schema()
        schema_str = str(definition.output_schema)
        assert "matches" in schema_str
        assert "total_matches" in schema_str
        assert "truncated" in schema_str


class TestMatchPathsIntegration:
    """Integration tests against a real HostBackend workspace."""

    @pytest.mark.asyncio
    async def test_pattern_returns_matching_paths_relative_to_workspace(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "src/main.py", "x\n")
        _write(temp_workspace, "src/sub/util.py", "x\n")
        _write(temp_workspace, "tests/test.py", "x\n")
        result = await backend.match_paths(pattern="src/**/*.py", path=Path("."))

        paths = {Path(m.path) for m in result.matches}
        assert Path("src/main.py") in paths
        assert Path("src/sub/util.py") in paths
        assert Path("tests/test.py") not in paths
        for m in result.matches:
            assert not Path(m.path).is_absolute()

    @pytest.mark.asyncio
    async def test_respects_include_hidden_flag_for_dotfiles(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, ".hidden/config.py", "x\n")
        _write(temp_workspace, "visible.py", "x\n")

        excluded = await backend.match_paths(
            pattern="**/*.py", path=Path("."), include_hidden=False
        )
        assert Path("visible.py") in {Path(m.path) for m in excluded.matches}
        assert Path(".hidden/config.py") not in {Path(m.path) for m in excluded.matches}

        included = await backend.match_paths(
            pattern="**/*.py", path=Path("."), include_hidden=True
        )
        paths = {Path(m.path) for m in included.matches}
        assert Path("visible.py") in paths
        assert Path(".hidden/config.py") in paths

    @pytest.mark.asyncio
    async def test_respects_include_ignored_flag_for_git_ignored_files(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        (temp_workspace / ".gitignore").write_text("ignored.py\n", newline="")
        _write(temp_workspace, "kept.py", "x\n")
        _write(temp_workspace, "ignored.py", "x\n")

        excluded = await backend.match_paths(
            pattern="**/*.py", path=Path("."), include_ignored=False
        )
        paths = {Path(m.path) for m in excluded.matches}
        assert Path("kept.py") in paths
        assert Path("ignored.py") not in paths

        included = await backend.match_paths(
            pattern="**/*.py", path=Path("."), include_ignored=True
        )
        paths = {Path(m.path) for m in included.matches}
        assert Path("kept.py") in paths
        assert Path("ignored.py") in paths

    @pytest.mark.asyncio
    async def test_respects_extension_filter(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "a.py", "x\n")
        _write(temp_workspace, "a.txt", "x\n")
        _write(temp_workspace, "a.pyi", "x\n")

        result = await backend.match_paths(
            pattern="**/*", path=Path("."), file_type="py"
        )
        paths = {Path(m.path) for m in result.matches if m.type == "file"}
        assert Path("a.py") in paths
        assert Path("a.txt") not in paths
        assert Path("a.pyi") not in paths

    @pytest.mark.asyncio
    async def test_respects_max_results_limit(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        for i in range(5):
            _write(temp_workspace, f"file{i}.py", "x\n")
        result = await backend.match_paths(
            pattern="**/*.py", path=Path("."), max_results=2
        )

        assert len(result.matches) == 2
        assert result.truncated is True

    @pytest.mark.asyncio
    async def test_honors_gitignore_rules(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        (temp_workspace / ".gitignore").write_text("ignored.py\n", newline="")
        _write(temp_workspace, "kept.py", "x\n")
        _write(temp_workspace, "ignored.py", "x\n")

        result = await backend.match_paths(
            pattern="**/*.py", path=Path("."), include_ignored=False
        )
        paths = {Path(m.path) for m in result.matches}
        assert Path("kept.py") in paths
        assert Path("ignored.py") not in paths

    @pytest.mark.asyncio
    async def test_honors_aiignore_rules(
        self, backend_with_config: HostBackend, temp_workspace: Path
    ) -> None:
        (temp_workspace / ".aiignore").write_text("secret.py\n", newline="")
        _write(temp_workspace, "kept.py", "x\n")
        _write(temp_workspace, "secret.py", "x\n")

        result = await backend_with_config.match_paths(
            pattern="**/*.py", path=Path("."), include_ignored=False
        )
        paths = {Path(m.path) for m in result.matches}
        assert Path("kept.py") in paths
        assert Path("secret.py") not in paths

    @pytest.mark.asyncio
    async def test_returns_paths_sorted_lexicographically(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "zebra.py", "x\n")
        _write(temp_workspace, "apple.py", "x\n")
        _write(temp_workspace, "mango.py", "x\n")

        result = await backend.match_paths(pattern="**/*.py", path=Path("."))

        paths = [m.path for m in result.matches]
        assert paths == sorted(paths)
        assert paths == ["apple.py", "mango.py", "zebra.py"]

    @pytest.mark.asyncio
    async def test_supports_offset_for_pagination(
        self, backend: HostBackend, temp_workspace: Path
    ) -> None:
        _write(temp_workspace, "a.py", "x\n")
        _write(temp_workspace, "b.py", "x\n")
        _write(temp_workspace, "c.py", "x\n")

        result = await backend.match_paths(pattern="**/*.py", path=Path("."))
        first_paths = {m.path for m in result.matches}
        assert first_paths == {"a.py", "b.py", "c.py"}

        impl_result = await match_paths_impl(
            backend,
            MatchPathsInput(pattern="**/*.py", path=".", offset=1),
        )
        impl_paths = {m.path for m in impl_result.matches}
        assert len(impl_paths) == 2
        assert impl_paths <= first_paths
