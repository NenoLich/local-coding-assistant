"""Unit tests for the search_text tool.

Covers the test cases defined in plan_docs/2026_09/tools_platform_tests.md
(section test_search_text_tool.py):

- literal query returns matches with file, line, column, context
- regex pattern returns regex matches
- include_globs / exclude_globs filtering
- case_sensitive flag
- context_lines before/after each match
- max_results limit (clamped to max_search_matches)
- internal max_bytes output limit and truncation metadata
- include_gitignored respects .gitignore/.aiignore
- next_offset when truncated for pagination
- falls back to Python when ripgrep not available
- output structure identical between ripgrep and Python backends
- LLM output excludes file_observations
- ToolDefinition output_schema uses model_llm_schema
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from local_coding_assistant.tools_platform.backends import host as host_backend_module
from local_coding_assistant.tools_platform.backends.host import HostBackend
from local_coding_assistant.tools_platform.contracts import (
    FileObservation,
    ObservationSource,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.search_text import (
    PatternType,
    SearchTextInput,
    SearchTextMatch,
    SearchTextOutput,
    register_search_text,
    search_text_impl,
)


def _match(path: str, line_number: int = 1, column: int = 0) -> SearchTextMatch:
    return SearchTextMatch(
        path=path,
        line_number=line_number,
        column=column,
        line_content=f"content of {path}",
        context_before=[],
        context_after=[],
    )


class _MockBackend:
    """Mock backend returning a controllable SearchTextOutput."""

    def __init__(self, output: SearchTextOutput | None = None) -> None:
        self.output = output or SearchTextOutput(
            matches=[],
            total_matches=0,
            truncated=False,
            next_offset=None,
        )
        self.search_text_kwargs: list[dict[str, Any]] = []
        self.observe_calls: list[tuple[Path, str | None]] = []

    async def search_text(self, **kwargs: Any) -> SearchTextOutput:
        self.search_text_kwargs.append(kwargs)
        return self.output

    async def observe_file(
        self, path: Path, session_id: str | None = None
    ) -> FileObservation:
        self.observe_calls.append((path, session_id))
        return FileObservation(
            path=str(path),
            hash="obs-hash",
            size=10,
            mtime=datetime.now(UTC),
            source=ObservationSource.READ,
        )


class TestSearchTextImpl:
    """Tests for search_text_impl (argument pass-through, pagination, observations)."""

    @pytest.mark.asyncio
    async def test_passes_input_arguments_to_backend(self) -> None:
        backend = _MockBackend()
        input_model = SearchTextInput(
            query="hello",
            path="src",
            pattern=PatternType.REGEX,
            includes=["*.py"],
            excludes=["*_test.py"],
            case_sensitive=True,
            context_lines=3,
            max_results=50,
            offset=5,
            include_gitignored=True,
        )
        await search_text_impl(backend, input_model, session_id="sess-1")

        assert len(backend.search_text_kwargs) == 1
        kwargs = backend.search_text_kwargs[0]
        assert kwargs["query"] == "hello"
        assert kwargs["path"] == Path("src")
        assert kwargs["pattern"] == "regex"
        assert kwargs["includes"] == ["*.py"]
        assert kwargs["excludes"] == ["*_test.py"]
        assert kwargs["case_sensitive"] is True
        assert kwargs["context_lines"] == 3
        assert kwargs["max_matches"] == 50
        assert kwargs["max_bytes"] == 262144
        assert kwargs["include_ignored"] is True

    @pytest.mark.asyncio
    async def test_skips_offset_matches(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[_match("a.py"), _match("b.py"), _match("c.py")],
                total_matches=3,
                truncated=True,
                next_offset=3,
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x", offset=1))

        assert [m.path for m in result.matches] == ["b.py", "c.py"]
        assert result.next_offset == 1 + 2

    @pytest.mark.asyncio
    async def test_no_offset_when_offset_zero(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[_match("a.py"), _match("b.py")],
                total_matches=2,
                truncated=False,
                next_offset=None,
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x", offset=0))
        assert [m.path for m in result.matches] == ["a.py", "b.py"]

    @pytest.mark.asyncio
    async def test_next_offset_none_when_not_truncated(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[_match("a.py")],
                total_matches=1,
                truncated=False,
                next_offset=None,
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x"))
        assert result.next_offset is None

    @pytest.mark.asyncio
    async def test_total_matches_passthrough(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[_match("a.py")],
                total_matches=42,
                truncated=False,
                next_offset=None,
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x"))
        assert result.total_matches == 42

    @pytest.mark.asyncio
    async def test_file_observations_for_unique_paths(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[
                    _match("a.py", 1),
                    _match("a.py", 5),
                    _match("b.py", 2),
                ],
                total_matches=3,
                truncated=False,
                next_offset=None,
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x"))

        assert len(result.file_observations) == 2
        observed_paths = {obs.path for obs in result.file_observations}
        assert observed_paths == {"a.py", "b.py"}
        assert len(backend.observe_calls) == 2

    @pytest.mark.asyncio
    async def test_to_llm_dict_excludes_file_observations(self) -> None:
        backend = _MockBackend(
            output=SearchTextOutput(
                matches=[_match("a.py")],
                total_matches=1,
                truncated=False,
                next_offset=None,
                file_observations=[
                    FileObservation(
                        path="a.py",
                        hash="h",
                        size=1,
                        mtime=datetime.now(UTC),
                        source=ObservationSource.READ,
                    )
                ],
            )
        )
        result = await search_text_impl(backend, SearchTextInput(query="x"))
        llm_dict = result.to_llm_dict()
        assert "file_observations" not in llm_dict
        assert set(llm_dict.keys()) == {
            "matches",
            "total_matches",
            "truncated",
            "next_offset",
        }
        assert len(llm_dict["matches"]) == 1

    @pytest.mark.asyncio
    async def test_tool_definition_output_schema_uses_model_llm_schema(self) -> None:
        registry = ToolRegistry()
        register_search_text(registry)
        definition = registry.get_tool("search_text")
        assert definition is not None
        assert definition.output_schema == SearchTextOutput.model_llm_schema()
        schema_str = str(definition.output_schema)
        assert "file_observations" not in schema_str
        assert "matches" in schema_str
        assert "total_matches" in schema_str
        assert "truncated" in schema_str


class TestSearchTextIntegration:
    """Integration tests against a real HostBackend workspace."""

    def _write(self, backend: HostBackend, name: str, content: str) -> Path:
        file_path = backend.get_workspace_root() / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, newline="")
        return file_path

    @pytest.mark.asyncio
    async def test_literal_query_returns_matches_with_file_line_column(
        self, backend: HostBackend
    ) -> None:
        self._write(backend, "src/main.py", 'def hello():\n    print("Hello, world")\n')
        result = await search_text_impl(
            backend,
            SearchTextInput(query="Hello", path=".", pattern=PatternType.LITERAL),
        )

        assert isinstance(result, SearchTextOutput)
        assert len(result.matches) >= 1
        match = result.matches[0]
        assert Path(match.path) == Path("src/main.py")
        assert match.line_number >= 1
        assert match.column >= 0
        assert "hello" in match.line_content.lower()

    @pytest.mark.asyncio
    async def test_regex_pattern_returns_regex_matches(
        self, backend: HostBackend
    ) -> None:
        self._write(
            backend,
            "src/main.py",
            "def hello():\n    pass\nclass Goodbye:\n    pass\n",
        )
        result = await search_text_impl(
            backend,
            SearchTextInput(query=r"def\s+\w+", path=".", pattern=PatternType.REGEX),
        )

        assert len(result.matches) >= 1
        assert any("def hello" in m.line_content for m in result.matches)

    @pytest.mark.asyncio
    async def test_respects_includes_filter(self, backend: HostBackend) -> None:
        self._write(backend, "src/a.py", "target\n")
        self._write(backend, "src/b.txt", "target\n")
        self._write(backend, "tests/c.py", "target\n")
        result = await search_text_impl(
            backend,
            SearchTextInput(query="target", path=".", includes=["*.py"]),
        )

        paths = {Path(m.path) for m in result.matches}
        assert Path("src/a.py") in paths
        assert Path("tests/c.py") in paths
        assert Path("src/b.txt") not in paths

    @pytest.mark.asyncio
    async def test_respects_excludes_filter(self, backend: HostBackend) -> None:
        self._write(backend, "src/unit_test.py", "target\n")
        self._write(backend, "src/real.py", "target\n")
        result = await search_text_impl(
            backend,
            SearchTextInput(query="target", path=".", excludes=["*_test.py"]),
        )

        paths = {Path(m.path) for m in result.matches}
        assert Path("src/real.py") in paths
        assert Path("src/unit_test.py") not in paths

    @pytest.mark.asyncio
    async def test_respects_case_sensitive_flag(self, backend: HostBackend) -> None:
        self._write(backend, "src/a.py", "Hello\n")
        sensitive = await search_text_impl(
            backend,
            SearchTextInput(query="hello", path=".", case_sensitive=True),
        )
        insensitive = await search_text_impl(
            backend,
            SearchTextInput(query="hello", path=".", case_sensitive=False),
        )

        assert len(sensitive.matches) == 0
        assert len(insensitive.matches) >= 1

    @pytest.mark.asyncio
    async def test_returns_context_lines_before_after(
        self, backend: HostBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the Python implementation (ripgrep returns empty context).
        monkeypatch.setattr(
            host_backend_module,
            "search_with_ripgrep",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("rg missing")),  # type: ignore[arg-type]
        )
        self._write(
            backend,
            "src/a.py",
            "line one\nline two\nTARGET\nline four\nline five\n",
        )
        result = await search_text_impl(
            backend,
            SearchTextInput(query="TARGET", path=".", context_lines=2),
        )

        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.context_before == ["line one", "line two"]
        assert match.context_after == ["line four", "line five"]

    @pytest.mark.asyncio
    async def test_respects_max_results_limit(self, backend: HostBackend) -> None:
        lines = "\n".join(f"match {i}" for i in range(20))
        self._write(backend, "src/a.py", lines)
        result = await search_text_impl(
            backend, SearchTextInput(query="match", path=".", max_results=5)
        )

        assert len(result.matches) <= 5

    @pytest.mark.asyncio
    async def test_include_gitignored_false_respects_gitignore(
        self, backend: HostBackend
    ) -> None:
        (backend.get_workspace_root() / ".gitignore").write_text(
            "ignored.py\n", newline=""
        )
        self._write(backend, "kept.py", "needle\n")
        self._write(backend, "ignored.py", "needle\n")
        result = await search_text_impl(
            backend,
            SearchTextInput(query="needle", path=".", include_gitignored=False),
        )

        paths = {m.path for m in result.matches}
        assert "kept.py" in paths
        assert "ignored.py" not in paths

    @pytest.mark.asyncio
    async def test_include_gitignored_true_includes_ignored_files(
        self, backend: HostBackend
    ) -> None:
        (backend.get_workspace_root() / ".gitignore").write_text(
            "ignored.py\n", newline=""
        )
        self._write(backend, "kept.py", "needle\n")
        self._write(backend, "ignored.py", "needle\n")
        result = await search_text_impl(
            backend,
            SearchTextInput(query="needle", path=".", include_gitignored=True),
        )

        paths = {m.path for m in result.matches}
        assert "kept.py" in paths
        assert "ignored.py" in paths

    @pytest.mark.asyncio
    async def test_returns_next_offset_when_truncated(
        self, backend: HostBackend
    ) -> None:
        lines = "\n".join(f"match {i}" for i in range(10))
        self._write(backend, "src/a.py", lines)
        result = await search_text_impl(
            backend,
            SearchTextInput(query="match", path=".", max_results=4, offset=0),
        )

        assert result.truncated is True
        assert result.next_offset is not None
        assert result.next_offset >= 1

        # Resume from the returned offset.
        second = await search_text_impl(
            backend,
            SearchTextInput(
                query="match", path=".", max_results=4, offset=result.next_offset
            ),
        )
        assert len(second.matches) <= 4

    @pytest.mark.asyncio
    async def test_falls_back_to_python_when_ripgrep_not_available(
        self, backend: HostBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            host_backend_module,
            "search_with_ripgrep",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("rg missing")),  # type: ignore[arg-type]
        )
        self._write(backend, "src/a.py", "needle\n")
        result = await search_text_impl(
            backend, SearchTextInput(query="needle", path=".")
        )

        assert isinstance(result, SearchTextOutput)
        assert len(result.matches) >= 1

    @pytest.mark.asyncio
    async def test_output_structure_identical_between_backends(
        self, backend: HostBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write(backend, "src/a.py", "needle\nmore\nneedle\n")

        # ripgrep path (default, rg is available in this environment).
        ripgrep_result = await search_text_impl(
            backend, SearchTextInput(query="needle", path=".")
        )

        # Python path (force fallback).
        monkeypatch.setattr(
            host_backend_module,
            "search_with_ripgrep",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("rg missing")),  # type: ignore[arg-type]
        )
        python_result = await search_text_impl(
            backend, SearchTextInput(query="needle", path=".")
        )

        assert set(ripgrep_result.model_dump().keys()) == set(
            python_result.model_dump().keys()
        )
        assert [m.path for m in ripgrep_result.matches] == [
            m.path for m in python_result.matches
        ]
        assert [m.line_number for m in ripgrep_result.matches] == [
            m.line_number for m in python_result.matches
        ]
        assert [m.column for m in ripgrep_result.matches] == [
            m.column for m in python_result.matches
        ]
