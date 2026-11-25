"""Unit tests for ValidationMiddleware.

Covers argument validation, path normalization, clamping, and dangerous
argument detection described in plan_docs/2026_09/tools_platform_tests.md.

The input models API uses tool-specific Pydantic input models (e.g.
``ReadFileInput``, ``SearchTextInput``) exposed through
``ToolImplementation.input_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from local_coding_assistant.tools_platform.contracts import ToolCallRequest
from local_coding_assistant.tools_platform.exceptions import (
    ToolPathViolationError,
    ToolValidationError,
)
from local_coding_assistant.tools_platform.middleware import MiddlewareContext
from local_coding_assistant.tools_platform.middleware.prepare.validation import (
    ValidationMiddleware,
)
from local_coding_assistant.tools_platform.registry import ToolRegistry
from local_coding_assistant.tools_platform.tools.register import register_all_tools
from local_coding_assistant.tools_platform.workspace import WorkspaceManager
from local_coding_assistant.tools_platform.workspace.models import PathPolicy

LIMITS = {
    "max_output_bytes": 262144,
    "max_read_lines": 400,
    "max_search_matches": 200,
}


def _make_workspace_manager(
    tmp_path: Path, **policy_overrides: Any
) -> WorkspaceManager:
    path_policy = PathPolicy(
        workspace_root=tmp_path,
        allowed_patterns=policy_overrides.get("allowed_patterns", ["**"]),
        denied_patterns=policy_overrides.get("denied_patterns", []),
        allow_symlinks=policy_overrides.get("allow_symlinks", False),
        max_depth=policy_overrides.get("max_depth"),
        require_relative=True,
    )
    return WorkspaceManager(
        workspace_id="test-ws",
        workspace_root=tmp_path,
        path_policy=path_policy,
        config_manager=None,
    )


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_all_tools(registry)
    return registry


def _make_context(
    wm: WorkspaceManager, registry: ToolRegistry, request: ToolCallRequest
) -> MiddlewareContext:
    return MiddlewareContext(
        request=request,
        registry=registry,
        approval_service=Mock(),
        workspace_manager=wm,
        backend=Mock(),
        scheduler=Mock(),
    )


def _request(
    tool_id: str,
    call_id: str = "c1",
    arguments: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_id=tool_id,
        call_id=call_id,
        session_id="sess_1",
        frame_id="frame_1",
        raw_arguments=arguments or {},
        workspace_id="sess_1",
    )


async def _prepare(
    wm: WorkspaceManager, registry: ToolRegistry, request: ToolCallRequest
):
    mw = ValidationMiddleware(wm, limits=LIMITS)
    context = _make_context(wm, registry, request)
    await mw.prepare(context)
    return context


@pytest.fixture
def registry() -> ToolRegistry:
    return _make_registry()


# ---------------------------------------------------------------------------
# Unknown tool_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_call_rejects_unknown_tool_id(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request("nonexistent_tool", arguments={"path": "src/main.py"})
    with pytest.raises(ToolValidationError):
        await _prepare(wm, registry, request)


# ---------------------------------------------------------------------------
# Input model validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_call_rejects_extra_fields(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    request = _request(
        "read_file",
        arguments={"path": "src/main.py", "extra_field": "should-be-rejected"},
    )
    with pytest.raises(ToolValidationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_accepts_read_file_input(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    request = _request(
        "read_file",
        arguments={"path": "src/main.py", "offset": 0, "limit": 10},
    )
    context = await _prepare(wm, registry, request)
    input_model = context.validated_call.input_model
    assert input_model.path == "src/main.py"
    assert input_model.offset == 0
    assert input_model.limit == 10


@pytest.mark.asyncio
async def test_validate_call_accepts_run_command_input(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "run_command",
        arguments={"argv": ["echo", "hi"], "cwd": "src"},
    )
    context = await _prepare(wm, registry, request)
    input_model = context.validated_call.input_model
    assert input_model.argv == ["echo", "hi"]
    assert input_model.cwd == "src"


@pytest.mark.asyncio
async def test_validate_call_accepts_search_text_input(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "search_text",
        arguments={"query": "foo", "max_results": 50},
    )
    context = await _prepare(wm, registry, request)
    input_model = context.validated_call.input_model
    assert input_model.query == "foo"
    assert input_model.max_results == 50


# ---------------------------------------------------------------------------
# Path normalization & boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_call_normalizes_relative_path(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    request = _request("read_file", arguments={"path": "src/main.py"})
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.path == "src/main.py"


@pytest.mark.asyncio
async def test_validate_call_rejects_absolute_path_outside_workspace(
    tmp_path, registry
):
    wm = _make_workspace_manager(tmp_path)
    request = _request("read_file", arguments={"path": "/etc/passwd"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_rejects_denied_patterns(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path, denied_patterns=["*.tmp", "temp/*"])
    (tmp_path / "file.tmp").write_text("x")
    request = _request("read_file", arguments={"path": "file.tmp"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_rejects_not_allowed_patterns(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path, allowed_patterns=["*.py", "*.txt"])
    (tmp_path / "notes.md").write_text("x")
    request = _request("read_file", arguments={"path": "notes.md"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_rejects_exceeding_max_depth(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path, max_depth=2)
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "file.py").write_text("x")
    request = _request("read_file", arguments={"path": "a/b/c/file.py"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_rejects_symlink_when_disallowed(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path, allow_symlinks=False)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "link.py").symlink_to("src/main.py")
    request = _request("read_file", arguments={"path": "link.py"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_allows_symlink_when_allowed(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path, allow_symlinks=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    (tmp_path / "link.py").symlink_to("src/main.py")
    request = _request("read_file", arguments={"path": "link.py"})
    context = await _prepare(wm, registry, request)
    assert context.validated_call is not None


@pytest.mark.asyncio
async def test_validate_call_rejects_dotdot_escaping_workspace(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request("read_file", arguments={"path": "../etc/passwd"})
    with pytest.raises(ToolPathViolationError):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_allows_dotdot_within_workspace(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file.py").write_text("x")
    request = _request("read_file", arguments={"path": "subdir/../file.py"})
    context = await _prepare(wm, registry, request)
    assert context.validated_call is not None


@pytest.mark.asyncio
async def test_validate_call_normalizes_cwd(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    request = _request("run_command", arguments={"argv": ["echo", "hi"], "cwd": "src"})
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.cwd == "src"


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_call_clamps_read_file_limit(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    request = _request(
        "read_file",
        arguments={"path": "src/main.py", "limit": 999999},
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.limit == 262144


@pytest.mark.asyncio
async def test_validate_call_clamps_search_text_max_results(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "search_text",
        arguments={"query": "foo", "max_results": 9999},
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.max_results == 200


@pytest.mark.asyncio
async def test_validate_call_clamps_search_symbols_limit(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "search_symbols",
        arguments={"query": "foo", "limit": 9999},
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.limit == 200


@pytest.mark.asyncio
async def test_validate_call_clamps_match_paths_max_results(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "match_paths",
        arguments={"pattern": "**/*.py", "max_results": 9999},
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.max_results == 200


# ---------------------------------------------------------------------------
# Dangerous arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_call_detects_shell_escape_non_query(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("x")
    request = _request(
        "read_file",
        arguments={"path": "src/main.py;rm -rf /"},
    )
    with pytest.raises(ValueError, match="Dangerous pattern"):
        await _prepare(wm, registry, request)


@pytest.mark.asyncio
async def test_validate_call_allows_shell_patterns_in_query(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "search_text",
        arguments={"query": "foo && bar", "path": "."},
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call is not None


@pytest.mark.asyncio
async def test_validate_call_accepts_create_file_with_shell_content(tmp_path, registry):
    wm = _make_workspace_manager(tmp_path)
    request = _request(
        "create_file",
        arguments={
            "path": "script.sh",
            "content": "echo a && echo b || true; echo c | tee d",
        },
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.content == (
        "echo a && echo b || true; echo c | tee d"
    )


@pytest.mark.asyncio
async def test_validate_call_accepts_replace_exact_with_shell_content(
    tmp_path, registry
):
    wm = _make_workspace_manager(tmp_path)
    (tmp_path / "file.txt").write_text("foo $(bar)")
    request = _request(
        "replace_exact",
        arguments={
            "path": "file.txt",
            "old_text": "foo $(bar)",
            "new_text": "baz && qux | quux",
            "expected_match_count": 1,
        },
    )
    context = await _prepare(wm, registry, request)
    assert context.validated_call.input_model.old_text == "foo $(bar)"
    assert context.validated_call.input_model.new_text == "baz && qux | quux"
