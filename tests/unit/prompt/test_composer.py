"""Golden and behavioral tests for PromptComposer.

Golden approach: inbound is a ``PromptContext`` (plus render settings such as
preset and ``session_context_resolution``); outbound is the ``RenderedPrompt``
produced by the real composer, compared against golden files.

Run ``pytest tests/unit/prompt/test_composer.py -v --update-goldens`` to
regenerate the golden ``expected_output.txt`` files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import serialize_rendered

GOLDEN_DIR = Path(__file__).parent / "golden"

CASES = [
    "static_default_full",
    "dynamic_default_full",
    "compaction_preset",
    "subagent_preset",
    "planner_preset",
    "executor_preset",
    "minimal_context",
    "sandbox_enabled",
    "model_gemini_provider",
    "personality_friendly",
    "system_reminder_reasoning_only",
    "static_no_user_input",
]


def _load_case(name: str) -> dict:
    case_dir = GOLDEN_DIR / name
    with open(case_dir / "input.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("case", CASES)
def test_prompt_composition_golden(render_prompt, case, request):
    """Render each context scenario and compare against its golden output."""
    data = _load_case(case)
    render_kwargs = data.get("render", {})
    rendered = render_prompt(data["context"], **render_kwargs)
    output = serialize_rendered(rendered)

    expected_file = GOLDEN_DIR / case / "expected_output.txt"
    if request.config.getoption("--update-goldens", default=False):
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        expected_file.write_text(output, encoding="utf-8")
        return

    expected = expected_file.read_text(encoding="utf-8")
    assert output == expected, (
        f"\nGenerated prompt does not match golden for {case}."
        f"\nExpected file: {expected_file}"
        "\nRun with `pytest tests/unit/prompt/test_composer.py -v --update-goldens` to regenerate"
        f"\nDiff (expected vs actual):\n{'=' * 40}\n{expected}"
        f"\n{'=' * 40}\n{output}"
        f"\n{'=' * 40}"
    )


# ---------------------------------------------------------------------------
# Full context shared by behavioral tests
# ---------------------------------------------------------------------------

FULL_CTX = {
    "session_id": "sess-alpha",
    "execution_mode": "classic_tools",
    "tool_call_mode": "classic",
    "agent_profile": {
        "name": "default",
        "kind": "default",
        "description": "Primary coding assistant focused on safe, step-by-step reasoning.",
        "goals": ["Deliver concise answers grounded in repository state"],
        "tone": "Confident, pragmatic, collaborative",
        "constraints": ["Never fabricate file paths"],
        "model_policy": "general",
    },
    "model_id": "gemini-2.0-flash",
    "model_provider": "google",
    "personality": "pragmatic",
    "workspace_root": "/workspace",
    "platform": "linux",
    "is_git_repo": True,
    "cwd": "/workspace/src",
    "current_date": 1735689600,
    "sandbox_enabled": False,
    "agents_md_content": [["AGENTS.md", "# Project rules\n- Keep it simple"]],
    "global_rules": "Never commit secrets to version control.",
    "config_instructions": ["Prefer type hints in all signatures."],
    "project_info": {
        "name": "locca",
        "description": "A local coding assistant.",
        "language_version": "3.12",
        "package_manager": "uv",
        "core_frameworks": ["pytest", "jinja2"],
        "build_backend": "hatchling",
        "linter_formatter": ["ruff", "mypy"],
        "test_frameworks": ["pytest"],
        "license": "MIT",
        "available_scripts": ["test", "lint"],
        "language_distribution": {"Python": 85.5, "YAML": 10.0, "Markdown": 4.5},
    },
    "repo_map_data": {
        "num_symbols": 1,
        "grouped_symbols": {
            "src/local_coding_assistant/prompt/composer.py": [
                {
                    "name": "PromptComposer",
                    "symbol_type": "class",
                    "line_number": 46,
                    "end_line_number": 356,
                    "signature": "class PromptComposer",
                    "parent_scope": None,
                }
            ]
        },
    },
    "memories": ["Prefer targeted edits over full rewrites."],
    "examples": [
        {
            "description": "edit example",
            "input": "Change x to y",
            "output": "Use the edit tool",
        }
    ],
    "tools": [{"name": "read_file", "description": "Read a file", "parameters": {}}],
    "active_skills": [],
    "metadata": {},
}


def test_static_mode_puts_all_context_in_system(render_prompt):
    """Static mode: every context block lands in the system prompt; user = input only."""
    rendered = render_prompt(
        FULL_CTX,
        session_context_resolution="static",
        user_input="What is the entry point?",
    )
    system_text = "\n".join(rendered.system_messages)
    assert "Project Metadata:" in system_text
    assert "Repo Map (top 1 symbols" in system_text
    assert "AGENTS.md Instructions" in system_text
    assert "Global Rules:" in system_text
    assert "Config Instructions:" in system_text
    assert "Relevant Memories:" in system_text
    assert "Reference Examples:" in system_text
    assert "CLASSIC TOOL USAGE" in system_text
    assert rendered.user_messages == ["What is the entry point?"]
    assert rendered.tool_schemas == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {},
            },
        }
    ]


def test_static_mode_system_prompt_is_cached(render_prompt):
    """Rendering the same static context twice returns the cached system prompt."""
    rendered_one = render_prompt(FULL_CTX, session_context_resolution="static")
    rendered_two = render_prompt(FULL_CTX, session_context_resolution="static")
    assert rendered_one == rendered_two
    # The composer caches the static system prompt on the instance; cannot introspect
    # the cache from outside, so behavioral equality is the contract.


def test_dynamic_mode_system_excludes_context_blocks(render_prompt):
    """Dynamic mode: system = core blocks only; context blocks move to user messages."""
    rendered = render_prompt(
        FULL_CTX,
        session_context_resolution="dynamic",
        user_input="What is the entry point?",
    )
    system_text = "\n".join(rendered.system_messages)
    user_text = "\n".join(rendered.user_messages)
    # Core blocks stay in system.
    assert "You are LOCCA" in system_text
    assert "Constraints:" in system_text
    assert "Output Format:" in system_text
    # Context blocks move to user.
    assert "Project Metadata:" in user_text
    assert "Repo Map" in user_text
    assert "AGENTS.md Instructions" in user_text
    assert rendered.user_messages[-1] == "What is the entry point?"


def test_compaction_preset_is_minimal(render_prompt):
    """Compaction preset: identity + compaction instructions + constraints only."""
    rendered = render_prompt(
        FULL_CTX, preset="compaction", session_context_resolution="static"
    )
    system_text = "\n".join(rendered.system_messages)
    assert "Compaction Agent:" in system_text
    assert "Summarize the conversation" in system_text
    assert "Repo Map" not in system_text
    assert "CLASSIC TOOL USAGE" not in system_text
    assert "AGENTS.md Instructions" not in system_text
    assert rendered.user_messages == []


def test_subagent_preset(render_prompt):
    """Subagent preset: identity + subagent instructions + constraints."""
    rendered = render_prompt(
        FULL_CTX, preset="subagent", session_context_resolution="static"
    )
    system_text = "\n".join(rendered.system_messages)
    assert "Subagent:" in system_text
    assert "Execute the assigned task" in system_text
    assert "Repo Map" not in system_text


def test_planner_preset_includes_extra(render_prompt):
    """Planner preset: default blocks plus planner enhancements."""
    rendered = render_prompt(
        FULL_CTX, preset="planner", session_context_resolution="static"
    )
    system_text = "\n".join(rendered.system_messages)
    assert "Planner Enhancements:" in system_text
    assert "numbered plan" in system_text
    assert "Repo Map" in system_text  # default context blocks still present


def test_executor_preset_includes_extra(render_prompt):
    """Executor preset: default blocks plus executor enhancements."""
    rendered = render_prompt(
        FULL_CTX, preset="executor", session_context_resolution="static"
    )
    system_text = "\n".join(rendered.system_messages)
    assert "Executor Enhancements:" in system_text
    assert "Run tests after each logical unit" in system_text


def test_minimal_context_still_renders(render_prompt):
    """Only required fields: identity + static_env + mode + constraints still render."""
    minimal = {
        "session_id": "sess-min",
        "execution_mode": "classic_tools",
        "tool_call_mode": "classic",
    }
    rendered = render_prompt(minimal, session_context_resolution="static")
    system_text = "\n".join(rendered.system_messages)
    assert "You are LOCCA" in system_text
    assert "Workspace root: " in system_text
    assert "CLASSIC TOOL USAGE" in system_text
    assert rendered.user_messages == []
    assert rendered.tool_schemas == []


def test_sandbox_enabled_renders_safety_and_reminder(render_prompt):
    """Sandbox enabled: safety block + system-reminder in metadata."""
    ctx = {**FULL_CTX, "sandbox_enabled": True}
    rendered = render_prompt(ctx, session_context_resolution="static")
    system_text = "\n".join(rendered.system_messages)
    assert "Sandbox:" in system_text
    reminder = rendered.metadata.get("system_reminder", "")
    assert reminder.startswith("<system-reminder>")
    assert "Sandbox is enabled" in reminder
    assert "Working directory:" in reminder  # cwd != workspace_root


def test_model_gemini_provider_renders_model_tuning(render_prompt):
    """model_provider=gemini selects base/model_gemini.jinja2 via path variable."""
    ctx = {**FULL_CTX, "model_provider": "gemini"}
    rendered = render_prompt(ctx, session_context_resolution="static")
    system_text = "\n".join(rendered.system_messages)
    assert "Model-specific tuning (Gemini)" in system_text
    assert "Model-specific tuning (default)" not in system_text


def test_model_google_provider_skips_missing_template(render_prompt):
    """model_provider=google has no matching template -> block renders empty."""
    ctx = {**FULL_CTX, "model_provider": "google"}
    rendered = render_prompt(ctx, session_context_resolution="static")
    system_text = "\n".join(rendered.system_messages)
    assert "Model-specific tuning" not in system_text


def test_personality_friendly_renders(render_prompt):
    """personality=friendly selects the friendly personality block."""
    ctx = {**FULL_CTX, "personality": "friendly"}
    rendered = render_prompt(ctx, session_context_resolution="static")
    system_text = "\n".join(rendered.system_messages)
    assert "Personality: Friendly & Collaborative" in system_text
    assert "Personality: Pragmatic & Direct" not in system_text


def test_system_reminder_reasoning_only_mode(render_prompt):
    """Reasoning-only mode injects PLANNING MODE system-reminder into metadata."""
    ctx = {**FULL_CTX, "execution_mode": "reasoning_only"}
    rendered = render_prompt(ctx, session_context_resolution="static")
    reminder = rendered.metadata.get("system_reminder", "")
    assert reminder.startswith("<system-reminder>")
    assert "PLANNING MODE" in reminder
    assert "do not execute tools" in reminder


def test_build_system_reminder_no_parts_when_defaults(render_prompt):
    """No sandbox, classic mode, cwd == workspace_root -> empty system-remainder."""
    ctx = {
        **FULL_CTX,
        "cwd": "/workspace",
        "sandbox_enabled": False,
        "execution_mode": "classic_tools",
    }
    rendered = render_prompt(ctx, session_context_resolution="static")
    assert rendered.metadata.get("system_reminder", "") == ""
