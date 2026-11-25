"""Unit tests for runtime-level run options and their config translation.

``RunOptions`` is the typed contract front-ends use for config-affecting
run options; ``RuntimeManager.apply_run_options`` is the single place that
translates them into config session overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_coding_assistant.runtime.manager import RuntimeManager
from local_coding_assistant.runtime.options import RunOptions


class _StubPathManager:
    """Minimal path manager exposing only the template dir."""

    def get_template_dir(self) -> Path:
        return Path(".")


class _RecordingConfigManager:
    """Minimal IConfigManager stub recording session overrides."""

    def __init__(self) -> None:
        self.overrides: dict[str, Any] = {}
        self.path_manager = _StubPathManager()
        self._global_config = object()

    @property
    def global_config(self) -> Any:
        return self._global_config

    def load_global_config(self) -> Any:
        return self._global_config

    def set_session_overrides(self, overrides: dict[str, Any]) -> None:
        self.overrides.update(overrides)


class TestRunOptionsTranslation:
    """RunOptions -> config session overrides mapping."""

    def test_empty_options_yield_no_overrides(self) -> None:
        assert RunOptions().to_session_overrides() == {}

    def test_model_override(self) -> None:
        assert RunOptions(model="gpt-5").to_session_overrides() == {
            "llm.model_name": "gpt-5"
        }

    def test_agent_mode_override(self) -> None:
        assert RunOptions(agent_mode="frame").to_session_overrides() == {
            "runtime.agent_mode": "frame"
        }

    def test_classic_tool_call_mode_does_not_touch_sandbox(self) -> None:
        assert RunOptions(tool_call_mode="classic").to_session_overrides() == {
            "runtime.tool_call_mode": "classic"
        }

    def test_ptc_tool_call_mode_enables_sandbox(self) -> None:
        assert RunOptions(tool_call_mode="ptc").to_session_overrides() == {
            "runtime.tool_call_mode": "ptc",
            "sandbox.enabled": True,
        }

    def test_ptc_with_named_sandbox_session(self) -> None:
        assert RunOptions(
            tool_call_mode="ptc", sandbox_session="shared"
        ).to_session_overrides() == {
            "runtime.tool_call_mode": "ptc",
            "sandbox.enabled": True,
            "sandbox.session_id": "shared",
            "sandbox.persistence": True,
        }

    def test_combined_overrides(self) -> None:
        overrides = RunOptions(
            model="gpt-5",
            agent_mode="frame",
            tool_call_mode="ptc",
            sandbox_session="shared",
        ).to_session_overrides()
        assert overrides == {
            "runtime.agent_mode": "frame",
            "runtime.tool_call_mode": "ptc",
            "sandbox.enabled": True,
            "sandbox.session_id": "shared",
            "sandbox.persistence": True,
            "llm.model_name": "gpt-5",
        }

    def test_auto_approve_override(self) -> None:
        assert RunOptions(auto_approve=True).to_session_overrides() == {
            "tools_platform.auto_approve": True
        }

    def test_auto_approve_false_clears_override(self) -> None:
        """An explicit false must clear a previously set true override."""
        assert RunOptions(auto_approve=False).to_session_overrides() == {
            "tools_platform.auto_approve": False
        }

    def test_auto_approve_not_set_emits_nothing(self) -> None:
        """Omitting auto_approve must not touch the config override."""
        assert RunOptions().to_session_overrides() == {}
        assert RunOptions(model="gpt-5").to_session_overrides() == {
            "llm.model_name": "gpt-5"
        }

    def test_combined_overrides_with_auto_approve(self) -> None:
        overrides = RunOptions(
            model="gpt-5",
            agent_mode="frame",
            tool_call_mode="ptc",
            sandbox_session="shared",
            auto_approve=True,
        ).to_session_overrides()
        assert overrides == {
            "runtime.agent_mode": "frame",
            "runtime.tool_call_mode": "ptc",
            "sandbox.enabled": True,
            "sandbox.session_id": "shared",
            "sandbox.persistence": True,
            "llm.model_name": "gpt-5",
            "tools_platform.auto_approve": True,
        }


class TestApplyRunOptions:
    """RuntimeManager.apply_run_options wiring."""

    def _make_manager(self) -> tuple[RuntimeManager, _RecordingConfigManager]:
        config = _RecordingConfigManager()
        return RuntimeManager(config_manager=config), config

    def test_applies_translated_overrides(self) -> None:
        manager, config = self._make_manager()

        manager.apply_run_options(
            RunOptions(model="gpt-5", tool_call_mode="ptc", sandbox_session="shared")
        )

        assert config.overrides == {
            "runtime.tool_call_mode": "ptc",
            "sandbox.enabled": True,
            "sandbox.session_id": "shared",
            "sandbox.persistence": True,
            "llm.model_name": "gpt-5",
        }

    def test_empty_options_leave_config_untouched(self) -> None:
        manager, config = self._make_manager()

        manager.apply_run_options(RunOptions())

        assert config.overrides == {}
