"""Unit tests for the ConfigManager-backed CLI config commands.

``config effective`` and ``config overrides`` read the resolved
ConfigManager (not the env layer) so front-ends can see the config the
runtime actually uses, including session overrides.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from local_coding_assistant.cli.commands.config import app

runner = CliRunner()


class _FakeResolvedConfig:
    """Resolved-config stand-in with attribute access and model_dump."""

    def __init__(self) -> None:
        self.llm = type("Llm", (), {"model_name": "test-model"})()
        self.runtime = type("Runtime", (), {"agent_mode": "frame"})()

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {
            "llm": {"model_name": self.llm.model_name},
            "runtime": {"agent_mode": self.runtime.agent_mode},
        }


class _FakeConfigManager:
    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._resolved = _FakeResolvedConfig()
        self._overrides = overrides or {}

    @property
    def global_config(self) -> _FakeResolvedConfig:
        return self._resolved

    @property
    def session_overrides(self) -> dict[str, Any]:
        return dict(self._overrides)


@contextmanager
def _patched_bootstrap(config_manager: _FakeConfigManager) -> Iterator[None]:
    """Patch bootstrap to yield a fake async context around config_manager."""
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = {"config": config_manager}
    mock_ctx.__aexit__.return_value = False
    with patch(
        "local_coding_assistant.cli.commands.config.bootstrap", return_value=mock_ctx
    ):
        yield


class TestConfigEffective:
    """config effective [KEY]"""

    def test_single_key_resolution(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["effective", "llm.model_name"])
            assert result.exit_code == 0, result.output
            assert "llm.model_name=test-model" in result.output

    def test_nested_section_value(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["effective", "runtime.agent_mode"])
            assert result.exit_code == 0, result.output
            assert "runtime.agent_mode=frame" in result.output

    def test_missing_key_fails(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["effective", "llm.nope"])
            assert result.exit_code == 1
            assert "No configuration found for key" in result.output

    def test_full_dump_json(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["effective", "--format", "json"])
            assert result.exit_code == 0, result.output
            payload = json.loads(result.output)
            assert payload["llm"]["model_name"] == "test-model"

    def test_full_dump_yaml_default(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["effective"])
            assert result.exit_code == 0, result.output
            assert "model_name: test-model" in result.output


class TestConfigOverrides:
    """config overrides"""

    def test_lists_session_overrides(self) -> None:
        overrides = {"llm.model_name": "gpt-5", "sandbox.enabled": True}
        with _patched_bootstrap(_FakeConfigManager(overrides)):
            result = runner.invoke(app, ["overrides"])
            assert result.exit_code == 0, result.output
            assert "llm.model_name=gpt-5" in result.output
            assert "sandbox.enabled=True" in result.output

    def test_reports_no_overrides(self) -> None:
        with _patched_bootstrap(_FakeConfigManager()):
            result = runner.invoke(app, ["overrides"])
            assert result.exit_code == 0, result.output
            assert "No active session overrides" in result.output
