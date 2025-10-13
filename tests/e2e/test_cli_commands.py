from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from local_coding_assistant.cli.main import app

runner = CliRunner()


def test_run_query_echoes_plain_text():
    # Set test mode environment variable to enable mock responses
    with patch.dict(
        os.environ, {"LOCCA_TEST_MODE": "true", "OPENAI_API_KEY": "test-key"}
    ):
        result = runner.invoke(
            app, ["run", "query", "hello world", "--log-level", "INFO"]
        )
        assert result.exit_code == 0
        # Expect our CLI prints a Response: header and then the echo from LLMManager
        assert "Response:" in result.stdout
        assert "[LLMManager] Echo:" in result.stdout
