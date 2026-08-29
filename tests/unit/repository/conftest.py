"""Conftest for repository unit tests."""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(tmp_path: Path) -> Path:
    """Fixture for temporary directory."""
    return tmp_path
