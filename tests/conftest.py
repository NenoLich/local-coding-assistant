from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_coding_assistant.config import EnvManager, PathManager

# Add project src/ to sys.path for imports like `local_coding_assistant.*`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# This fixture ensures the test environment is set for all tests
@pytest.fixture(autouse=True)
def ensure_test_environment():
    """Ensure all tests run with the correct environment settings."""
    # Force test environment
    os.environ["LOCCA_ENV"] = "test"
    os.environ["LOCCA_TEST_MODE"] = "true"

    # Verify the environment is set correctly
    assert os.environ.get("LOCCA_ENV") == "test", (
        f"Test environment not set correctly. LOCCA_ENV={os.environ.get('LOCCA_ENV')}"
    )


def test_environment_setup():
    """Verify the test environment is properly configured."""
    assert os.environ.get("LOCCA_ENV") == "test", (
        f"Expected LOCCA_ENV='test', got '{os.environ.get('LOCCA_ENV')}'. "
        "Check if environment is being overridden in test setup or fixtures."
    )


def pytest_configure(config):
    """Configure pytest for async tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for async tests."""
    for item in items:
        # Check if test function is async
        if hasattr(item, "function") and asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


@pytest.fixture(scope="session")
def path_manager():
    """Session-scoped PathManager instance for tests."""
    return PathManager(is_testing=True)


@pytest.fixture
def test_configs(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """Fixture that provides test config files and directories."""
    # Create a new PathManager instance for testing
    test_path_manager = PathManager(
        is_development=False,
        is_testing=True,
        project_root=tmp_path,  # Use tmp_path as the project root
    )

    # Create an EnvManager that uses our test PathManager
    test_env_manager = EnvManager(load_env=False, path_manager=test_path_manager)

    # Get the test config directory
    test_config_dir = test_path_manager.get_config_dir()
    test_config_dir.mkdir(parents=True, exist_ok=True)

    # Create config files with the expected names
    default_config = test_config_dir / "tools.default.yaml"
    local_config = test_config_dir / "tools.local.yaml"

    # Create modules directory
    modules_dir = test_path_manager.get_module_dir()
    modules_dir.mkdir(parents=True, exist_ok=True)

    return {
        "default": default_config,
        "local": local_config,
        "config_dir": test_config_dir,
        "modules_dir": modules_dir,
        "path_manager": test_path_manager,
        "env_manager": test_env_manager,  # Include the test env manager
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
