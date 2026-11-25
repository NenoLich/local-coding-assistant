"""Integration tests for PathManager behavior."""

import pytest

from local_coding_assistant.config.path_manager import PathManager


@pytest.fixture
def path_manager(path_manager_integration: PathManager) -> PathManager:
    """Alias fixture providing integration PathManager instance."""
    return path_manager_integration


def test_initializes_environment_directories(path_manager: PathManager) -> None:
    """PathManager should create expected directories for testing environment."""
    project_root = path_manager.get_project_root()

    expected_dirs = {
        "config": project_root / "tests" / "configs",
        "data": project_root / "tests" / "data",
        "cache": project_root / "tests" / "cache",
        "logs": project_root / "tests" / "logs",
        "modules": project_root / "tests" / "modules",
    }

    for name, directory in expected_dirs.items():
        assert directory.is_dir(), f"Expected {name} directory to exist at {directory}"


def test_resolve_special_paths_uses_environment_directories(
    path_manager: PathManager,
) -> None:
    """Special @-prefixed paths should resolve to environment-specific directories."""
    cases = {
        "@config/settings.yaml": path_manager.get_config_dir() / "settings.yaml",
        "@data/dataset.json": path_manager.get_data_dir() / "dataset.json",
        "@cache/state.db": path_manager.get_cache_dir() / "state.db",
        "@log/run.log": path_manager.get_log_dir() / "run.log",
        "@module/tool.py": path_manager.get_module_dir() / "tool.py",
        "@project/docs/readme.md": path_manager.get_project_root()
        / "docs"
        / "readme.md",
    }

    for raw, expected in cases.items():
        assert path_manager.resolve_path(raw) == expected


def test_resolve_path_with_base_and_parent_creation(path_manager: PathManager) -> None:
    """resolve_path should respect custom base directory and create parent folders when asked."""
    project_root = path_manager.get_project_root()
    custom_base = project_root / "custom"
    resolved = path_manager.resolve_path("config.json", base_dir=custom_base)
    assert resolved == custom_base / "config.json"

    nested = path_manager.resolve_path("logs/output/app.log", ensure_parent=True)
    assert nested == project_root / "logs" / "output" / "app.log"
    assert nested.parent.is_dir()


def test_unknown_prefix_falls_back_to_project_root(path_manager: PathManager) -> None:
    """Unknown @ prefixes should fall back to being relative to project root."""
    project_root = path_manager.get_project_root()
    unknown = path_manager.resolve_path("@unknown/file.txt")
    assert unknown == project_root / "@unknown" / "file.txt"
