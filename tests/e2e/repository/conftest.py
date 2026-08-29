"""Pytest configuration and fixtures for repository e2e tests."""

import tempfile
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from local_coding_assistant.config import ConfigManager, EnvManager, PathManager
from local_coding_assistant.repository.service import RepositoryContextService


@pytest.fixture
def test_project_structure(tmp_path: Path) -> Path:
    """Create a test project structure with multiple files."""
    # Create directory structure
    src_dir = tmp_path / "src" / "myproject"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Create Python files
    (src_dir / "__init__.py").write_text("")

    (src_dir / "main.py").write_text("""
def main():
    \"\"\"Main entry point.\"\"\"
    print("Hello, World!")

class Application:
    def __init__(self):
        self.name = "MyApp"
    
    def run(self):
        return f"Running {self.name}"
""")

    (src_dir / "utils.py").write_text("""
def helper_function(x: int) -> int:
    \"\"\"A helper function.\"\"\"
    return x * 2

class Helper:
    def process(self, data: str) -> str:
        return data.upper()
""")

    (src_dir / "models.py").write_text("""
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
    
    def get_display_name(self) -> str:
        return f"{self.name} <{self.email}>"
""")

    # Create config files
    (tmp_path / "pyproject.toml").write_text("""
[project]
name = "myproject"
version = "0.1.0"
description = "A test project"
dependencies = ["pydantic>=2.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
""")

    (tmp_path / "README.md").write_text("""
# My Project

A test project for e2e testing.
""")

    # Initialize git repository
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=tmp_path, capture_output=True
    )

    return tmp_path


@pytest.fixture
def repository_config(test_configs: dict[str, Any]) -> dict[str, Any]:
    """Create repository configuration for testing."""
    config_dir = test_configs["config_dir"]

    repo_config = {
        "enabled": True,
        "storage": {
            "mode": "test",
            "persistent_db_path": ".locca/repo_context.db",
            "temp_db_dir": "@data/temp",
            "temp_db_naming_strategy": "session_id",
        },
        "file_filter": {
            "supported_extensions": [
                ".py",
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".rs",
                ".go",
                ".c",
                ".cpp",
                ".h",
                ".hpp",
            ],
            "max_file_size_kb": 500,
        },
        "file_monitoring": {
            "enabled": False,  # Disable for e2e tests to avoid background threads
            "use_git_tracking": True,
            "notify_on_changes": True,
            "notification_types": ["modified", "created", "deleted"],
        },
        "ast_parser": {"language_pack": "tree-sitter-language-pack"},
        "repo_map": {
            "max_symbols": 100,
            "include_docstrings": True,
            "include_imports": False,
            "included_symbol_types": [
                "function",
                "method",
                "class",
                "interface",
                "type_alias",
            ],
        },
        "indexing": {
            "startup_enabled": True,
            "startup_parallel_workers": 4,
            "agent_edit_enabled": True,
            "user_edit_enabled": True,
            "debounce_window": 2.0,
            "max_cumulative_file_size_kb": 10000,
        },
        "call_graph": {
            "relationship_weights": {
                "python": {
                    "call": 1.5,
                    "type_reference": 1.2,
                    "inherits": 1.0,
                    "belongs_to": 0.2,
                    "contains": 0.0,
                },
                "default": {
                    "call": 1.5,
                    "type_reference": 1.2,
                    "inherits": 1.0,
                    "belongs_to": 0.2,
                    "contains": 0.0,
                },
            }
        },
    }

    repo_config_file = config_dir / "repository.yaml"
    with open(repo_config_file, "w") as f:
        yaml.dump(repo_config, f, default_flow_style=False)

    return repo_config


@pytest.fixture
def config_manager_with_repo(
    test_configs: dict[str, Any], repository_config: dict[str, Any]
) -> Iterator[ConfigManager]:
    """Create a ConfigManager with repository configuration."""
    env_manager = test_configs["env_manager"]

    # Create ConfigManager with env_manager (which contains path_manager)
    config_manager = ConfigManager(env_manager=env_manager)

    # Register capabilities and load global config
    config_manager.register_capability(["repository"])
    config_manager.load_global_config()

    yield config_manager


@pytest.fixture
def repository_service(
    test_project_structure: Path, config_manager_with_repo: ConfigManager
) -> Iterator[RepositoryContextService]:
    """Create a RepositoryContextService instance for testing."""
    service = RepositoryContextService(
        target_project_root=test_project_structure,
        config_manager=config_manager_with_repo,
    )

    yield service

    # Cleanup
    service.close()
