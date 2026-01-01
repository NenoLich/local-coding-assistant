"""Unit tests for PathManager functionality."""

from __future__ import annotations

import site
from pathlib import Path

from pytest import MonkeyPatch

from local_coding_assistant.config.path_manager import PathManager


class TestPathManager:
    """Tests for PathManager behavior across environments."""

    def test_testing_environment_directories(self, tmp_path: Path) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        assert manager.get_project_root() == tmp_path
        assert manager.get_config_dir() == tmp_path / "tests" / "configs"
        assert manager.get_data_dir() == tmp_path / "tests" / "data"
        assert manager.get_cache_dir() == tmp_path / "tests" / "cache"
        assert manager.get_log_dir() == tmp_path / "tests" / "logs"
        assert manager.get_module_dir() == tmp_path / "tests" / "modules"

        for directory in (
            manager.get_config_dir(),
            manager.get_data_dir(),
            manager.get_cache_dir(),
            manager.get_log_dir(),
            manager.get_module_dir(),
        ):
            assert directory.is_dir()

    def test_development_environment_directories(self, tmp_path: Path) -> None:
        manager = PathManager(is_development=True, project_root=tmp_path)

        assert manager.get_project_root() == tmp_path
        assert manager.get_config_dir() == tmp_path / "config"
        assert manager.get_data_dir() == tmp_path / "data"
        assert manager.get_cache_dir() == tmp_path / ".cache"
        assert manager.get_log_dir() == tmp_path / "logs"
        assert manager.get_module_dir() == tmp_path / "config" / "modules"

        for directory in (
            manager.get_config_dir(),
            manager.get_data_dir(),
            manager.get_cache_dir(),
            manager.get_log_dir(),
            manager.get_module_dir(),
        ):
            assert directory.is_dir()

    def test_production_environment_directories(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            PathManager, "_get_system_config_dir", lambda self: tmp_path / "config"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_data_dir", lambda self: tmp_path / "data"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_cache_dir", lambda self: tmp_path / "cache"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_log_dir", lambda self: tmp_path / "logs"
        )

        manager = PathManager(project_root=tmp_path)

        assert manager.is_production is True
        assert manager.get_config_dir() == tmp_path / "config"
        assert manager.get_data_dir() == tmp_path / "data" / "local-coding-assistant"
        assert manager.get_cache_dir() == tmp_path / "cache" / "local-coding-assistant"
        assert manager.get_log_dir() == tmp_path / "logs" / "local-coding-assistant"

        for directory in (
            manager.get_config_dir(),
            manager.get_data_dir(),
            manager.get_cache_dir(),
            manager.get_log_dir(),
        ):
            assert directory.is_dir()

    def test_resolve_path_with_absolute_and_relative_inputs(
        self, tmp_path: Path
    ) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        absolute = tmp_path / "absolute.txt"
        assert manager.resolve_path(absolute) == absolute

        relative = manager.resolve_path("relative/file.yaml")
        assert relative == tmp_path / "relative" / "file.yaml"

    def test_resolve_path_with_base_dir(self, tmp_path: Path) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        base = tmp_path / "custom"
        result = manager.resolve_path("config.json", base_dir=base)
        assert result == base / "config.json"

    def test_resolve_path_with_special_prefixes(self, tmp_path: Path) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        cases = {
            "@config/settings.yaml": manager.get_config_dir() / "settings.yaml",
            "@data/dataset.json": manager.get_data_dir() / "dataset.json",
            "@cache/cache.db": manager.get_cache_dir() / "cache.db",
            "@log/session.log": manager.get_log_dir() / "session.log",
            "@module/plugin.py": manager.get_module_dir() / "plugin.py",
            "@project/docs/readme.md": tmp_path / "docs" / "readme.md",
        }

        for raw, expected in cases.items():
            resolved = manager.resolve_path(raw)
            assert resolved == expected

    def test_resolve_path_unknown_prefix_falls_back_to_project_root(
        self, tmp_path: Path
    ) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        unknown = manager.resolve_path("@unknown/file.txt")
        assert unknown == tmp_path / "@unknown" / "file.txt"

    def test_resolve_path_invalid_marker_format(self, tmp_path: Path) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        invalid = manager.resolve_path("@invalidprefix")
        assert invalid == tmp_path / "@invalidprefix"

    def test_resolve_path_ensure_parent_creates_directory(self, tmp_path: Path) -> None:
        manager = PathManager(is_testing=True, project_root=tmp_path)

        target = manager.resolve_path("logs/output/app.log", ensure_parent=True)
        assert target == tmp_path / "logs" / "output" / "app.log"
        assert target.parent.is_dir()

    def test_module_dir_production_site_packages(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            PathManager, "_get_system_config_dir", lambda self: tmp_path / "config"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_data_dir", lambda self: tmp_path / "data"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_cache_dir", lambda self: tmp_path / "cache"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_log_dir", lambda self: tmp_path / "logs"
        )
        monkeypatch.setattr(
            site, "getsitepackages", lambda: [str(tmp_path / "site-packages")]
        )

        manager = PathManager(project_root=tmp_path)
        module_dir = manager.get_module_dir()

        assert module_dir == (
            tmp_path / "site-packages" / "local_coding_assistant" / "modules"
        )

    def test_module_dir_production_fallback(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            PathManager, "_get_system_config_dir", lambda self: tmp_path / "config"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_data_dir", lambda self: tmp_path / "data"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_cache_dir", lambda self: tmp_path / "cache"
        )
        monkeypatch.setattr(
            PathManager, "_get_system_log_dir", lambda self: tmp_path / "logs"
        )
        monkeypatch.setattr(site, "getsitepackages", lambda: [])

        manager = PathManager(project_root=tmp_path)
        module_dir = manager.get_module_dir()

        assert (
            module_dir
            == Path("lib") / "site-packages" / "local_coding_assistant" / "modules"
        )
