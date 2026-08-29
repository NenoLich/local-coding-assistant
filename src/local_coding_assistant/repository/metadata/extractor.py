"""Main metadata extractor that coordinates all parsers."""

import time
from pathlib import Path
from typing import ClassVar

from local_coding_assistant.repository.file_filter import FileFilter
from local_coding_assistant.repository.metadata.detector import MetadataFileDetector
from local_coding_assistant.repository.metadata.go_parsers import parse_go_mod
from local_coding_assistant.repository.metadata.java_parsers import (
    parse_build_gradle,
    parse_pom_xml,
)
from local_coding_assistant.repository.metadata.javascript_parsers import (
    parse_package_json,
    parse_tsconfig_json,
)
from local_coding_assistant.repository.metadata.python_parsers import (
    parse_pyproject_toml,
    parse_requirements_txt,
    parse_setup_py,
)
from local_coding_assistant.repository.metadata.rust_parsers import parse_cargo_toml
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.extractor")


class MetadataExtractor:
    """Extracts project metadata from configuration files."""

    # Parser mapping
    PARSERS: ClassVar = {
        "parse_pyproject_toml": parse_pyproject_toml,
        "parse_setup_py": parse_setup_py,
        "parse_requirements_txt": parse_requirements_txt,
        "parse_package_json": parse_package_json,
        "parse_tsconfig_json": parse_tsconfig_json,
        "parse_cargo_toml": parse_cargo_toml,
        "parse_go_mod": parse_go_mod,
        "parse_pom_xml": parse_pom_xml,
        "parse_build_gradle": parse_build_gradle,
    }

    def __init__(self, file_filter: FileFilter | None = None) -> None:
        """Initialize the metadata extractor.

        Args:
            file_filter: Optional file filter. If None, creates a new one.
        """
        self._file_filter = file_filter or FileFilter()
        self._parsed_config_files: set[str] = set()
        self._cached_info: ProjectInfo | None = None
        self._last_parse_time: float = 0

    def extract(
        self,
        tracked_files: set[Path],
        target_project_root: str | Path,
        force_refresh: bool = False,
    ) -> ProjectInfo:
        """Extract project metadata from configuration files.

        Args:
            tracked_files: Set of tracked files in the project.
            target_project_root: Path to the project root directory.
            force_refresh: Force re-parsing all files even if cache is valid.

        Returns:
            ProjectInfo object with extracted metadata.
        """
        current_time = time.time()
        target_project_root_path = self._normalize_project_root(target_project_root)
        _, filtered_tracked_files = self._file_filter.filter_files(
            list(tracked_files),
            apply_extension_filter=False,
            apply_size_filter=False,
            apply_test_file_filter=True,
        )

        config_files = MetadataFileDetector.find_config_files(
            filtered_tracked_files, target_project_root_path
        )
        self._invalidate_cache_for_missing_files(config_files)

        project_info = self._get_or_create_project_info(force_refresh)
        self._register_manifest_files(
            project_info, filtered_tracked_files, target_project_root_path
        )
        self._parse_config_files(
            project_info, config_files, target_project_root_path, force_refresh
        )

        self._cached_info = project_info
        self._last_parse_time = current_time
        return project_info

    def _normalize_project_root(self, target_project_root: str | Path) -> Path:
        """Normalize project root type to a pathlib Path."""
        return (
            Path(target_project_root)
            if isinstance(target_project_root, str)
            else target_project_root
        )

    def _invalidate_cache_for_missing_files(
        self, config_files: list[tuple[str, str]]
    ) -> None:
        """Clear cached metadata when tracked config files are no longer present."""
        for parsed_file in self._parsed_config_files:
            if not any(parsed_file == config_file for config_file, _ in config_files):
                self._cached_info = None
                self._parsed_config_files.clear()
                break

    def _get_or_create_project_info(self, force_refresh: bool) -> ProjectInfo:
        """Return cached project info or a fresh one."""
        if force_refresh or self._cached_info is None:
            return ProjectInfo()
        return self._cached_info

    def _register_manifest_files(
        self,
        project_info: ProjectInfo,
        filtered_tracked_files: list[Path],
        target_project_root_path: Path,
    ) -> None:
        """Register all discovered manifest files on the project info."""
        project_info.manifest_files.update(
            MetadataFileDetector.detect_manifest_files(
                filtered_tracked_files, target_project_root_path
            )
        )

    def _parse_config_files(
        self,
        project_info: ProjectInfo,
        config_files: list[tuple[str, str]],
        target_project_root_path: Path,
        force_refresh: bool,
    ) -> None:
        """Parse config files and merge their metadata into the project info."""
        for rel_path, parser_name in config_files:
            file_path = target_project_root_path / rel_path
            if not file_path.exists():
                continue

            project_info.manifest_files.add(rel_path)
            if not self._needs_reparse(file_path, force_refresh):
                continue

            try:
                parser = self.PARSERS.get(parser_name)
                if parser is None:
                    log.debug(f"Unknown parser: {parser_name}")
                    continue

                info = parser(file_path)
                self._merge_parser_info(project_info, info)
                self._parsed_config_files.add(rel_path)
            except Exception as e:
                log.debug(f"Failed to parse {rel_path}. {e}")

    def _needs_reparse(self, file_path: Path, force_refresh: bool) -> bool:
        """Decide if a config file needs to be reparsed."""
        if force_refresh:
            return True
        if self._cached_info is None:
            return True

        try:
            return file_path.stat().st_mtime > self._last_parse_time
        except (OSError, FileNotFoundError):
            return True

    def _merge_parser_info(self, project_info: ProjectInfo, info: ProjectInfo) -> None:
        """Merge parsed metadata while keeping the richer values."""
        for field_name, field_value in info.model_dump().items():
            current_value = getattr(project_info, field_name)
            if field_value is None:
                continue

            if isinstance(field_value, list) and isinstance(current_value, list):
                setattr(project_info, field_name, current_value + field_value)
            elif isinstance(field_value, set) and isinstance(current_value, set):
                setattr(project_info, field_name, current_value.union(field_value))
            elif current_value is None:
                setattr(project_info, field_name, field_value)
