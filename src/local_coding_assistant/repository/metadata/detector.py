"""File detection for metadata extraction."""

import fnmatch
from pathlib import Path
from typing import ClassVar

from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.detector")


class MetadataFileDetector:
    """Detects and locates configuration and manifest files in a project."""

    # Config file patterns to parse (with subdirectory support)
    CONFIG_FILE_PATTERNS: ClassVar[list[str]] = [
        "pyproject.toml",
        "setup.py",
        "requirements*.txt",
        "package.json",
        "tsconfig.json",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
    ]

    # Manifest files to detect (not parsed, just tracked)
    MANIFEST_FILES: ClassVar[list[str]] = [
        "Makefile",
        "makefile",
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".dockerignore",
        "Dockerfile.dev",
        "Dockerfile.prod",
        ".gitignore",
        ".gitattributes",
        ".env.example",
        ".env.template",
        "Procfile",
        "vercel.json",
        "netlify.toml",
        ".nvmrc",
        ".python-version",
        ".ruby-version",
        ".node-version",
        "go.sum",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "tox.ini",
        "setup.cfg",
        "MANIFEST.in",
        "Jenkinsfile",
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        ".gitlab-ci.yml",
        "azure-pipelines.yml",
        "cloudbuild.yaml",
    ]

    # Map patterns to parser function names
    PATTERN_TO_PARSER: ClassVar[dict[str, str]] = {
        "pyproject.toml": "parse_pyproject_toml",
        "setup.py": "parse_setup_py",
        "requirements*.txt": "parse_requirements_txt",
        "package.json": "parse_package_json",
        "tsconfig.json": "parse_tsconfig_json",
        "Cargo.toml": "parse_cargo_toml",
        "go.mod": "parse_go_mod",
        "pom.xml": "parse_pom_xml",
        "build.gradle": "parse_build_gradle",
    }

    @staticmethod
    def detect_manifest_files(
        tracked_files: list[Path], target_project_root: str | Path
    ) -> set[str]:
        """Detect manifest files in the project.

        Args:
            tracked_files: List of tracked files in the project.
            target_project_root: Root directory of the project.

        Returns:
            Set of manifest file paths (relative to project root).
        """
        manifest_files = set()

        for tracked_file in tracked_files:
            rel_path = (
                tracked_file.relative_to(target_project_root)
                if tracked_file.is_absolute()
                else tracked_file
            )
            file_name = rel_path.name
            file_str = str(rel_path)

            # Check exact matches
            if file_name in MetadataFileDetector.MANIFEST_FILES:
                manifest_files.add(file_str)
            # Check pattern matches (e.g., .github/workflows/*.yml)
            else:
                for pattern in MetadataFileDetector.MANIFEST_FILES:
                    if "*" in pattern:
                        if fnmatch.fnmatch(file_str, pattern):
                            manifest_files.add(file_str)

        return manifest_files

    @staticmethod
    def find_config_files(
        tracked_files: list[Path], target_project_root: str | Path
    ) -> list[tuple[str, str]]:
        """Find config files in the project using patterns.

        Args:
            tracked_files: List of tracked files in the project.
            target_project_root: Root directory of the project.

        Returns:
            List of (relative_path, parser_name) tuples.
        """
        config_files = []

        for tracked_file in tracked_files:
            rel_path = (
                tracked_file.relative_to(target_project_root)
                if tracked_file.is_absolute()
                else tracked_file
            )
            file_str = str(rel_path)
            file_name = rel_path.name

            for pattern, parser_name in MetadataFileDetector.PATTERN_TO_PARSER.items():
                if "*" in pattern:
                    if fnmatch.fnmatch(file_str, pattern):
                        config_files.append((file_str, parser_name))
                        break
                else:
                    if fnmatch.fnmatch(file_name, pattern):
                        config_files.append((file_str, parser_name))
                        break

        return config_files
