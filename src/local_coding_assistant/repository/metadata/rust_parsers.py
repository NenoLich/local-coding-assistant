"""Rust project metadata parsers."""

from pathlib import Path

import tomli

from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.rust_parsers")


def _extract_deps_from_section(section: dict) -> set[str]:
    """Extract dependency names from a Cargo.toml section.

    Handles both inline string versions and detailed dependency tables.
    e.g., serde = "1.0" OR serde = { version = "1.0", features = ["derive"] }

    Args:
        section: Cargo.toml dependency section.

    Returns:
        Set of dependency names.
    """
    deps = set()
    if not isinstance(section, dict):
        return deps

    for dep_name in section.keys():
        deps.add(dep_name)
    return deps


def parse_cargo_toml(file_path: Path) -> ProjectInfo:
    """Parse Cargo.toml file.

    Args:
        file_path: Path to Cargo.toml.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    info.package_manager = "cargo"
    try:
        with file_path.open("rb") as f:
            data = tomli.load(f)
    except Exception as e:
        log.debug(f"Failed to parse Cargo.toml: {e}")
        return info

    # Package Metadata
    if "package" in data:
        pkg = data["package"]
        info.name = pkg.get("name")
        info.description = pkg.get("description")
        info.language_version_edition = pkg.get("edition")
        version = pkg.get("version")
        if version:
            info.language_version = "rust " + version

    # Workspace Detection
    if "workspace" in data:
        info.is_workspace = True
        workspace = data["workspace"]
        if "members" in workspace and isinstance(workspace["members"], list):
            info.workspace_members = set(workspace["members"])

    # Dependencies
    deps_sections = [
        data.get("dependencies", {}),
        data.get("dev-dependencies", {}),
        data.get("build-dependencies", {}),
        data.get("workspace", {}).get("dependencies", {}),
    ]

    for section in deps_sections:
        info.clean_dependencies.update(_extract_deps_from_section(section))

    # Infer Frameworks & Paradigms
    FrameworkInference.infer_rust_frameworks(info)

    return info
