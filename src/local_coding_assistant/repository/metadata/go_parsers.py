"""Go project metadata parsers."""

from pathlib import Path

from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.go_parsers")


def _add_dependency(info: ProjectInfo, dependency_name: str) -> None:
    """Add a Go dependency to the raw and clean dependency sets."""
    if not dependency_name:
        return
    info.raw_dependencies.add(dependency_name)
    info.clean_dependencies.add(dependency_name)


def _apply_go_mod_metadata(info: ProjectInfo, line: str) -> bool:
    """Apply module metadata directives such as module name, go version, and workspace markers."""
    if line.startswith("module "):
        info.name = line.split(" ", 1)[1].strip()
        return True
    if line.startswith("go "):
        version = line.split(" ", 1)[1].strip()
        if version:
            info.language_version = "go " + version
        return True
    if line.startswith("use "):
        info.is_workspace = True
        return True
    return False


def _apply_go_mod_line(info: ProjectInfo, line: str, in_require_block: bool) -> bool:
    """Apply a single Go module line to the ProjectInfo."""
    if not line or line.startswith("//"):
        return in_require_block
    if _apply_go_mod_metadata(info, line):
        return in_require_block
    if line.startswith("require ("):
        return True
    if line == ")":
        return False

    if line.startswith("require "):
        parts = line.split()
        if len(parts) >= 2:
            _add_dependency(info, parts[1])
        return in_require_block
    if in_require_block:
        parts = line.split()
        if parts:
            _add_dependency(info, parts[0])
    return in_require_block


def parse_go_mod(file_path: Path) -> ProjectInfo:
    """Parse go.mod file.

    Args:
        file_path: Path to go.mod.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    info.package_manager = "go modules"
    info.build_backend = "go build"

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.debug(f"Failed to parse go.mod: {e}")
        return info

    in_require_block = False
    for line in content.splitlines():
        in_require_block = _apply_go_mod_line(info, line.strip(), in_require_block)

    FrameworkInference.infer_go_frameworks(info)
    return info
