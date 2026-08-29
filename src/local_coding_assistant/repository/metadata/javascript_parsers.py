"""JavaScript/TypeScript project metadata parsers."""

import json
from pathlib import Path

from local_coding_assistant.repository.metadata.common import clean_json5
from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.javascript_parsers")


def parse_package_json(file_path: Path) -> ProjectInfo:
    """Parse package.json file.

    Args:
        file_path: Path to package.json.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    try:
        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.debug(f"Failed to parse package.json: {e}")
        return info

    # Project Metadata
    info.name = data.get("name")
    info.description = data.get("description")

    # Extract Node engine requirements
    if "engines" in data and isinstance(data["engines"], dict):
        version = data["engines"].get("node")
        if version:
            info.language_version = "JS " + version

    # Package Manager Detection (Corepack Standard)
    pkg_manager_string = data.get("packageManager", "")
    if pkg_manager_string:
        info.package_manager = pkg_manager_string.split("@")[0]
    else:
        info.package_manager = "npm"

    # Available Scripts
    if "scripts" in data and isinstance(data["scripts"], dict):
        info.available_scripts = list(data["scripts"].keys())

    # Dependencies
    deps = data.get("dependencies", {})
    dev_deps = data.get("devDependencies", {})

    if not isinstance(deps, dict):
        deps = {}
    if not isinstance(dev_deps, dict):
        dev_deps = {}

    # In JS, the dict keys are already perfectly clean package names
    info.clean_dependencies = set(deps.keys()).union(set(dev_deps.keys()))

    # Detect Frameworks & Tools
    FrameworkInference.infer_javascript_frameworks(info)
    FrameworkInference.infer_javascript_tooling(info)

    return info


def parse_tsconfig_json(file_path: Path) -> ProjectInfo:
    """Parse tsconfig.json file.

    Args:
        file_path: Path to tsconfig.json.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()

    try:
        raw_content = file_path.read_text(encoding="utf-8")
        clean_content = clean_json5(raw_content)
        data = json.loads(clean_content)
    except Exception as e:
        log.debug(f"Failed to parse tsconfig.json: {e}")
        return info

    compiler_options = data.get("compilerOptions", {})

    # Compilation Targets & Modules
    info.ts_config_target = compiler_options.get("target")
    info.ts_config_module = compiler_options.get("module")
    info.ts_config_module_resolution = compiler_options.get("moduleResolution")
    info.ts_config_jsx_mode = compiler_options.get("jsx")

    # Strictness
    if compiler_options.get("strict") is True:
        info.ts_config_strict_mode = True
    else:
        info.ts_config_strict_mode = False

    # Path Aliases
    paths = compiler_options.get("paths", {})
    if paths:
        info.ts_config_path_aliases = list(paths.keys())

    return info
