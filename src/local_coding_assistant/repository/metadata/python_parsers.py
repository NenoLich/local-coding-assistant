"""Python project metadata parsers."""

import ast
from pathlib import Path

import tomli

from local_coding_assistant.repository.metadata.common import extract_base_package_name
from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.python_parsers")


def _apply_pyproject_project_metadata(data: dict, info: ProjectInfo) -> None:
    """Apply PEP 621 project metadata to ProjectInfo."""
    project = data.get("project")
    if not isinstance(project, dict):
        return

    info.name = project.get("name")
    info.description = project.get("description")
    version = project.get("requires-python")
    if version:
        info.language_version = "python " + str(version)

    dependencies = project.get("dependencies")
    if isinstance(dependencies, list):
        info.raw_dependencies.update(dependencies)

    optional_dependencies = project.get("optional-dependencies")
    if isinstance(optional_dependencies, dict):
        for deps in optional_dependencies.values():
            if isinstance(deps, list):
                info.raw_dependencies.update(deps)


def _apply_dependency_groups(data: dict, info: ProjectInfo) -> None:
    """Apply PEP 735 dependency groups to project metadata."""
    dependency_groups = data.get("dependency-groups")
    if not isinstance(dependency_groups, dict):
        return

    for group_deps in dependency_groups.values():
        if isinstance(group_deps, list):
            info.raw_dependencies.update(d for d in group_deps if isinstance(d, str))


def _apply_poetry_metadata(tool: dict, info: ProjectInfo) -> None:
    """Apply Poetry metadata and dependency configuration."""
    poetry = tool.get("poetry")
    if not isinstance(poetry, dict):
        return

    info.package_manager = "poetry"
    info.name = info.name or poetry.get("name")
    info.description = info.description or poetry.get("description")

    if "dependencies" in poetry and isinstance(poetry["dependencies"], dict):
        info.raw_dependencies.update(poetry["dependencies"].keys())
    if "dev-dependencies" in poetry and isinstance(poetry["dev-dependencies"], dict):
        info.raw_dependencies.update(poetry["dev-dependencies"].keys())
    if "group" in poetry and isinstance(poetry["group"], dict):
        for group_data in poetry["group"].values():
            if "dependencies" in group_data and isinstance(
                group_data["dependencies"], dict
            ):
                info.raw_dependencies.update(group_data["dependencies"].keys())


def _apply_build_system_metadata(data: dict, info: ProjectInfo) -> None:
    """Apply pyproject build-system metadata."""
    build_system = data.get("build-system")
    if not isinstance(build_system, dict):
        return

    build_backend = build_system.get("build-backend", "")
    info.build_backend = build_backend
    if info.package_manager:
        return

    if "hatchling" in build_backend:
        info.package_manager = "hatch"
    elif "flit" in build_backend:
        info.package_manager = "flit"
    elif "poetry" in build_backend:
        info.package_manager = "poetry"
    elif "setuptools" in build_backend:
        info.package_manager = "pip/setuptools"


def _apply_tool_metadata(tool: dict, info: ProjectInfo) -> None:
    """Apply project tool-specific metadata such as linters and test tooling."""
    if "black" in tool:
        info.linter_formatter.add("Black")
    if "ruff" in tool:
        info.linter_formatter.add("Ruff")
    if "isort" in tool:
        info.linter_formatter.add("isort")
    if "mypy" in tool:
        info.linter_formatter.add("MyPy")
    if "ty" in tool:
        info.linter_formatter.add("ty")
    if "pytest" in tool:
        info.test_frameworks.add("pytest")


def _apply_pyproject_scripts(data: dict, info: ProjectInfo) -> None:
    """Extract available script entries from the project metadata."""
    scripts_data = data.get("scripts")
    project_data = data.get("project")
    if isinstance(project_data, dict) and "scripts" in project_data:
        scripts_data = project_data.get("scripts")

    if isinstance(scripts_data, dict):
        info.available_scripts = list(scripts_data.keys())
    elif isinstance(scripts_data, list):
        info.available_scripts = scripts_data
    elif isinstance(scripts_data, str):
        info.available_scripts = [scripts_data]


def _apply_license_metadata(data: dict, info: ProjectInfo) -> None:
    """Extract licensing metadata when provided."""
    license_data = data.get("license")
    if isinstance(license_data, str):
        info.license = license_data
    elif isinstance(license_data, dict):
        if license_data.get("file"):
            info.license = license_data["file"]
        elif license_data.get("text"):
            info.license = license_data["text"]


def parse_pyproject_toml(file_path: Path) -> ProjectInfo:
    """Parse pyproject.toml file.

    Args:
        file_path: Path to pyproject.toml.

    Returns:
        ProjectInfo with extracted data.
    """
    with file_path.open("rb") as f:
        data = tomli.load(f)

    info = ProjectInfo()
    _apply_pyproject_project_metadata(data, info)
    _apply_dependency_groups(data, info)

    tool = data.get("tool", {})
    if "uv" in tool:
        info.package_manager = "uv"
    elif "rye" in tool:
        info.package_manager = "rye"
    elif "pdm" in tool:
        info.package_manager = "pdm"
    elif "poetry" in tool:
        _apply_poetry_metadata(tool, info)

    _apply_build_system_metadata(data, info)
    _apply_tool_metadata(tool, info)

    info.clean_dependencies = {
        extract_base_package_name(dep)
        for dep in info.raw_dependencies
        if dep and str(dep).lower() != "python"
    }

    FrameworkInference.infer_python_frameworks(info)
    FrameworkInference.infer_python_tooling(info)
    _apply_pyproject_scripts(data, info)
    _apply_license_metadata(data, info)

    return info


def _apply_setup_string_keyword(info: ProjectInfo, kw: ast.keyword) -> None:
    """Apply string-valued setup() metadata keywords."""
    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
        info.name = str(kw.value.value)
        return
    if kw.arg == "description" and isinstance(kw.value, ast.Constant):
        info.description = str(kw.value.value)
        return
    if kw.arg == "python_requires" and isinstance(kw.value, ast.Constant):
        info.language_version = "python " + str(kw.value.value)


def _apply_setup_dependency_keyword(info: ProjectInfo, kw: ast.keyword) -> None:
    """Apply dependency-related setup() metadata keywords."""
    if kw.arg == "install_requires" and isinstance(kw.value, (ast.List, ast.Tuple)):
        for elt in kw.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                info.raw_dependencies.add(elt.value)
        return
    if kw.arg == "extras_require" and isinstance(kw.value, ast.Dict):
        for val_node in kw.value.values:
            if isinstance(val_node, (ast.List, ast.Tuple)):
                for elt in val_node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        info.raw_dependencies.add(elt.value)


def _apply_setup_call_keyword(info: ProjectInfo, kw: ast.keyword) -> None:
    """Apply a keyword argument from a setup() call to project metadata."""
    _apply_setup_string_keyword(info, kw)
    _apply_setup_dependency_keyword(info, kw)


def parse_setup_py(file_path: Path) -> ProjectInfo:
    """Safely parse setup.py using AST.

    Never use eval() or exec() to read setup.py due to arbitrary code execution risks.

    Args:
        file_path: Path to setup.py.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo(package_manager="pip/setuptools", build_backend="setuptools")

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception as e:
        log.debug(f"Failed to parse setup.py AST: {e}")
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                _apply_setup_call_keyword(info, kw)

    info.clean_dependencies = {
        extract_base_package_name(dep)
        for dep in info.raw_dependencies
        if dep and str(dep).lower() != "python"
    }

    FrameworkInference.infer_python_frameworks(info)
    if "pytest" in info.clean_dependencies:
        info.test_frameworks.add("pytest")

    return info


def parse_requirements_txt(file_path: Path) -> ProjectInfo:
    """Parse requirements.txt file.

    Args:
        file_path: Path to requirements.txt.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    info.package_manager = "pip"

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.debug(f"Failed to read {file_path.name}: {e}")
        return info

    for line in content.splitlines():
        line = line.strip()
        # Skip empty lines, comments, and pip flags
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        info.raw_dependencies.add(line)

    # Clean dependencies
    info.clean_dependencies = {
        extract_base_package_name(dep)
        for dep in info.raw_dependencies
        if dep and str(dep).lower() != "python"
    }

    # Infer Frameworks
    FrameworkInference.infer_python_frameworks(info)

    if "pytest" in info.clean_dependencies:
        info.test_frameworks.add("pytest")

    return info
