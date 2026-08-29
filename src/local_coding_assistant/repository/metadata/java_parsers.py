"""Java project metadata parsers."""

import re
from pathlib import Path
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET  # noqa: N817

from local_coding_assistant.repository.metadata.common import strip_xml_namespace
from local_coding_assistant.repository.metadata.framework_inference import (
    FrameworkInference,
)
from local_coding_assistant.repository.models import ProjectInfo
from local_coding_assistant.utils.logging import get_logger

log = get_logger("repository.metadata.java_parsers")


def _apply_pom_artifact_id(info: ProjectInfo, elem: Element) -> None:
    """Apply artifactId metadata from a Maven element."""
    if not info.name:
        info.name = elem.text


def _apply_pom_description(info: ProjectInfo, elem: Element) -> None:
    """Apply description metadata from a Maven element."""
    if not info.description:
        info.description = elem.text


def _apply_pom_modules(info: ProjectInfo, elem: Element) -> None:
    """Apply workspace module detection from a Maven modules element."""
    info.is_workspace = True
    for module in elem:
        module_name = strip_xml_namespace(module.tag)
        if module_name == "module" and module.text:
            info.workspace_members.add(module.text)


def _apply_pom_java_version(info: ProjectInfo, elem: Element) -> None:
    """Apply the Java language version from a Maven property element."""
    if elem.text:
        info.language_version = "java " + elem.text


def _apply_pom_dependency(info: ProjectInfo, elem: Element) -> None:
    """Apply dependency extraction from a Maven dependency element."""
    artifact_id = None
    for child in elem:
        if strip_xml_namespace(child.tag) == "artifactId":
            artifact_id = child.text
            break
    if artifact_id:
        info.raw_dependencies.add(artifact_id)
        info.clean_dependencies.add(artifact_id.lower())


def _apply_pom_metadata(info: ProjectInfo, tag_name: str, elem: Element) -> None:
    """Update project metadata based on a parsed POM element."""
    if tag_name == "artifactId":
        _apply_pom_artifact_id(info, elem)
        return
    if tag_name == "description":
        _apply_pom_description(info, elem)
        return
    if tag_name == "modules":
        _apply_pom_modules(info, elem)
        return
    if tag_name in {"java.version", "maven.compiler.source"}:
        _apply_pom_java_version(info, elem)
        return
    if tag_name == "dependency":
        _apply_pom_dependency(info, elem)


def parse_pom_xml(file_path: Path) -> ProjectInfo:
    """Parse pom.xml file (basic).

    Args:
        file_path: Path to pom.xml.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    info.package_manager = "maven"
    info.build_backend = "maven"

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        log.debug(f"Failed to parse pom.xml: {e}")
        return info

    for elem in root.iter():
        _apply_pom_metadata(info, strip_xml_namespace(elem.tag), elem)

    FrameworkInference.infer_java_frameworks(info)
    return info


def parse_build_gradle(file_path: Path) -> ProjectInfo:
    """Parse build.gradle file (basic).

    Args:
        file_path: Path to build.gradle.

    Returns:
        ProjectInfo with extracted data.
    """
    info = ProjectInfo()
    info.package_manager = "gradle"
    info.build_backend = "gradle"

    java_version_pattern = re.compile(
        r'sourceCompatibility\s*=?\s*[\'"]?(?:JavaVersion\.VERSION_)?([\d\.]+)[\'"]?'
    )
    dep_pattern = re.compile(
        r'(?:implementation|api|compileOnly|runtimeOnly|testImplementation|testCompile)\s*\(?[\'"]([^\'":]+:[^\'":]+)'
    )

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.debug(f"Failed to read gradle file: {e}")
        return info

    for line in content.splitlines():
        line = line.strip()

        # Skip comments
        if line.startswith("//"):
            continue

        # Java Version
        java_match = java_version_pattern.search(line)
        if java_match and not info.language_version:
            version = java_match.group(1).replace("_", ".")
            if version:
                info.language_version = "java " + version

        # Dependencies
        dep_match = dep_pattern.search(line)
        if dep_match:
            full_artifact = dep_match.group(1)

            if ":" in full_artifact:
                artifact_name = full_artifact.split(":")[1]
                info.raw_dependencies.add(full_artifact)
                info.clean_dependencies.add(artifact_name.lower())

    # Frameworks
    FrameworkInference.infer_java_frameworks(info)

    return info
