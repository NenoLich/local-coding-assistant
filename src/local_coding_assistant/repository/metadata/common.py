"""Shared utilities for metadata extraction."""

import re


def extract_base_package_name(pep508_string: str) -> str:
    """Extract base package name from a PEP 508 dependency string.

    Extracts 'fastapi' from 'fastapi[all]>=0.100.0; python_version < "3.11"'.
    This is crucial so the AI doesn't hallucinate version numbers.

    Args:
        pep508_string: PEP 508 dependency specification string.

    Returns:
        Base package name in lowercase.
    """
    match = re.match(r"^([a-zA-Z0-9_.-]+)", pep508_string.strip())
    if match:
        return match.group(1).lower()
    return pep508_string.strip().lower()


def clean_json5(raw_json: str) -> str:
    """Remove comments and trailing commas from a JSON5-like string.

    Converts JSON5 to strict JSON so it can be parsed by Python's json module.

    Args:
        raw_json: Raw JSON5 string.

    Returns:
        Clean JSON string.
    """
    # Remove multi-line comments /* ... */
    clean_str = re.sub(r"/\*.*?\*/", "", raw_json, flags=re.DOTALL)

    # Remove single-line comments // ...
    clean_str = re.sub(r"//.*", "", clean_str)

    # Remove trailing commas (e.g., "key": "value", } -> "key": "value" })
    clean_str = re.sub(r",\s*([\]}])", r"\1", clean_str)

    return clean_str


def strip_xml_namespace(tag: str) -> str:
    """Strip XML namespace from a tag.

    Maven heavily uses XML namespaces like {http://maven.apache.org/POM/4.0.0}project.
    This strips it out so we can just look for 'project' or 'dependency'.

    Args:
        tag: XML tag with potential namespace.

    Returns:
        Tag name without namespace.
    """
    return tag.split("}")[-1] if "}" in tag else tag
