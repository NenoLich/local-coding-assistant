"""Shared helpers for the AST extraction modules."""

from typing import Any

JS_TS_LANGUAGES = frozenset({"javascript", "typescript", "jsx", "tsx"})
TEST_MODULE_NAMES = frozenset({"tests", "test"})


def node_text(node: Any) -> str:
    """Decode a tree-sitter node's text payload to a string."""
    return node.text.decode("utf-8")


def start_line(node: Any) -> int:
    """Return the 1-based line number where ``node`` starts."""
    return node.start_point[0] + 1


def is_pruned_test_module(node: Any) -> bool:
    """Return True for Rust ``mod tests`` / ``mod test`` blocks, which are skipped."""
    if node.type != "mod_item":
        return False
    name_node = node.child_by_field_name("name")
    return bool(name_node) and node_text(name_node) in TEST_MODULE_NAMES
