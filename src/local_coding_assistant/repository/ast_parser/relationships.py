"""Caller-callee relationship extraction from AST nodes."""

from typing import Any, ClassVar

from local_coding_assistant.repository.models import CallRelationship

from .common import is_pruned_test_module, node_text, start_line
from .inheritance import InheritanceExtractor
from .types import TypeExtractor

_SCOPE_NODE_TYPES = frozenset(
    {
        "class_definition",
        "class_declaration",
        "impl_item",
        "function_definition",
        "function_declaration",
        "method_definition",
        "function_item",
    }
)
_INHERITANCE_NODE_TYPES = frozenset(
    {"class_definition", "class_declaration", "struct_item", "impl_item"}
)
_MEMBER_EXPRESSION_NODES = frozenset(
    {"attribute", "member_expression", "field_expression"}
)


class CallRelationshipExtractor:
    """Extracts caller-callee relationships from AST nodes."""

    PYTHON_TYPE_NODES: ClassVar[set[str]] = {"type"}  # Python type hints
    TS_TYPE_NODES: ClassVar[set[str]] = {"type_annotation"}
    RUST_TYPE_NODES: ClassVar[set[str]] = {"type_identifier", "scoped_type_identifier"}
    GO_TYPE_NODES: ClassVar[set[str]] = {"type_identifier"}

    LANGUAGE_TYPE_MAPPINGS: ClassVar[dict[str, set[str]]] = {
        "python": PYTHON_TYPE_NODES,
        "typescript": TS_TYPE_NODES,
        "rust": RUST_TYPE_NODES,
        "go": GO_TYPE_NODES,
    }

    # Language-specific call expression node types
    PYTHON_CALL_TYPES: ClassVar[set[str]] = {"call"}
    JAVASCRIPT_CALL_TYPES: ClassVar[set[str]] = {"call_expression"}
    RUST_CALL_TYPES: ClassVar[set[str]] = {"call_expression"}
    GO_CALL_TYPES: ClassVar[set[str]] = {"call_expression"}

    LANGUAGE_CALL_MAPPINGS: ClassVar[dict[str, set[str]]] = {
        "python": PYTHON_CALL_TYPES,
        "javascript": JAVASCRIPT_CALL_TYPES,
        "typescript": JAVASCRIPT_CALL_TYPES,
        "tsx": JAVASCRIPT_CALL_TYPES,
        "jsx": JAVASCRIPT_CALL_TYPES,
        "rust": RUST_CALL_TYPES,
        "go": GO_CALL_TYPES,
    }

    @classmethod
    def _scope_name(cls, node: Any, node_type: str) -> str | None:
        """Resolve the name of a scope node using the field, impl_item, or child scan."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return node_text(name_node)

        if node_type == "impl_item":
            # Rust stores the struct being implemented in the 'type' field
            type_node = node.child_by_field_name("type")
            if type_node:
                # If it has generics (e.g., GenerationHandler<'a>), dig one level deeper
                if type_node.type == "generic_type":
                    inner_node = type_node.child_by_field_name("type")
                    if inner_node:
                        return node_text(inner_node)
                    return None
                return node_text(type_node)

        # Fallback to scanning children (for edge cases)
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "type_annotation"):
                return node_text(child)
        return None

    @classmethod
    def _enter_scope(
        cls,
        relationships: list[CallRelationship],
        scope_stack: list[tuple[str, int]],
        node: Any,
        name: str,
        node_type: str,
        language: str,
        file_path: str,
    ) -> None:
        """Push a scope and emit belongs_to/inherits edges for the new scope."""
        current_scope = (
            ".".join(scope_name for scope_name, _ in scope_stack)
            if scope_stack
            else None
        )
        new_scope = f"{current_scope}.{name}" if current_scope else name
        line = start_line(node)
        scope_stack.append((name, line))

        if current_scope:
            # Create organic hierarchy edges
            relationships.append(
                CallRelationship(
                    caller_name=new_scope,
                    caller_line=line,
                    callee_name=current_scope,
                    callee_line=line,
                    relationship_type="belongs_to",
                    language=language,
                    file_path=file_path,
                )
            )

        if node_type in _INHERITANCE_NODE_TYPES:
            for base_class in InheritanceExtractor.extract_inheritance(
                node, name, language
            ):
                relationships.append(
                    CallRelationship(
                        caller_name=new_scope,
                        caller_line=line,
                        callee_name=base_class,
                        callee_line=line,
                        relationship_type="inherits",
                        language=language,
                        file_path=file_path,
                    )
                )

    @classmethod
    def _resolve_callee(cls, node: Any) -> tuple[str | None, str | None]:
        """Return (callee_name, object_name) for the first relevant child of a call node."""
        for child in node.children:
            if child.type in ("identifier", "name"):
                return node_text(child), None
            if child.type in _MEMBER_EXPRESSION_NODES:
                method_node = (
                    child.child_by_field_name("attribute")
                    or child.child_by_field_name("property")
                    or child.child_by_field_name("field")
                )
                if method_node:
                    callee = node_text(method_node)
                else:
                    callee = None
                    # Fallback: scan backwards for the right-most identifier
                    for grandchild in reversed(child.children):
                        if grandchild.type in (
                            "identifier",
                            "property_identifier",
                            "field_identifier",
                        ):
                            callee = node_text(grandchild)
                            break
                value_node = child.child_by_field_name(
                    "value"
                ) or child.child_by_field_name("object")
                obj = node_text(value_node) if value_node else None
                return callee, obj
        return None, None

    @classmethod
    def _resolve_method_target(
        cls, callee: str | None, obj: str | None, scope_stack: list[tuple[str, int]]
    ) -> str | None:
        """Resolve self/this method calls to the enclosing class's fully qualified name."""
        if callee and obj in ("self", "this") and scope_stack:
            class_name = scope_stack[0][0]  # The root class in the current scope
            return f"{class_name}.{callee}"
        return callee

    @classmethod
    def _type_reference_edges(
        cls, node: Any, caller: str, caller_line: int, language: str, file_path: str
    ) -> list[CallRelationship]:
        """Build type_reference edges for the type-annotation nodes in the call tree."""
        edges: list[CallRelationship] = []
        for type_name in TypeExtractor.extract_types(node_text(node), language):
            edges.append(
                CallRelationship(
                    caller_name=caller,
                    caller_line=caller_line,
                    callee_name=type_name,
                    callee_line=start_line(node),
                    relationship_type="type_reference",
                    language=language,
                    file_path=file_path,
                )
            )
        return edges

    @classmethod
    def extract_call_relationships(
        cls, root_node: Any, language: str, file_path: str
    ) -> list[CallRelationship]:
        """Extract caller-callee relationships from the AST.

        Args:
            root_node: Tree-sitter root node.
            language: Programming language name.
            file_path: Path to the file being parsed.

        Returns:
            List of CallRelationship objects.
        """
        relationships: list[CallRelationship] = []
        call_types = cls.LANGUAGE_CALL_MAPPINGS.get(language.lower(), set())
        type_nodes = cls.LANGUAGE_TYPE_MAPPINGS.get(language.lower(), set())
        builtin_callees = TypeExtractor.LANGUAGE_BUILTIN_CALLEES.get(
            language.lower(), set()
        )
        # Use a stack to track nested classes and functions (e.g., ["BaseDriver", "generate"])
        scope_stack: list[tuple[str, int]] = []

        def traverse(node: Any) -> None:
            if node is None or is_pruned_test_module(node):
                return

            node_type = node.type
            is_scope = node_type in _SCOPE_NODE_TYPES

            # 1. Enter Scope (Class or Function)
            if is_scope:
                name = cls._scope_name(node, node_type)
                if name:
                    cls._enter_scope(
                        relationships,
                        scope_stack,
                        node,
                        name,
                        node_type,
                        language,
                        file_path,
                    )
                else:
                    scope_stack.append(("<anonymous>", start_line(node)))

            caller = (
                ".".join(scope_name for scope_name, _ in scope_stack)
                if scope_stack
                else None
            )
            caller_line = scope_stack[-1][1] if scope_stack else 0

            # 2. Detect Calls (Execution Edges)
            if node_type in call_types and caller:
                callee, obj = cls._resolve_callee(node)
                callee = cls._resolve_method_target(callee, obj, scope_stack)
                if callee and callee not in builtin_callees:
                    relationships.append(
                        CallRelationship(
                            caller_name=caller,
                            caller_line=caller_line,
                            callee_name=callee,
                            callee_line=start_line(node),
                            relationship_type="call",
                            language=language,
                            file_path=file_path,
                        )
                    )

            # 3. Detect Type References (Interfaces, Traits, Models)
            if node_type in type_nodes and caller:
                relationships.extend(
                    cls._type_reference_edges(
                        node, caller, caller_line, language, file_path
                    )
                )

            # 4. Recurse down the tree
            for child in node.children:
                traverse(child)

            # 5. Exit Scope
            if is_scope:
                scope_stack.pop()

        traverse(root_node)
        return relationships
