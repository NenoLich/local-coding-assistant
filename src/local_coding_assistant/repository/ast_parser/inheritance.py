"""Inheritance extraction from AST nodes for different languages."""

from typing import Any

from .common import JS_TS_LANGUAGES, node_text


class InheritanceExtractor:
    """Extracts inheritance relationships from AST nodes for different languages."""

    @classmethod
    def extract_python_inheritance(cls, node: Any, class_name: str) -> list[str]:
        """Extract base classes from Python class definition.

        Args:
            node: Tree-sitter class_definition node.
            class_name: Name of the class being defined.

        Returns:
            List of base class names.
        """
        if node is None:
            return []

        base_classes: list[str] = []

        # Check for superclasses (inheritance)
        argument_list = node.child_by_field_name("superclasses")
        if argument_list:
            for child in argument_list.children:
                # Plain identifiers, `type` nodes, and dotted names like 'module.Class'
                if child.type in ("identifier", "type", "attribute"):
                    base_classes.append(node_text(child))

        return base_classes

    @classmethod
    def extract_js_ts_inheritance(cls, node: Any, class_name: str) -> list[str]:
        """Extract base classes/interfaces from JavaScript/TypeScript class declaration.

        Args:
            node: Tree-sitter class_declaration node.
            class_name: Name of the class being defined.

        Returns:
            List of base class/interface names.
        """
        if node is None:
            return []

        base_classes: list[str] = []

        # Check for heritage clause (extends/implements)
        heritage_clause = node.child_by_field_name("heritage")
        if heritage_clause:
            for child in heritage_clause.children:
                # Plain identifiers and dotted names like 'module.Class'
                if child.type in (
                    "identifier",
                    "property_identifier",
                    "type_identifier",
                    "member_expression",
                ):
                    base_classes.append(node_text(child))

        return base_classes

    @classmethod
    def _rust_impl_traits(cls, node: Any) -> list[str]:
        """Extract traits implemented by an impl block."""
        trait_type = node.child_by_field_name("type")
        if not trait_type:
            return []

        trait_text = node_text(trait_type)
        # Format: 'Trait for Struct'
        if " for " in trait_text:
            return [trait_text.split(" for ")[0].strip()]
        return [trait_text]

    @classmethod
    def _rust_struct_trait_bounds(cls, node: Any) -> list[str]:
        """Extract trait bounds from a struct's generic parameters."""
        traits: list[str] = []
        type_parameters = node.child_by_field_name("type_parameters")
        if not type_parameters:
            return traits

        for child in type_parameters.children:
            if child.type == "bounded_type":
                for bound_child in child.children:
                    if bound_child.type == "type_identifier":
                        traits.append(node_text(bound_child))
        return traits

    @classmethod
    def extract_rust_inheritance(cls, node: Any, struct_name: str) -> list[str]:
        """Extract traits from Rust impl block or struct definition.

        Args:
            node: Tree-sitter impl_item or struct_item node.
            struct_name: Name of the struct being defined or implemented.

        Returns:
            List of trait names being implemented.
        """
        if node is None:
            return []

        if node.type == "impl_item":
            return cls._rust_impl_traits(node)
        if node.type == "struct_item":
            return cls._rust_struct_trait_bounds(node)
        return []

    @classmethod
    def extract_go_inheritance(cls, node: Any, struct_name: str) -> list[str]:
        """Extract embedded interfaces from Go struct declaration.

        Args:
            node: Tree-sitter type_declaration node.
            struct_name: Name of the struct being defined.

        Returns:
            List of embedded interface names.
        """
        if node is None:
            return []

        embedded: list[str] = []

        # Go uses embedding for inheritance-like behavior: fields with a type
        # but no name are embedded interfaces.
        for child in node.children:
            if child.type == "field_declaration":
                type_node = child.child_by_field_name("type")
                if type_node and not child.child_by_field_name("name"):
                    embedded.append(node_text(type_node))

        return embedded

    @classmethod
    def extract_inheritance(
        cls, node: Any, class_name: str, language: str
    ) -> list[str]:
        """Extract inheritance relationships based on language.

        Args:
            node: Tree-sitter node representing a class/struct definition.
            class_name: Name of the class/struct being defined.
            language: Programming language name.

        Returns:
            List of base class/trait/interface names.
        """
        lang_lower = language.lower()

        if lang_lower == "python":
            return cls.extract_python_inheritance(node, class_name)
        if lang_lower in JS_TS_LANGUAGES:
            return cls.extract_js_ts_inheritance(node, class_name)
        if lang_lower == "rust":
            return cls.extract_rust_inheritance(node, class_name)
        if lang_lower == "go":
            return cls.extract_go_inheritance(node, class_name)
        return []
