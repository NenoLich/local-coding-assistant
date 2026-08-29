"""Symbol extraction from AST nodes."""

from typing import Any, ClassVar

from local_coding_assistant.repository.models import ASTNode, SymbolType

from .common import JS_TS_LANGUAGES, is_pruned_test_module, node_text

_IDENTIFIER_NODE_TYPES = frozenset(
    {
        "identifier",
        "name",
        "property_identifier",
        "field_identifier",
        "type_identifier",
    }
)
_VARIABLE_NODE_TYPES = frozenset(
    {"lexical_declaration", "variable_declaration", "variable_declarator"}
)
_BLOCK_NODE_TYPES = frozenset(
    {"block", "statement_block", "compound_statement", "function_body", "{"}
)
# Container nodes we must not descend into when resolving an identifier name
_SKIP_CONTAINERS = frozenset(
    {
        "block",
        "statement_block",
        "arrow_function",
        "function_declaration",
        "function_item",
        "function_signature_item",
        "method_signature",
        "method_spec",
        "method_definition",
        "method_elem",
        "interface_body",
        "declaration_list",
        "object_type",
    }
)


def _find_identifier_text(node: Any, is_root: bool = True) -> str | None:
    """Depth-first search for the first identifier-like child of ``node``."""
    if node is None:
        return None

    if node.type in _IDENTIFIER_NODE_TYPES:
        return node_text(node)

    # "trait_item" is a container node (like a class or interface body); we must
    # allow the search to look past it for the trait name, but still block
    # descent into true child method/function definition blocks.
    if not is_root and node.type in _SKIP_CONTAINERS:
        return None

    for child in node.children:
        text = _find_identifier_text(child, is_root=False)
        if text:
            return text
    return None


class SymbolExtractor:
    """Extracts symbols from AST nodes."""

    # Node types that should have signatures extracted
    PYTHON_SIGNATURE_TYPES: ClassVar[set[str]] = {
        "function_definition",
        "method_definition",
        "class_definition",
        "type_alias",
    }
    JAVASCRIPT_SIGNATURE_TYPES: ClassVar[set[str]] = {
        "function_declaration",
        "arrow_function",
        "method_definition",
        "method_signature",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
    }
    RUST_SIGNATURE_TYPES: ClassVar[set[str]] = {
        "function_item",
        "function_signature_item",
        "struct_item",
        "trait_item",
        "type_alias",
    }
    GO_SIGNATURE_TYPES: ClassVar[set[str]] = {
        "function_declaration",
        "method_declaration",
        "method_spec",
        "method_elem",
        "type_declaration",
    }

    # Language-specific node type mappings
    PYTHON_SYMBOL_TYPES: ClassVar[dict[str, SymbolType]] = {
        "function_definition": SymbolType.FUNCTION,
        "class_definition": SymbolType.CLASS,
        "method_definition": SymbolType.METHOD,
        "assignment": SymbolType.VARIABLE,
        "parameter": SymbolType.PARAMETER,
        "import_statement": SymbolType.IMPORT,
        "import_from_statement": SymbolType.IMPORT,
        "type_alias": SymbolType.TYPE_ALIAS,
    }

    JAVASCRIPT_SYMBOL_TYPES: ClassVar[dict[str, SymbolType]] = {
        "function_declaration": SymbolType.FUNCTION,
        "function_expression": SymbolType.FUNCTION,
        "arrow_function": SymbolType.FUNCTION,
        "class_declaration": SymbolType.CLASS,
        "method_definition": SymbolType.METHOD,
        "method_signature": SymbolType.METHOD,
        "variable_declaration": SymbolType.VARIABLE,
        "import_statement": SymbolType.IMPORT,
        "interface_declaration": SymbolType.INTERFACE,
        "type_alias_declaration": SymbolType.TYPE_ALIAS,
    }

    RUST_SYMBOL_TYPES: ClassVar[dict[str, SymbolType]] = {
        "function_item": SymbolType.FUNCTION,
        "function_signature_item": SymbolType.FUNCTION,
        "struct_item": SymbolType.CLASS,
        "trait_item": SymbolType.INTERFACE,
        "enum_item": SymbolType.CLASS,
        "impl_item": SymbolType.CLASS,
        "let_declaration": SymbolType.VARIABLE,
        "use_declaration": SymbolType.IMPORT,
        "type_alias": SymbolType.TYPE_ALIAS,
    }

    GO_SYMBOL_TYPES: ClassVar[dict[str, SymbolType]] = {
        "function_declaration": SymbolType.FUNCTION,
        "type_declaration": SymbolType.INTERFACE,
        "method_declaration": SymbolType.METHOD,
        "method_spec": SymbolType.METHOD,
        "method_elem": SymbolType.METHOD,
        "var_declaration": SymbolType.VARIABLE,
        "import_declaration": SymbolType.IMPORT,
    }

    LANGUAGE_MAPPINGS: ClassVar[dict[str, dict[str, SymbolType]]] = {
        "python": PYTHON_SYMBOL_TYPES,
        "javascript": JAVASCRIPT_SYMBOL_TYPES,
        "typescript": JAVASCRIPT_SYMBOL_TYPES,
        "tsx": JAVASCRIPT_SYMBOL_TYPES,
        "jsx": JAVASCRIPT_SYMBOL_TYPES,
        "rust": RUST_SYMBOL_TYPES,
        "go": GO_SYMBOL_TYPES,
    }

    SIGNATURE_MAPPINGS: ClassVar[dict[str, set[str]]] = {
        "python": PYTHON_SIGNATURE_TYPES,
        "javascript": JAVASCRIPT_SIGNATURE_TYPES,
        "typescript": JAVASCRIPT_SIGNATURE_TYPES,
        "tsx": JAVASCRIPT_SIGNATURE_TYPES,
        "jsx": JAVASCRIPT_SIGNATURE_TYPES,
        "rust": RUST_SIGNATURE_TYPES,
        "go": GO_SIGNATURE_TYPES,
    }

    @classmethod
    def get_symbol_type(cls, node_type: str, language: str) -> SymbolType | None:
        """Get the symbol type for a given node type and language.

        Args:
            node_type: Tree-sitter node type.
            language: Programming language name.

        Returns:
            SymbolType if the node type represents a symbol, None otherwise.
        """
        mapping = cls.LANGUAGE_MAPPINGS.get(language.lower())
        if mapping:
            return mapping.get(node_type)
        return None

    @classmethod
    def _arrow_function_signature(cls, node: Any) -> str | None:
        """Extract the signature of a JS/TS arrow function from its declaring variable."""
        ancestor = node.parent
        while ancestor:
            if ancestor.type in ("lexical_declaration", "variable_declaration"):
                body_node = node.child_by_field_name("body")
                if not body_node:
                    return None
                full_text = node_text(ancestor)
                offset = body_node.start_byte - ancestor.start_byte
                sig = full_text[:offset].strip()
                if sig.endswith("=>"):
                    sig = sig[:-2].strip()
                return sig
            ancestor = ancestor.parent
        return None

    @classmethod
    def _body_node(cls, node: Any) -> Any:
        """Find the physical code block body of a node, if any."""
        for field_name in ("body", "block"):
            body_node = node.child_by_field_name(field_name)
            if body_node:
                return body_node
        for child in node.children:
            if child.type in _BLOCK_NODE_TYPES:
                return child
        return None

    @classmethod
    def extract_signature(cls, node: Any, language: str) -> str | None:
        """Extract a single-line signature from a symbol node."""
        lang_lower = language.lower()
        signature_types = cls.SIGNATURE_MAPPINGS.get(lang_lower, set())
        if node.type not in signature_types:
            return None

        if lang_lower in JS_TS_LANGUAGES and node.type == "arrow_function":
            sig = cls._arrow_function_signature(node)
            if sig is not None:
                return sig

        # Structural body truncation
        signature_text = node_text(node).strip()
        body_node = cls._body_node(node)
        if body_node:
            end_offset = body_node.start_byte - node.start_byte
            signature_text = signature_text[:end_offset].strip()

        # Sanitization and cleanup
        if lang_lower == "python" and signature_text.endswith(":"):
            signature_text = signature_text[:-1].strip()
        if signature_text.endswith(";") or signature_text.endswith("{"):
            signature_text = signature_text[:-1].strip()

        return signature_text

    @classmethod
    def _variable_context(
        cls, node: Any, node_type: str, parent_name: str | None
    ) -> str | None:
        """Resolve the variable name that should flow down to arrow functions."""
        if node_type in _VARIABLE_NODE_TYPES:
            extracted_name = _find_identifier_text(node)
            if extracted_name:
                return extracted_name
        return parent_name

    @classmethod
    def _symbol_type_for(
        cls, node: Any, node_type: str, lang_lower: str, parent_scope: str | None
    ) -> SymbolType | None:
        """Resolve the symbol type, upgrading class-nested functions to methods."""
        symbol_type = cls.get_symbol_type(node_type, lang_lower)

        # A function_definition nested inside a class should be a method
        if (
            lang_lower == "python"
            and node_type == "function_definition"
            and parent_scope
        ):
            ancestor = node.parent
            while ancestor:
                if ancestor.type == "class_definition":
                    symbol_type = SymbolType.METHOD
                    break
                ancestor = ancestor.parent

        return symbol_type

    @classmethod
    def _go_method_context(
        cls, node: Any, parent_scope: str | None
    ) -> tuple[str | None, str | None]:
        """Return (method_name, receiver_scope) for a Go method declaration."""
        name = None
        name_node = node.child_by_field_name("name")
        if name_node:
            name = node_text(name_node)

        scope = parent_scope
        receiver_node = node.child_by_field_name("receiver")
        if receiver_node and receiver_node.named_child_count > 0:
            param_decl = receiver_node.named_child(0)
            if param_decl and param_decl.type == "parameter_declaration":
                type_node = param_decl.child_by_field_name("type")
                if type_node:
                    scope = node_text(type_node).lstrip("*")

        return name, scope

    @classmethod
    def _impl_name(cls, node: Any) -> str | None:
        """Resolve the name of a Rust impl_item from its 'type' field."""
        type_node = node.child_by_field_name("type")
        if not type_node:
            return None
        if type_node.type == "generic_type":
            inner = type_node.child_by_field_name("type")
            if inner:
                return node_text(inner)
            return node_text(type_node)
        return node_text(type_node)

    @classmethod
    def _resolve_symbol_context(
        cls, node: Any, node_type: str, lang_lower: str, parent_scope: str | None
    ) -> tuple[str | None, str | None]:
        """Return (name, effective_parent_scope) for a symbol node."""
        name = _find_identifier_text(node)

        if node_type == "method_declaration" and lang_lower == "go":
            go_name, parent_scope = cls._go_method_context(node, parent_scope)
            if go_name:
                name = go_name

        if node_type == "impl_item":
            impl_name = cls._impl_name(node)
            if impl_name:
                name = impl_name

        return name, parent_scope

    @classmethod
    def _record_symbol(
        cls,
        symbols: list[ASTNode],
        node: Any,
        name: str | None,
        symbol_type: SymbolType,
        parent_scope: str | None,
        file_path: str,
        language: str,
    ) -> None:
        """Append a symbol node, or extend the struct's end line for impl_item nodes."""
        if node.type == "impl_item":
            struct_node = next((n for n in symbols if n.name == name), None)
            if struct_node:
                struct_node.end_line_number = node.end_point[0] + 1
            return

        start_point = node.start_point
        end_point = node.end_point
        symbols.append(
            ASTNode(
                type=node.type,
                name=name,
                line_number=start_point[0] + 1,
                end_line_number=end_point[0] + 1,
                column=start_point[1],
                metadata={
                    "symbol_type": symbol_type.value,
                    "parent_scope": parent_scope,
                    "file_path": file_path,
                },
                signature=cls.extract_signature(node, language),
            )
        )

    @classmethod
    def extract_symbols(
        cls, root_node: Any, language: str, file_path: str
    ) -> list[ASTNode]:
        """Extract symbol nodes from the AST, gracefully handling syntactically incomplete structures.

        Args:
            root_node: Tree-sitter root node.
            language: Programming language name.
            file_path: Path to the file being parsed.

        Returns:
            List of ASTNode objects representing symbols.
        """
        symbols: list[ASTNode] = []
        lang_lower = language.lower()

        def traverse(
            node: Any, parent_scope: str | None = None, parent_name: str | None = None
        ) -> None:
            if node is None or is_pruned_test_module(node):
                return

            node_type = node.type
            if node_type == "ERROR":
                for child in node.children:
                    traverse(child, parent_scope, parent_name)
                return

            symbol_type = cls._symbol_type_for(
                node, node_type, lang_lower, parent_scope
            )
            current_var_name = cls._variable_context(node, node_type, parent_name)

            if symbol_type:
                name, effective_parent_scope = cls._resolve_symbol_context(
                    node, node_type, lang_lower, parent_scope
                )
                if node_type == "arrow_function" and current_var_name:
                    name = current_var_name
                cls._record_symbol(
                    symbols,
                    node,
                    name,
                    symbol_type,
                    effective_parent_scope,
                    file_path,
                    language,
                )

                new_parent_scope = (
                    f"{parent_scope}.{name}"
                    if parent_scope and name
                    else name or parent_scope
                )
                child_parent_name = (
                    current_var_name if node_type in _VARIABLE_NODE_TYPES else None
                )
                for child in node.children:
                    traverse(child, new_parent_scope, child_parent_name)
            else:
                for child in node.children:
                    traverse(child, parent_scope, current_var_name)

        traverse(root_node)
        return symbols
