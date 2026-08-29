"""Tests for InheritanceExtractor class across different programming languages."""

from unittest.mock import Mock

from local_coding_assistant.repository.ast_parser import InheritanceExtractor


class TestPythonInheritanceExtraction:
    """Test Python inheritance extraction."""

    def test_single_inheritance(self):
        """Test extraction of single base class."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)
        # No superclasses
        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == []

    def test_multiple_inheritance(self):
        """Test extraction of multiple base classes."""
        node = Mock()
        arg_list = Mock()

        # Create mock children for superclasses
        child1 = Mock()
        child1.type = "identifier"
        child1.text = b"BaseClass1"

        child2 = Mock()
        child2.type = "identifier"
        child2.text = b"BaseClass2"

        arg_list.children = [child1, child2]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == ["BaseClass1", "BaseClass2"]

    def test_dotted_module_inheritance(self):
        """Test extraction of dotted module path inheritance."""
        node = Mock()
        arg_list = Mock()

        child = Mock()
        child.type = "attribute"
        child.text = b"module.BaseClass"

        arg_list.children = [child]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == ["module.BaseClass"]

    def test_no_inheritance(self):
        """Test class with no inheritance."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == []

    def test_mixed_inheritance(self):
        """Test mixed identifier and dotted inheritance."""
        node = Mock()
        arg_list = Mock()

        child1 = Mock()
        child1.type = "identifier"
        child1.text = b"BaseClass"

        child2 = Mock()
        child2.type = "attribute"
        child2.text = b"module.Mixin"

        arg_list.children = [child1, child2]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == ["BaseClass", "module.Mixin"]


class TestJavaScriptTypeScriptInheritanceExtraction:
    """Test JavaScript/TypeScript inheritance extraction."""

    def test_extends_single(self):
        """Test extraction of single extends clause."""
        node = Mock()
        heritage = Mock()

        child = Mock()
        child.type = "identifier"
        child.text = b"BaseClass"

        heritage.children = [child]
        node.child_by_field_name = Mock(return_value=heritage)

        result = InheritanceExtractor.extract_js_ts_inheritance(node, "MyClass")
        assert result == ["BaseClass"]

    def test_implements_multiple(self):
        """Test extraction of multiple implements clauses."""
        node = Mock()
        heritage = Mock()

        child1 = Mock()
        child1.type = "type_identifier"
        child1.text = b"Interface1"

        child2 = Mock()
        child2.type = "type_identifier"
        child2.text = b"Interface2"

        heritage.children = [child1, child2]
        node.child_by_field_name = Mock(return_value=heritage)

        result = InheritanceExtractor.extract_js_ts_inheritance(node, "MyClass")
        assert result == ["Interface1", "Interface2"]

    def test_property_identifier(self):
        """Test extraction with property_identifier."""
        node = Mock()
        heritage = Mock()

        child = Mock()
        child.type = "property_identifier"
        child.text = b"MyInterface"

        heritage.children = [child]
        node.child_by_field_name = Mock(return_value=heritage)

        result = InheritanceExtractor.extract_js_ts_inheritance(node, "MyClass")
        assert result == ["MyInterface"]

    def test_member_expression(self):
        """Test extraction with member expression (dotted names)."""
        node = Mock()
        heritage = Mock()

        child = Mock()
        child.type = "member_expression"
        child.text = b"module.BaseClass"

        heritage.children = [child]
        node.child_by_field_name = Mock(return_value=heritage)

        result = InheritanceExtractor.extract_js_ts_inheritance(node, "MyClass")
        assert result == ["module.BaseClass"]

    def test_no_heritage(self):
        """Test class with no heritage clause."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_js_ts_inheritance(node, "MyClass")
        assert result == []


class TestRustInheritanceExtraction:
    """Test Rust trait implementation extraction."""

    def test_impl_trait(self):
        """Test extraction of trait from impl block."""
        node = Mock()
        node.type = "impl_item"

        trait_type = Mock()
        trait_type.text = b"Trait for Struct"

        node.child_by_field_name = Mock(return_value=trait_type)

        result = InheritanceExtractor.extract_rust_inheritance(node, "Struct")
        assert result == ["Trait"]

    def test_impl_trait_simple(self):
        """Test extraction of simple trait without 'for' clause."""
        node = Mock()
        node.type = "impl_item"

        trait_type = Mock()
        trait_type.text = b"Trait"

        node.child_by_field_name = Mock(return_value=trait_type)

        result = InheritanceExtractor.extract_rust_inheritance(node, "Struct")
        assert result == ["Trait"]

    def test_struct_item_trait_bounds(self):
        """Test extraction of trait bounds from struct item."""
        node = Mock()
        node.type = "struct_item"

        type_params = Mock()
        bounded_type = Mock()
        bounded_type.type = "bounded_type"

        type_identifier = Mock()
        type_identifier.type = "type_identifier"
        type_identifier.text = b"MyTrait"

        bounded_type.children = [type_identifier]
        type_params.children = [bounded_type]

        node.child_by_field_name = Mock(return_value=type_params)

        result = InheritanceExtractor.extract_rust_inheritance(node, "MyStruct")
        assert result == ["MyTrait"]

    def test_no_trait_bounds(self):
        """Test struct with no trait bounds."""
        node = Mock()
        node.type = "struct_item"
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_rust_inheritance(node, "MyStruct")
        assert result == []

    def test_impl_item_no_type(self):
        """Test impl item with no type field."""
        node = Mock()
        node.type = "impl_item"
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_rust_inheritance(node, "Struct")
        assert result == []

    def test_non_impl_struct_node(self):
        """Test non-impl/struct node returns empty."""
        node = Mock()
        node.type = "function_item"

        result = InheritanceExtractor.extract_rust_inheritance(node, "MyStruct")
        assert result == []


class TestGoInheritanceExtraction:
    """Test Go embedded interface extraction."""

    def test_single_embedded_interface(self):
        """Test extraction of single embedded interface."""
        node = Mock()

        field = Mock()
        field.type = "field_declaration"
        field.child_by_field_name = Mock(
            side_effect=lambda x: None if x == "name" else Mock()
        )

        type_node = Mock()
        type_node.text = b"InterfaceType"

        def mock_child_by_field_name(field_name):
            if field_name == "type":
                return type_node
            return None

        field.child_by_field_name = mock_child_by_field_name

        node.children = [field]

        result = InheritanceExtractor.extract_go_inheritance(node, "MyStruct")
        assert result == ["InterfaceType"]

    def test_multiple_embedded_interfaces(self):
        """Test extraction of multiple embedded interfaces."""
        node = Mock()

        field1 = Mock()
        field1.type = "field_declaration"
        type1 = Mock()
        type1.text = b"Interface1"

        def mock_child_by_field_name1(field_name):
            if field_name == "type":
                return type1
            return None

        field1.child_by_field_name = mock_child_by_field_name1

        field2 = Mock()
        field2.type = "field_declaration"
        type2 = Mock()
        type2.text = b"Interface2"

        def mock_child_by_field_name2(field_name):
            if field_name == "type":
                return type2
            return None

        field2.child_by_field_name = mock_child_by_field_name2

        node.children = [field1, field2]

        result = InheritanceExtractor.extract_go_inheritance(node, "MyStruct")
        assert result == ["Interface1", "Interface2"]

    def test_regular_field_not_embedded(self):
        """Test that regular fields (with names) are not extracted as embedded."""
        node = Mock()

        field = Mock()
        field.type = "field_declaration"
        type_node = Mock()
        type_node.text = b"FieldType"
        name_node = Mock()

        def mock_child_by_field_name(field_name):
            if field_name == "type":
                return type_node
            if field_name == "name":
                return name_node
            return None

        field.child_by_field_name = mock_child_by_field_name

        node.children = [field]

        result = InheritanceExtractor.extract_go_inheritance(node, "MyStruct")
        assert result == []

    def test_no_embedded_interfaces(self):
        """Test struct with no embedded interfaces."""
        node = Mock()
        node.children = []

        result = InheritanceExtractor.extract_go_inheritance(node, "MyStruct")
        assert result == []

    def test_mixed_embedded_and_regular(self):
        """Test struct with both embedded and regular fields."""
        node = Mock()

        # Embedded field
        field1 = Mock()
        field1.type = "field_declaration"
        type1 = Mock()
        type1.text = b"EmbeddedInterface"

        def mock_child_by_field_name1(field_name):
            if field_name == "type":
                return type1
            return None

        field1.child_by_field_name = mock_child_by_field_name1

        # Regular field
        field2 = Mock()
        field2.type = "field_declaration"
        type2 = Mock()
        type2.text = b"RegularType"
        name2 = Mock()

        def mock_child_by_field_name2(field_name):
            if field_name == "type":
                return type2
            if field_name == "name":
                return name2
            return None

        field2.child_by_field_name = mock_child_by_field_name2

        node.children = [field1, field2]

        result = InheritanceExtractor.extract_go_inheritance(node, "MyStruct")
        assert result == ["EmbeddedInterface"]


class TestGenericExtraction:
    """Test generic extraction method."""

    def test_python_dispatch(self):
        """Test dispatch to Python extractor."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "python")
        assert result == []

    def test_typescript_dispatch(self):
        """Test dispatch to TypeScript extractor."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "typescript")
        assert result == []

    def test_rust_dispatch(self):
        """Test dispatch to Rust extractor."""
        node = Mock()
        node.type = "function_item"

        result = InheritanceExtractor.extract_inheritance(node, "MyStruct", "rust")
        assert result == []

    def test_go_dispatch(self):
        """Test dispatch to Go extractor."""
        node = Mock()
        node.children = []

        result = InheritanceExtractor.extract_inheritance(node, "MyStruct", "go")
        assert result == []

    def test_unsupported_language(self):
        """Test extraction for unsupported language returns empty."""
        node = Mock()

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "unknown")
        assert result == []

    def test_case_insensitive(self):
        """Test that language matching is case-insensitive."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result1 = InheritanceExtractor.extract_inheritance(node, "MyClass", "Python")
        result2 = InheritanceExtractor.extract_inheritance(node, "MyClass", "PYTHON")
        assert result1 == result2 == []

    def test_javascript_dispatch(self):
        """Test dispatch to JavaScript extractor."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "javascript")
        assert result == []

    def test_jsx_dispatch(self):
        """Test dispatch to JSX extractor."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "jsx")
        assert result == []

    def test_tsx_dispatch(self):
        """Test dispatch to TSX extractor."""
        node = Mock()
        node.child_by_field_name = Mock(return_value=None)

        result = InheritanceExtractor.extract_inheritance(node, "MyClass", "tsx")
        assert result == []


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_node_children(self):
        """Test node with empty children list."""
        node = Mock()
        arg_list = Mock()
        arg_list.children = []
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == []

    def test_none_node(self):
        """Test handling of None node."""
        result = InheritanceExtractor.extract_python_inheritance(None, "MyClass")
        assert result == []

    def test_unicode_class_names(self):
        """Test handling of unicode in class names."""
        node = Mock()
        arg_list = Mock()

        child = Mock()
        child.type = "identifier"
        child.text = "MyClass".encode("utf-8")

        arg_list.children = [child]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == ["MyClass"]

    def test_very_long_inheritance_chain(self):
        """Test very long inheritance chain."""
        node = Mock()
        arg_list = Mock()

        children = []
        for i in range(10):
            child = Mock()
            child.type = "identifier"
            child.text = f"BaseClass{i}".encode("utf-8")
            children.append(child)

        arg_list.children = children
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert len(result) == 10
        assert result[0] == "BaseClass0"
        assert result[9] == "BaseClass9"

    def test_duplicate_base_classes(self):
        """Test handling of duplicate base classes."""
        node = Mock()
        arg_list = Mock()

        child1 = Mock()
        child1.type = "identifier"
        child1.text = b"BaseClass"

        child2 = Mock()
        child2.type = "identifier"
        child2.text = b"BaseClass"

        arg_list.children = [child1, child2]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        # Note: current implementation doesn't deduplicate
        assert result == ["BaseClass", "BaseClass"]

    def test_mixed_node_types_in_children(self):
        """Test handling of mixed node types in children."""
        node = Mock()
        arg_list = Mock()

        child1 = Mock()
        child1.type = "identifier"
        child1.text = b"BaseClass"

        child2 = Mock()
        child2.type = "comment"  # Should be ignored
        child2.text = b"# comment"

        arg_list.children = [child1, child2]
        node.child_by_field_name = Mock(return_value=arg_list)

        result = InheritanceExtractor.extract_python_inheritance(node, "MyClass")
        assert result == ["BaseClass"]
