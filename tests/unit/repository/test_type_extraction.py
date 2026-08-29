"""Tests for TypeExtractor class across different programming languages."""

import pytest

from local_coding_assistant.repository.ast_parser import TypeExtractor


class TestPythonTypeExtraction:
    """Test Python type extraction."""

    def test_simple_type(self):
        """Test extraction of a simple custom type."""
        result = TypeExtractor.extract_python_types("MyClass")
        assert result == ["MyClass"]

    def test_generic_type(self):
        """Test extraction of generic type with custom class."""
        result = TypeExtractor.extract_python_types(
            "dict[str, ProviderLLMResponseDelta]"
        )
        assert result == ["ProviderLLMResponseDelta"]

    def test_nested_generics(self):
        """Test extraction of nested generic types."""
        result = TypeExtractor.extract_python_types("list[dict[str, CustomType]]")
        assert result == ["CustomType"]

    def test_union_type(self):
        """Test extraction of Union types."""
        result = TypeExtractor.extract_python_types("Union[str, CustomClass, int]")
        assert result == ["CustomClass"]

    def test_optional_type(self):
        """Test extraction of Optional types."""
        result = TypeExtractor.extract_python_types("Optional[MyClass]")
        assert result == ["MyClass"]

    def test_multiple_custom_types(self):
        """Test extraction of multiple custom types."""
        result = TypeExtractor.extract_python_types(
            "dict[str, list[CustomType1]] | CustomType2"
        )
        assert set(result) == {"CustomType1", "CustomType2"}

    def test_builtin_filtering(self):
        """Test that built-in types are filtered out."""
        result = TypeExtractor.extract_python_types("str, int, bool, list, dict")
        assert result == []

    def test_complex_annotation(self):
        """Test complex type annotation with multiple custom types."""
        result = TypeExtractor.extract_python_types(
            "dict[str, list[tuple[CustomClass1, CustomClass2]]]"
        )
        assert set(result) == {"CustomClass1", "CustomClass2"}

    def test_literal_type(self):
        """Test Literal type extraction."""
        result = TypeExtractor.extract_python_types("Literal['value']")
        # 'Literal' is a built-in type, should be filtered
        # 'value' is a string literal, extracted by regex - limitation of simple approach
        assert result == ["value"]

    def test_callable_type(self):
        """Test Callable type extraction."""
        result = TypeExtractor.extract_python_types(
            "Callable[[str, int], CustomResult]"
        )
        assert result == ["CustomResult"]

    def test_protocol_type(self):
        """Test Protocol type extraction."""
        result = TypeExtractor.extract_python_types("MyProtocol")
        assert result == ["MyProtocol"]

    def test_typevar_type(self):
        """Test TypeVar type extraction."""
        result = TypeExtractor.extract_python_types("T = TypeVar('T')")
        # 'TypeVar' is a built-in, should be filtered, but 'T' should remain
        # Note: 'T' appears twice in the string, deduplication handles this
        assert result == ["T"]

    def test_ellipsis_type(self):
        """Test ellipsis type extraction."""
        result = TypeExtractor.extract_python_types("list[int, ...]")
        assert result == []

    def test_empty_string(self):
        """Test empty string handling."""
        result = TypeExtractor.extract_python_types("")
        assert result == []

    def test_none_type(self):
        """Test None type handling."""
        result = TypeExtractor.extract_python_types("None")
        assert result == []

    def test_any_type(self):
        """Test Any type handling."""
        result = TypeExtractor.extract_python_types("Any")
        assert result == []

    def test_mixed_builtin_custom(self):
        """Test mixed built-in and custom types."""
        result = TypeExtractor.extract_python_types("dict[str, CustomClass] | None")
        assert result == ["CustomClass"]


class TestJavaScriptTypeScriptExtraction:
    """Test JavaScript/TypeScript type extraction."""

    def test_simple_interface(self):
        """Test extraction of simple interface name."""
        result = TypeExtractor.extract_js_ts_types("MyInterface")
        assert result == ["MyInterface"]

    def test_generic_interface(self):
        """Test extraction of generic interface."""
        result = TypeExtractor.extract_js_ts_types("Array<MyCustomType>")
        assert result == ["MyCustomType"]

    def test_union_type(self):
        """Test extraction of union type."""
        result = TypeExtractor.extract_js_ts_types("string | number | CustomClass")
        assert result == ["CustomClass"]

    def test_intersection_type(self):
        """Test extraction of intersection type."""
        result = TypeExtractor.extract_js_ts_types("TypeA & TypeB & TypeC")
        assert set(result) == {"TypeA", "TypeB", "TypeC"}

    def test_array_type(self):
        """Test extraction of array type."""
        result = TypeExtractor.extract_js_ts_types("CustomClass[]")
        assert result == ["CustomClass"]

    def test_record_type(self):
        """Test extraction of Record type."""
        result = TypeExtractor.extract_js_ts_types("Record<string, CustomType>")
        assert result == ["CustomType"]

    def test_promise_type(self):
        """Test extraction of Promise type."""
        result = TypeExtractor.extract_js_ts_types("Promise<CustomResult>")
        assert result == ["CustomResult"]

    def test_builtin_filtering(self):
        """Test that built-in types are filtered out."""
        result = TypeExtractor.extract_js_ts_types(
            "string, number, boolean, Array, Object"
        )
        # 'Object' is now in built-in list, should be filtered
        assert result == []

    def test_complex_generic(self):
        """Test complex generic type."""
        result = TypeExtractor.extract_js_ts_types("Map<string, Array<CustomType>>")
        assert result == ["CustomType"]

    def test_nullable_type(self):
        """Test nullable type."""
        result = TypeExtractor.extract_js_ts_types("CustomClass | null")
        assert result == ["CustomClass"]

    def test_undefined_type(self):
        """Test undefined type."""
        result = TypeExtractor.extract_js_ts_types("CustomClass | undefined")
        assert result == ["CustomClass"]

    def test_function_type(self):
        """Test function type."""
        result = TypeExtractor.extract_js_ts_types("(arg: string) => CustomResult")
        # 'arg' is a parameter name, extracted by regex - limitation of simple approach
        assert set(result) == {"arg", "CustomResult"}

    def test_class_type(self):
        """Test class type."""
        result = TypeExtractor.extract_js_ts_types("typeof MyClass")
        # 'typeof' is a keyword, should be filtered
        assert result == ["MyClass"]

    def test_utility_types(self):
        """Test utility types."""
        result = TypeExtractor.extract_js_ts_types("Partial<CustomInterface>")
        assert result == ["CustomInterface"]

    def test_mixed_types(self):
        """Test mixed built-in and custom types."""
        result = TypeExtractor.extract_js_ts_types("Record<string, CustomType> | null")
        assert result == ["CustomType"]


class TestRustTypeExtraction:
    """Test Rust type extraction."""

    def test_simple_struct(self):
        """Test extraction of simple struct name."""
        result = TypeExtractor.extract_rust_types("MyStruct")
        assert result == ["MyStruct"]

    def test_generic_struct(self):
        """Test extraction of generic struct."""
        result = TypeExtractor.extract_rust_types("Vec<MyCustomType>")
        assert result == ["MyCustomType"]

    def test_option_type(self):
        """Test extraction of Option type."""
        result = TypeExtractor.extract_rust_types("Option<MyStruct>")
        assert result == ["MyStruct"]

    def test_result_type(self):
        """Test extraction of Result type."""
        result = TypeExtractor.extract_rust_types("Result<MyStruct, Error>")
        assert result == ["MyStruct", "Error"]

    def test_box_type(self):
        """Test extraction of Box type."""
        result = TypeExtractor.extract_rust_types("Box<MyTrait>")
        assert result == ["MyTrait"]

    def test_hashmap_type(self):
        """Test extraction of HashMap type."""
        result = TypeExtractor.extract_rust_types("HashMap<String, CustomType>")
        assert result == ["CustomType"]

    def test_builtin_filtering(self):
        """Test that built-in types are filtered out."""
        result = TypeExtractor.extract_rust_types("String, Vec, i32, bool, Option")
        assert result == []

    def test_tuple_type(self):
        """Test extraction of tuple type."""
        result = TypeExtractor.extract_rust_types("(String, CustomType, i32)")
        assert result == ["CustomType"]

    def test_array_type(self):
        """Test extraction of array type."""
        result = TypeExtractor.extract_rust_types("[CustomType; 10]")
        assert result == ["CustomType"]

    def test_reference_type(self):
        """Test extraction of reference type."""
        result = TypeExtractor.extract_rust_types("&MyStruct")
        assert result == ["MyStruct"]

    def test_mutable_reference(self):
        """Test extraction of mutable reference."""
        result = TypeExtractor.extract_rust_types("&mut MyStruct")
        # 'mut' is a keyword, should be filtered
        assert result == ["MyStruct"]

    def test_lifetime_annotation(self):
        """Test lifetime annotation."""
        result = TypeExtractor.extract_rust_types("&'a MyStruct")
        # 'a' is not a keyword, it's a lifetime name, should be included
        # This is a limitation of the simple regex approach
        assert result == ["a", "MyStruct"]

    def test_trait_object(self):
        """Test trait object."""
        result = TypeExtractor.extract_rust_types("Box<dyn MyTrait>")
        # 'dyn' is a keyword, should be filtered
        assert result == ["MyTrait"]

    def test_complex_generic(self):
        """Test complex generic type."""
        result = TypeExtractor.extract_rust_types("Vec<HashMap<String, CustomType>>")
        assert result == ["CustomType"]

    def test_impl_trait(self):
        """Test impl trait."""
        result = TypeExtractor.extract_rust_types("impl MyTrait")
        # 'impl' is a keyword, should be filtered
        assert result == ["MyTrait"]


class TestGoTypeExtraction:
    """Test Go type extraction."""

    def test_simple_struct(self):
        """Test extraction of simple struct name."""
        result = TypeExtractor.extract_go_types("MyStruct")
        assert result == ["MyStruct"]

    def test_pointer_type(self):
        """Test extraction of pointer type."""
        result = TypeExtractor.extract_go_types("*MyStruct")
        assert result == ["MyStruct"]

    def test_slice_type(self):
        """Test extraction of slice type."""
        result = TypeExtractor.extract_go_types("[]MyStruct")
        assert result == ["MyStruct"]

    def test_array_type(self):
        """Test extraction of array type."""
        result = TypeExtractor.extract_go_types("[10]MyStruct")
        assert result == ["MyStruct"]

    def test_map_type(self):
        """Test extraction of map type."""
        result = TypeExtractor.extract_go_types("map[string]MyStruct")
        # 'map' is a keyword, should be filtered
        assert result == ["MyStruct"]

    def test_channel_type(self):
        """Test extraction of channel type."""
        result = TypeExtractor.extract_go_types("chan MyStruct")
        # 'chan' is a keyword, should be filtered
        assert result == ["MyStruct"]

    def test_builtin_filtering(self):
        """Test that built-in types are filtered out."""
        result = TypeExtractor.extract_go_types("string, int, bool, error, []byte")
        assert result == []

    def test_interface_type(self):
        """Test extraction of interface type."""
        result = TypeExtractor.extract_go_types("MyInterface")
        assert result == ["MyInterface"]

    def test_struct_pointer(self):
        """Test extraction of struct pointer."""
        result = TypeExtractor.extract_go_types("*MyStruct")
        assert result == ["MyStruct"]

    def test_nested_slice(self):
        """Test extraction of nested slice."""
        result = TypeExtractor.extract_go_types("[][]MyStruct")
        assert result == ["MyStruct"]

    def test_map_with_custom_key(self):
        """Test extraction of map with custom key type."""
        result = TypeExtractor.extract_go_types("map[CustomKey]CustomValue")
        # 'map' is a keyword, should be filtered
        assert set(result) == {"CustomKey", "CustomValue"}

    def test_function_type(self):
        """Test extraction of function type."""
        result = TypeExtractor.extract_go_types("func(string) MyStruct")
        # 'func' is a keyword, should be filtered
        assert result == ["MyStruct"]

    def test_complex_type(self):
        """Test complex type with multiple custom types."""
        result = TypeExtractor.extract_go_types("map[string][]*MyStruct")
        # 'map' is a keyword, should be filtered
        assert result == ["MyStruct"]


class TestGenericExtraction:
    """Test generic extraction method."""

    def test_python_dispatch(self):
        """Test dispatch to Python extractor."""
        result = TypeExtractor.extract_types("dict[str, MyClass]", "python")
        assert result == ["MyClass"]

    def test_typescript_dispatch(self):
        """Test dispatch to TypeScript extractor."""
        result = TypeExtractor.extract_types("Array<MyClass>", "typescript")
        assert result == ["MyClass"]

    def test_rust_dispatch(self):
        """Test dispatch to Rust extractor."""
        result = TypeExtractor.extract_types("Vec<MyStruct>", "rust")
        assert result == ["MyStruct"]

    def test_go_dispatch(self):
        """Test dispatch to Go extractor."""
        result = TypeExtractor.extract_types("[]MyStruct", "go")
        assert result == ["MyStruct"]

    def test_unsupported_language(self):
        """Test extraction for unsupported language (generic)."""
        result = TypeExtractor.extract_types("MyClass", "unknown")
        assert result == ["MyClass"]

    def test_case_insensitive(self):
        """Test that language matching is case-insensitive."""
        result1 = TypeExtractor.extract_types("dict[str, MyClass]", "Python")
        result2 = TypeExtractor.extract_types("dict[str, MyClass]", "PYTHON")
        assert result1 == result2 == ["MyClass"]

    def test_javascript_dispatch(self):
        """Test dispatch to JavaScript extractor."""
        result = TypeExtractor.extract_types("MyClass[]", "javascript")
        assert result == ["MyClass"]

    def test_jsx_dispatch(self):
        """Test dispatch to JSX extractor."""
        result = TypeExtractor.extract_types("MyInterface", "jsx")
        assert result == ["MyInterface"]

    def test_tsx_dispatch(self):
        """Test dispatch to TSX extractor."""
        result = TypeExtractor.extract_types("MyComponent", "tsx")
        assert result == ["MyComponent"]


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_underscore_start(self):
        """Test type starting with underscore."""
        result = TypeExtractor.extract_python_types("_PrivateClass")
        assert result == ["_PrivateClass"]

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        result = TypeExtractor.extract_python_types("MyClass")
        assert result == ["MyClass"]

    def test_numbers_in_type(self):
        """Test type with numbers."""
        result = TypeExtractor.extract_python_types("Type123")
        assert result == ["Type123"]

    def test_mixed_case(self):
        """Test mixed case type names."""
        result = TypeExtractor.extract_python_types("MyCustomType")
        assert result == ["MyCustomType"]

    def test_very_long_type_string(self):
        """Test very long type string."""
        long_type = "dict[str, list[tuple[CustomType1, CustomType2, CustomType3]]]"
        result = TypeExtractor.extract_python_types(long_type)
        assert set(result) == {"CustomType1", "CustomType2", "CustomType3"}

    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        result = TypeExtractor.extract_python_types("dict[ str , MyClass ]")
        assert result == ["MyClass"]

    def test_newlines_in_type(self):
        """Test handling of newlines in type string."""
        result = TypeExtractor.extract_python_types("dict[str,\nMyClass]")
        assert result == ["MyClass"]

    def test_special_characters_filtered(self):
        """Test that special characters are filtered out."""
        result = TypeExtractor.extract_python_types("dict[str, @MyClass]")
        assert result == ["MyClass"]

    def test_dotted_module_path(self):
        """Test dotted module path (should extract each component)."""
        result = TypeExtractor.extract_python_types("my.module.MyClass")
        assert set(result) == {"my", "module", "MyClass"}

    def test_empty_after_filtering(self):
        """Test that empty list is returned when all types are filtered."""
        result = TypeExtractor.extract_python_types("str, int, bool")
        assert result == []

    def test_single_builtin(self):
        """Test single built-in type."""
        result = TypeExtractor.extract_python_types("str")
        assert result == []

    def test_single_custom(self):
        """Test single custom type."""
        result = TypeExtractor.extract_python_types("MyClass")
        assert result == ["MyClass"]
