"""Type name extraction from AST type annotation strings.

Extracts user-defined type names while filtering out language built-ins
and keywords.
"""

import re
from typing import ClassVar

from .common import JS_TS_LANGUAGES


class TypeExtractor:
    """Extracts type names from type strings for different programming languages."""

    # Python built-in types to filter out
    PYTHON_BUILTINS: ClassVar[set[str]] = {
        "str",
        "int",
        "bool",
        "float",
        "None",
        "Any",
        "dict",
        "list",
        "set",
        "tuple",
        "bytes",
        "Optional",
        "Union",
        "Dict",
        "List",
        "Set",
        "Tuple",
        "Mapping",
        "Sequence",
        "Iterable",
        "Callable",
        "Awaitable",
        "AsyncIterator",
        "Generator",
        "AsyncGenerator",
        "ContextManager",
        "AsyncContextManager",
        "TypeVar",
        "Protocol",
        "Final",
        "Literal",
        "ClassVar",
    }

    PYTHON_BUILTIN_CALLEES: ClassVar[set[str]] = {
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "isinstance",
        "issubclass",
    }

    # Python keywords to filter out
    PYTHON_KEYWORDS: ClassVar[set[str]] = {
        "or",
        "and",
        "not",
        "in",
        "is",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "def",
        "class",
        "return",
        "yield",
        "from",
        "import",
        "as",
        "with",
        "lambda",
    }

    # JavaScript/TypeScript built-in types to filter out
    JS_TS_BUILTINS: ClassVar[set[str]] = {
        "string",
        "number",
        "boolean",
        "null",
        "undefined",
        "void",
        "any",
        "unknown",
        "never",
        "object",
        "Object",
        "Array",
        "Function",
        "Promise",
        "Record",
        "Map",
        "Set",
        "Date",
        "RegExp",
        "Error",
        "Readonly",
        "Partial",
        "Required",
        "Pick",
        "Omit",
    }

    # JavaScript/TypeScript keywords to filter out
    JS_TS_KEYWORDS: ClassVar[set[str]] = {
        "typeof",
        "instanceof",
        "new",
        "this",
        "super",
        "extends",
        "implements",
        "interface",
        "type",
        "enum",
        "const",
        "let",
        "var",
        "function",
        "return",
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "break",
        "continue",
        "try",
        "catch",
        "finally",
        "throw",
        "async",
        "await",
        "yield",
    }

    # Rust built-in types to filter out
    RUST_BUILTINS: ClassVar[set[str]] = {
        "str",
        "String",
        "bool",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "char",
        "Box",
        "Vec",
        "HashMap",
        "HashSet",
        "Option",
        "Result",
        "BTreeMap",
        "BTreeSet",
        "LinkedList",
        "VecDeque",
        "BinaryHeap",
    }

    RUST_BUILTINS_CALLEES: ClassVar[set[str]] = {
        "Ok",
        "Some",
        "Err",
        "as_mut",
        "as_ref",
        "extend",
        "push",
        "take",
        "to_vec",
        "is_empty",
        "unwrap",
        "len",
        "iter",
        "iter_mut",
        "next",
        "next_back",
        "unwrap_or",
        "unwrap_or_else",
        "unwrap_or_default",
        "unwrap_unchecked",
    }

    # Rust keywords to filter out
    RUST_KEYWORDS: ClassVar[set[str]] = {
        "mut",
        "const",
        "static",
        "let",
        "fn",
        "pub",
        "crate",
        "mod",
        "use",
        "struct",
        "enum",
        "union",
        "trait",
        "impl",
        "type",
        "where",
        "for",
        "while",
        "loop",
        "match",
        "if",
        "else",
        "return",
        "break",
        "continue",
        "async",
        "await",
        "move",
        "dyn",
        "ref",
        "unsafe",
    }

    # Go built-in types to filter out
    GO_BUILTINS: ClassVar[set[str]] = {
        "string",
        "bool",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintptr",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "byte",
        "rune",
        "error",
        "any",
        "comparable",
    }

    # Go keywords to filter out
    GO_KEYWORDS: ClassVar[set[str]] = {
        "map",
        "chan",
        "func",
        "go",
        "select",
        "defer",
        "goto",
        "return",
        "break",
        "continue",
        "if",
        "else",
        "switch",
        "case",
        "fallthrough",
        "for",
        "range",
        "const",
        "var",
        "type",
        "struct",
        "interface",
        "package",
        "import",
    }

    LANGUAGE_BUILTIN_MAPPINGS: ClassVar[dict[str, set[str]]] = {
        "python": PYTHON_BUILTINS,
        "javascript": JS_TS_BUILTINS,
        "typescript": JS_TS_BUILTINS,
        "jsx": JS_TS_BUILTINS,
        "tsx": JS_TS_BUILTINS,
        "rust": RUST_BUILTINS,
        "go": GO_BUILTINS,
    }

    LANGUAGE_BUILTIN_CALLEES: ClassVar[dict[str, set[str]]] = {
        "python": PYTHON_BUILTINS | PYTHON_BUILTIN_CALLEES,
        "javascript": JS_TS_BUILTINS,
        "typescript": JS_TS_BUILTINS,
        "jsx": JS_TS_BUILTINS,
        "tsx": JS_TS_BUILTINS,
        "rust": RUST_BUILTINS | RUST_BUILTINS_CALLEES,
        "go": GO_BUILTINS,
    }

    _TYPE_TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(r"[a-zA-Z_]\w*")

    @classmethod
    def _extract_tokens(
        cls, raw_type_string: str, builtins: set[str], keywords: set[str]
    ) -> list[str]:
        """Extract unique type tokens, filtering out built-ins and keywords."""
        tokens = cls._TYPE_TOKEN_RE.findall(raw_type_string)
        filtered = [t for t in tokens if t not in builtins and t not in keywords]
        return list(dict.fromkeys(filtered))

    @classmethod
    def extract_python_types(cls, raw_type_string: str) -> list[str]:
        """Extract type names from Python type hint strings.

        Args:
            raw_type_string: Raw type string from AST node.

        Returns:
            List of extracted type names with built-ins filtered out.
        """
        return cls._extract_tokens(
            raw_type_string, cls.PYTHON_BUILTINS, cls.PYTHON_KEYWORDS
        )

    @classmethod
    def extract_js_ts_types(cls, raw_type_string: str) -> list[str]:
        """Extract type names from JavaScript/TypeScript type annotation strings.

        Args:
            raw_type_string: Raw type string from AST node.

        Returns:
            List of extracted type names with built-ins filtered out.
        """
        return cls._extract_tokens(
            raw_type_string, cls.JS_TS_BUILTINS, cls.JS_TS_KEYWORDS
        )

    @classmethod
    def extract_rust_types(cls, raw_type_string: str) -> list[str]:
        """Extract type names from Rust type identifier strings.

        Args:
            raw_type_string: Raw type string from AST node.

        Returns:
            List of extracted type names with built-ins filtered out.
        """
        return cls._extract_tokens(
            raw_type_string, cls.RUST_BUILTINS, cls.RUST_KEYWORDS
        )

    @classmethod
    def extract_go_types(cls, raw_type_string: str) -> list[str]:
        """Extract type names from Go type identifier strings.

        Args:
            raw_type_string: Raw type string from AST node.

        Returns:
            List of extracted type names with built-ins filtered out.
        """
        return cls._extract_tokens(raw_type_string, cls.GO_BUILTINS, cls.GO_KEYWORDS)

    @classmethod
    def extract_types(cls, raw_type_string: str, language: str) -> list[str]:
        """Extract type names from type strings based on language.

        Args:
            raw_type_string: Raw type string from AST node.
            language: Programming language name.

        Returns:
            List of extracted type names with built-ins filtered out.
        """
        lang_lower = language.lower()

        if lang_lower == "python":
            return cls.extract_python_types(raw_type_string)
        if lang_lower in JS_TS_LANGUAGES:
            return cls.extract_js_ts_types(raw_type_string)
        if lang_lower == "rust":
            return cls.extract_rust_types(raw_type_string)
        if lang_lower == "go":
            return cls.extract_go_types(raw_type_string)
        # Generic extraction for unsupported languages
        return cls._TYPE_TOKEN_RE.findall(raw_type_string)
