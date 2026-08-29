"""Pydantic models for repository context data."""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class SymbolType(str, Enum):
    """Types of symbols that can be extracted from code."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    PARAMETER = "parameter"
    IMPORT = "import"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    CONSTANT = "constant"
    MODULE = "module"


class ImportType(str, Enum):
    """Types of import statements."""

    MODULE = "module"
    FROM_IMPORT = "from_import"
    RELATIVE = "relative"


class SymbolDetail(BaseModel):
    """Detailed information about a symbol."""

    symbol_id: int
    file_id: int
    name: str
    symbol_type: SymbolType
    line_number: int
    end_line_number: int | None = None
    parent_scope: str | None = None
    docstring: str | None = None
    content: str | None = None
    signature: str | None = None
    file_path: str | None = None


@dataclass
class ASTNode:
    """Lightweight AST node representation."""

    type: str
    name: str | None = None
    line_number: int = 0
    end_line_number: int | None = None
    column: int = 0
    children: list["ASTNode"] | None = None
    metadata: dict | None = None
    signature: str | None = None

    def to_symbol_detail(
        self, file_content: str, language: str, file_id: int | None = None
    ) -> SymbolDetail:
        """Convert ASTNode to SymbolDetail.

        Args:
            file_content: Content of the file node parsed from.
            language: Programming language of the file.
            file_id: ID of the file in the database.

        Returns:
            SymbolDetail: Detailed information about the symbol.
        """
        symbol_type_str = (
            self.metadata.get("symbol_type", "unknown") if self.metadata else "unknown"
        )
        try:
            symbol_type = SymbolType(symbol_type_str)
        except ValueError:
            symbol_type = SymbolType.VARIABLE

        docstring = self._extract_docstring(file_content, language)
        symbol_content = self._extract_symbol_content(file_content)

        return SymbolDetail(
            symbol_id=0,
            file_id=file_id or 0,  # Should match file_id from database
            name=self.name or "unknown",
            symbol_type=symbol_type,
            line_number=self.line_number,
            end_line_number=self.end_line_number,
            parent_scope=self.metadata.get("parent_scope") if self.metadata else None,
            docstring=docstring,
            content=symbol_content,
            signature=self.signature,
        )

    def _extract_docstring(self, content: str, language: str) -> str | None:
        """Extract docstring from a symbol.

        Args:
            content: Full file content.
            language: Programming language.

        Returns:
            Docstring if found, None otherwise.
        """
        # This is a simplified implementation
        # A full implementation would parse the AST to find docstring nodes
        # For now, return None as docstring extraction is complex
        return None

    def _extract_symbol_content(self, content: str) -> str:
        """Extract the code content for a symbol.

        Args:
            content: Full file content.

        Returns:
            Symbol code content.
        """
        lines = content.split("\n")
        start_line = self.line_number - 1  # Convert to 0-based
        end_line = self.end_line_number if self.end_line_number else start_line + 1

        if start_line < 0 or start_line >= len(lines):
            return ""

        # Extract lines from start to end
        symbol_lines = lines[start_line:end_line]
        return "\n".join(symbol_lines)


class FileMetadata(BaseModel):
    """Metadata about a source file."""

    path: str
    language: str
    last_modified: int
    last_indexed: int
    hash: str
    size_bytes: int = 0
    content: str | None = None


@dataclass
class CallRelationship:
    """Represents a caller-callee relationship."""

    caller_name: str | None
    caller_line: int
    callee_name: str | None
    callee_line: int
    relationship_type: str
    language: str
    file_path: str


class SymbolResult(BaseModel):
    """Search result for a symbol."""

    symbol_id: int
    name: str
    symbol_type: SymbolType
    file_path: str
    line_number: int
    end_line_number: int | None = None
    parent_scope: str | None = None
    docstring: str | None = None
    signature: str | None = None
    rank: float = 0.0


class ImportInfo(BaseModel):
    """Information about an import statement."""

    file_id: int
    import_statement: str
    import_type: ImportType
    imported_symbols: list[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    """Extracted project metadata."""

    name: str | None = None
    description: str | None = None
    package_manager: str | None = None
    core_frameworks: set[str] = Field(default_factory=set)
    build_backend: str | None = None
    raw_dependencies: set[str] = Field(default_factory=set)
    clean_dependencies: set[str] = Field(default_factory=set)
    linter_formatter: set[str] = Field(default_factory=set)
    manifest_files: set[str] = Field(default_factory=set)
    test_frameworks: set[str] = Field(default_factory=set)
    license: str | None = None
    language_version: str | None = None
    language_version_edition: str | None = None
    available_scripts: list[str] = Field(default_factory=list)

    ts_config_target: str | None = None
    ts_config_module: str | None = None
    ts_config_module_resolution: str | None = None
    ts_config_jsx_mode: str | None = None
    ts_config_strict_mode: bool | None = None
    ts_config_path_aliases: list[str] = Field(default_factory=list)

    is_workspace: bool | None = None
    workspace_members: set[str] = Field(default_factory=set)
