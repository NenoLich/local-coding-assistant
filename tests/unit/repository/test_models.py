"""Unit tests for repository models."""

import pytest

from local_coding_assistant.repository.models import (
    ASTNode,
    FileMetadata,
    ImportInfo,
    ImportType,
    ProjectInfo,
    SymbolDetail,
    SymbolResult,
    SymbolType,
)


class TestSymbolType:
    """Test cases for SymbolType enum."""

    def test_symbol_type_values(self) -> None:
        """Test that symbol type enum has correct values."""
        assert SymbolType.FUNCTION == "function"
        assert SymbolType.CLASS == "class"
        assert SymbolType.METHOD == "method"
        assert SymbolType.VARIABLE == "variable"
        assert SymbolType.IMPORT == "import"


class TestImportType:
    """Test cases for ImportType enum."""

    def test_import_type_values(self) -> None:
        """Test that import type enum has correct values."""
        assert ImportType.MODULE == "module"
        assert ImportType.FROM_IMPORT == "from_import"
        assert ImportType.RELATIVE == "relative"


class TestASTNode:
    """Test cases for ASTNode dataclass."""

    def test_ast_node_creation(self) -> None:
        """Test creating an AST node."""
        node = ASTNode(
            type="function_definition",
            name="test_function",
            line_number=10,
            end_line_number=20,
            column=4,
            children=None,
            metadata={"symbol_type": "function"},
        )
        assert node.type == "function_definition"
        assert node.name == "test_function"
        assert node.line_number == 10
        assert node.end_line_number == 20
        assert node.column == 4


class TestFileMetadata:
    """Test cases for FileMetadata model."""

    def test_file_metadata_creation(self) -> None:
        """Test creating file metadata."""
        metadata = FileMetadata(
            path="test.py",
            language="python",
            last_modified=1234567890,
            last_indexed=1234567890,
            hash="abc123",
            size_bytes=100,
        )
        assert metadata.path == "test.py"
        assert metadata.language == "python"
        assert metadata.last_modified == 1234567890
        assert metadata.size_bytes == 100


class TestSymbolDetail:
    """Test cases for SymbolDetail model."""

    def test_symbol_detail_creation(self) -> None:
        """Test creating symbol detail."""
        symbol = SymbolDetail(
            symbol_id=1,
            file_id=10,
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            line_number=10,
            end_line_number=20,
            parent_scope=None,
            docstring="A test function",
            content="def test_function(): pass",
            signature="def test_function() -> None",
        )
        assert symbol.symbol_id == 1
        assert symbol.file_id == 10
        assert symbol.name == "test_function"
        assert symbol.symbol_type == SymbolType.FUNCTION
        assert symbol.signature == "def test_function() -> None"


class TestSymbolResult:
    """Test cases for SymbolResult model."""

    def test_symbol_result_creation(self) -> None:
        """Test creating symbol result."""
        result = SymbolResult(
            symbol_id=1,
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            file_path="test.py",
            line_number=10,
            parent_scope=None,
            docstring="A test function",
            signature="def test_function() -> None",
            rank=0.95,
        )
        assert result.symbol_id == 1
        assert result.name == "test_function"
        assert result.file_path == "test.py"
        assert result.signature == "def test_function() -> None"
        assert result.rank == 0.95


class TestImportInfo:
    """Test cases for ImportInfo model."""

    def test_import_info_creation(self) -> None:
        """Test creating import info."""
        import_info = ImportInfo(
            file_id=10,
            import_statement="import os",
            import_type=ImportType.MODULE,
            imported_symbols=[],
        )
        assert import_info.file_id == 10
        assert import_info.import_statement == "import os"
        assert import_info.import_type == ImportType.MODULE


class TestProjectInfo:
    """Test cases for ProjectInfo model."""

    def test_project_info_creation(self) -> None:
        """Test creating project info."""
        info = ProjectInfo(
            name="test_project",
            description="A test project",
            package_manager="pip",
            core_frameworks={"Django", "FastAPI"},
            build_backend="setuptools",
            clean_dependencies={"requests", "pytest"},
            test_frameworks={"pytest"},
            license="MIT",
        )
        assert info.name == "test_project"
        assert len(info.core_frameworks) == 2
        assert info.clean_dependencies == {"requests", "pytest"}
        assert info.license == "MIT"

    def test_project_info_defaults(self) -> None:
        """Test project info with default values."""
        info = ProjectInfo()
        assert info.name is None
        assert info.description is None
        assert info.core_frameworks == set()
        assert info.clean_dependencies == set()
