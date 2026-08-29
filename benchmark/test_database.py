"""Benchmarks for database indexing performance."""

from pathlib import Path

from local_coding_assistant.repository.ast_parser import ASTParser
from local_coding_assistant.repository.database import RepositoryDatabase
from local_coding_assistant.repository.models import (
    FileMetadata,
    ImportInfo,
    ImportType,
    SymbolDetail,
    SymbolType,
)


def test_db_initialization(benchmark, temp_db_path: Path):
    """Benchmark database schema initialization."""

    def init_db():
        db = RepositoryDatabase(temp_db_path)
        # DB initializes on first connection
        return db

    benchmark(init_db)


def test_add_file_metadata(
    benchmark, repository_db: RepositoryDatabase, sample_python_code
):
    """Benchmark adding file metadata to database."""
    metadata = FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1024,
        content=sample_python_code,
    )

    file_id = benchmark(repository_db.add_or_update_file, metadata)
    assert file_id > 0


def test_add_single_symbol(
    benchmark, repository_db: RepositoryDatabase, sample_python_code
):
    """Benchmark adding a single symbol to database."""
    # First add a file
    metadata = FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1024,
        content=sample_python_code,
    )
    file_id = repository_db.add_or_update_file(metadata)

    # Benchmark symbol addition
    def add_symbol():
        symbol = SymbolDetail(
            symbol_id=0,  # Will be assigned by DB
            file_id=file_id,
            name="test_function",
            symbol_type=SymbolType.FUNCTION,
            line_number=10,
            end_line_number=15,
            parent_scope=None,
            docstring="Test function",
            signature="def test_function():",
            content="def test_function():\n    pass",
        )
        return repository_db.add_symbol(symbol)

    symbol_id = benchmark(add_symbol)
    assert symbol_id > 0


def test_add_import(benchmark, repository_db: RepositoryDatabase, sample_python_code):
    """Benchmark adding an import to database."""
    # First add a file
    metadata = FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1024,
        content=sample_python_code,
    )
    file_id = repository_db.add_or_update_file(metadata)

    # Benchmark import addition
    import_info = ImportInfo(
        file_id=file_id,
        import_statement="import os",
        import_type=ImportType.MODULE,
        imported_symbols=["os"],
    )

    import_id = benchmark(repository_db.add_import, import_info)
    assert import_id > 0


def test_add_call_relationship(
    benchmark, repository_db: RepositoryDatabase, sample_python_code
):
    """Benchmark adding a single call relationship to database."""
    # First add a file
    metadata = FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1024,
        content=sample_python_code,
    )
    file_id = repository_db.add_or_update_file(metadata)

    # Benchmark call relationship addition
    call_id = benchmark(
        repository_db.add_call_relationship,
        file_id,
        "caller_func",
        10,
        "callee_func",
        20,
        "call",
        "python",
    )
    assert call_id > 0


def test_add_batch_call_relationships(
    benchmark, repository_db: RepositoryDatabase, sample_python_code
):
    """Benchmark adding multiple call relationships in batch."""
    # First add a file
    metadata = FileMetadata(
        path="test.py",
        language="python",
        last_modified=1234567890,
        last_indexed=1234567890,
        hash="abc123",
        size_bytes=1024,
        content=sample_python_code,
    )
    file_id = repository_db.add_or_update_file(metadata)

    # Create batch of relationships
    relationships = [
        ("caller1", 10, "callee1", 20, "call", "python"),
        ("caller2", 30, "callee2", 40, "call", "python"),
        ("caller3", 50, "callee3", 60, "call", "python"),
        ("caller4", 70, "callee4", 80, "call", "python"),
        ("caller5", 90, "callee5", 100, "call", "python"),
    ]

    benchmark(repository_db.add_call_relationships, file_id, relationships)


def test_index_file_end_to_end_python_small(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a small Python file."""
    file_path = benchmark_data_dir / "small" / "test_small.py"
    code = file_path.read_text(encoding="utf-8")

    def index_file():
        # Parse the file
        symbols, file_meta, relationships = ast_parser.parse(
            file_path, content=code, language="python"
        )

        file_id = repository_db.add_or_update_file(file_meta)

        # Add symbols (convert ASTNode to SymbolDetail)
        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,  # Will be assigned by DB
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            repository_db.add_symbol(symbol)

        # Add call relationships
        if relationships:
            rel_tuples = [
                (
                    rel.caller_name,
                    rel.caller_line,
                    rel.callee_name,
                    rel.callee_line,
                    rel.relationship_type,
                    rel.language,
                )
                for rel in relationships
            ]
            repository_db.add_call_relationships(file_id, rel_tuples)

        return file_id

    file_id = benchmark(index_file)
    assert file_id > 0


def test_index_file_end_to_end_python_medium(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    def index_file():
        # Parse the file
        symbols, metadata, relationships = ast_parser.parse(
            file_path, content=code, language="python"
        )

        file_id = repository_db.add_or_update_file(metadata)

        # Add symbols (convert ASTNode to SymbolDetail)
        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            repository_db.add_symbol(symbol)

        # Add call relationships
        if relationships:
            rel_tuples = [
                (
                    rel.caller_name,
                    rel.caller_line,
                    rel.callee_name,
                    rel.callee_line,
                    rel.relationship_type,
                    rel.language,
                )
                for rel in relationships
            ]
            repository_db.add_call_relationships(file_id, rel_tuples)

        return file_id

    file_id = benchmark(index_file)
    assert file_id > 0


def test_index_file_end_to_end_javascript_small(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a small JavaScript file."""
    file_path = benchmark_data_dir / "small" / "test_small.js"
    code = file_path.read_text(encoding="utf-8")

    def index_file():
        # Parse the file
        symbols, metadata, relationships = ast_parser.parse(
            file_path, content=code, language="javascript"
        )

        file_id = repository_db.add_or_update_file(metadata)

        # Add symbols (convert ASTNode to SymbolDetail)
        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            repository_db.add_symbol(symbol)

        # Add call relationships
        if relationships:
            rel_tuples = [
                (
                    rel.caller_name,
                    rel.caller_line,
                    rel.callee_name,
                    rel.callee_line,
                    rel.relationship_type,
                    rel.language,
                )
                for rel in relationships
            ]
            repository_db.add_call_relationships(file_id, rel_tuples)

        return file_id

    file_id = benchmark(index_file)
    assert file_id > 0


def test_index_file_end_to_end_go_small(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a small Go file."""
    file_path = benchmark_data_dir / "small" / "test_small.go"
    code = file_path.read_text(encoding="utf-8")

    def index_file():
        # Parse the file
        symbols, metadata, relationships = ast_parser.parse(
            file_path, content=code, language="go"
        )

        file_id = repository_db.add_or_update_file(metadata)

        # Add symbols (convert ASTNode to SymbolDetail)
        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            repository_db.add_symbol(symbol)

        # Add call relationships
        if relationships:
            rel_tuples = [
                (
                    rel.caller_name,
                    rel.caller_line,
                    rel.callee_name,
                    rel.callee_line,
                    rel.relationship_type,
                    rel.language,
                )
                for rel in relationships
            ]
            repository_db.add_call_relationships(file_id, rel_tuples)

        return file_id

    file_id = benchmark(index_file)
    assert file_id > 0


def test_add_file_10_medium_files(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)
    metadatas = []

    for file_path in files[:i]:
        symbols, metadata, _ = parser.parse(file_path, content=code, language="python")
        metadatas.append(metadata)

    def add_file():
        # Parse the file
        for meta in metadatas:
            _ = repository_db.add_or_update_file(meta)

        return True

    success = benchmark(add_file)

    assert success


def test_add_files_10_medium_files(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)
    metadatas = []

    for file_path in files[:i]:
        symbols, file_metadata, _ = parser.parse(
            file_path, content=code, language="python"
        )
        metadatas.append(file_metadata)

    def add_file():
        # Parse the file
        _ = repository_db.add_or_update_files(metadatas)

        return True

    success = benchmark(add_file)

    assert success


def test_add_symbol_10_medium_files(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)
    symbols_to_index = []

    for file_path in files[:i]:
        symbols, metadata, _ = parser.parse(file_path, content=code, language="python")
        file_id = repository_db.add_or_update_file(metadata)

        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            symbols_to_index.append(symbol)

    def add_symbols():
        # Parse the file
        for symbol in symbols_to_index:
            repository_db.add_symbol(symbol)

        return True

    success = benchmark(add_symbols)

    assert success


def test_add_symbols_10_medium_files(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)
    symbols_to_index = []

    for file_path in files[:i]:
        symbols, metadata, _ = parser.parse(file_path, content=code, language="python")
        file_id = repository_db.add_or_update_file(metadata)

        for ast_node in symbols:
            symbol_type_str = (
                ast_node.metadata.get("symbol_type", "function")
                if ast_node.metadata
                else "function"
            )
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            symbol = SymbolDetail(
                symbol_id=0,
                file_id=file_id,
                name=ast_node.name or "unknown",
                symbol_type=symbol_type,
                line_number=ast_node.line_number,
                end_line_number=ast_node.end_line_number,
                parent_scope=ast_node.metadata.get("parent_scope")
                if ast_node.metadata
                else None,
                docstring=ast_node.metadata.get("docstring")
                if ast_node.metadata
                else None,
                content=None,
                signature=ast_node.signature,
                file_path=str(file_path),
            )
            symbols_to_index.append(symbol)

    def add_symbols():
        # Parse the file
        repository_db.add_symbols(symbols_to_index)

        return True

    success = benchmark(add_symbols)

    assert success


def test_index_file_call_relationships_10_batch_medium(
    benchmark,
    repository_db: RepositoryDatabase,
    benchmark_data_dir: Path,
    ast_parser: ASTParser,
):
    """Benchmark end-to-end indexing of a medium Python file."""
    file_path = benchmark_data_dir / "medium" / "test_medium.py"
    code = file_path.read_text(encoding="utf-8")

    parser = ASTParser()
    files = list((benchmark_data_dir / "medium").glob("test_medium.*"))
    i = 10
    files_len = len(files)
    files = files * ((i // files_len) + 1)
    relationships = []

    for file_path in files[:i]:
        _, metadata, relationship = parser.parse(
            file_path, content=code, language="python"
        )
        file_id = repository_db.add_or_update_file(metadata)

        relationships.extend(relationship)

    def index_calls():
        # Parse the file
        if relationships:
            rel_tuples = [
                (
                    rel.caller_name,
                    rel.caller_line,
                    rel.callee_name,
                    rel.callee_line,
                    rel.relationship_type,
                    rel.language,
                )
                for rel in relationships
            ]
            repository_db.add_call_relationships(file_id, rel_tuples)

        return True

    success = benchmark(index_calls)

    assert success
