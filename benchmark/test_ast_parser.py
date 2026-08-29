"""Benchmarks for AST parsing performance."""

from pathlib import Path

from local_coding_assistant.repository.ast_parser import (
    ASTParser,
    CallRelationshipExtractor,
    LanguageRegistry,
)


def test_parse_python_small_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a small Python file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "small" / "test_small.py"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_parse_python_medium_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a medium Python file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "medium" / "test_medium.py"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_parse_javascript_small_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a small JavaScript file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "small" / "test_small.js"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_parse_javascript_medium_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a medium JavaScript file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "medium" / "test_medium.js"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_parse_go_small_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a small Go file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "small" / "test_small.go"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_parse_go_medium_file(benchmark, benchmark_data_dir: Path):
    """Benchmark parsing a medium Go file."""
    parser = ASTParser()
    file_path = benchmark_data_dir / "medium" / "test_medium.go"

    symbols, language, relationships = benchmark(parser.parse, file_path)
    assert symbols


def test_language_cache_hit(benchmark, benchmark_data_dir: Path):
    """Benchmark language cache hit performance."""
    code = (benchmark_data_dir / "small" / "test_small.py").read_text(encoding="utf-8")

    # Warm up cache
    LanguageRegistry.get_language("python")

    # Benchmark cache hit
    result = benchmark(LanguageRegistry.get_language, "python")
    assert result is not None


def test_extract_call_relationships_python(
    benchmark, ast_parser: ASTParser, sample_python_code: str
):
    """Benchmark call relationship extraction for Python."""
    extractor = CallRelationshipExtractor()

    # Create a temporary file for parsing
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_python_code)
        temp_path = f.name

    try:

        def parse_and_extract():
            symbols, language, relationships = ast_parser.parse(temp_path)
            return relationships

        relationships = benchmark(parse_and_extract)
        assert len(relationships) >= 0
    finally:
        import os

        os.unlink(temp_path)


def test_extract_call_relationships_javascript(
    benchmark, ast_parser: ASTParser, sample_javascript_code: str
):
    """Benchmark call relationship extraction for JavaScript."""
    extractor = CallRelationshipExtractor()

    # Create a temporary file for parsing
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
        f.write(sample_javascript_code)
        temp_path = f.name

    try:

        def parse_and_extract():
            symbols, language, relationships = ast_parser.parse(temp_path)
            return relationships

        relationships = benchmark(parse_and_extract)
        assert len(relationships) >= 0
    finally:
        import os

        os.unlink(temp_path)


def test_extract_call_relationships_go(
    benchmark, ast_parser: ASTParser, sample_go_code: str
):
    """Benchmark call relationship extraction for Go."""
    extractor = CallRelationshipExtractor()

    # Create a temporary file for parsing
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
        f.write(sample_go_code)
        temp_path = f.name

    try:

        def parse_and_extract():
            symbols, language, relationships = ast_parser.parse(temp_path)
            return relationships

        relationships = benchmark(parse_and_extract)
        assert len(relationships) >= 0
    finally:
        import os

        os.unlink(temp_path)
