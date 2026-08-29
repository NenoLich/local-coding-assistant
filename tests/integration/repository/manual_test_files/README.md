# Manual Test Files Directory

This directory is used for manually testing the repository context pipeline on arbitrary files.

## How to Use

1. **Add test files**: Drop any files you want to test into this directory.

2. **Run the standalone script** (recommended for quick testing):
   ```bash
   uv run python scripts/test_repo_context_manual.py
   ```

   Or run the pytest integration test:
   ```bash
   pytest tests/integration/repository/test_manual_files.py::test_manual_files_pipeline -v -s
   ```

3. **View output**: The script will show you:
   - AST parsing results (symbols found, call relationships)
   - Database indexing
   - Repo map with PageRank ranking
   - Project metadata extraction
   - Final context string

## File Types

### Code Files (AST Parsing + Indexing)
These files are parsed with the AST parser and indexed into the database:
- Python: `.py`
- JavaScript: `.js`, `.jsx`
- TypeScript: `.ts`, `.tsx`
- Rust: `.rs`
- Go: `.go`
- C/C++: `.c`, `.cpp`, `.h`, `.hpp`

### Config Files (Metadata Only)
These files are only used for metadata extraction (not AST parsed):
- Python: `pyproject.toml`, `setup.py`, `requirements*.txt`
- JavaScript/TypeScript: `package.json`, `tsconfig.json`
- Rust: `Cargo.toml`
- Go: `go.mod`
- Java: `pom.xml`, `build.gradle`
- Other: `README.md`, `.gitignore`, etc.

**Important**: Config files must be named exactly as expected by the metadata extractor:
- Use `pyproject.toml` (not `NOT_USED_pyproject.toml`)
- Use `package.json` (not `example_package.json`)
- Use `requirements.txt` or `requirements-dev.txt` (not `my_requirements.txt`)

## Example Files

- `example_python.py` - A sample Python file with classes and functions
- `pyproject.toml` - A sample pyproject.toml for metadata extraction
- `README.md` - Documentation file (metadata only)

## Clean Up

You can add or remove files from this directory at any time. The script automatically processes all files present in the directory.
