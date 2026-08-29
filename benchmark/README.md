# Repository Context Service Benchmarks

This directory contains performance benchmarks for the repository context service components.

## What is Benchmarked

- **AST Parsing**: Tree-sitter-based parsing performance across languages (Python, JavaScript, Go)
- **Database Indexing**: SQLite FTS5 database operations for symbols, imports, and call relationships
- **Call Graph Computation**: NetworkX-based call graph construction and PageRank ranking

## Test Data Structure

```
benchmark/data/
├── small/      # Files < 10KB
│   ├── test_small.py
│   ├── test_small.js
│   └── test_small.go
└── medium/     # Files 10-100KB
    ├── test_medium.py
    ├── test_medium.js
    └── test_medium.go
```

## Running Benchmarks

### Run all benchmarks
```bash
pytest benchmark/ --benchmark-only
```

### Run specific benchmark file
```bash
pytest benchmark/test_ast_parser.py --benchmark-only
pytest benchmark/test_database.py --benchmark-only
pytest benchmark/test_call_graph.py --benchmark-only
```

### Run specific benchmark test
```bash
pytest benchmark/test_ast_parser.py::test_parse_python_small_file --benchmark-only
```

### Compare with previous run
```bash
# First run (saves baseline)
pytest benchmark/ --benchmark-only --benchmark-save=baseline

# Second run (compares with baseline)
pytest benchmark/ --benchmark-only --benchmark-compare=baseline
```

### Generate histogram
```bash
pytest benchmark/ --benchmark-only --benchmark-histogram
```

### Generate detailed output
```bash
pytest benchmark/ --benchmark-only --benchmark-sort=name
```

## Benchmark Configuration

Benchmark settings are configured in `pyproject.toml`:

```toml
[tool.pytest-benchmark]
min_rounds = 5
max_time = 1.0
min_time = 0.005
histogram = true
save_data = true
json_file = ".benchmarks/{benchmark_name}.json"
```

## Understanding Results

Example output:
```
Name (time in us)                     Min          Max      Mean      StdDev    Median      IQR  Outliers  OPS (Kops/s)  Rounds  Iterations
test_parse_python_small_file     577.8000  20,595.8001  674.3572  1,094.4972  598.2500  22.3001      1;48        1.4829     334           1
```

- **Min**: Minimum execution time (best case) - 577.8 microseconds (0.578ms)
- **Max**: Maximum execution time (worst case) - 20,595.8 microseconds (20.6ms)
- **Mean**: Average execution time - 674.4 microseconds (0.674ms)
- **StdDev**: Standard deviation (lower = more consistent) - 1,094.5 microseconds (high variance here)
- **Median**: Median execution time - 598.3 microseconds
- **IQR**: Interquartile range (middle 50% of data) - 22.3 microseconds
- **Outliers**: Number of outliers below/above 1.5 IQR (1 low, 48 high)
- **OPS**: Operations per second (higher = better) - 1,482.9 ops/sec
- **Rounds**: Number of benchmark iterations - 334
- **Iterations**: Number of operations per round - 1

**Time units**: us = microseconds (1/1000 of a millisecond), ms = milliseconds, s = seconds

## Tips for Accurate Results

1. Close heavy applications before running benchmarks
2. Run multiple times to account for system variance
3. Use `--benchmark-warmup` to warm up caches
4. Increase `--benchmark-min-rounds` for more statistical significance
5. For regression testing, consider using an isolated environment

## Adding New Benchmarks

1. Create test data in `benchmark/data/`
2. Add benchmark function to appropriate test file
3. Use `benchmark` fixture from pytest-benchmark
4. Follow naming convention: `test_<component>_<operation>_<size>`

Example:
```python
def test_parse_python_large_file(benchmark, benchmark_data_dir: Path):
    parser = ASTParser()
    code = (benchmark_data_dir / "large" / "test_large.py").read_text(encoding="utf-8")
    result = benchmark(parser.parse, code, "python")
    assert result.symbols
```

## Benchmark Levels

The suite includes three levels of benchmarks:

### 1. Unit-Level Benchmarks
- Individual operations (single file parsing, single DB insert, single relationship)
- Fast, high-iteration tests for micro-optimizations
- Files: `test_ast_parser.py`, `test_database.py`, `test_call_graph.py`

### 2. Integration-Level Benchmarks
- Multi-file operations (10, 50, 100 files)
- Realistic workload simulation
- File: `test_multi_file.py` (planned)

### 3. Full Pipeline Benchmarks
- End-to-end: parse + index + call graph construction
- System-level performance measurement
- File: `test_full_pipeline.py` (planned)

## Database Initialization

DB initialization is **not** included in operation benchmarks. It's measured separately:
- `test_db_initialization` - measures schema creation overhead
- Operation benchmarks use pre-initialized DBs to isolate operation cost

## CI Integration

For CI regression testing:

```bash
# In CI pipeline
pytest benchmark/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:20%
```

This will fail the build if performance regresses by more than 20% compared to baseline.
