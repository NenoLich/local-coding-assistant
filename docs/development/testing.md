# Testing

LOCCA maintains a comprehensive test suite to ensure reliability and prevent regressions.

## Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── providers/       # Provider-specific tests
│   └── ...              # Other unit tests
├── integration/         # Integration tests
│   └── ...              # Module interaction tests
├── e2e/                 # End-to-end tests
│   ├── conftest.py      # Test fixtures
│   └── ...              # Full workflow tests
└── ...
```

## Running Tests

### All Tests
```bash
# Using taskipy
uv run task test-all

# Using pytest directly
uv run pytest -v
```

### Specific Test Categories
```bash
# Unit tests only
uv run task test-unit

# Integration tests
uv run task test-integration

# End-to-end tests
uv run task test-e2e
```

### Test with Coverage
```bash
uv run pytest --cov=local_coding_assistant --cov-report=html
```

## Writing Tests

### Unit Tests
- Place in `tests/unit/`
- Test individual functions and classes
- Mock external dependencies
- Use `pytest.mark.asyncio` for async tests

### Integration Tests
- Place in `tests/integration/`
- Test module interactions
- May use real dependencies with proper isolation

### End-to-End Tests
- Place in `tests/e2e/`
- Test complete workflows
- Use real configurations where safe

## Test Conventions

### Naming
- Test files: `test_*.py`
- Test functions: `test_*`
- Use descriptive names

### Assertions
- Assert on complete objects rather than individual fields
- Use appropriate assertion methods
- Provide clear failure messages

### Fixtures
- Use centralized fixtures in `conftest.py`
- Share fixtures across test files
- Clean up after tests

## Golden Tests

For tests with complex outputs, use golden tests:

- Store expected outputs in `golden/` directories
- Update goldens when behavior changes intentionally
- Compare against stored expectations

## Mocking

### External Dependencies
- Mock LLM providers for unit tests
- Use `pytest-mock` for patching
- Isolate logic from external services

### Test Data
- Use realistic but deterministic test data
- Avoid hardcoded values where possible
- Use factories for complex objects

## Continuous Integration

Tests run automatically on:
- Pull requests
- Main branch pushes
- Release builds

## Coverage Goals

- Aim for high unit test coverage
- Critical paths should have integration tests
- Key user workflows need e2e tests

## Debugging Tests

### Running Single Tests
```bash
uv run pytest tests/unit/test_specific.py::test_function -v
```

### Debugging Failures
```bash
uv run pytest --pdb tests/failing_test.py
```

### Verbose Output
```bash
uv run pytest -v -s tests/
```

## Contributing Tests

When adding features:
1. Write tests first (TDD)
2. Cover happy path and error cases
3. Update existing tests if behavior changes
4. Run full test suite before submitting

## Test Infrastructure

- **pytest**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **pytest-golden**: Golden test support

## Performance Testing

For performance-critical code:
- Use benchmarks in tests
- Monitor execution times
- Set performance budgets

## Next Steps

- [Contributing](contributing.md)
- [Coding Conventions](coding-conventions.md)
