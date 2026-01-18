"""Pytest configuration for unit tests."""

from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest with custom options and settings."""
    # Add a marker for golden tests
    config.addinivalue_line(
        "markers",
        "golden_test: mark test as a golden test (updates golden files)",
    )


def pytest_addoption(parser):
    """Add custom command line options for tests."""
    # Only add the option if it doesn't already exist
    if not any(opt.dest == "update_goldens" for opt in parser._anonymous.options):
        parser.addoption(
            "--update-goldens",
            action="store_true",
            default=False,
            help="Update golden test files",
        )


@pytest.fixture
def golden(request):
    """Fixture for golden tests."""

    class Golden:
        def __init__(self, test_name):
            self.test_name = test_name
            self.golden_dir = Path(__file__).parent / "golden"
            self.golden_dir.mkdir(exist_ok=True)
            self.golden_file = self.golden_dir / f"{test_name}.golden"

        def assert_match(self, actual, update=None, normalize=False):
            """Assert that the actual output matches the golden file."""
            update = (
                update
                if update is not None
                else request.config.getoption("--update-goldens")
            )

            if update or not self.golden_file.exists():
                self.golden_file.parent.mkdir(parents=True, exist_ok=True)
                self.golden_file.write_text(actual, encoding="utf-8")
                # Instead of skipping, we can just continue or let the user know
                print(f"\nGolden file updated: {self.golden_file}")
                return

            expected = self.golden_file.read_text(encoding="utf-8")

            if normalize:
                import re

                actual = re.sub(r"\s+", " ", actual).strip()
                expected = re.sub(r"\s+", " ", expected).strip()

            assert actual == expected, (
                f"Output does not match golden file. "
                f"Run with --update-goldens to update golden files.\n"
                f"Expected:\n{expected}\n\nGot:\n{actual}"
            )

    return Golden(request.node.name)
