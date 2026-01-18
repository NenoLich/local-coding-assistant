"""Test configuration and fixtures for prompt composition tests."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from jinja2 import Environment, FileSystemLoader

# Use mock implementation for testing
from .mocks import MockPromptComposer as PromptComposer
from local_coding_assistant.config.path_manager import PathManager

# Disable warnings about redefining the PromptComposer name
# pylint: disable=redefined-outer-name


@pytest.fixture(scope="session")
def path_manager():
    """Create a PathManager instance for testing."""
    return PathManager(is_testing=True)


@pytest.fixture
def template_env(path_manager):
    """Create a Jinja2 environment with the template directory."""
    template_dir = path_manager.get_template_dir()
    return Environment(loader=FileSystemLoader(template_dir))


@pytest.fixture
def composer(template_env):
    """Create a MockPromptComposer instance for testing."""
    return PromptComposer(template_env=template_env)


@pytest.fixture
def test_case_dir(request, path_manager):
    """
    Fixture to get the directory of the current test case.

    To use this fixture, a test should be parametrized with 'case_dir' parameter.
    """
    if hasattr(request, "param") and request.param:
        test_dir = (
            path_manager.get_project_root() / "tests" / "unit" / "prompt" / "golden"
        )
        return test_dir / request.param
    return None


@pytest.fixture
def load_test_case(test_case_dir):
    """Load test case data from the golden directory."""
    if test_case_dir is None:
        return {}

    def _load():
        input_file = test_case_dir / "input.json"
        expected_file = test_case_dir / "expected_output.txt"

        with open(input_file) as f:
            input_data = json.load(f)

        with open(expected_file) as f:
            expected_output = f.read().strip()

        return {"input": input_data, "expected_output": expected_output}

    return _load


def pytest_generate_tests(metafunc):
    """
    Generate test cases from golden directory.

    This function is automatically called by pytest to generate test cases
    based on the contents of the golden directory.
    """
    if "test_case_dir" in metafunc.fixturenames:
        golden_dir = Path(__file__).parent / "golden"
        test_cases = [f.name for f in golden_dir.iterdir() if f.is_dir()]
        metafunc.parametrize("test_case_dir", test_cases, indirect=True)
