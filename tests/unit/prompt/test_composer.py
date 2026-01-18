"""Unit tests for the PromptComposer class."""

import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Update golden files instead of testing against them",
    )


# Test cases will be automatically discovered and parametrized by the conftest.py
# using the 'case_dir' fixture and pytest_generate_tests


def test_prompt_composition(composer, test_case_dir, load_test_case, request):
    """Test that prompt composition matches expected output for each test case."""
    if test_case_dir is None:
        pytest.skip("No test case directory provided")

    test_data = load_test_case()

    # Generate the prompt
    result = composer.compose(**test_data["input"])

    # Get the expected output file path
    expected_file = test_case_dir / "expected_output.txt"

    # Update golden files if in update mode
    if request.config.getoption("--update-goldens"):
        expected_file.parent.mkdir(parents=True, exist_ok=True)
        with open(expected_file, "w", encoding="utf-8") as f:
            f.write(result)
        return  # Skip assertion when updating

    # Read expected output
    with open(expected_file, "r", encoding="utf-8") as f:
        expected_output = f.read()

    # Verify the result matches expected output
    assert isinstance(result, str), "Composed prompt should be a string"
    assert result == expected_output, (
        f"\n\nGenerated prompt does not match expected output for {test_case_dir.name}.\n"
        f"Expected file: {expected_file}\n"
        "Run with `pytest tests/unit/prompt/test_composer.py -v --update-goldens` to update golden files\n"
        f"Diff (expected vs actual):\n{'=' * 40}\n"
        f"{expected_output}\n"
        f"{'=' * 40}\n"
        f"{result}\n"
        f"{'=' * 40}"
    )


def test_composer_with_default_templates(composer):
    """Test that the composer works with default template values."""
    # Test with minimal input
    result = composer.compose(
        system_core="Test system core", agent_identity="Test agent identity"
    )

    assert isinstance(result, str)
    assert "Test system core" in result
    assert "Test agent identity" in result


def test_composer_with_all_components(composer):
    """Test that the composer works with all optional components."""
    result = composer.compose(
        system_core="Core system instructions",
        agent_identity="Agent identity details",
        execution_rules="Execution rules here",
        constraints=["Constraint 1", "Constraint 2"],
        skills=["Skill 1", "Skill 2"],
        tools=["Tool 1", "Tool 2"],
        examples=["Example 1", "Example 2"],
        memories=["Memory 1", "Memory 2"],
    )

    assert isinstance(result, str)
    assert "Core system instructions" in result
    assert "Agent identity details" in result
    assert "Execution rules here" in result
    assert "Constraint 1" in result
    assert "Skill 1" in result
    assert "Tool 1" in result
    assert "Example 1" in result
    assert "Memory 1" in result
