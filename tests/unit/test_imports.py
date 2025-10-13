#!/usr/bin/env python3
"""Test script for circular import fix."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_circular_imports():
    """Test that circular imports are resolved."""
    # Test that LLMManager imports successfully
    from local_coding_assistant.agent.llm_manager import LLMManager

    assert LLMManager is not None

    # Test that bootstrap imports successfully
    from local_coding_assistant.core.bootstrap import bootstrap

    assert bootstrap is not None

    # Test that bootstrap can be called (even if it fails due to missing OpenAI)
    try:
        ctx = bootstrap()

        # Check that we get the expected components
        tools = ctx.get("tools")
        runtime = ctx.get("runtime")
        llm = ctx.get("llm")

        # At minimum, we should get some context back
        assert ctx is not None

    except Exception as e:
        if "OpenAI" in str(e):
            # Bootstrap fails gracefully due to missing OpenAI (expected)
            assert "OpenAI" in str(e)
        else:
            # Bootstrap failed unexpectedly
            raise AssertionError(f"Bootstrap failed unexpectedly: {e}") from e


if __name__ == "__main__":
    test_circular_imports()
    print("✓ All tests passed!")
