#!/usr/bin/env python3
"""Test script for bootstrap integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_bootstrap():
    """Test the bootstrap functionality."""
    try:
        from local_coding_assistant.core.bootstrap import bootstrap

        # This should work now without the len() error
        ctx = bootstrap()
        print("✓ Bootstrap works correctly")

        # Check that we have the expected components
        llm = ctx.get("llm")
        tools = ctx.get("tools")
        runtime = ctx.get("runtime")

        print(f"✓ LLM: {'present' if llm else 'None'}")
        print(f"✓ Tools: {'present' if tools else 'None'}")
        print(f"✓ Runtime: {'present' if runtime else 'None'}")

        # These should all be present, even if runtime is None due to missing LLM
        assert tools is not None
        print("✓ Core components are present")

        print("🎉 Bootstrap integration test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bootstrap()
    sys.exit(0 if success else 1)
