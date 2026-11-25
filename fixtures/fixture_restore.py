#!/usr/bin/env python
"""Manual fixture restore utility for LOCCA testing.

Usage:
    # Before running agent
    python tests/fixtures/fixture_restore.py snapshot tests/fixtures/projects/python_cli_tool

    # Run your agent
    locca run query "Add export command" --target-project-root tests/fixtures/projects/python_cli_tool -m mock-model

    # After, restore
    python tests/fixtures/fixture_restore.py restore tests/fixtures/projects/python_cli_tool

Or with context manager in Python:
    from tests.fixtures.fixture_manager import FixtureManager
    with FixtureManager(Path("tests/fixtures/projects/python_cli_tool")) as fm:
        # run agent
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.fixture_manager import (
    FixtureManager,
    restore_fixture_snapshot,
    snapshot_file_contents,
)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python fixture_restore.py [snapshot|restore|changes] <fixture_path> [snapshot_file]"
        )
        sys.exit(1)

    action = sys.argv[1]
    fixture_path = Path(sys.argv[2])
    snapshot_file = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else fixture_path / ".fixture_snapshot.json"
    )

    if action == "snapshot":
        state = snapshot_file_contents(fixture_path)
        import json

        snapshot_file.write_text(json.dumps(state, indent=2))
        print(f"Snapshot saved to {snapshot_file}")
        print(f"Files snapshotted: {len(state)}")

    elif action == "restore":
        import json

        if not snapshot_file.exists():
            print(f"Error: Snapshot file not found: {snapshot_file}")
            sys.exit(1)
        state = json.loads(snapshot_file.read_text())
        restore_fixture_snapshot(fixture_path, state)
        print(f"Restored {len(state)} files from {snapshot_file}")

    elif action == "changes":
        import json

        if not snapshot_file.exists():
            print(f"Error: Snapshot file not found: {snapshot_file}")
            sys.exit(1)
        state = json.loads(snapshot_file.read_text())
        fm = FixtureManager(fixture_path)
        created, modified = fm.get_changes(state)
        print(f"Created: {created}")
        print(f"Modified: {modified}")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
