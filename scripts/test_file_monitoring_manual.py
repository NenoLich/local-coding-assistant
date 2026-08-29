#!/usr/bin/env python
"""Manual test script for file monitoring system.

This script starts the file monitoring service on a test directory and prints
file change events to stdout. Use this to manually verify the file monitoring
system is working correctly.

Usage:
    python scripts/test_file_monitoring_manual.py

Then, in another terminal, manipulate files in tests/data/file_monitoring_test/
to see the events being detected.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from local_coding_assistant.repository.file_monitoring import (
    FileChangeEvent,
    FileChangeType,
    FileMonitoringService,
)
from local_coding_assistant.repository.file_scope import FileScope


def print_event(event: FileChangeEvent) -> None:
    """Print file change event to stdout."""
    change_type_emoji = {
        FileChangeType.MODIFIED: "✏️",
        FileChangeType.CREATED: "➕",
        FileChangeType.DELETED: "🗑️",
        FileChangeType.MOVED: "📦",
    }
    emoji = change_type_emoji.get(event.change_type, "❓")

    tracked_status = "tracked" if event.is_tracked else "not tracked"
    print(
        f"{emoji} [{event.change_type.value.upper()}] {event.path} ({tracked_status})"
    )


def main() -> None:
    """Main entry point."""
    # Set up test directory
    test_dir = Path(__file__).parent.parent / "tests" / "data" / "file_monitoring_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Starting file monitoring on: {test_dir}")
    print(f"📁 Test directory: {test_dir.absolute()}")
    print("\n📝 Instructions:")
    print("  1. Open another terminal")
    print("  2. Create, modify, or delete files in the test directory")
    print("  3. Watch for events printed below")
    print("  4. Press Ctrl+C to stop monitoring\n")
    print("=" * 60)
    print("Monitoring started. Waiting for file changes...")
    print("=" * 60)

    # Initialize file monitoring service
    file_scope = FileScope(
        test_dir, tracking_strategy="walk_only", ignore_dirs={".git"}
    )
    service = FileMonitoringService(
        target_project_root=test_dir,
        debounce_window=0.5,  # Shorter debounce for manual testing
        file_scope=file_scope,
    )

    # Register callback to print events
    service.register_notification_callback(print_event)

    # Start monitoring
    service.start()

    try:
        # Keep running until interrupted
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping file monitoring...")
        service.stop()
        print("✅ Monitoring stopped. Goodbye!")

    finally:
        service.stop()


if __name__ == "__main__":
    main()
