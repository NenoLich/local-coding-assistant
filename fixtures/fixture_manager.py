"""Fixture management utilities for test isolation."""

from pathlib import Path


def snapshot_files(path: Path) -> set[str]:
    """Get a snapshot of all files in the directory."""
    files = set()
    for f in path.rglob("*"):
        if f.is_file() and not any(part.startswith(".") for part in f.parts):
            try:
                rel = f.relative_to(path)
                files.add(str(rel))
            except ValueError:
                pass
    return files


def snapshot_file_contents(path: Path) -> dict[str, str]:
    """Get a snapshot of file contents."""
    contents = {}
    for f in path.rglob("*"):
        if f.is_file() and not any(part.startswith(".") for part in f.parts):
            try:
                rel = f.relative_to(path)
                contents[str(rel)] = f.read_text(encoding="utf-8", errors="ignore")
            except (ValueError, OSError):
                pass
    return contents


def restore_fixture_snapshot(fixture_path: Path, snapshot: dict[str, str]) -> None:
    """Restore fixture to a previous state from a content snapshot.

    Removes files not in snapshot, restores content of files in snapshot.
    """
    # Remove files not in snapshot
    for f in fixture_path.rglob("*"):
        if f.is_file() and not any(part.startswith(".") for part in f.parts):
            try:
                rel = f.relative_to(fixture_path)
                if str(rel) not in snapshot:
                    f.unlink()
            except (ValueError, OSError):
                pass

    # Restore files from snapshot
    for rel_path, content in snapshot.items():
        target = fixture_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


class FixtureManager:
    """Context manager for running code against a fixture with automatic restore.

    Usage:
        with FixtureManager(fixture_path) as fm:
            # Run agent/LLM against fixture
            # ... make changes ...
            # Automatically restored on exit

    Or manually:
        fm = FixtureManager(fixture_path)
        snapshot = fm.snapshot()
        try:
            # run agent
        finally:
            fm.restore(snapshot)
    """

    def __init__(self, fixture_path: Path):
        self.fixture_path = Path(fixture_path)
        self._snapshot: dict[str, str] | None = None

    def snapshot(self) -> dict[str, str]:
        """Take a snapshot of current fixture state."""
        self._snapshot = snapshot_file_contents(self.fixture_path)
        return self._snapshot

    def restore(self, snapshot: dict[str, str] | None = None) -> None:
        """Restore fixture to given snapshot (or last taken snapshot)."""
        if snapshot is None:
            snapshot = self._snapshot
        if snapshot is not None:
            restore_fixture_snapshot(self.fixture_path, snapshot)

    def __enter__(self) -> "FixtureManager":
        self.snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.restore()
        return False  # Don't suppress exceptions

    def get_changes(
        self, snapshot: dict[str, str] | None = None
    ) -> tuple[list[str], list[str]]:
        """Get files created and modified since snapshot."""
        if snapshot is None:
            snapshot = self._snapshot
        if snapshot is None:
            return [], []

        current = snapshot_file_contents(self.fixture_path)
        before_files = set(snapshot.keys())
        after_files = set(current.keys())

        created = list(after_files - before_files)
        modified = [f for f in before_files & after_files if snapshot[f] != current[f]]

        return created, modified
