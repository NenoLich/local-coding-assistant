from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import pathspec
except ImportError:  # pragma: no cover - optional dependency
    pathspec = None  # type: ignore[assignment]


def _hash_file(path: Path) -> str:
    """Return a content hash for an ignore file."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_ignore_files(
    workspace_root: Path, ignore_files: list[str] | None = None
) -> list[Path]:
    """Collect ignore file paths in configured precedence order.

    For each configured ignore filename (lower index = lower priority),
    collects instances from the upward chain (workspace root -> filesystem
    root) and nested instances via rglob. Returns paths in loading order
    so that later (higher priority) files are considered after earlier ones.
    """
    if ignore_files is None:
        ignore_files = [".gitignore"]

    collected: list[Path] = []
    seen: set[Path] = set()

    for ignore_file in ignore_files:
        current = workspace_root.resolve()
        while True:
            candidate = current / ignore_file
            if candidate.exists() and candidate not in seen:
                collected.append(candidate)
                seen.add(candidate)
            if current.parent == current:
                break
            current = current.parent

        try:
            for candidate in workspace_root.rglob(ignore_file):
                if candidate not in seen:
                    collected.append(candidate)
                    seen.add(candidate)
        except Exception:  # noqa: S110 - filesystem errors during spec loading are non-fatal
            pass

    return collected


def _load_specs(ignore_paths: list[Path]) -> list:
    """Parse ignore files into (spec, base_dir) tuples."""
    specs: list = []
    if pathspec is None:
        return specs
    for ignore_path in ignore_paths:
        try:
            lines = ignore_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
            base_dir = ignore_path.parent
            spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
            specs.append((spec, base_dir))
        except Exception:  # noqa: S112 - skip malformed ignore files
            continue
    return specs


def is_ignored(path: Path, workspace_root: Path, specs: list) -> bool:
    """Check if a path is ignored based on loaded ignore specs."""
    if not specs:
        return False

    try:
        path.relative_to(workspace_root)
    except ValueError:
        return True

    for spec, base_dir in specs:
        try:
            path_rel_to_ignore = path.relative_to(base_dir)
            path_str = str(path_rel_to_ignore)
            if spec.match_file(path_str):
                return True
        except ValueError:
            continue

    return False


class IgnoreSpecsCache:
    """Caches ignore specs per workspace, invalidating on content change.

    Keyed by (workspace_root, ignore_files). On each call the current set of
    ignore files and their content hashes are compared against the cached
    snapshot; specs are only re-parsed when the manifest changed.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[list, dict[str, str]]] = {}

    def specs(
        self,
        workspace_root: Path,
        ignore_files: list[str] | None = None,
    ) -> list:
        """Return cached ignore specs, re-parsing only when files changed."""
        if ignore_files is None:
            ignore_files = [".gitignore"]

        key = (str(workspace_root.resolve()), tuple(ignore_files))
        ignore_paths = collect_ignore_files(workspace_root, ignore_files)
        current_hashes = {str(p): _hash_file(p) for p in ignore_paths}

        cached = self._cache.get(key)
        if cached is not None:
            cached_specs, cached_hashes = cached
            if cached_hashes == current_hashes:
                return cached_specs

        specs = _load_specs(ignore_paths)
        self._cache[key] = (specs, current_hashes)
        return specs

    def clear(self) -> None:
        """Clear all cached specs (mainly for tests)."""
        self._cache.clear()


# Default cache used by the module-level convenience functions.
_default_cache = IgnoreSpecsCache()


def load_ignore_specs(
    workspace_root: Path, ignore_files: list[str] | None = None
) -> list:
    """Load cached ignore specs using the default cache."""
    return _default_cache.specs(workspace_root, ignore_files)


def clear_ignore_specs_cache() -> None:
    """Clear the default ignore specs cache (mainly for tests)."""
    _default_cache.clear()
