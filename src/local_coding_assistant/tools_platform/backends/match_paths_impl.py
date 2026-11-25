from __future__ import annotations

from pathlib import Path

from local_coding_assistant.tools_platform.backends.ignore_policy import (
    is_ignored,
    load_ignore_specs,
)
from local_coding_assistant.tools_platform.tools.match_paths import (
    MatchPathResult,
    MatchPathsOutput,
)


def _should_include_path(
    path: Path,
    workspace_root: Path,
    include_hidden: bool,
    include_ignored: bool,
    file_type: str | None,
    specs: list,
    max_results: int,
    current_count: int,
) -> bool:
    """Decide whether a glob-matched path should be included in results."""
    # Check result limit
    if current_count >= max_results:
        return False

    # Skip hidden files if not included
    if not include_hidden:
        if any(part.startswith(".") for part in path.parts):
            return False

    # Skip ignored files if not included
    if not include_ignored:
        if is_ignored(path, workspace_root, specs):
            return False

    # Filter by file type if specified
    if file_type:
        if path.is_file() and path.suffix.lstrip(".") != file_type:
            return False

    # Must be a file or directory
    if not (path.is_file() or path.is_dir()):
        return False

    # Must be within workspace
    try:
        path.relative_to(workspace_root)
    except ValueError:
        # Path is outside workspace, skip
        return False

    return True


def _build_match_result(path: Path, workspace_root: Path) -> MatchPathResult | None:
    """Build a MatchPathResult for a path, or None if it cannot be classified."""
    if path.is_file():
        path_type = "file"
        size = path.stat().st_size
    elif path.is_dir():
        path_type = "directory"
        size = None
    else:
        return None

    try:
        rel_path = path.relative_to(workspace_root)
    except ValueError:
        return None

    return MatchPathResult(
        path=str(rel_path),
        type=path_type,
        size=size,
    )


async def match_paths(
    pattern: str,
    search_path: Path,
    workspace_root: Path,
    include_hidden: bool = False,
    include_ignored: bool = False,
    file_type: str | None = None,
    max_results: int = 200,
    ignore_files: list[str] | None = None,
) -> MatchPathsOutput:
    """Implementation of match_paths tool

    Args:
        pattern: Glob pattern to match files
        search_path: Search path
        workspace_root: Root of the workspace
        include_hidden: Include hidden files/directories
        include_ignored: Include ignored files
        file_type: Filter by file extension (e.g., 'py')
        max_results: Maximum number of results to return
        ignore_files: List of ignore filenames in precedence order

    Returns:
        MatchPathsOutput with matches

    Raises:
        FileNotFoundError: If search path does not exist
    """
    if not search_path.exists():
        return MatchPathsOutput(
            matches=[],
            total_matches=0,
            truncated=False,
        )

    # Load ignore specs unless ignored files are explicitly included
    specs: list = []
    if not include_ignored:
        specs = load_ignore_specs(workspace_root, ignore_files)

    matches = []
    total_matches = 0

    # Use glob to match files
    if not pattern.startswith("**"):
        # If pattern doesn't have recursive wildcard, add it for recursive search
        pattern = f"**/{pattern}"

    for path in search_path.glob(pattern):
        if not _should_include_path(
            path,
            workspace_root,
            include_hidden,
            include_ignored,
            file_type,
            specs,
            max_results,
            len(matches),
        ):
            continue

        result = _build_match_result(path, workspace_root)
        if result is None:
            continue

        matches.append(result)
        total_matches += 1

    matches.sort(key=lambda m: m.path)

    truncated = len(matches) >= max_results

    return MatchPathsOutput(
        matches=matches,
        total_matches=total_matches,
        truncated=truncated,
    )
