from __future__ import annotations

import asyncio
import fnmatch
import json
import re
from pathlib import Path
from typing import Any

from local_coding_assistant.tools_platform.backends.ignore_policy import (
    is_ignored,
    load_ignore_specs,
)
from local_coding_assistant.tools_platform.tools.search_text import (
    SearchTextMatch,
    SearchTextOutput,
)


async def search_with_ripgrep(
    query: str,
    path: Path,
    pattern: str,
    includes: list[str],
    excludes: list[str],
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    max_bytes: int,
    workspace_root: Path,
    include_ignored: bool = False,
    ignore_files: list[str] | None = None,
) -> SearchTextOutput:
    """Search using ripgrep command.

    Args:
        query: Search query
        path: Path to search
        pattern: Pattern type (literal or regex)
        includes: Glob patterns to include
        excludes: Glob patterns to exclude
        case_sensitive: Case-sensitive search
        context_lines: Number of context lines
        max_matches: Maximum number of matches
        max_bytes: Maximum output bytes
        workspace_root: Workspace root path
        include_ignored: Include ignored files
        ignore_files: List of ignore files

    Returns:
        SearchTextOutput with matches
    """
    cmd = _build_ripgrep_command(
        query=query,
        path=path,
        pattern=pattern,
        case_sensitive=case_sensitive,
        context_lines=context_lines,
        max_matches=max_matches,
        includes=includes,
        excludes=excludes,
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace_root),
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # No matches or error
        if process.returncode == 1:
            return SearchTextOutput(
                matches=[],
                total_matches=0,
                truncated=False,
                next_offset=None,
            )
        raise RuntimeError(f"ripgrep failed: {stderr.decode('utf-8', errors='ignore')}")

    # Parse match records. ripgrep nests match fields under a "data" key:
    # {"type":"match","data":{"path":{"text":...},"line_number":...,
    #  "lines":{"text":...},"submatches":[...]}}
    match_records = _parse_ripgrep_match_records(stdout)
    matches = _build_ripgrep_matches(
        match_records=match_records,
        workspace_root=workspace_root,
        max_matches=max_matches,
    )

    # Apply ignore policy manually (consistent with the Python backend),
    # rather than relying on ripgrep's internal ignore handling.
    if not include_ignored:
        specs = load_ignore_specs(workspace_root, ignore_files)
        matches = [
            m
            for m in matches
            if not is_ignored(workspace_root / m.path, workspace_root, specs)
        ]

    # Populate context lines from file contents (consistent across backends).
    _populate_ripgrep_context(
        matches=matches,
        workspace_root=workspace_root,
        context_lines=context_lines,
    )

    total_bytes = sum(len(m.model_dump_json()) for m in matches)
    truncated = len(matches) >= max_matches or total_bytes >= max_bytes

    return SearchTextOutput(
        matches=matches,
        total_matches=len(matches),
        truncated=truncated,
        next_offset=len(matches) if truncated else None,
    )


def _build_ripgrep_command(
    *,
    query: str,
    path: Path,
    pattern: str,
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    includes: list[str],
    excludes: list[str],
) -> list[str]:
    """Build the ripgrep command line.

    ``--no-ignore`` disables ripgrep's internal ignore policy so ignore
    handling is done explicitly (and identically) via ignore_policy.py in
    both the ripgrep and Python backends.
    """
    cmd = ["rg", "--no-ignore"]

    if pattern == "literal":
        cmd.append("--fixed-strings")
    else:
        cmd.append("--regex")

    if not case_sensitive:
        cmd.append("--ignore-case")

    cmd.extend(["--context", str(context_lines)])
    cmd.extend(["--json", "--line-number"])
    cmd.extend(["--max-count", str(max_matches)])

    for include in includes:
        cmd.extend(["--glob", include])
    for exclude in excludes:
        cmd.extend(["--glob", f"!{exclude}"])

    cmd.append(query)
    cmd.append(str(path))
    return cmd


def _parse_ripgrep_match_records(stdout: bytes) -> list[dict[str, Any]]:
    """Parse ripgrep JSON output, returning only ``match`` records."""
    records: list[dict[str, Any]] = []
    for line in stdout.decode("utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "match":
            records.append(data)
    return records


def _build_ripgrep_matches(
    *,
    match_records: list[dict[str, Any]],
    workspace_root: Path,
    max_matches: int,
) -> list[SearchTextMatch]:
    """Convert ripgrep match records into :class:`SearchTextMatch` objects."""
    matches: list[SearchTextMatch] = []
    for data in match_records:
        if len(matches) >= max_matches:
            break
        md = data["data"]
        path_str = md["path"]["text"]
        line_number = md["line_number"]
        line_content = md["lines"]["text"].rstrip("\n")

        column = 0
        if md.get("submatches"):
            column = md["submatches"][0]["start"]

        rel_path = str(Path(path_str).relative_to(workspace_root))
        matches.append(
            SearchTextMatch(
                path=rel_path,
                line_number=line_number,
                column=column,
                line_content=line_content,
                context_before=[],
                context_after=[],
            )
        )
    return matches


def _populate_ripgrep_context(
    *,
    matches: list[SearchTextMatch],
    workspace_root: Path,
    context_lines: int,
) -> None:
    """Populate context lines for each match from on-disk file contents."""
    path_contents: dict[str, list[str]] = {}
    for m in matches:
        if m.path not in path_contents:
            try:
                path_contents[m.path] = (
                    (workspace_root / m.path)
                    .read_text(encoding="utf-8", errors="ignore")
                    .splitlines()
                )
            except Exception:
                path_contents[m.path] = []
    for m in matches:
        lines = path_contents.get(m.path, [])
        idx = m.line_number - 1
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + 1 + context_lines)
        m.context_before = lines[start:idx]
        m.context_after = lines[idx + 1 : end]


async def search_with_python(
    query: str,
    path: Path,
    pattern: str,
    includes: list[str],
    excludes: list[str],
    case_sensitive: bool,
    context_lines: int,
    max_matches: int,
    max_bytes: int,
    workspace_root: Path,
    include_ignored: bool = False,
    ignore_files: list[str] | None = None,
) -> SearchTextOutput:
    """Search using Python implementation.

    Args:
        query: Search query
        path: Path to search
        pattern: Pattern type (literal or regex)
        includes: Glob patterns to include
        excludes: Glob patterns to exclude
        case_sensitive: Case-sensitive search
        context_lines: Number of context lines
        max_matches: Maximum number of matches
        max_bytes: Maximum output bytes
        workspace_root: Workspace root path
        include_ignored: Include ignored files
        ignore_files: List of ignore files

    Returns:
        SearchTextOutput with matches
    """
    regex = _compile_search_pattern(
        pattern=pattern, query=query, case_sensitive=case_sensitive
    )

    specs: list = (
        [] if include_ignored else load_ignore_specs(workspace_root, ignore_files)
    )

    matches: list[SearchTextMatch] = []
    total_bytes = 0

    # Walk directory
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip ignored files if not included
        if not include_ignored and is_ignored(file_path, workspace_root, specs):
            continue

        rel_path = file_path.relative_to(workspace_root)

        # Apply include/exclude filters
        if _path_matches_filters(rel_path, includes, excludes):
            continue

        # Read file
        lines = _read_file_lines(file_path)
        if lines is None:
            continue

        # Search for matches
        remaining = max_matches - len(matches)
        file_matches = _find_matches_in_lines(
            lines=lines,
            regex=regex,
            context_lines=context_lines,
            rel_path=str(rel_path),
            remaining=remaining,
        )
        matches.extend(file_matches)
        total_bytes += sum(len(m.model_dump_json()) for m in file_matches)

        if len(matches) >= max_matches or total_bytes >= max_bytes:
            break

    truncated = len(matches) >= max_matches or total_bytes >= max_bytes

    return SearchTextOutput(
        matches=matches,
        total_matches=len(matches),
        truncated=truncated,
        next_offset=len(matches) if truncated else None,
    )


def _compile_search_pattern(
    *, pattern: str, query: str, case_sensitive: bool
) -> re.Pattern[str]:
    """Compile the search regex, escaping literal queries."""
    flags = 0 if case_sensitive else re.IGNORECASE
    if pattern == "literal":
        regex_pattern = re.escape(query)
    else:
        regex_pattern = query
    try:
        return re.compile(regex_pattern, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}") from e


def _path_matches_filters(
    rel_path: Path, includes: list[str], excludes: list[str]
) -> bool:
    """Return True when the path should be skipped based on include/exclude globs."""
    rel_str = str(rel_path)
    if includes and not any(fnmatch.fnmatch(rel_str, inc) for inc in includes):
        return True
    if excludes and any(fnmatch.fnmatch(rel_str, exc) for exc in excludes):
        return True
    return False


def _read_file_lines(file_path: Path) -> list[str] | None:
    """Read a file's lines, returning None when it cannot be read."""
    try:
        with file_path.open(
            mode="r", encoding="utf-8", newline="", errors="ignore"
        ) as f:
            return f.read().splitlines()
    except Exception:
        return None


def _find_matches_in_lines(
    *,
    lines: list[str],
    regex: re.Pattern[str],
    context_lines: int,
    rel_path: str,
    remaining: int,
) -> list[SearchTextMatch]:
    """Find regex matches within a file's lines, up to ``remaining`` results."""
    matches: list[SearchTextMatch] = []
    for line_idx, line in enumerate(lines, start=1):
        if len(matches) >= remaining:
            break
        for match in regex.finditer(line):
            if len(matches) >= remaining:
                break
            start_ctx = max(0, line_idx - 1 - context_lines)
            end_ctx = min(len(lines), line_idx + context_lines)
            matches.append(
                SearchTextMatch(
                    path=rel_path,
                    line_number=line_idx,
                    column=match.start(),
                    line_content=line,
                    context_before=lines[start_ctx : line_idx - 1],
                    context_after=lines[line_idx:end_ctx],
                )
            )
    return matches
