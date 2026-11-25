"""Line-ending normalization for content crossing the LLM/disk boundary.

This module provides a config-driven normalizer that converts CRLF/CR line
endings to LF so LLM-generated edits match on-disk content regardless of the
file's native line endings.

Baseline hashes and snapshots intentionally remain on raw disk bytes; only
LLM-facing reads and the edit match path are normalized.
"""

from __future__ import annotations

import difflib
from enum import Enum


def _indent(text: str, prefix: str) -> str:
    """Add prefix to each line of text."""
    return "".join(prefix + line for line in text.splitlines(keepends=True))


class LineEnding(str, Enum):
    """Detected line-ending style of a piece of content."""

    LF = "lf"
    CRLF = "crlf"
    CR = "cr"
    NONE = "none"

    @property
    def value_str(self) -> str:
        """The actual string representation for writing to disk."""
        return {
            LineEnding.LF: "\n",
            LineEnding.CRLF: "\r\n",
            LineEnding.CR: "\r",
            LineEnding.NONE: "\n",  # Safe default if content had no newlines
        }[self]


def detect_line_ending(content: str) -> LineEnding:
    """Detect the dominant line ending in content by frequency count."""
    # Sample-based detection for massive buffers
    sample = content[:16384] if len(content) > 16384 else content

    crlf_count = sample.count("\r\n")
    # Total \n minus \r\n gives standalone \n
    lf_count = sample.count("\n") - crlf_count
    # Total \r minus \r\n gives standalone \r
    cr_count = sample.count("\r") - crlf_count

    counts = {
        LineEnding.CRLF: crlf_count,
        LineEnding.LF: lf_count,
        LineEnding.CR: cr_count,
    }

    max_count = max(counts.values())
    if max_count == 0:
        return LineEnding.NONE

    # In case of ties (e.g. 50 LF vs 50 CRLF), prefer LF as the modern default
    if counts[LineEnding.LF] == max_count:
        return LineEnding.LF
    if counts[LineEnding.CRLF] == max_count:
        return LineEnding.CRLF
    return LineEnding.CR


def normalize_to_lf(content: str) -> str:
    """Normalize all line endings in ``content`` to LF."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(content: str, target: LineEnding) -> str:
    """Restore LF-terminated ``content`` to the target line ending.

    Assumes ``content`` already uses LF endings (e.g. after
    :func:`normalize_to_lf`). Content with no newlines is returned unchanged.
    """
    if target == LineEnding.CRLF:
        return content.replace("\n", "\r\n")
    if target == LineEnding.CR:
        return content.replace("\n", "\r")
    return content


class LineNormalizer:
    """Applies line-ending normalization when enabled.

    Attributes:
        enabled: Whether normalization is active. When disabled all methods
            are pass-through.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def normalize(self, content: str) -> str:
        """Normalize ``content`` to LF when enabled, otherwise pass-through."""
        if not self.enabled:
            return content
        return normalize_to_lf(content)

    def restore(self, content: str, original: LineEnding) -> str:
        """Restore normalized ``content`` back to ``original`` line endings.

        Pass-through when disabled or when ``original`` is LF/none.
        """
        if not self.enabled:
            return content
        if original in (LineEnding.LF, LineEnding.NONE):
            return content
        return restore_line_endings(content, original)


def generate_failure_hint(
    content_lf: str, old_text_lf: str, expected: int, actual: int
) -> str:
    """Generate a human-readable hint when old_text does not match file content.

    All inputs should already be normalized to LF.
    """
    if actual > expected:
        return _ambiguous_match_hint(content_lf, old_text_lf, expected, actual)
    if actual > 0 and actual < expected:
        return (
            f"Match count mismatch: expected {expected} matches, found {actual}. "
            "The old_text appears fewer times than expected. "
            "Verify the file content or adjust expected_match_count."
        )
    return _no_match_hint(content_lf, old_text_lf)


def _ambiguous_match_hint(
    content_lf: str, old_text_lf: str, expected: int, actual: int
) -> str:
    match_lines = _find_match_lines(content_lf, old_text_lf, actual)
    context_str = _build_context_snippets(content_lf, match_lines)
    hint = (
        f"Ambiguous match: found {actual} matches for old_text, expected {expected}. "
        "Please include more surrounding context lines in old_text to make it unique."
    )
    if context_str:
        hint += f"\nMatches found at:\n{context_str}"
    if actual > 5:
        hint += f"\n  ... and {actual - 5} more matches"
    return hint


def _find_match_lines(content_lf: str, old_text_lf: str, actual: int) -> list[int]:
    content_lines = content_lf.splitlines(keepends=True)
    old_lines = old_text_lf.splitlines(keepends=True)
    match_lines = []
    offset = 0
    for i, line in enumerate(content_lines):
        if line.startswith(old_lines[0].rstrip("\n")):
            if content_lf.count(old_text_lf, offset) > 0:
                match_lines.append(i + 1)
                if len(match_lines) >= actual:
                    break
        offset += len(line)
    return match_lines


def _build_context_snippets(content_lf: str, match_lines: list[int]) -> str:
    content_lines = content_lf.splitlines(keepends=True)
    snippets = []
    for line_num in match_lines[:5]:
        start = max(0, line_num - 3)
        end = min(len(content_lines), line_num + 2)
        context = "".join(content_lines[start:end])
        snippets.append(f"  Line {line_num}:\n{_indent(context, '    ')}")
    return "\n".join(snippets)


def _no_match_hint(content_lf: str, old_text_lf: str) -> str:
    lines_old = old_text_lf.splitlines()
    first_line = lines_old[0].strip() if lines_old else ""

    # Use split("\n") instead of splitlines() so trailing \r (from CRLF files)
    # is preserved in content lines and visible in repr().
    content_lines_split = content_lf.split("\n")
    matching_lines = [
        line
        for line in content_lines_split
        if line.rstrip("\r").strip() == first_line and line.strip() != ""
    ]
    if matching_lines:
        return _indent_mismatch_hint(
            content_lf, old_text_lf, matching_lines[0], lines_old[0]
        )
    return _fuzzy_match_hint(content_lf, old_text_lf)


def _indent_mismatch_hint(
    content_lf: str, old_text_lf: str, file_line: str, old_first: str
) -> str:
    if file_line.endswith("\r") and not old_first.endswith("\r"):
        return (
            "No exact match found: the matching file line ends with CRLF "
            "but old_text uses LF. This is a line-ending mismatch. "
            f"File line: {file_line!r}, old_text first line: {old_first!r}. "
            "Enable line-ending normalization or match the file's CRLF endings."
        )
    return (
        "No exact match found, but found lines with matching content and "
        "different indentation/whitespace. "
        f"File has: {file_line!r}, but old_text started with: {old_first!r}. "
        "Verify tabs vs. spaces, exact indentation, and line endings."
    )


def _fuzzy_match_hint(content_lf: str, old_text_lf: str) -> str:
    content_lines = content_lf.splitlines(keepends=True)
    n_lines = len(old_text_lf.splitlines())
    best_ratio = 0.0
    best_candidate = ""

    for i in range(max(1, len(content_lines) - n_lines + 1)):
        window = "".join(content_lines[i : i + n_lines])
        ratio = difflib.SequenceMatcher(None, window, old_text_lf).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = window

    if best_ratio > 0.75:
        diff = "".join(
            difflib.unified_diff(
                best_candidate.splitlines(keepends=True),
                old_text_lf.splitlines(keepends=True),
                fromfile="file_actual",
                tofile="old_text_provided",
                n=1,
            )
        )
        return (
            f"No exact match found. Found a similar block "
            f"({int(best_ratio * 100)}% similarity). Differences:\n{diff}"
            "Please check for outdated surrounding context."
        )

    return (
        "No match found for old_text. The file does not contain the specified "
        "lines. Run read_file on the target line range to verify current contents."
    )
