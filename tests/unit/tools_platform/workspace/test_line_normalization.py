"""Unit tests for line normalization logic.

Covers the pure functions and classes in
``local_coding_assistant.tools_platform.workspace.line_normalization``:

- LineEnding enum + value_str
- detect_line_ending (CRLF/LF/CR/NONE detection, frequency-based)
- normalize_to_lf
- restore_line_endings
- LineNormalizer (enabled/disabled passthrough)
- generate_failure_hint (ambiguity, indentation, fuzzy match, no match)
"""

from __future__ import annotations

from local_coding_assistant.tools_platform.workspace.line_normalization import (
    LineEnding,
    LineNormalizer,
    detect_line_ending,
    generate_failure_hint,
    normalize_to_lf,
    restore_line_endings,
)

# ---------------------------------------------------------------------------
# LineEnding
# ---------------------------------------------------------------------------


def test_line_ending_values() -> None:
    assert LineEnding.LF.value == "lf"
    assert LineEnding.CRLF.value == "crlf"
    assert LineEnding.CR.value == "cr"
    assert LineEnding.NONE.value == "none"


def test_line_ending_value_str() -> None:
    assert LineEnding.LF.value_str == "\n"
    assert LineEnding.CRLF.value_str == "\r\n"
    assert LineEnding.CR.value_str == "\r"
    assert LineEnding.NONE.value_str == "\n"


# ---------------------------------------------------------------------------
# detect_line_ending
# ---------------------------------------------------------------------------


def test_detect_lf() -> None:
    assert detect_line_ending("line1\nline2\nline3\n") == LineEnding.LF


def test_detect_crlf() -> None:
    assert detect_line_ending("line1\r\nline2\r\nline3\r\n") == LineEnding.CRLF


def test_detect_cr() -> None:
    assert detect_line_ending("line1\rline2\rline3\r") == LineEnding.CR


def test_detect_none() -> None:
    assert detect_line_ending("no newlines here") == LineEnding.NONE


def test_detect_empty_string() -> None:
    assert detect_line_ending("") == LineEnding.NONE


def test_detect_mixed_prefers_lf_on_tie() -> None:
    # Equal LF and CRLF counts -> tie prefers LF per implementation.
    content = "a\r\nb\n"  # 1 CRLF, 1 LF -> tie -> LF
    assert detect_line_ending(content) == LineEnding.LF


def test_detect_dominant_crlf() -> None:
    content = "a\r\nb\r\nc\nd\r\n"  # 3 CRLF vs 1 LF -> CRLF wins
    assert detect_line_ending(content) == LineEnding.CRLF


def test_detect_standalone_cr() -> None:
    content = "a\rc\r"
    assert detect_line_ending(content) == LineEnding.CR


# ---------------------------------------------------------------------------
# normalize_to_lf
# ---------------------------------------------------------------------------


def test_normalize_to_lf_crlf() -> None:
    assert normalize_to_lf("a\r\nb\r\nc") == "a\nb\nc"


def test_normalize_to_lf_cr() -> None:
    assert normalize_to_lf("a\rb\rc") == "a\nb\nc"


def test_normalize_to_lf_already_lf() -> None:
    assert normalize_to_lf("a\nb\nc") == "a\nb\nc"


def test_normalize_to_lf_no_newlines() -> None:
    assert normalize_to_lf("plain text") == "plain text"


def test_normalize_to_lf_empty() -> None:
    assert normalize_to_lf("") == ""


# ---------------------------------------------------------------------------
# restore_line_endings
# ---------------------------------------------------------------------------


def test_restore_to_crlf() -> None:
    assert restore_line_endings("a\nb\nc", LineEnding.CRLF) == "a\r\nb\r\nc"


def test_restore_to_cr() -> None:
    assert restore_line_endings("a\nb\nc", LineEnding.CR) == "a\rb\rc"


def test_restore_to_lf_passthrough() -> None:
    assert restore_line_endings("a\nb\nc", LineEnding.LF) == "a\nb\nc"


def test_restore_to_none_passthrough() -> None:
    assert restore_line_endings("a\nb\nc", LineEnding.NONE) == "a\nb\nc"


def test_restore_no_newlines() -> None:
    assert restore_line_endings("no newlines", LineEnding.CRLF) == "no newlines"


def test_restore_empty() -> None:
    assert restore_line_endings("", LineEnding.CRLF) == ""


# ---------------------------------------------------------------------------
# LineNormalizer
# ---------------------------------------------------------------------------


def test_normalizer_enabled_normalizes() -> None:
    n = LineNormalizer(enabled=True)
    assert n.normalize("a\r\nb\r\nc") == "a\nb\nc"


def test_normalizer_disabled_passthrough() -> None:
    n = LineNormalizer(enabled=False)
    assert n.normalize("a\r\nb\r\nc") == "a\r\nb\r\nc"


def test_normalizer_restore_enabled_crlf() -> None:
    n = LineNormalizer(enabled=True)
    assert n.restore("a\nb\nc", LineEnding.CRLF) == "a\r\nb\r\nc"


def test_normalizer_restore_disabled_passthrough() -> None:
    n = LineNormalizer(enabled=False)
    assert n.restore("a\nb\nc", LineEnding.CRLF) == "a\nb\nc"


def test_normalizer_restore_lf_passthrough() -> None:
    n = LineNormalizer(enabled=True)
    assert n.restore("a\nb\nc", LineEnding.LF) == "a\nb\nc"


def test_normalizer_restore_none_passthrough() -> None:
    n = LineNormalizer(enabled=True)
    assert n.restore("a\nb\nc", LineEnding.NONE) == "a\nb\nc"


# ---------------------------------------------------------------------------
# generate_failure_hint
# ---------------------------------------------------------------------------


def test_hint_too_many_matches() -> None:
    content = "line2\nline2\nline3\n"
    hint = generate_failure_hint(content, "line2\n", expected=1, actual=2)
    assert "Ambiguous match" in hint
    assert "2 matches" in hint
    assert "expected 1" in hint


def test_hint_indentation_mismatch() -> None:
    content = "  line2\nline3\n"
    hint = generate_failure_hint(content, "line2\n", expected=1, actual=0)
    assert "indentation/whitespace" in hint.lower()
    assert "line2" in hint


def test_hint_crlf_line_ending_mismatch() -> None:
    """CRLF file content vs LF old_text produces explicit line-ending hint."""
    content = "import json\r\nfrom pathlib import Path\r\n"
    old_text = "import json\n"
    hint = generate_failure_hint(content, old_text, expected=1, actual=0)
    assert "CRLF" in hint
    assert "LF" in hint
    assert "line-ending mismatch" in hint.lower()
    # repr() must expose the trailing \r
    assert "'import json\\r'" in hint


def test_hint_no_match_found() -> None:
    content = "XYZ completely different content here\nABC\n"
    hint = generate_failure_hint(content, "line2\n", expected=1, actual=0)
    assert "No match found" in hint
    assert "read_file" in hint


def test_hint_fuzzy_similar_block() -> None:
    content = "line2\nline3\n"
    hint = generate_failure_hint(content, "line2X\n", expected=1, actual=0)
    assert "No exact match found" in hint
    assert "similarity" in hint.lower()
    assert "line2" in hint
    assert "Please check" in hint


def test_hint_fuzzy_low_similarity_falls_back() -> None:
    content = "completely different content here\n"
    hint = generate_failure_hint(content, "line2\n", expected=1, actual=0)
    assert "No match found" in hint


def test_hint_expected_matches_actual_is_success_path() -> None:
    # When actual == expected this function isn't called, but ensure it doesn't
    # crash if called with equal counts (returns no-match style hint).
    content = "line1\nline3\n"
    hint = generate_failure_hint(content, "line2\n", expected=1, actual=1)
    assert hint  # non-empty diagnostic
