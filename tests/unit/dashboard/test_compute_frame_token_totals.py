"""
Unit tests for compute_frame_token_totals.
"""

import json

from local_coding_assistant.dashboard.models.historical import (
    compute_frame_token_totals,
)


class TestComputeFrameTokenTotals:
    def test_sums_llm_complete_events(self):
        events = [
            {
                "type": "llm_complete",
                "data_json": json.dumps(
                    {"input_tokens": 100, "output_tokens": 50, "cached_tokens": 10}
                ),
            },
            {
                "type": "llm_complete",
                "data_json": json.dumps(
                    {"input_tokens": 30, "output_tokens": 20, "cached_tokens": 5}
                ),
            },
            {"type": "turn_start", "data_json": json.dumps({"user_input": "hi"})},
        ]
        totals = compute_frame_token_totals(events)
        assert totals == {"input_tokens": 130, "output_tokens": 70, "cache_tokens": 15}

    def test_ignores_non_llm_complete_events(self):
        events = [
            {"type": "turn_start", "data_json": "{}"},
            {"type": "tool_result", "data_json": "{}"},
        ]
        assert compute_frame_token_totals(events) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_tokens": 0,
        }

    def test_empty_list_returns_zeros(self):
        assert compute_frame_token_totals([]) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_tokens": 0,
        }

    def test_handles_data_json_as_dict(self):
        events = [
            {
                "type": "llm_complete",
                "data_json": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "cached_tokens": 1,
                },
            },
        ]
        totals = compute_frame_token_totals(events)
        assert totals == {"input_tokens": 7, "output_tokens": 3, "cache_tokens": 1}

    def test_missing_fields_default_to_zero(self):
        events = [
            {"type": "llm_complete", "data_json": json.dumps({})},
        ]
        totals = compute_frame_token_totals(events)
        assert totals == {"input_tokens": 0, "output_tokens": 0, "cache_tokens": 0}

    def test_invalid_data_json_string_defaults_to_empty(self):
        events = [
            {"type": "llm_complete", "data_json": "not-json"},
        ]
        totals = compute_frame_token_totals(events)
        assert totals == {"input_tokens": 0, "output_tokens": 0, "cache_tokens": 0}
