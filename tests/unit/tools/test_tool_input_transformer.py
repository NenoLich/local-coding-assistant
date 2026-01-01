"""Unit tests for ToolInputTransformer edge cases."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from local_coding_assistant.tools.tool_runtime import ToolInputTransformer


class MultiValueModel(BaseModel):
    numbers: list[float]


class SingleFieldModel(BaseModel):
    value: int


class SequenceFieldModel(BaseModel):
    items: list[int]


class MixedModel(BaseModel):
    first: int
    second: int = 5


@pytest.fixture
def numbers_field_info():
    return MultiValueModel.model_fields["numbers"]


@pytest.fixture
def value_field_info():
    return SingleFieldModel.model_fields["value"]


@pytest.fixture
def sequence_field_info():
    return SequenceFieldModel.model_fields["items"]


def test_process_multi_value_field_converts_to_list(numbers_field_info):
    data = {"numbers": "1"}

    ToolInputTransformer._process_multi_value_field(
        data,
        "numbers",
        numbers_field_info,
        lambda _: ["1", "2", "3"],
    )

    assert data["numbers"] == [1.0, 2.0, 3.0]


def test_process_multi_value_field_handles_conversion_error(numbers_field_info):
    data = {"numbers": "not-a-number"}

    ToolInputTransformer._process_multi_value_field(
        data,
        "numbers",
        numbers_field_info,
        lambda _: ["a", "b"],
    )

    assert data["numbers"] == "not-a-number"


@pytest.mark.parametrize(
    "args,expected",
    [([5], {"value": 5}), ([1, 2], {"value": [1, 2]})],
)
def test_handle_single_field_returns_expected(value_field_info, args, expected):
    assert (
        ToolInputTransformer._handle_single_field(args, "value", value_field_info)
        == expected
    )


def test_handle_single_field_for_list_field(sequence_field_info):
    result = ToolInputTransformer._handle_single_field(
        [1, 2, 3],
        "items",
        sequence_field_info,
    )

    assert result == {"items": [1, 2, 3]}


def test_map_args_to_fields_uses_defaults():
    result = ToolInputTransformer._map_args_to_fields(
        [1],
        MixedModel.model_fields,
    )

    assert result == {"first": 1, "second": 5}


def test_transform_positional_args_handles_numbers_field():
    transformed = ToolInputTransformer._transform_positional_args(
        {"args": ["1", "2"]},
        MultiValueModel.model_fields,
    )

    assert transformed == {"numbers": [1.0, 2.0]}


def test_transform_positional_args_single_field_as_list(sequence_field_info):
    transformed = ToolInputTransformer._transform_positional_args(
        {"args": [1, 2]},
        {"items": sequence_field_info},
    )

    assert transformed == {"items": [1, 2]}


def test_transform_positional_args_maps_by_position():
    transformed = ToolInputTransformer._transform_positional_args(
        {"args": [10, 20]},
        MixedModel.model_fields,
    )

    assert transformed == {"first": 10, "second": 20}


@pytest.mark.parametrize(
    "field_info,is_collection",
    [
        (MultiValueModel.model_fields["numbers"], True),
        (SingleFieldModel.model_fields["value"], False),
    ],
)
def test_is_list_or_tuple_field_detection(field_info, is_collection):
    assert ToolInputTransformer._is_list_or_tuple_field(field_info) is is_collection
