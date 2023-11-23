# coding: utf-8
import filecmp
from typing import Any, Dict

import numpy as np
import pyarrow as pa
import pytest

import lightgbm as lgb

from .utils import np_assert_array_equal

# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #

_INTEGER_TYPES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
]
_FLOAT_TYPES = [
    pa.float32(),
    pa.float64(),
]


def generate_simple_arrow_table() -> pa.Table:
    columns = [
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint8()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int8()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint16()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int16()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.uint64()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.int64()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.float32()),
        pa.chunked_array([[1, 2, 3, 4, 5]], type=pa.float64()),
    ]
    return pa.Table.from_arrays(columns, names=[f"col_{i}" for i in range(len(columns))])


def generate_dummy_arrow_table() -> pa.Table:
    col1 = pa.chunked_array([[1, 2, 3], [4, 5]], type=pa.uint8())
    col2 = pa.chunked_array([[0.5, 0.6], [0.1, 0.8, 1.5]], type=pa.float32())
    return pa.Table.from_arrays([col1, col2], names=["a", "b"])


def generate_random_arrow_table(num_columns: int, num_datapoints: int, seed: int) -> pa.Table:
    columns = [generate_random_arrow_array(num_datapoints, seed + i) for i in range(num_columns)]
    names = [f"col_{i}" for i in range(num_columns)]
    return pa.Table.from_arrays(columns, names=names)


def generate_random_arrow_array(num_datapoints: int, seed: int) -> pa.ChunkedArray:
    generator = np.random.default_rng(seed)
    data = generator.standard_normal(num_datapoints)

    # Set random nulls
    indices = generator.choice(len(data), size=num_datapoints // 10)
    data[indices] = None

    # Split data into <=2 random chunks
    split_points = np.sort(generator.choice(np.arange(1, num_datapoints), 2, replace=False))
    split_points = np.concatenate([[0], split_points, [num_datapoints]])
    chunks = [data[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    # Turn chunks into array
    return pa.chunked_array([data], type=pa.float32())


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
    }


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- DATASET ------------------------------------------- #


@pytest.mark.parametrize(
    ("arrow_table_fn", "dataset_params"),
    [  # Use lambda functions here to minimize memory consumption
        (lambda: generate_simple_arrow_table(), dummy_dataset_params()),
        (lambda: generate_dummy_arrow_table(), dummy_dataset_params()),
        (lambda: generate_random_arrow_table(3, 1000, 42), {}),
        (lambda: generate_random_arrow_table(100, 10000, 43), {}),
    ],
)
def test_dataset_construct_fuzzy(tmp_path, arrow_table_fn, dataset_params):
    arrow_table = arrow_table_fn()

    arrow_dataset = lgb.Dataset(arrow_table, params=dataset_params)
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(arrow_table.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    arrow_dataset._dump_text(tmp_path / "arrow.txt")
    pandas_dataset._dump_text(tmp_path / "pandas.txt")
    assert filecmp.cmp(tmp_path / "arrow.txt", tmp_path / "pandas.txt")


# -------------------------------------------- FIELDS ------------------------------------------- #


def test_dataset_construct_fields_fuzzy():
    arrow_table = generate_random_arrow_table(3, 1000, 42)
    arrow_labels = generate_random_arrow_array(1000, 42)
    arrow_weights = generate_random_arrow_array(1000, 42)
    arrow_groups = pa.chunked_array([[300, 400, 50], [250]], type=pa.int32())

    arrow_dataset = lgb.Dataset(
        arrow_table, label=arrow_labels, weight=arrow_weights, group=arrow_groups
    )
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(
        arrow_table.to_pandas(),
        label=arrow_labels.to_numpy(),
        weight=arrow_weights.to_numpy(),
        group=arrow_groups.to_numpy(),
    )
    pandas_dataset.construct()

    # Check for equality
    for field in ("label", "weight", "group"):
        np_assert_array_equal(
            arrow_dataset.get_field(field), pandas_dataset.get_field(field), strict=True
        )
    np_assert_array_equal(arrow_dataset.get_label(), pandas_dataset.get_label(), strict=True)
    np_assert_array_equal(arrow_dataset.get_weight(), pandas_dataset.get_weight(), strict=True)


# -------------------------------------------- LABELS ------------------------------------------- #


@pytest.mark.parametrize(
    ["array_type", "label_data"],
    [(pa.array, [0, 1, 0, 0, 1]), (pa.chunked_array, [[0], [1, 0, 0, 1]])],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_labels(array_type, label_data, arrow_type):
    data = generate_dummy_arrow_table()
    labels = array_type(label_data, type=arrow_type)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)


# ------------------------------------------- WEIGHTS ------------------------------------------- #


def test_dataset_construct_weights_none():
    data = generate_dummy_arrow_table()
    weight = pa.array([1, 1, 1, 1, 1])
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None
    assert dataset.get_field("weight") is None


@pytest.mark.parametrize(
    ["array_type", "weight_data"],
    [(pa.array, [3, 0.7, 1.5, 0.5, 0.1]), (pa.chunked_array, [[3], [0.7, 1.5, 0.5, 0.1]])],
)
@pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()])
def test_dataset_construct_weights(array_type, weight_data, arrow_type):
    data = generate_dummy_arrow_table()
    weights = array_type(weight_data, type=arrow_type)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_weight(), strict=True)


# -------------------------------------------- GROUPS ------------------------------------------- #


@pytest.mark.parametrize(
    ["array_type", "group_data"],
    [
        (pa.array, [2, 3]),
        (pa.chunked_array, [[2], [3]]),
        (pa.chunked_array, [[], [2, 3]]),
        (pa.chunked_array, [[2], [], [3], []]),
    ],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES)
def test_dataset_construct_groups(array_type, group_data, arrow_type):
    data = generate_dummy_arrow_table()
    groups = array_type(group_data, type=arrow_type)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 2, 5], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_field("group"), strict=True)
