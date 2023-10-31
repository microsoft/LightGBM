# coding: utf-8
import filecmp
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pyarrow as pa
import pytest

import lightgbm as lgb

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
_FLOATING_POINT_TYPES = [pa.float32(), pa.float64()]


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


def assert_arrays_equal(lhs: np.ndarray, rhs: np.ndarray):
    assert lhs.dtype == rhs.dtype and np.array_equal(lhs, rhs)


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
def test_dataset_construct_fuzzy(
    tmp_path: Path, arrow_table_fn: Callable[[], pa.Table], dataset_params: Dict[str, Any]
):
    arrow_table = arrow_table_fn()

    arrow_dataset = lgb.Dataset(arrow_table, params=dataset_params)
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(arrow_table.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    arrow_dataset._dump_text(tmp_path / "arrow.txt")
    pandas_dataset._dump_text(tmp_path / "pandas.txt")
    assert filecmp.cmp(tmp_path / "arrow.txt", tmp_path / "pandas.txt")


@pytest.mark.parametrize("field", ["label", "weight", "init_score"])
def test_dataset_construct_fields_fuzzy(field: str):
    arrow_table = generate_random_arrow_table(3, 1000, 42)
    arrow_array = generate_random_arrow_array(1000, 42)

    arrow_dataset = lgb.Dataset(arrow_table, **{field: arrow_array})
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(arrow_table.to_pandas(), **{field: arrow_array.to_numpy()})
    pandas_dataset.construct()

    assert_arrays_equal(arrow_dataset.get_field(field), pandas_dataset.get_field(field))
    assert_arrays_equal(
        getattr(arrow_dataset, f"get_{field}")(), getattr(pandas_dataset, f"get_{field}")()
    )


@pytest.mark.parametrize(
    ["array_type", "label_data"],
    [(pa.array, [0, 1, 0, 0, 1]), (pa.chunked_array, [[0], [1, 0, 0, 1]])],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOATING_POINT_TYPES)
def test_dataset_construct_labels(array_type: Any, label_data: Any, arrow_type: Any):
    data = generate_dummy_arrow_table()
    labels = array_type(label_data, type=arrow_type)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    assert_arrays_equal(expected, dataset.get_label())


def test_dataset_construct_weights_none():
    data = generate_dummy_arrow_table()
    weight = pa.array([1, 1, 1, 1, 1])
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None


@pytest.mark.parametrize(
    ["array_type", "weight_data"],
    [(pa.array, [3, 0.7, 1.5, 0.5, 0.1]), (pa.chunked_array, [[3], [0.7, 1.5, 0.5, 0.1]])],
)
@pytest.mark.parametrize("arrow_type", _FLOATING_POINT_TYPES)
def test_dataset_construct_weights(array_type: Any, weight_data: Any, arrow_type: Any):
    data = generate_dummy_arrow_table()
    weights = array_type(weight_data, type=arrow_type)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    assert_arrays_equal(expected, dataset.get_weight())


@pytest.mark.parametrize(
    ["array_type", "group_data"], [(pa.array, [2, 3]), (pa.chunked_array, [[2], [3]])]
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES)
def test_dataset_construct_groups(array_type: Any, group_data: Any, arrow_type: Any):
    data = generate_dummy_arrow_table()
    groups = array_type(group_data, type=arrow_type)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 2, 5], dtype=np.int32)
    assert_arrays_equal(expected, dataset.get_field("group"))


@pytest.mark.parametrize(
    ["array_type", "init_score_data"],
    [(pa.array, [0, 1, 2, 3, 3]), (pa.chunked_array, [[0, 1, 2], [3, 3]])],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOATING_POINT_TYPES)
def test_dataset_construct_init_scores_array(
    array_type: Any, init_score_data: Any, arrow_type: Any
):
    data = generate_dummy_arrow_table()
    init_scores = array_type(init_score_data, type=arrow_type)
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 3], dtype=np.float64)
    assert_arrays_equal(expected, dataset.get_init_score())


def test_dataset_construct_init_scores_table():
    data = generate_dummy_arrow_table()
    init_scores = pa.Table.from_arrays(
        [
            generate_random_arrow_array(5, seed=1),
            generate_random_arrow_array(5, seed=2),
            generate_random_arrow_array(5, seed=3),
        ],
        names=["a", "b", "c"],
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    actual = dataset.get_init_score()
    assert actual.dtype == np.float64
    assert actual.shape == (5, 3)


# ------------------------------------------ PREDICTION ----------------------------------------- #


def test_predict():
    data = generate_random_arrow_table(10, 10000, 42)
    labels = generate_random_arrow_array(10000, 43)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    booster = lgb.train({}, dataset, num_boost_round=1)

    out_arrow = booster.predict(data)
    out_pandas = booster.predict(data.to_pandas())
    assert_arrays_equal(out_arrow, out_pandas)
