# coding: utf-8
import filecmp
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pyarrow as pa
import pytest

import lightgbm as lgb

# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #


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
    return pa.Table.from_arrays(columns, names=[str(i) for i in range(len(columns))])


def generate_dummy_arrow_table() -> pa.Table:
    col1 = pa.chunked_array([[1, 2, 3], [4, 5]], type=pa.uint8())
    col2 = pa.chunked_array([[0.5, 0.6], [0.1, 0.8, 1.5]], type=pa.float32())
    return pa.Table.from_arrays([col1, col2], names=["a", "b"])


def generate_random_arrow_table(num_columns: int, num_datapoints: int, seed: int) -> pa.Table:
    columns = [generate_random_arrow_array(num_datapoints, seed + i) for i in range(num_columns)]
    names = [str(i) for i in range(num_columns)]
    return pa.Table.from_arrays(columns, names=names)


def generate_random_arrow_array(num_datapoints: int, seed: int) -> pa.ChunkedArray:
    generator = np.random.default_rng(seed)
    data = generator.standard_normal(num_datapoints)

    # Set random nulls
    indices = generator.choice(len(data), size=num_datapoints // 10)
    data[indices] = None

    # Split data into random chunks
    n_chunks = generator.integers(1, num_datapoints // 3)
    split_points = np.sort(generator.choice(np.arange(1, num_datapoints), n_chunks, replace=False))
    split_points = np.concatenate([[0], split_points, [num_datapoints]])
    chunks = [data[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)]
    chunks = [chunk for chunk in chunks if len(chunk) > 0]

    # Turn chunks into array
    return pa.chunked_array(chunks, type=pa.float32())


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
    }


def arrays_equal(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    return lhs.dtype == rhs.dtype and np.array_equal(lhs, rhs)


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- DATASET ------------------------------------------- #


def test_dataset_construct_smoke():
    data = generate_random_arrow_table(10, 10000, 42)
    label = generate_random_arrow_array(10000, 43)
    weight = generate_random_arrow_array(10000, 44)
    init_scores = generate_random_arrow_array(10000, 45)

    dataset = lgb.Dataset(data, label=label, weight=weight, init_score=init_scores)
    dataset.construct()


@pytest.mark.parametrize(
    ("arrow_table", "dataset_params"),
    [
        (generate_simple_arrow_table(), dummy_dataset_params()),
        (generate_dummy_arrow_table(), dummy_dataset_params()),
        (generate_random_arrow_table(3, 1000, 42), {}),
        (generate_random_arrow_table(100, 10000, 43), {}),
    ],
)
def test_dataset_construct_fuzzy(arrow_table: pa.Table, dataset_params: Dict[str, Any]):
    arrow_dataset = lgb.Dataset(arrow_table, params=dataset_params)
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(arrow_table.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    with tempfile.TemporaryDirectory() as t:
        tmpdir = Path(t)
        arrow_dataset._dump_text(tmpdir / "arrow.txt")
        pandas_dataset._dump_text(tmpdir / "pandas.txt")
        assert filecmp.cmp(tmpdir / "arrow.txt", tmpdir / "pandas.txt")


def test_dataset_construct_labels():
    data = generate_dummy_arrow_table()
    labels = pa.chunked_array([[0], [1, 0, 0, 1]], type=pa.uint8())
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    dataset._dump_text("out.txt")

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    assert arrays_equal(expected, dataset.get_label())


def test_dataset_construct_weights():
    data = generate_dummy_arrow_table()
    weights = pa.chunked_array([[0.3], [0.6, 1.0, 4.0], [2.5]], type=pa.float32())
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0.3, 0.6, 1.0, 4.0, 2.5], dtype=np.float32)
    assert arrays_equal(expected, dataset.get_weight())


def test_dataset_construct_groups():
    data = generate_dummy_arrow_table()
    groups = pa.chunked_array([[2], [1, 2]], type=pa.uint8())
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 2, 3, 5], dtype=np.int32)
    assert arrays_equal(expected, dataset.get_field("group"))


def test_dataset_construct_init_scores_1d():
    data = generate_dummy_arrow_table()
    init_scores = pa.chunked_array([[1.0, 2.0], [1.0, 1.0, 1.0]], type=pa.float32())
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([1.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
    assert arrays_equal(expected, dataset.get_init_score())


def test_dataset_construct_init_scores_2d():
    data = generate_dummy_arrow_table()
    init_scores = pa.Table.from_arrays(
        [
            pa.chunked_array([[1.0, 2.0], [1.0, 1.0, 1.0]], type=pa.float32()),
            pa.array([3.5, 3.5, 3.5, 3.5, 3.5], type=pa.float32()),
        ],
        names=["a", "b"],
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array(
        [[1.0, 3.5], [2.0, 3.5], [1.0, 3.5], [1.0, 3.5], [1.0, 3.5]], dtype=np.float64
    )
    assert arrays_equal(expected, dataset.get_init_score())


# ------------------------------------------ PREDICTION ----------------------------------------- #


def test_predict():
    data = generate_random_arrow_table(10, 10000, 42)
    labels = generate_random_arrow_array(10000, 43)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    booster = lgb.train({}, dataset, num_boost_round=1)

    out_arrow = booster.predict(data)
    out_pandas = booster.predict(data.to_pandas())
    assert arrays_equal(out_arrow, out_pandas)
