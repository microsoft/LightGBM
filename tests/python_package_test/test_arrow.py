# coding: utf-8
import filecmp
from typing import Any, Dict, Optional

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


def generate_simple_arrow_table(empty_chunks: bool = False) -> pa.Table:
    c: list[list[int]] = [[]] if empty_chunks else []
    columns = [
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.uint8()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.int8()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.uint16()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.int16()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.uint32()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.int32()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.uint64()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.int64()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.float32()),
        pa.chunked_array(c + [[1, 2, 3]] + c + [[4, 5]] + c, type=pa.float64()),
    ]
    return pa.Table.from_arrays(columns, names=[f"col_{i}" for i in range(len(columns))])


def generate_nullable_arrow_table() -> pa.Table:
    columns = [
        pa.chunked_array([[1, None, 3, 4, 5]], type=pa.float32()),
        pa.chunked_array([[None, 2, 3, 4, 5]], type=pa.float32()),
        pa.chunked_array([[1, 2, 3, 4, None]], type=pa.float32()),
        pa.chunked_array([[None, None, None, None, None]], type=pa.float32()),
    ]
    return pa.Table.from_arrays(columns, names=[f"col_{i}" for i in range(len(columns))])


def generate_dummy_arrow_table() -> pa.Table:
    col1 = pa.chunked_array([[1, 2, 3], [4, 5]], type=pa.uint8())
    col2 = pa.chunked_array([[0.5, 0.6], [0.1, 0.8, 1.5]], type=pa.float32())
    return pa.Table.from_arrays([col1, col2], names=["a", "b"])


def generate_random_arrow_table(
    num_columns: int,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pa.Table:
    columns = [
        generate_random_arrow_array(
            num_datapoints, seed + i, generate_nulls=generate_nulls, values=values
        )
        for i in range(num_columns)
    ]
    names = [f"col_{i}" for i in range(num_columns)]
    return pa.Table.from_arrays(columns, names=names)


def generate_random_arrow_array(
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pa.ChunkedArray:
    generator = np.random.default_rng(seed)
    data = (
        generator.standard_normal(num_datapoints)
        if values is None
        else generator.choice(values, size=num_datapoints, replace=True)
    )

    # Set random nulls
    if generate_nulls:
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
        (lambda: generate_simple_arrow_table(empty_chunks=True), dummy_dataset_params()),
        (lambda: generate_dummy_arrow_table(), dummy_dataset_params()),
        (lambda: generate_nullable_arrow_table(), dummy_dataset_params()),
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
    arrow_labels = generate_random_arrow_array(1000, 42, generate_nulls=False)
    arrow_weights = generate_random_arrow_array(1000, 42, generate_nulls=False)
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
    [
        (pa.array, [0, 1, 0, 0, 1]),
        (pa.chunked_array, [[0], [1, 0, 0, 1]]),
        (pa.chunked_array, [[], [0], [1, 0, 0, 1]]),
        (pa.chunked_array, [[0], [], [1, 0], [], [], [0, 1], []]),
    ],
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
    [
        (pa.array, [3, 0.7, 1.5, 0.5, 0.1]),
        (pa.chunked_array, [[3], [0.7, 1.5, 0.5, 0.1]]),
        (pa.chunked_array, [[], [3], [0.7, 1.5, 0.5, 0.1]]),
        (pa.chunked_array, [[3], [0.7], [], [], [1.5, 0.5, 0.1], []]),
    ],
)
@pytest.mark.parametrize("arrow_type", _FLOAT_TYPES)
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


# ----------------------------------------- INIT SCORES ----------------------------------------- #


@pytest.mark.parametrize(
    ["array_type", "init_score_data"],
    [
        (pa.array, [0, 1, 2, 3, 3]),
        (pa.chunked_array, [[0, 1, 2], [3, 3]]),
        (pa.chunked_array, [[], [0, 1, 2], [3, 3]]),
        (pa.chunked_array, [[0, 1], [], [], [2], [3, 3], []]),
    ],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_init_scores_array(
    array_type: Any, init_score_data: Any, arrow_type: Any
):
    data = generate_dummy_arrow_table()
    init_scores = array_type(init_score_data, type=arrow_type)
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 3], dtype=np.float64)
    np_assert_array_equal(expected, dataset.get_init_score(), strict=True)


def test_dataset_construct_init_scores_table():
    data = generate_dummy_arrow_table()
    init_scores = pa.Table.from_arrays(
        [
            generate_random_arrow_array(5, seed=1, generate_nulls=False),
            generate_random_arrow_array(5, seed=2, generate_nulls=False),
            generate_random_arrow_array(5, seed=3, generate_nulls=False),
        ],
        names=["a", "b", "c"],
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    actual = dataset.get_init_score()
    expected = init_scores.to_pandas().to_numpy().astype(np.float64)
    np_assert_array_equal(expected, actual, strict=True)


# ------------------------------------------ PREDICTION ----------------------------------------- #


def assert_equal_predict_arrow_pandas(booster: lgb.Booster, data: pa.Table):
    p_arrow = booster.predict(data)
    p_pandas = booster.predict(data.to_pandas())
    np_assert_array_equal(p_arrow, p_pandas, strict=True)

    p_raw_arrow = booster.predict(data, raw_score=True)
    p_raw_pandas = booster.predict(data.to_pandas(), raw_score=True)
    np_assert_array_equal(p_raw_arrow, p_raw_pandas, strict=True)

    p_leaf_arrow = booster.predict(data, pred_leaf=True)
    p_leaf_pandas = booster.predict(data.to_pandas(), pred_leaf=True)
    np_assert_array_equal(p_leaf_arrow, p_leaf_pandas, strict=True)

    p_pred_contrib_arrow = booster.predict(data, pred_contrib=True)
    p_pred_contrib_pandas = booster.predict(data.to_pandas(), pred_contrib=True)
    np_assert_array_equal(p_pred_contrib_arrow, p_pred_contrib_pandas, strict=True)

    p_first_iter_arrow = booster.predict(data, start_iteration=0, num_iteration=1, raw_score=True)
    p_first_iter_pandas = booster.predict(
        data.to_pandas(), start_iteration=0, num_iteration=1, raw_score=True
    )
    np_assert_array_equal(p_first_iter_arrow, p_first_iter_pandas, strict=True)


def test_predict_regression():
    data = generate_random_arrow_table(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_arrow_array(10000, 43, generate_nulls=False),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_arrow_pandas(booster, data)


def test_predict_binary_classification():
    data = generate_random_arrow_table(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_arrow_array(10000, 43, generate_nulls=False, values=np.arange(2)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_arrow_pandas(booster, data)


def test_predict_multiclass_classification():
    data = generate_random_arrow_table(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_arrow_array(10000, 43, generate_nulls=False, values=np.arange(5)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "multiclass", "num_leaves": 7, "num_class": 5},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_arrow_pandas(booster, data)


def test_predict_ranking():
    data = generate_random_arrow_table(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_arrow_array(10000, 43, generate_nulls=False, values=np.arange(4)),
        group=np.array([1000, 2000, 3000, 4000]),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_arrow_pandas(booster, data)
