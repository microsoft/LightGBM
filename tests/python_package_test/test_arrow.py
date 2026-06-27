# coding: utf-8
import filecmp
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

import lightgbm as lgb

from .utils import np_assert_array_equal

pa = pytest.importorskip("pyarrow")


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
        pa.chunked_array(c + [[True, True, False]] + c + [[False, True]] + c, type=pa.bool_()),
    ]
    return pa.Table.from_arrays(columns, names=[f"col_{i}" for i in range(len(columns))])


def generate_nullable_arrow_table(dtype: Any) -> pa.Table:
    columns = [
        pa.chunked_array([[1, None, 3, 4, 5]], type=dtype),
        pa.chunked_array([[None, 2, 3, 4, 5]], type=dtype),
        pa.chunked_array([[1, 2, 3, 4, None]], type=dtype),
        pa.chunked_array([[None, None, None, None, None]], type=dtype),
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
        generate_random_arrow_array(num_datapoints, seed + i, generate_nulls=generate_nulls, values=values)
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
    return pa.chunked_array(chunks, type=pa.float32())


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
    }


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- DATASET ------------------------------------------- #


def assert_datasets_equal(tmp_path: Path, lhs: lgb.Dataset, rhs: lgb.Dataset):
    lhs._dump_text(tmp_path / "arrow.txt")
    rhs._dump_text(tmp_path / "pandas.txt")
    assert filecmp.cmp(tmp_path / "arrow.txt", tmp_path / "pandas.txt")


@pytest.mark.parametrize(
    ("arrow_table_fn", "dataset_params"),
    [  # Use lambda functions here to minimize memory consumption
        (lambda: generate_simple_arrow_table(), dummy_dataset_params()),
        (lambda: generate_simple_arrow_table(empty_chunks=True), dummy_dataset_params()),
        (lambda: generate_dummy_arrow_table(), dummy_dataset_params()),
        (lambda: generate_nullable_arrow_table(pa.float32()), dummy_dataset_params()),
        (lambda: generate_nullable_arrow_table(pa.int32()), dummy_dataset_params()),
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

    assert_datasets_equal(tmp_path, arrow_dataset, pandas_dataset)


def test_dataset_construct_fuzzy_boolean(tmp_path):
    boolean_data = generate_random_arrow_table(10, 10000, 42, generate_nulls=False, values=np.array([True, False]))

    float_schema = pa.schema([pa.field(f"col_{i}", pa.float32()) for i in range(len(boolean_data.columns))])
    float_data = boolean_data.cast(float_schema)

    arrow_dataset = lgb.Dataset(boolean_data)
    arrow_dataset.construct()

    pandas_dataset = lgb.Dataset(float_data.to_pandas())
    pandas_dataset.construct()

    assert_datasets_equal(tmp_path, arrow_dataset, pandas_dataset)


# -------------------------------------------- FIELDS ------------------------------------------- #


def test_dataset_construct_fields_fuzzy():
    arrow_table = generate_random_arrow_table(3, 1000, 42)
    arrow_labels = generate_random_arrow_array(1000, 42, generate_nulls=False)
    arrow_weights = generate_random_arrow_array(1000, 42, generate_nulls=False)
    arrow_groups = pa.chunked_array([[300, 400, 50], [250]], type=pa.int32())

    arrow_dataset = lgb.Dataset(arrow_table, label=arrow_labels, weight=arrow_weights, group=arrow_groups)
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
        np_assert_array_equal(arrow_dataset.get_field(field), pandas_dataset.get_field(field), strict=True)
    np_assert_array_equal(arrow_dataset.get_label(), pandas_dataset.get_label(), strict=True)
    np_assert_array_equal(arrow_dataset.get_weight(), pandas_dataset.get_weight(), strict=True)


# -------------------------------------------- LABELS ------------------------------------------- #


@pytest.mark.parametrize(
    "label_data",
    [
        [[0, 1, 0, 0, 1]],
        [[0], [1, 0, 0, 1]],
        [[], [0], [1, 0, 0, 1]],
        [[0], [], [1, 0], [], [], [0, 1], []],
    ],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_labels(label_data, arrow_type):
    data = generate_dummy_arrow_table()
    labels = pa.chunked_array(label_data, type=arrow_type)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)


@pytest.mark.parametrize(
    "label_data",
    [
        [[False, True, False, False, True]],
        [[False], [True, False, False, True]],
        [[], [False], [True, False, False, True]],
        [[False], [], [True, False], [], [], [False, True], []],
    ],
)
def test_dataset_construct_labels_boolean(label_data):
    data = generate_dummy_arrow_table()
    labels = pa.chunked_array(label_data, type=pa.bool_())
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)


# ------------------------------------------- WEIGHTS ------------------------------------------- #


def test_dataset_construct_weights_none():
    data = generate_dummy_arrow_table()
    weight = pa.chunked_array([[1, 1, 1, 1, 1]])
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None
    assert dataset.get_field("weight") is None


@pytest.mark.parametrize(
    "weight_data",
    [
        [[3, 0.7, 1.5, 0.5, 0.1]],
        [[3], [0.7, 1.5, 0.5, 0.1]],
        [[], [3], [0.7, 1.5, 0.5, 0.1]],
        [[3], [0.7], [], [], [1.5, 0.5, 0.1], []],
    ],
)
@pytest.mark.parametrize("arrow_type", _FLOAT_TYPES)
def test_dataset_construct_weights(weight_data, arrow_type):
    data = generate_dummy_arrow_table()
    weights = pa.chunked_array(weight_data, type=arrow_type)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_weight(), strict=True)


# -------------------------------------------- GROUPS ------------------------------------------- #


@pytest.mark.parametrize(
    "group_data",
    [
        [[2, 3]],
        [[2], [3]],
        [[], [2, 3]],
        [[2], [], [3], []],
    ],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES)
def test_dataset_construct_groups(group_data, arrow_type):
    data = generate_dummy_arrow_table()
    groups = pa.chunked_array(group_data, type=arrow_type)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 2, 5], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_field("group"), strict=True)


# ----------------------------------------- INIT SCORES ----------------------------------------- #


@pytest.mark.parametrize(
    "init_score_data",
    [
        [[0, 1, 2, 3, 3]],
        [[0, 1, 2], [3, 3]],
        [[], [0, 1, 2], [3, 3]],
        [[0, 1], [], [], [2], [3, 3], []],
    ],
)
@pytest.mark.parametrize("arrow_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_init_scores_array(init_score_data, arrow_type):
    data = generate_dummy_arrow_table()
    init_scores = pa.chunked_array(init_score_data, type=arrow_type)
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
    p_first_iter_pandas = booster.predict(data.to_pandas(), start_iteration=0, num_iteration=1, raw_score=True)
    np_assert_array_equal(p_first_iter_arrow, p_first_iter_pandas, strict=True)


def test_predict_regression():
    data_float = generate_random_arrow_table(10, 10000, 42)
    data_bool = generate_random_arrow_table(1, 10000, 42, generate_nulls=False, values=np.array([True, False]))
    data = pa.Table.from_arrays(data_float.columns + data_bool.columns, names=data_float.schema.names + ["col_bool"])

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


def test_arrow_feature_name_auto():
    data = generate_dummy_arrow_table()
    dataset = lgb.Dataset(
        data,
        label=pa.chunked_array([[0, 1, 0, 0, 1]]),
        params=dummy_dataset_params(),
        categorical_feature=["a"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["a", "b"]


def test_arrow_feature_name_manual():
    data = generate_dummy_arrow_table()
    dataset = lgb.Dataset(
        data,
        label=pa.chunked_array([[0, 1, 0, 0, 1]]),
        params=dummy_dataset_params(),
        feature_name=["c", "d"],
        categorical_feature=["c"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["c", "d"]


def pyarrow_array_equal(arr1: pa.ChunkedArray, arr2: pa.ChunkedArray) -> bool:
    """Similar to ``np.array_equal()``, but for ``pyarrow.Array`` objects.

    ``pyarrow.Array`` objects with identical values do not compare equal if any of those
    values are nulls. This function treats them as equal.
    """
    if len(arr1) != len(arr2):
        return False

    np1 = arr1.to_numpy()
    np2 = arr2.to_numpy()
    return np.array_equal(np1, np2, equal_nan=True)


def test_get_data_arrow_table():
    original_table = generate_simple_arrow_table()
    dataset = lgb.Dataset(original_table, free_raw_data=False)
    dataset.construct()

    returned_data = dataset.get_data()
    assert isinstance(returned_data, pa.Table)
    assert returned_data.schema == original_table.schema
    assert returned_data.shape == original_table.shape

    for column_name in original_table.column_names:
        original_column = original_table[column_name]
        returned_column = returned_data[column_name]

        assert original_column.type == returned_column.type
        assert original_column.num_chunks == returned_column.num_chunks
        assert pyarrow_array_equal(original_column, returned_column)

        for i in range(original_column.num_chunks):
            original_chunk_array = pa.chunked_array([original_column.chunk(i)])
            returned_chunk_array = pa.chunked_array([returned_column.chunk(i)])
            assert pyarrow_array_equal(original_chunk_array, returned_chunk_array)


def test_get_data_arrow_table_subset(rng):
    original_table = generate_random_arrow_table(num_columns=3, num_datapoints=1000, seed=42)
    dataset = lgb.Dataset(original_table, free_raw_data=False)
    dataset.construct()

    subset_size = 100
    used_indices = rng.choice(a=original_table.shape[0], size=subset_size, replace=False)
    used_indices = sorted(used_indices)

    subset_dataset = dataset.subset(used_indices).construct()
    expected_subset = original_table.take(used_indices)
    subset_data = subset_dataset.get_data()

    assert isinstance(subset_data, pa.Table)
    assert subset_data.schema == expected_subset.schema
    assert subset_data.shape == expected_subset.shape
    assert len(subset_data) == len(used_indices)
    assert subset_data.shape == (subset_size, 3)

    for column_name in expected_subset.column_names:
        expected_col = expected_subset[column_name]
        returned_col = subset_data[column_name]
        assert expected_col.type == returned_col.type
        assert pyarrow_array_equal(expected_col, returned_col)


# ------------------------------------------- CATEGORICAL ----------------------------------------- #


def test_arrow_categorical_basic():
    """Explicit categorical_feature constructs successfully and metadata is captured."""
    df = pa.table(
        {
            "cat_col": pa.array(["a", "b", "a", "c", "b"]).dictionary_encode(),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat_col"], params={"min_data_in_bin": 1})
    ds.construct()

    assert ds.pandas_categorical is not None
    assert len(ds.pandas_categorical) == 1
    assert sorted(ds.pandas_categorical[0]) == ["a", "b", "c"]


def test_arrow_categorical_doesnt_modify_original():
    """Construction must not mutate the input Table."""
    original_table = pa.table(
        {
            "cat_col": pa.array(["a", "b", "a", "c"]).dictionary_encode(),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    y = [0, 1, 0, 1]

    original_values = original_table["cat_col"].to_pylist()
    original_type = original_table["cat_col"].type

    ds = lgb.Dataset(original_table, label=y, categorical_feature=["cat_col"], params={"min_data_in_bin": 1})
    ds.construct()

    assert original_table["cat_col"].to_pylist() == original_values
    assert original_table["cat_col"].type == original_type


def test_arrow_categorical_multiple_columns():
    """Two categorical columns alongside a numeric column are both encoded."""
    df = pa.table(
        {
            "cat1": pa.array(["a", "b", "a", "c"]).dictionary_encode(),
            "cat2": pa.array(["x", "x", "y", "z"]).dictionary_encode(),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    y = [0, 1, 0, 1]

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat1", "cat2"], params={"min_data_in_bin": 1})
    ds.construct()

    assert len(ds.pandas_categorical) == 2
    assert sorted(ds.pandas_categorical[0]) == ["a", "b", "c"]
    assert sorted(ds.pandas_categorical[1]) == ["x", "y", "z"]


def test_arrow_categorical_validation_uses_train_mapping():
    """A valid table whose categorical column has a *different* category ordering must
    still be encoded using train's category-to-code mapping."""
    train_values = ["a", "b", "c"] * 30
    train_labels = [0, 1, 0] * 30
    valid_values = ["c", "a", "c", "b", "a", "b", "c"] * 3

    train_table = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 2] * 30), pa.array(["a", "b", "c"])
            ),
            "num_col": pa.array([float(i % 5) for i in range(len(train_values))]),
        }
    )
    valid_table = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(
                pa.array([2, 0, 2, 1, 0, 1, 2] * 3), pa.array(["a", "b", "c"])
            ),
            "num_col": pa.array([float(i % 5) for i in range(len(valid_values))]),
        }
    )

    train_ds = lgb.Dataset(
        train_table, label=train_labels, categorical_feature=["cat_col"], params={"min_data_in_bin": 1}
    )
    bst = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, train_ds, num_boost_round=20)
    assert train_ds.pandas_categorical == [["a", "b", "c"]]

    # Reference: encode valid_values with train's mapping (a=0, b=1, c=2) and feed as a
    # plain numpy array so the categorical path is bypassed entirely.
    train_code = {c: i for i, c in enumerate(["a", "b", "c"])}
    pre_encoded = np.array(
        [[float(train_code[v]), float(i % 5)] for i, v in enumerate(valid_values)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(bst.predict(valid_table), bst.predict(pre_encoded))


def test_arrow_categorical_matches_pandas(tmp_path):
    """Arrow-built Datasets (train + valid) match the pandas-built equivalents."""
    pd = pytest.importorskip("pandas")

    train_values = ["a", "b", "c", "a"]
    valid_values = ["c", "a", "c"]

    arrow_train = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 2, 0]), pa.array(["a", "b", "c"])
            ),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0]),
        }
    )
    arrow_valid = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(
                pa.array([2, 0, 2]), pa.array(["a", "b", "c"])
            ),
            "num_col": pa.array([5.0, 6.0, 7.0]),
        }
    )
    pandas_train = pd.DataFrame(
        {
            "cat_col": pd.Categorical(train_values, categories=["a", "b", "c"], ordered=False),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pandas_valid = pd.DataFrame(
        {
            "cat_col": pd.Categorical(valid_values, categories=["a", "b", "c"], ordered=False),
            "num_col": [5.0, 6.0, 7.0],
        }
    )

    params = {"min_data_in_bin": 1}
    arrow_train_ds = lgb.Dataset(arrow_train, label=[0, 1, 0, 1], categorical_feature=["cat_col"], params=params)
    arrow_train_ds.construct()
    arrow_valid_ds = lgb.Dataset(arrow_valid, label=[1, 0, 1], reference=arrow_train_ds, params=params)
    arrow_valid_ds.construct()
    pandas_train_ds = lgb.Dataset(pandas_train, label=[0, 1, 0, 1], categorical_feature=["cat_col"], params=params)
    pandas_train_ds.construct()
    pandas_valid_ds = lgb.Dataset(pandas_valid, label=[1, 0, 1], reference=pandas_train_ds, params=params)
    pandas_valid_ds.construct()

    assert arrow_train_ds.pandas_categorical == pandas_train_ds.pandas_categorical
    assert_datasets_equal(tmp_path, arrow_train_ds, pandas_train_ds)
    assert_datasets_equal(tmp_path, arrow_valid_ds, pandas_valid_ds)


def test_arrow_categorical_high_cardinality():
    """Construction works with a large number of unique categories."""
    rng = np.random.default_rng(42)
    categories = [f"cat_{i}" for i in range(1000)]
    values = categories + rng.choice(categories, size=4000).tolist()
    indices = [categories.index(v) for v in values]

    df = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(pa.array(indices), pa.array(categories)),
            "num_col": pa.array(rng.uniform(0, 10, size=5000)),
        }
    )
    y = rng.integers(0, 2, size=5000)

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat_col"])
    ds.construct()

    assert ds.num_data() == 5000
    assert ds.num_feature() == 2
    assert len(ds.pandas_categorical[0]) == 1000


def test_arrow_categorical_prediction_and_persistence(tmp_path):
    """End-to-end: train, predict, save/load, predictions match."""
    train_values = ["a", "b", "a", "c", "b", "c"] * 10
    test_values = ["a", "b", "c", "a"]
    cats = sorted(set(train_values))
    train_indices = [cats.index(v) for v in train_values]
    test_indices = [cats.index(v) for v in test_values]

    train_table = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(pa.array(train_indices), pa.array(cats)),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 10),
        }
    )
    train_y = [0, 1, 0, 1, 0, 1] * 10
    test_table = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(pa.array(test_indices), pa.array(cats)),
            "num_col": pa.array([1.5, 2.5, 3.5, 4.5]),
        }
    )

    train_ds = lgb.Dataset(train_table, label=train_y, categorical_feature=["cat_col"])
    bst = lgb.train({"objective": "binary", "verbose": -1}, train_ds, num_boost_round=10)

    preds = bst.predict(test_table)
    assert preds.shape == (4,)
    assert all(0 <= p <= 1 for p in preds)

    model_path = tmp_path / "categorical_model.txt"
    bst.save_model(model_path)
    loaded_bst = lgb.Booster(model_file=model_path)
    assert loaded_bst.pandas_categorical == bst.pandas_categorical
    np.testing.assert_allclose(preds, loaded_bst.predict(test_table))


def test_arrow_pandas_categorical_predictions_match():
    """Arrow-trained and pandas-trained models give identical predictions."""
    pd = pytest.importorskip("pandas")

    cats = sorted(["cat_a", "cat_b", "cat_c"])
    values = ["cat_a", "cat_b", "cat_c", "cat_a", "cat_b"] * 20
    indices = [cats.index(v) for v in values]

    arrow_table = pa.table(
        {
            "cat_col": pa.DictionaryArray.from_arrays(pa.array(indices), pa.array(cats)),
            "num_col": pa.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20),
        }
    )
    pandas_df = pd.DataFrame(
        {
            "cat_col": pd.Categorical(values, categories=cats, ordered=False),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        }
    )
    y = [0, 1, 0, 1, 0] * 20

    arrow_ds = lgb.Dataset(arrow_table, label=y, categorical_feature=["cat_col"])
    arrow_bst = lgb.train({"objective": "binary", "verbose": -1, "seed": 42}, arrow_ds, num_boost_round=10)

    pandas_ds = lgb.Dataset(pandas_df, label=y, categorical_feature=["cat_col"])
    pandas_bst = lgb.train({"objective": "binary", "verbose": -1, "seed": 42}, pandas_ds, num_boost_round=10)

    np.testing.assert_allclose(arrow_bst.predict(arrow_table), pandas_bst.predict(pandas_df), rtol=1e-10)


def test_arrow_categorical_auto_detected():
    """categorical_feature='auto' picks up dictionary-encoded columns."""
    df = pa.table(
        {
            "cat_col": pa.array(["x", "y", "x"]).dictionary_encode(),
            "num_col": pa.array([1.0, 2.0, 3.0]),
        }
    )
    y = [0, 1, 0]

    ds = lgb.Dataset(df, label=y, categorical_feature="auto", params={"min_data_in_bin": 1})
    ds.construct()

    assert ds.params.get("categorical_column") == [0]
    assert len(ds.pandas_categorical) == 1
