# coding: utf-8
import filecmp
import itertools
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

import lightgbm as lgb

from .utils import np_assert_array_equal

pl = pytest.importorskip("polars")


# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #

_INTEGER_TYPES = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
_FLOAT_TYPES = [pl.Float32, pl.Float64]


def generate_simple_polars_frame() -> pl.DataFrame:
    values = [1, 2, 3, 4, 5]
    bool_values = [True, True, False, False, True]
    columns = {f"col_{i}": pl.Series(values, dtype=dtype) for i, dtype in enumerate(_INTEGER_TYPES + _FLOAT_TYPES)}
    columns[f"col_{len(columns)}"] = pl.Series(bool_values, dtype=pl.Boolean)
    return pl.DataFrame(columns)


def generate_nullable_polars_frame(dtype: Any) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col_0": pl.Series([1, None, 3, 4, 5], dtype=dtype),
            "col_1": pl.Series([None, 2, 3, 4, 5], dtype=dtype),
            "col_2": pl.Series([1, 2, 3, 4, None], dtype=dtype),
            "col_3": pl.Series([None, None, None, None, None], dtype=dtype),
        }
    )


def generate_dummy_polars_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.UInt8),
            "b": pl.Series([0.5, 0.6, 0.1, 0.8, 1.5], dtype=pl.Float32),
        }
    )


def generate_random_polars_frame(
    num_columns: int,
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            f"col_{i}": generate_random_polars_series(
                num_datapoints, seed + i, generate_nulls=generate_nulls, values=values
            )
            for i in range(num_columns)
        }
    )


def generate_random_polars_series(
    num_datapoints: int,
    seed: int,
    generate_nulls: bool = True,
    values: Optional[np.ndarray] = None,
) -> pl.Series:
    generator = np.random.default_rng(seed)
    data = (
        generator.standard_normal(num_datapoints).astype(np.float32)
        if values is None
        else generator.choice(values, size=num_datapoints, replace=True)
    )
    series = pl.Series("col", data, dtype=pl.Float32)
    if generate_nulls:
        indices = generator.choice(len(data), size=num_datapoints // 10)
        series = series.scatter(indices, None)
    return series


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
    lhs._dump_text(tmp_path / "polars.txt")
    rhs._dump_text(tmp_path / "pandas.txt")
    assert filecmp.cmp(tmp_path / "polars.txt", tmp_path / "pandas.txt")


@pytest.mark.parametrize(
    ("polars_frame_fn", "dataset_params"),
    [  # Use lambda functions here to minimize memory consumption
        (lambda: generate_simple_polars_frame(), dummy_dataset_params()),
        (lambda: generate_dummy_polars_frame(), dummy_dataset_params()),
        (lambda: generate_nullable_polars_frame(pl.Float32), dummy_dataset_params()),
        (lambda: generate_nullable_polars_frame(pl.Int32), dummy_dataset_params()),
        (lambda: generate_random_polars_frame(3, 1000, 42), {}),
        (lambda: generate_random_polars_frame(100, 10000, 43), {}),
    ],
)
def test_dataset_construct_fuzzy(tmp_path, polars_frame_fn, dataset_params):
    polars_frame = polars_frame_fn()

    polars_dataset = lgb.Dataset(polars_frame, params=dataset_params)
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(polars_frame.to_pandas(), params=dataset_params)
    pandas_dataset.construct()

    assert_datasets_equal(tmp_path, polars_dataset, pandas_dataset)


def test_dataset_construct_fuzzy_boolean(tmp_path):
    boolean_data = generate_random_polars_frame(10, 10000, 42, generate_nulls=False, values=np.array([True, False]))
    float_data = boolean_data.cast(pl.Float32)

    polars_dataset = lgb.Dataset(boolean_data)
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(float_data.to_pandas())
    pandas_dataset.construct()

    assert_datasets_equal(tmp_path, polars_dataset, pandas_dataset)


# -------------------------------------------- FIELDS ------------------------------------------- #


def test_dataset_construct_fields_fuzzy():
    polars_frame = generate_random_polars_frame(3, 1000, 42)
    polars_labels = generate_random_polars_series(1000, 42, generate_nulls=False)
    polars_weights = generate_random_polars_series(1000, 42, generate_nulls=False)
    polars_groups = pl.Series("group", [300, 400, 50, 250], dtype=pl.Int32)

    polars_dataset = lgb.Dataset(polars_frame, label=polars_labels, weight=polars_weights, group=polars_groups)
    polars_dataset.construct()

    pandas_dataset = lgb.Dataset(
        polars_frame.to_pandas(),
        label=polars_labels.to_numpy(),
        weight=polars_weights.to_numpy(),
        group=polars_groups.to_numpy(),
    )
    pandas_dataset.construct()

    # Check for equality
    for field in ("label", "weight", "group"):
        np_assert_array_equal(polars_dataset.get_field(field), pandas_dataset.get_field(field), strict=True)
    np_assert_array_equal(polars_dataset.get_label(), pandas_dataset.get_label(), strict=True)
    np_assert_array_equal(polars_dataset.get_weight(), pandas_dataset.get_weight(), strict=True)


# -------------------------------------------- LABELS ------------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_labels(polars_type):
    data = generate_dummy_polars_frame()
    labels = pl.Series("label", [0, 1, 0, 0, 1], dtype=polars_type)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)


def test_dataset_construct_labels_boolean():
    data = generate_dummy_polars_frame()
    labels = pl.Series("label", [False, True, False, False, True], dtype=pl.Boolean)
    dataset = lgb.Dataset(data, label=labels, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 0, 0, 1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_label(), strict=True)


# ------------------------------------------- WEIGHTS ------------------------------------------- #


def test_dataset_construct_weights_none():
    data = generate_dummy_polars_frame()
    weight = pl.Series("weight", [1, 1, 1, 1, 1], dtype=pl.Float32)
    dataset = lgb.Dataset(data, weight=weight, params=dummy_dataset_params())
    dataset.construct()
    assert dataset.get_weight() is None
    assert dataset.get_field("weight") is None


@pytest.mark.parametrize("polars_type", _FLOAT_TYPES)
def test_dataset_construct_weights(polars_type):
    data = generate_dummy_polars_frame()
    weights = pl.Series("weight", [3, 0.7, 1.5, 0.5, 0.1], dtype=polars_type)
    dataset = lgb.Dataset(data, weight=weights, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([3, 0.7, 1.5, 0.5, 0.1], dtype=np.float32)
    np_assert_array_equal(expected, dataset.get_weight(), strict=True)


# -------------------------------------------- GROUPS ------------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES)
def test_dataset_construct_groups(polars_type):
    data = generate_dummy_polars_frame()
    groups = pl.Series("group", [2, 3], dtype=polars_type)
    dataset = lgb.Dataset(data, group=groups, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 2, 5], dtype=np.int32)
    np_assert_array_equal(expected, dataset.get_field("group"), strict=True)


# ----------------------------------------- INIT SCORES ----------------------------------------- #


@pytest.mark.parametrize("polars_type", _INTEGER_TYPES + _FLOAT_TYPES)
def test_dataset_construct_init_scores_array(polars_type):
    data = generate_dummy_polars_frame()
    init_scores = pl.Series("init_score", [0, 1, 2, 3, 3], dtype=polars_type)
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    expected = np.array([0, 1, 2, 3, 3], dtype=np.float64)
    np_assert_array_equal(expected, dataset.get_init_score(), strict=True)


def test_dataset_construct_init_scores_table():
    data = generate_dummy_polars_frame()
    init_scores = pl.DataFrame(
        {
            "a": generate_random_polars_series(5, seed=1, generate_nulls=False),
            "b": generate_random_polars_series(5, seed=2, generate_nulls=False),
            "c": generate_random_polars_series(5, seed=3, generate_nulls=False),
        }
    )
    dataset = lgb.Dataset(data, init_score=init_scores, params=dummy_dataset_params())
    dataset.construct()

    actual = dataset.get_init_score()
    expected = init_scores.to_numpy().astype(np.float64)
    np_assert_array_equal(expected, actual, strict=True)


# ------------------------------------------ PREDICTION ----------------------------------------- #


def assert_equal_predict_polars_pandas(booster: lgb.Booster, data: pl.DataFrame):
    pandas_data = data.to_pandas()

    p_polars = booster.predict(data)
    p_pandas = booster.predict(pandas_data)
    np_assert_array_equal(p_polars, p_pandas, strict=True)

    p_raw_polars = booster.predict(data, raw_score=True)
    p_raw_pandas = booster.predict(pandas_data, raw_score=True)
    np_assert_array_equal(p_raw_polars, p_raw_pandas, strict=True)

    p_leaf_polars = booster.predict(data, pred_leaf=True)
    p_leaf_pandas = booster.predict(pandas_data, pred_leaf=True)
    np_assert_array_equal(p_leaf_polars, p_leaf_pandas, strict=True)

    p_pred_contrib_polars = booster.predict(data, pred_contrib=True)
    p_pred_contrib_pandas = booster.predict(pandas_data, pred_contrib=True)
    np_assert_array_equal(p_pred_contrib_polars, p_pred_contrib_pandas, strict=True)

    p_first_iter_polars = booster.predict(data, start_iteration=0, num_iteration=1, raw_score=True)
    p_first_iter_pandas = booster.predict(pandas_data, start_iteration=0, num_iteration=1, raw_score=True)
    np_assert_array_equal(p_first_iter_polars, p_first_iter_pandas, strict=True)


def test_predict_regression():
    data_float = generate_random_polars_frame(10, 10000, 42)
    data_bool = generate_random_polars_frame(1, 10000, 42, generate_nulls=False, values=np.array([True, False]))
    data = data_float.with_columns(data_bool["col_0"].alias("col_bool"))

    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(10000, 43, generate_nulls=False),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "regression", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_binary_classification():
    data = generate_random_polars_frame(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(10000, 43, generate_nulls=False, values=np.arange(2)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_multiclass_classification():
    data = generate_random_polars_frame(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(10000, 43, generate_nulls=False, values=np.arange(5)),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "multiclass", "num_leaves": 7, "num_class": 5},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_predict_ranking():
    data = generate_random_polars_frame(10, 10000, 42)
    dataset = lgb.Dataset(
        data,
        label=generate_random_polars_series(10000, 43, generate_nulls=False, values=np.arange(4)),
        group=np.array([1000, 2000, 3000, 4000]),
        params=dummy_dataset_params(),
    )
    booster = lgb.train(
        {"objective": "lambdarank", "num_leaves": 7},
        dataset,
        num_boost_round=5,
    )
    assert_equal_predict_polars_pandas(booster, data)


def test_polars_feature_name_auto():
    data = generate_dummy_polars_frame()
    dataset = lgb.Dataset(
        data,
        label=pl.Series("label", [0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        categorical_feature=["a"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["a", "b"]


def test_polars_feature_name_manual():
    data = generate_dummy_polars_frame()
    dataset = lgb.Dataset(
        data,
        label=pl.Series("label", [0, 1, 0, 0, 1]),
        params=dummy_dataset_params(),
        feature_name=["c", "d"],
        categorical_feature=["c"],
    )
    booster = lgb.train({"num_leaves": 7}, dataset, num_boost_round=5)
    assert booster.feature_name() == ["c", "d"]


def test_get_data_polars_frame():
    from polars.testing import assert_frame_equal  # noqa: PLC0415

    original_frame = generate_simple_polars_frame()
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    returned_data = dataset.get_data()
    assert isinstance(returned_data, pl.DataFrame)
    assert returned_data.schema == original_frame.schema
    assert returned_data.shape == original_frame.shape
    assert_frame_equal(returned_data, original_frame)


def test_get_data_polars_frame_subset(rng):
    from polars.testing import assert_frame_equal  # noqa: PLC0415

    original_frame = generate_random_polars_frame(num_columns=3, num_datapoints=1000, seed=42)
    dataset = lgb.Dataset(original_frame, free_raw_data=False)
    dataset.construct()

    subset_size = 100
    used_indices = rng.choice(a=original_frame.shape[0], size=subset_size, replace=False)
    used_indices = sorted(used_indices)

    subset_dataset = dataset.subset(used_indices).construct()
    expected_subset = original_frame[used_indices]
    subset_data = subset_dataset.get_data()

    assert isinstance(subset_data, pl.DataFrame)
    assert subset_data.schema == expected_subset.schema
    assert subset_data.shape == expected_subset.shape
    assert len(subset_data) == len(used_indices)
    assert subset_data.shape == (subset_size, 3)
    assert_frame_equal(subset_data, expected_subset)


# ------------------------------------------- CATEGORICAL ----------------------------------------- #

# Starting with polars 1.41, pl.Categorical columns share a process-wide categories dictionary
# by default, so categories from unrelated columns leak into one another's `cat.get_categories()`
# output; pl.Categories was added to restore per-column scoping. We use a unique scope per helper
# call when available so each test sees an isolated dictionary and can assert exact equality.
# On older polars (<1.41), each pl.Categorical column already has its own local dictionary, so
# no extra scoping is needed.
_HAS_POLARS_CATEGORIES = hasattr(pl, "Categories")
_cat_scope_counter = itertools.count()


def _polars_cat_series(values, cat_type, categories=None):
    """Build a polars categorical-like Series for the given dtype family.

    cat_type: "categorical" -> pl.Categorical (unordered, scoped per call when supported)
              "enum"        -> pl.Enum (ordered, fixed category list)
    """
    if cat_type == "categorical":
        if _HAS_POLARS_CATEGORIES:
            scope = pl.Categories(name=f"lgbm_test_{next(_cat_scope_counter)}")
            return pl.Series(values, dtype=pl.Categorical(categories=scope))
        return pl.Series(values, dtype=pl.Categorical)
    cats = categories if categories is not None else sorted(set(values))
    return pl.Series(values).cast(pl.Enum(cats))


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_basic(cat_type):
    """Explicit categorical_feature constructs successfully and metadata is captured."""
    df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(["a", "b", "a", "c", "b"], cat_type),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat_col"], params={"min_data_in_bin": 1})
    ds.construct()

    assert ds.pandas_categorical is not None
    assert len(ds.pandas_categorical) == 1
    assert sorted(ds.pandas_categorical[0]) == ["a", "b", "c"]


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_doesnt_modify_original(cat_type):
    """Construction must not mutate the input DataFrame."""
    original_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(["a", "b", "a", "c"], cat_type),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = [0, 1, 0, 1]

    original_values = original_df["cat_col"].to_list()
    original_dtype = original_df["cat_col"].dtype

    ds = lgb.Dataset(original_df, label=y, categorical_feature=["cat_col"], params={"min_data_in_bin": 1})
    ds.construct()

    assert original_df["cat_col"].to_list() == original_values
    assert original_df["cat_col"].dtype == original_dtype


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_multiple_columns(cat_type):
    """Two categorical columns alongside a numeric column are both encoded."""
    df = pl.DataFrame(
        {
            "cat1": _polars_cat_series(["a", "b", "a", "c"], cat_type),
            "cat2": _polars_cat_series(["x", "x", "y", "z"], cat_type),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = [0, 1, 0, 1]

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat1", "cat2"], params={"min_data_in_bin": 1})
    ds.construct()

    assert len(ds.pandas_categorical) == 2
    assert sorted(ds.pandas_categorical[0]) == ["a", "b", "c"]
    assert sorted(ds.pandas_categorical[1]) == ["x", "y", "z"]


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_validation_alignment(cat_type):
    """Booster predictions on a polars valid frame match a numerically pre-encoded equivalent using train's codes."""
    cats = ["a", "b", "c"]
    train_values = ["a", "b", "c"] * 30
    train_labels = [0, 1, 0] * 30
    # subset of train's cats AND in different row order
    valid_values = ["c", "a", "c", "b", "a", "b", "c"] * 3
    train_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(train_values, cat_type, categories=cats),
            "num_col": [float(i % 5) for i in range(len(train_values))],
        }
    )
    valid_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(valid_values, cat_type, categories=cats),
            "num_col": [float(i % 5) for i in range(len(valid_values))],
        }
    )
    pre_encoded_valid_df = pl.DataFrame(
        {
            "cat_col": pl.Series([float(cats.index(v)) for v in valid_values], dtype=pl.Float64),
            "num_col": [float(i % 5) for i in range(len(valid_values))],
        }
    )

    train_ds = lgb.Dataset(train_df, label=train_labels, categorical_feature=["cat_col"], params={"min_data_in_bin": 1})
    bst = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, train_ds, num_boost_round=20)

    assert train_ds.pandas_categorical == [cats]
    assert bst.pandas_categorical == [cats]
    np.testing.assert_allclose(bst.predict(valid_df), bst.predict(pre_encoded_valid_df))


@pytest.mark.parametrize(
    ("cat_type", "valid_values"),
    [
        ("categorical", ["a", "z", "c"]),  # unseen "z" -> null (Enum can't represent unseen)
        ("enum", ["c", "a", "c"]),
    ],
)
def test_polars_categorical_matches_pandas(tmp_path, cat_type, valid_values):
    """Polars-built Datasets (train + valid) match the pandas-built equivalents, including unseen-category handling."""
    pd = pytest.importorskip("pandas")

    cats = ["a", "b", "c"]
    train_values = ["a", "b", "c", "a"]
    polars_train = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(train_values, cat_type, categories=cats),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    polars_valid = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(valid_values, cat_type, categories=cats),
            "num_col": [5.0, 6.0, 7.0],
        }
    )
    pandas_train = pd.DataFrame(
        {
            "cat_col": pd.Categorical(train_values, categories=cats, ordered=cat_type == "enum"),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pandas_valid = pd.DataFrame(
        {
            "cat_col": pd.Categorical(valid_values, categories=cats, ordered=cat_type == "enum"),
            "num_col": [5.0, 6.0, 7.0],
        }
    )

    params = {"min_data_in_bin": 1}
    polars_train_ds = lgb.Dataset(polars_train, label=[0, 1, 0, 1], categorical_feature=["cat_col"], params=params)
    polars_train_ds.construct()
    polars_valid_ds = lgb.Dataset(polars_valid, label=[1, 0, 1], reference=polars_train_ds, params=params)
    polars_valid_ds.construct()
    pandas_train_ds = lgb.Dataset(pandas_train, label=[0, 1, 0, 1], categorical_feature=["cat_col"], params=params)
    pandas_train_ds.construct()
    pandas_valid_ds = lgb.Dataset(pandas_valid, label=[1, 0, 1], reference=pandas_train_ds, params=params)
    pandas_valid_ds.construct()

    assert polars_train_ds.pandas_categorical == pandas_train_ds.pandas_categorical
    assert_datasets_equal(tmp_path, polars_train_ds, pandas_train_ds)
    assert_datasets_equal(tmp_path, polars_valid_ds, pandas_valid_ds)


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_high_cardinality(cat_type):
    """Construction works with a large number of unique categories."""
    rng = np.random.default_rng(42)
    categories = [f"cat_{i}" for i in range(1000)]
    # include every category at least once so the inferred set is exactly 1000
    values = categories + rng.choice(categories, size=4000).tolist()

    df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(values, cat_type, categories=categories),
            "num_col": rng.uniform(0, 10, size=5000),
        }
    )
    y = rng.integers(0, 2, size=5000)

    ds = lgb.Dataset(df, label=y, categorical_feature=["cat_col"])
    ds.construct()

    assert ds.num_data() == 5000
    assert ds.num_feature() == 2
    assert len(ds.pandas_categorical[0]) == 1000


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_categorical_prediction_and_persistence(tmp_path, cat_type):
    """End-to-end: train, predict, save/load, predictions match."""
    train_values = ["a", "b", "a", "c", "b", "c"] * 10
    test_values = ["a", "b", "c", "a"]
    cats = sorted(set(train_values))

    train_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(train_values, cat_type, categories=cats),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 10,
        }
    )
    train_y = [0, 1, 0, 1, 0, 1] * 10
    test_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(test_values, cat_type, categories=cats),
            "num_col": [1.5, 2.5, 3.5, 4.5],
        }
    )

    train_ds = lgb.Dataset(train_df, label=train_y, categorical_feature=["cat_col"])
    bst = lgb.train({"objective": "binary", "verbose": -1}, train_ds, num_boost_round=10)

    preds = bst.predict(test_df)
    assert preds.shape == (4,)
    assert all(0 <= p <= 1 for p in preds)

    model_path = tmp_path / "categorical_model.txt"
    bst.save_model(model_path)
    loaded_bst = lgb.Booster(model_file=model_path)
    assert loaded_bst.pandas_categorical == bst.pandas_categorical
    np.testing.assert_allclose(preds, loaded_bst.predict(test_df))


@pytest.mark.parametrize("cat_type", ["categorical", "enum"])
def test_polars_pandas_categorical_predictions_match(cat_type):
    """Polars-trained and pandas-trained models give identical predictions."""
    pd = pytest.importorskip("pandas")

    cats = sorted(["cat_a", "cat_b", "cat_c"])
    values = ["cat_a", "cat_b", "cat_c", "cat_a", "cat_b"] * 20

    polars_df = pl.DataFrame(
        {
            "cat_col": _polars_cat_series(values, cat_type, categories=cats),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        }
    )
    pandas_df = pd.DataFrame(
        {
            "cat_col": pd.Categorical(values, categories=cats, ordered=(cat_type == "enum")),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
        }
    )
    y = [0, 1, 0, 1, 0] * 20

    polars_ds = lgb.Dataset(polars_df, label=y, categorical_feature=["cat_col"])
    polars_bst = lgb.train({"objective": "binary", "verbose": -1, "seed": 42}, polars_ds, num_boost_round=10)

    pandas_ds = lgb.Dataset(pandas_df, label=y, categorical_feature=["cat_col"])
    pandas_bst = lgb.train({"objective": "binary", "verbose": -1, "seed": 42}, pandas_ds, num_boost_round=10)

    np.testing.assert_allclose(polars_bst.predict(polars_df), pandas_bst.predict(pandas_df), rtol=1e-10)


def test_polars_categorical_auto_detected():
    """categorical_feature='auto' picks up unordered pl.Categorical columns."""
    df = pl.DataFrame(
        {
            "cat_unordered": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "num_col": [1.0, 2.0, 3.0],
        }
    )
    y = [0, 1, 0]

    ds = lgb.Dataset(df, label=y, categorical_feature="auto", params={"min_data_in_bin": 1})
    ds.construct()

    assert ds.params.get("categorical_column") == [0]
    assert len(ds.pandas_categorical) == 1


def test_polars_enum_not_auto_detected():
    """categorical_feature='auto' does NOT pick up pl.Enum (ordered), but metadata is captured."""
    df = pl.DataFrame(
        {
            "cat_ordered": pl.Series(["low", "medium", "high", "low"]).cast(pl.Enum(["low", "medium", "high"])),
            "num_col": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = [0, 1, 0, 1]

    ds = lgb.Dataset(df, label=y, categorical_feature="auto", params={"min_data_in_bin": 1})
    ds.construct()

    assert "categorical_column" not in ds.params
    assert len(ds.pandas_categorical) == 1
