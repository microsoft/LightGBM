# coding: utf-8
from typing import Any, Dict
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import lightgbm as lgb

from .utils import assert_datasets_equal

pd = pytest.importorskip("pandas")


# ----------------------------------------------------------------------------------------------- #
#                                            UTILITIES                                            #
# ----------------------------------------------------------------------------------------------- #


def dummy_dataset_params() -> Dict[str, Any]:
    return {
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
    }


# ----------------------------------------------------------------------------------------------- #
#                                            UNIT TESTS                                           #
# ----------------------------------------------------------------------------------------------- #

# ------------------------------------------- CATEGORICAL ----------------------------------------- #


def test_pandas_categorical_encoding(tmp_path):
    cat1_categories = ["a", "b", "c"]
    cat1_values = ["a", "b", "c", "b", "a"]
    cat2_categories = ["b", "c", "d"]
    cat2_values = ["b", "c", "c", "d", "d"]
    ordered_categories = ["high", "low", "mid"]
    ordered_values = ["low", "high", "mid", "high", "low"]

    df = pd.DataFrame(
        {
            "cat1": pd.Categorical(cat1_values, categories=cat1_categories, ordered=False),
            "cat2": pd.Categorical(cat2_values, categories=cat2_categories, ordered=False),
            "cat3": pd.Categorical(ordered_values, categories=ordered_categories, ordered=True),
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()

    assert ds.num_data() == 5
    assert ds.num_feature() == 4
    assert ds.get_feature_name() == ["cat1", "cat2", "cat3", "num_col"]

    assert ds.categorical_feature == "auto"
    assert len(ds.pandas_categorical) == 3
    assert ds.pandas_categorical[0] == cat1_categories
    assert ds.pandas_categorical[1] == cat2_categories
    assert ds.pandas_categorical[2] == ordered_categories
    assert ds.params["categorical_column"] == [0, 1]  # ordered categorical not treated as categorical by default

    # Verify correct encodings
    ref_df = pd.DataFrame(
        {
            "cat1": [cat1_categories.index(v) for v in cat1_values],  # [0, 1, 2, 1, 0]
            "cat2": [cat2_categories.index(v) for v in cat2_values],  # [0, 1, 1, 2, 2],
            "cat3": [ordered_categories.index(v) for v in ordered_values],  # [1, 0, 2, 0, 1],
            "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ref_ds = lgb.Dataset(ref_df, label=y, categorical_feature=[0, 1], params=dummy_dataset_params())
    ref_ds.construct()

    assert_datasets_equal(tmp_path, ds, ref_ds)


def test_pandas_categorical_encoding_unseen_category(tmp_path):
    train_categories = ["a", "b", "c"]
    train_values = ["a", "b", "c", "a", "b"]
    valid_values = ["a", "c", "d", "d", "a"]  # "d" is unseen in training data

    train_df = pd.DataFrame({"cat_col": pd.Categorical(train_values), "num_col": [1.0, 2.0, 3.0, 4.0, 5.0]})
    valid_df = pd.DataFrame({"cat_col": pd.Categorical(valid_values), "num_col": [6.0, 7.0, 8.0, 9.0, 10.0]})

    train_ds = lgb.Dataset(train_df, label=[0, 1, 0, 1, 0], params=dummy_dataset_params())
    valid_ds = lgb.Dataset(valid_df, label=[1, 0, 1, 0, 1], reference=train_ds, params=dummy_dataset_params())
    train_ds.construct()
    valid_ds.construct()

    # Verify unseen category is encoded as NaN
    ref_valid_df = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["a", "c", None, None, "a"], categories=train_categories),
            "num_col": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    ref_valid_ds = lgb.Dataset(ref_valid_df, label=[1, 0, 1, 0, 1], reference=train_ds, params=dummy_dataset_params())
    ref_valid_ds.construct()

    assert_datasets_equal(tmp_path, valid_ds, ref_valid_ds)


def test_pandas_categorical_encoding_registered_but_unobserved(tmp_path):
    train_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", "b", "b"], categories=["a", "b", "c", "d"]),
            "ordered_col": pd.Categorical(["e", "g", "g"], categories=["e", "f", "g", "h"], ordered=True),
        }
    )
    valid_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", "c", "b"], categories=["a", "b", "c", "d"]),
            "ordered_col": pd.Categorical(["e", "f", "h"], categories=["e", "f", "g", "h"], ordered=True),
        }
    )

    train_ds = lgb.Dataset(train_df, label=[0, 1, 0], params=dummy_dataset_params())
    valid_ds = lgb.Dataset(valid_df, label=[0, 1, 0], reference=train_ds, params=dummy_dataset_params())
    train_ds.construct()
    valid_ds.construct()

    assert train_ds.pandas_categorical[0] == ["a", "b", "c", "d"]
    assert train_ds.pandas_categorical[1] == ["e", "f", "g", "h"]
    assert train_ds.params["categorical_column"] == [0]  # only unordered column is treated as categorical

    # Python-side encoding: both ordered and unordered columns use all registered categories to encode
    valid_df_encoded = lgb.basic._data_from_pandas(
        data=valid_df,
        feature_name="auto",
        categorical_feature="auto",
        pandas_categorical=train_ds.pandas_categorical,
    )[0]
    assert valid_df_encoded[:, 0].tolist() == [0.0, 2.0, 1.0]  # a -> 0, c -> 2, b -> 1
    assert valid_df_encoded[:, 1].tolist() == [0.0, 1.0, 3.0]  # e -> 0, f -> 1, h -> 3

    # C++ binning
    # - Unordered columns: only codes observed during training are binned. Unseen codes are treated as missing.
    # - Ordered columns: treats as continuous. Unseen values interpolate (e<f<g) or clip (h clipped to g).
    ref_valid_df = pd.DataFrame(
        {
            "unordered_col": pd.Categorical(["a", None, "b"], categories=["a", "b", "c", "d"]),
            "ordered_col": pd.Categorical(["e", "g", "g"], categories=["e", "f", "g", "h"], ordered=True),
        }
    )
    ref_valid_ds = lgb.Dataset(ref_valid_df, label=[0, 1, 0], reference=train_ds, params=dummy_dataset_params())
    ref_valid_ds.construct()

    assert_datasets_equal(tmp_path, valid_ds, ref_valid_ds)


def test_pandas_categorical_with_missing_values(tmp_path):
    categories = ["a", "b"]
    values_none = ["a", "b", None, "a", None]
    values_nan = ["b", "a", np.nan, "b", np.nan]

    df = pd.DataFrame(
        {
            "cat_none": pd.Categorical(values_none, categories=categories),
            "cat_nan": pd.Categorical(values_nan, categories=categories),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = [0, 1, 0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()
    assert ds.pandas_categorical == [categories, categories]

    ref_df = pd.DataFrame(
        {
            "cat_none": [0.0, 1.0, np.nan, 0.0, np.nan],
            "cat_nan": [1.0, 0.0, np.nan, 1.0, np.nan],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ref_ds = lgb.Dataset(ref_df, label=y, categorical_feature=[0, 1], params=dummy_dataset_params())
    ref_ds.construct()
    assert_datasets_equal(tmp_path, ds, ref_ds)


def test_pandas_dataset_construction_with_high_cardinality_categorical_succeeds(rng):
    X = pd.DataFrame({"x1": rng.integers(low=0, high=5_000, size=(10_000,))})
    y = rng.uniform(size=(10_000,))
    ds = lgb.Dataset(X, y, categorical_feature=["x1"])
    ds.construct()
    assert ds.num_data() == 10_000
    assert ds.num_feature() == 1


@pytest.mark.parametrize(
    "feature_name",
    [
        pytest.param(["x1"], id="feature-name"),
        pytest.param([42], id="feature-index"),
        pytest.param("auto", id="auto"),
    ],
)
@pytest.mark.parametrize("categories", ["seen", "unseen"])
def test_pandas_categorical_code_conversion_doesnt_modify_original_data(feature_name, categories, rng):
    X = rng.choice(a=["a", "b"], size=(100, 1))
    df = pd.DataFrame(X.copy(), columns=["x1"], dtype="category")
    if categories == "seen":
        pandas_categorical = [["a", "b"]]
    else:
        pandas_categorical = [["a"]]
    data = lgb.basic._data_from_pandas(
        data=df,
        feature_name=feature_name,
        categorical_feature="auto",
        pandas_categorical=pandas_categorical,
    )[0]
    # check that the original data wasn't modified
    np.testing.assert_equal(df["x1"], X[:, 0])
    # check that the built data has the codes
    if categories == "seen":
        # if all categories were seen during training we just take the codes
        codes = df["x1"].cat.codes
    else:
        # if we only saw 'a' during training we just replace its code
        # and leave the rest as nan
        a_code = df["x1"].cat.categories.get_loc("a")
        codes = np.where(df["x1"] == "a", a_code, np.nan)
    np.testing.assert_equal(codes, data[:, 0])


# ---------------------------------------- DTYPE VALIDATION --------------------------------------- #


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pd.Int8Dtype(), [1, 2, 3]),
        (pd.Int16Dtype(), [1, 2, 3]),
        (pd.Int32Dtype(), [1, 2, 3]),
        (pd.Int64Dtype(), [1, 2, 3]),
        (pd.UInt8Dtype(), [1, 2, 3]),
        (pd.UInt16Dtype(), [1, 2, 3]),
        (pd.UInt32Dtype(), [1, 2, 3]),
        (pd.UInt64Dtype(), [1, 2, 3]),
        (pd.Float32Dtype(), [1.0, 2.0, 3.0]),
        (pd.Float64Dtype(), [1.0, 2.0, 3.0]),
        (pd.BooleanDtype(), [True, False, True]),
        (pd.SparseDtype(), [1.0, 2.0, 3.0]),
        # Categorical dtypes are supported, but tested separately
    ],
)
def test_pandas_supported_dtypes(tmp_path, dtype, values):
    df = pd.DataFrame({"test_col": pd.Series(values, dtype=dtype), "num_col": [4.0, 5.0, 6.0]})
    y = [0, 1, 0]

    ds = lgb.Dataset(df, label=y, params=dummy_dataset_params())
    ds.construct()

    assert ds.num_data() == 3
    assert ds.num_feature() == 2
    assert ds.get_feature_name() == ["test_col", "num_col"]
    assert ds.get_label().tolist() == y

    # Verify values are preserved
    ref_df = pd.DataFrame({"test_col": values, "num_col": [4.0, 5.0, 6.0]})
    ref_ds = lgb.Dataset(ref_df, label=y, params=dummy_dataset_params())
    ref_ds.construct()

    assert_datasets_equal(tmp_path, ds, ref_ds)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pd.StringDtype(), ["a", "b", "c"]),
        (pd.DatetimeTZDtype(tz=ZoneInfo("UTC")), ["2020-01-01", "2020-01-02", "2020-01-03"]),
        (pd.PeriodDtype(freq="Y"), [pd.Period("2024"), pd.Period("2025"), pd.Period("2026")]),
        (pd.IntervalDtype(subtype="int64"), [pd.Interval(0, 1), pd.Interval(1, 2), pd.Interval(2, 3)]),
    ],
)
def test_pandas_unsupported_dtypes(dtype, values):
    df = pd.DataFrame({"test_col": pd.Series(values, dtype=dtype), "num_col": [1.0, 2.0, 3.0]})
    y = [0, 1, 0]

    with pytest.raises(ValueError, match="pandas dtypes must be int, float or bool"):
        lgb.Dataset(df, label=y).construct()
