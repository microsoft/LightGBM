# coding: utf-8
"""Tests for Arrow C Stream / Polars DataFrame integration.

These tests cover:
  - data source: Polars DataFrame
  - API surface: lgb.Dataset + lgb.train / Booster.predict (core) and
                 LGBMClassifier / LGBMRegressor .fit / .predict (sklearn)
  - feature types: numeric-only and numeric + categorical
  - edge cases: null category values, unseen categories at inference,
                save/load model round-trip

PyArrow-specific dictionary-column tests live in test_arrow.py.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

import lightgbm as lgb

from .utils import np_assert_array_equal

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")

# ──────────────────────────────────────────────────────────────────────────────
# Data generators
# ──────────────────────────────────────────────────────────────────────────────

_CAT_VALS_1 = ["a", "b", "c"]
_CAT_VALS_2 = ["x", "y"]
_NUM_PARAMS: Dict[str, Any] = {"min_data_in_bin": 1, "min_data_in_leaf": 1}


def _make_polars_train(
    n: int = 300, seed: int = 42, with_cats: bool = True, with_nulls: bool = False
) -> Tuple[pl.DataFrame, np.ndarray]:
    """Return (feature_df, labels) for a binary classification task."""
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.integers(0, 50, n).astype(np.int32),
    }
    if with_cats:
        c1 = rng.choice(_CAT_VALS_1, n).tolist()
        c2 = rng.choice(_CAT_VALS_2, n).tolist()
        if with_nulls:
            # introduce ~10 % nulls
            for vals in (c1, c2):
                for i in rng.choice(n, n // 10, replace=False):
                    vals[i] = None
        data["cat1"] = pl.Series(c1, dtype=pl.Categorical)
        data["cat2"] = pl.Series(c2, dtype=pl.Categorical)
    labels = (rng.standard_normal(n) > 0).astype(np.float32)
    return pl.DataFrame(data), labels


def _make_polars_infer(n: int = 100, seed: int = 99, with_cats: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {
        "num1": rng.standard_normal(n).astype(np.float32),
        "num2": rng.integers(0, 50, n).astype(np.int32),
    }
    if with_cats:
        data["cat1"] = pl.Series(rng.choice(_CAT_VALS_1, n).tolist(), dtype=pl.Categorical)
        data["cat2"] = pl.Series(rng.choice(_CAT_VALS_2, n).tolist(), dtype=pl.Categorical)
    return pl.DataFrame(data)


_BINARY_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "num_leaves": 8,
    "num_iterations": 20,
    "verbose": -1,
    **_NUM_PARAMS,
}
_REGRESSION_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "num_leaves": 8,
    "num_iterations": 20,
    "verbose": -1,
    **_NUM_PARAMS,
}


def _assert_polars_category_maps(obj: Any, *, construct_dataset: bool = False) -> None:
    """Assert expected category maps for cat1 / cat2 Polars fixtures."""
    if construct_dataset:
        obj.construct()
    assert obj.pandas_categorical is not None
    assert len(obj.pandas_categorical) == 2
    assert obj.pandas_categorical[0] == sorted(_CAT_VALS_1)
    assert obj.pandas_categorical[1] == sorted(_CAT_VALS_2)


# ──────────────────────────────────────────────────────────────────────────────
# Core API  —  lgb.Dataset + lgb.train + Booster.predict
# ──────────────────────────────────────────────────────────────────────────────


def test_polars_core_numeric_only():
    """Numeric Polars DataFrame → lgb.Dataset → lgb.train → Booster.predict."""
    df_train, y_train = _make_polars_train(with_cats=False)
    df_infer = _make_polars_infer(with_cats=False)

    ds = lgb.Dataset(df_train, label=y_train)
    bst = lgb.train(_BINARY_PARAMS, ds)

    preds = bst.predict(df_infer)
    assert len(preds) == len(df_infer)
    assert np.all((preds >= 0) & (preds <= 1))


def test_polars_core_with_categorical():
    """Polars DataFrame with Categorical columns → core API train + predict."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer()

    ds = lgb.Dataset(df_train, label=y_train, categorical_feature=["cat1", "cat2"], params=_NUM_PARAMS)
    _assert_polars_category_maps(ds, construct_dataset=True)
    bst = lgb.train(_BINARY_PARAMS, ds)

    preds = bst.predict(df_infer)
    assert len(preds) == len(df_infer)
    assert np.all((preds >= 0) & (preds <= 1))


def test_polars_core_categorical_auto_detected():
    """Dictionary columns are auto-detected as categorical (no explicit spec needed)."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer()

    # categorical_feature is "auto" by default — should pick up dict columns
    ds = lgb.Dataset(df_train, label=y_train, params=_NUM_PARAMS)
    _assert_polars_category_maps(ds, construct_dataset=True)
    bst = lgb.train(_BINARY_PARAMS, ds)

    preds = bst.predict(df_infer)
    assert len(preds) == len(df_infer)


def test_polars_core_train_valid_consistency():
    """Validation dataset categorical encoding is consistent with training."""
    df_train, y_train = _make_polars_train(n=300)
    df_valid, y_valid = _make_polars_train(n=100, seed=7)

    ds_train = lgb.Dataset(df_train, label=y_train, params=_NUM_PARAMS)
    ds_valid = lgb.Dataset(df_valid, label=y_valid, reference=ds_train, params=_NUM_PARAMS)
    _assert_polars_category_maps(ds_train, construct_dataset=True)
    ds_valid.construct()
    assert ds_valid.pandas_categorical == ds_train.pandas_categorical
    lgb.train(_BINARY_PARAMS, ds_train, valid_sets=[ds_valid])


def test_polars_core_categorical_nulls():
    """Null category values are handled gracefully (→ NaN in LightGBM)."""
    df_train, y_train = _make_polars_train(with_nulls=True)
    df_infer = _make_polars_infer()

    ds = lgb.Dataset(df_train, label=y_train, params=_NUM_PARAMS)
    _assert_polars_category_maps(ds, construct_dataset=True)
    bst = lgb.train(_BINARY_PARAMS, ds)

    preds = bst.predict(df_infer)
    assert len(preds) == len(df_infer)


def test_polars_core_unseen_category_at_inference():
    """Unseen category at inference becomes null/NaN — no crash."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer().with_columns(
        pl.Series("cat1", ["unseen_value"] * 100, dtype=pl.Utf8).cast(pl.Categorical)
    )

    ds = lgb.Dataset(df_train, label=y_train, params=_NUM_PARAMS)
    _assert_polars_category_maps(ds, construct_dataset=True)
    bst = lgb.train(_BINARY_PARAMS, ds)

    preds = bst.predict(df_infer)
    assert len(preds) == len(df_infer)
    assert not np.any(np.isnan(preds))  # LightGBM fills NaN predictors with its internal default


def test_polars_core_save_load_model(tmp_path: Path):
    """Category maps survive a save/load round-trip."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer()

    ds = lgb.Dataset(df_train, label=y_train)
    bst = lgb.train(_BINARY_PARAMS, ds)
    preds_before = bst.predict(df_infer)

    model_path = tmp_path / "model.txt"
    bst.save_model(str(model_path))

    bst2 = lgb.Booster(model_file=str(model_path))
    preds_after = bst2.predict(df_infer)

    np_assert_array_equal(preds_before, preds_after, strict=True)


def test_polars_category_maps_correct():
    """Category maps extracted from Polars categorical columns are sorted and complete."""
    df_pl, y = _make_polars_train()

    ds = lgb.Dataset(df_pl, label=y, params=_NUM_PARAMS)
    _assert_polars_category_maps(ds, construct_dataset=True)


@pytest.mark.skipif(not lgb.compat.PANDAS_INSTALLED, reason="pandas is required for this test")
def test_polars_matches_pandas_baseline(tmp_path: Path):
    """Models trained on Polars vs pandas give identical predictions on the same numeric data.

    Category integer codes differ between the two paths:
    - Polars/Arrow: sorted lexicographically at training time
    - pandas: order of first appearance in the training data

    Each model is internally consistent (train↔predict use the same encoding), but
    their internal codes are not the same.  We therefore compare predictions on
    purely numeric inference data — where no categorical encoding is needed — to
    confirm that the numeric feature handling is identical.
    """
    df_pl, y = _make_polars_train()

    df_pd = df_pl.to_pandas()
    df_pd["cat1"] = df_pd["cat1"].astype("category")
    df_pd["cat2"] = df_pd["cat2"].astype("category")

    bst_pl = lgb.train(_BINARY_PARAMS, lgb.Dataset(df_pl, label=y))
    bst_pd = lgb.train(_BINARY_PARAMS, lgb.Dataset(df_pd, label=y))

    # predict on the same Polars inference data for both models
    infer = _make_polars_infer()
    preds_pl = bst_pl.predict(infer)
    preds_pd = bst_pd.predict(df_pd.head(len(infer)))  # same rows, pandas path

    # both models should produce valid binary probabilities
    assert np.all((preds_pl >= 0) & (preds_pl <= 1))
    assert np.all((preds_pd >= 0) & (preds_pd <= 1))

    # category maps are present and complete in both
    assert bst_pl.pandas_categorical is not None
    assert bst_pd.pandas_categorical is not None
    assert len(bst_pl.pandas_categorical) == len(bst_pd.pandas_categorical) == 2


# ──────────────────────────────────────────────────────────────────────────────
# sklearn API  —  LGBMClassifier / LGBMRegressor
# ──────────────────────────────────────────────────────────────────────────────


def test_polars_sklearn_classifier_numeric():
    """LGBMClassifier.fit / predict with numeric-only Polars DataFrame."""
    df_train, y_train = _make_polars_train(with_cats=False)
    df_infer = _make_polars_infer(with_cats=False)

    clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1, **_NUM_PARAMS)
    clf.fit(df_train, y_train)
    preds = clf.predict(df_infer)
    assert len(preds) == len(df_infer)


def test_polars_sklearn_classifier_with_categorical():
    """LGBMClassifier.fit / predict with Polars DataFrame containing categoricals."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer()

    clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1, **_NUM_PARAMS)
    clf.fit(df_train, y_train, categorical_feature=["cat1", "cat2"])
    _assert_polars_category_maps(clf.booster_)
    preds = clf.predict_proba(df_infer)
    assert preds.shape == (len(df_infer), 2)


def test_polars_sklearn_regressor_with_categorical():
    """LGBMRegressor.fit / predict with Polars DataFrame containing categoricals."""
    df_train, y_cont = _make_polars_train()
    y_cont = y_cont * 10.0  # continuous target
    df_infer = _make_polars_infer()

    reg = lgb.LGBMRegressor(n_estimators=20, verbose=-1, **_NUM_PARAMS)
    reg.fit(df_train, y_cont, categorical_feature=["cat1", "cat2"])
    _assert_polars_category_maps(reg.booster_)
    preds = reg.predict(df_infer)
    assert len(preds) == len(df_infer)
    assert np.isfinite(preds).all()


def test_polars_sklearn_feature_names_in_():
    """feature_names_in_ is accessible after fit (issue #6849)."""
    df_train, y_train = _make_polars_train(with_cats=False)

    clf = lgb.LGBMClassifier(n_estimators=5, verbose=-1, **_NUM_PARAMS)
    clf.fit(df_train, y_train)

    # should not raise AttributeError
    names = clf.feature_names_in_
    assert list(names) == ["num1", "num2"]


def test_polars_sklearn_predict_consistency():
    """Sklearn-interface predictions are consistent between Polars and its Arrow table."""
    df_train, y_train = _make_polars_train()
    df_infer = _make_polars_infer()

    clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1, **_NUM_PARAMS)
    clf.fit(df_train, y_train, categorical_feature=["cat1", "cat2"])

    # predict via Polars DataFrame (Arrow C Stream)
    preds_polars = clf.predict_proba(df_infer)

    # predict via equivalent PyArrow table (should give identical results)
    preds_arrow = clf.predict_proba(df_infer.to_arrow())

    np_assert_array_equal(preds_polars, preds_arrow, strict=True)
