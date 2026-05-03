# coding: utf-8
"""Regression tests for LightGBM issue #7122.

CUDA training previously SIGFPE'd when the product of (low bin count) x
(many features) caused DivideCUDAFeatureGroups to pack >504 columns into a
single feature partition, after which
    block_dim_y = NUM_THREADS_PER_BLOCK / max_num_column_per_partition
truncated to 0 in CalcConstructHistogramKernelDim.

The fix introduces MAX_NUM_COLUMN_PER_PARTITION (252) as a second split
condition in DivideCUDAFeatureGroups so block_dim_y stays >= 2.

These tests are gated on a working CUDA build; they auto-skip on CPU-only
installs.

NOTE on the failure mode: pre-fix, the bug is a host-side integer divide by
zero (SIGFPE) inside the LightGBM C++ extension, which terminates the Python
interpreter. If you run this file against an unpatched build, expect the
pytest worker to die mid-test with "Floating point exception (core dumped)"
rather than a clean assertion failure. Treat any such crash on the failure
matrix as a regression of #7122.
"""
import numpy as np
import pytest

import lightgbm as lgb


def _cuda_training_works():
    """Return True if the loaded lightgbm can train with device_type='cuda'.

    Detects: missing CUDA build, no GPU present, driver/arch mismatch (e.g. a
    wheel compiled without sm_120 on Blackwell hosts). Cached so we pay it
    once per test session.
    """
    # Probe with small continuous data — won't trigger #7122 even on an
    # unpatched build, so this detection step is safe to run.
    try:
        rng = np.random.default_rng(0)
        ds = lgb.Dataset(rng.standard_normal((128, 4)), rng.standard_normal(128))
        lgb.train(
            {"device_type": "cuda", "objective": "regression", "verbose": -1},
            ds,
            num_boost_round=2,
        )
    except Exception:
        return False
    return True


_CUDA_OK = _cuda_training_works()
pytestmark = pytest.mark.skipif(
    not _CUDA_OK, reason="CUDA-enabled LightGBM build with a working GPU is required"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Smallest training shape that reliably exercises DivideCUDAFeatureGroups and
# the histogram kernel launch. The bug is shape-dependent (n_features,
# n_unique), not row-count dependent, so we keep rows small for speed.
_N_ROWS = 1500
_N_BOOST_ROUNDS = 5

# Cap from the patch (include/LightGBM/cuda/cuda_row_data.hpp).
_MAX_NUM_COLUMN_PER_PARTITION = 252

# NUM_THREADS_PER_BLOCK from src/treelearner/cuda/cuda_histogram_constructor.hpp.
# Kept here as a literal so the test still serves as documentation if that
# header changes.
_NUM_THREADS_PER_BLOCK = 504


def _make_discrete_data(n_rows, n_features, n_unique, seed=0):
    """Integer-valued feature matrix with `n_unique` distinct values per
    column, cast to float32 (matches numerai-style quantized inputs)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, n_unique, size=(n_rows, n_features)).astype(np.float32)
    y = rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32)
    return X, y


def _train_cuda(X, y, **overrides):
    params = {
        "device_type": "cuda",
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 0,
    }
    params.update(overrides)
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=_N_BOOST_ROUNDS)


def _train_cpu(X, y, **overrides):
    params = {
        "device_type": "cpu",
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "seed": 0,
        "deterministic": True,
        "num_threads": 1,
    }
    params.update(overrides)
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=_N_BOOST_ROUNDS)


# ---------------------------------------------------------------------------
# 1. Regression: shapes from the issue's failure matrix must train without
#    crashing. Pre-fix, every one of these SIGFPE'd on the first boosting
#    round.
# ---------------------------------------------------------------------------

# (n_unique, n_features) drawn from the table in #7122 plus the
# higher-resolution boundary set contributed by KalliopeMain. Every row here
# was a confirmed crash before the patch.
_FAILURE_MATRIX = [
    (3, 700),
    (4, 700),
    (5, 600),
    (5, 700),
    (6, 600),
    (6, 700),
    (7, 600),
    (8, 600),
    (8, 700),
    # KalliopeMain's narrower scan: 8..12 unique x 505..510 features all
    # crashed pre-fix.
    (8, 505),
    (8, 510),
    (10, 505),
    (12, 505),
    (12, 510),
]


@pytest.mark.parametrize("n_unique,n_features", _FAILURE_MATRIX)
def test_no_sigfpe_on_documented_failure_matrix(n_unique, n_features):
    """Every (n_unique, n_features) pair from the issue's crash table must
    now train end-to-end."""
    X, y = _make_discrete_data(_N_ROWS, n_features, n_unique)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert preds.shape == (_N_ROWS,)
    assert np.all(np.isfinite(preds)), "predictions contain NaN or inf"


# ---------------------------------------------------------------------------
# 2. Numerai-shape regression: the real-world payload that motivated the fix.
# ---------------------------------------------------------------------------


def test_numerai_shape_no_sigfpe():
    """5 unique values x 705 features is the shape that triggered SIGFPE on
    the real Numerai dataset. Pre-fix, this crashes on the first split."""
    X, y = _make_discrete_data(_N_ROWS, n_features=705, n_unique=5)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert np.all(np.isfinite(preds))
    # Sanity: prediction range non-degenerate (not all-equal).
    assert preds.max() - preds.min() > 1e-6


# ---------------------------------------------------------------------------
# 3. Boundary tests around the 252-column cap. These exercise the new split
#    condition specifically (low bin counts ensure the bin-count condition
#    never fires, so any split has to come from the column-count cap).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_features",
    [
        _MAX_NUM_COLUMN_PER_PARTITION - 1,  # 251 — no split forced by cap
        _MAX_NUM_COLUMN_PER_PARTITION,      # 252 — exactly at cap
        _MAX_NUM_COLUMN_PER_PARTITION + 1,  # 253 — must split
        _MAX_NUM_COLUMN_PER_PARTITION * 2,  # 504 — exactly at old SIGFPE edge
        _MAX_NUM_COLUMN_PER_PARTITION * 2 + 1,  # 505 — KalliopeMain's edge
        _MAX_NUM_COLUMN_PER_PARTITION * 4,  # 1008 — multi-partition
    ],
)
def test_partition_boundaries_low_bin(n_features):
    """Low-bin (5 unique) data at boundary feature counts. Pre-fix, anything
    >= ~505 features SIGFPE'd; >= 253 features now produces multiple
    partitions but must still train."""
    X, y = _make_discrete_data(_N_ROWS, n_features=n_features, n_unique=5)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert preds.shape == (_N_ROWS,)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# 4. The patch must not regress shapes that were already passing.
# ---------------------------------------------------------------------------

# Shapes the issue reports as "Pass" pre-fix. We verify they still train and
# produce finite, non-degenerate predictions after the patch.
_PASSING_MATRIX = [
    (2, 500),
    (2, 700),
    (3, 500),
    (4, 500),
    (5, 500),
    (13, 510),  # first bin count that naturally splits via bin-count condition
    (16, 510),
]


@pytest.mark.parametrize("n_unique,n_features", _PASSING_MATRIX)
def test_previously_passing_shapes_still_pass(n_unique, n_features):
    X, y = _make_discrete_data(_N_ROWS, n_features, n_unique)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert np.all(np.isfinite(preds))
    assert preds.max() - preds.min() > 1e-6


# ---------------------------------------------------------------------------
# 5. Existing high-bin (continuous) code path is unchanged: a single feature
#    with >max_num_bin_per_partition bins still gets its own partition, and
#    moderate-feature continuous data still trains.
# ---------------------------------------------------------------------------


def test_high_bin_continuous_unaffected():
    """Continuous-valued data with default max_bin=255. The bin-count split
    condition is the active one here; the new column-count cap should be a
    no-op."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((_N_ROWS, 50)).astype(np.float32)
    y = rng.standard_normal(_N_ROWS).astype(np.float32)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert np.all(np.isfinite(preds))
    assert preds.max() - preds.min() > 1e-6


def test_single_feature_with_many_bins():
    """One feature, many distinct values — exercises the >max_num_bin_per_partition
    branch (large_bin_partitions_) in DivideCUDAFeatureGroups, untouched by
    this patch."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((_N_ROWS, 1)).astype(np.float32)
    y = rng.standard_normal(_N_ROWS).astype(np.float32)
    booster = _train_cuda(X, y, max_bin=255)
    preds = booster.predict(X)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# 6. Stability: same data + same seed -> tightly-matching predictions across
#    runs. The CUDA path is not bit-deterministic (reduction order varies
#    across thread schedules), but the new partition boundaries should not
#    introduce gross run-to-run drift on top of that baseline noise.
# ---------------------------------------------------------------------------


def test_cuda_training_stable_low_bin():
    X, y = _make_discrete_data(_N_ROWS, n_features=600, n_unique=5)
    booster_a = _train_cuda(X, y)
    booster_b = _train_cuda(X, y)
    preds_a = booster_a.predict(X)
    preds_b = booster_b.predict(X)
    # Float-reduction noise empirically lands at ~1e-8 on RTX-class hardware
    # for this shape; 1e-5 is generous and would flag any logic-level
    # nondeterminism without false-positiving on schedule jitter.
    np.testing.assert_allclose(preds_a, preds_b, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. CPU/CUDA prediction parity. The CUDA reduction order differs from CPU,
#    so we don't expect bit-identical output (per LightGBM's own docs and
#    issue-tracker discussions). We do require predictions to track each
#    other within a small tolerance — guards against the patch silently
#    miscomputing partition offsets.
# ---------------------------------------------------------------------------


def test_cpu_cuda_prediction_parity_low_bin():
    X, y = _make_discrete_data(_N_ROWS, n_features=600, n_unique=5)
    booster_cuda = _train_cuda(X, y)
    booster_cpu = _train_cpu(X, y)
    preds_cuda = booster_cuda.predict(X)
    preds_cpu = booster_cpu.predict(X)
    # Correlation is the right invariant: reduction-order differences shift
    # absolute values slightly but should not change the rank ordering of
    # predictions in a meaningful way after only a handful of trees on the
    # same data.
    corr = float(np.corrcoef(preds_cuda, preds_cpu)[0, 1])
    assert corr > 0.95, f"CPU/CUDA predictions diverged: corr={corr:.4f}"


# ---------------------------------------------------------------------------
# 8. Stress: shapes that are far past the old SIGFPE boundary, to make sure
#    multi-partition splitting works for many partitions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [800, 1200, 2000])
def test_many_partitions_low_bin(n_features):
    """At 5 unique values, n_features=2000 forces ceil(2000/252)=8 partitions
    via the column-count cap. Pre-fix all of these SIGFPE'd."""
    X, y = _make_discrete_data(_N_ROWS, n_features=n_features, n_unique=5)
    booster = _train_cuda(X, y)
    preds = booster.predict(X)
    assert preds.shape == (_N_ROWS,)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# 9. Invariant documented in the patch's header comment AND machine-checked
#    by a static_assert in src/treelearner/cuda/cuda_histogram_constructor.cpp.
#    The cap must stay strictly less than NUM_THREADS_PER_BLOCK so block_dim_y
#    >= 1; with the chosen 252 we get block_dim_y >= 2. The static_assert is
#    the primary guard — this test mirrors the contract on the Python side so
#    a reader of the test file sees the same relationship.
# ---------------------------------------------------------------------------


def test_max_num_column_per_partition_invariant():
    assert _MAX_NUM_COLUMN_PER_PARTITION < _NUM_THREADS_PER_BLOCK
    # block_dim_y is computed as integer division; we want >= 2 worth of
    # y-parallelism in the histogram kernel.
    assert _NUM_THREADS_PER_BLOCK // _MAX_NUM_COLUMN_PER_PARTITION >= 2
