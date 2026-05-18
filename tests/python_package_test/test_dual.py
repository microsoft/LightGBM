# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import platform

import numpy as np
import pytest
from sklearn.metrics import log_loss

import lightgbm as lgb

from .utils import load_breast_cancer


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", "0") != "1",
    reason="Set LIGHTGBM_TEST_DUAL_CPU_GPU=1 to test using CPU and GPU training from the same package.",
)
def test_cpu_and_gpu_work():
    # If compiled appropriately, the same installation will support both GPU and CPU.
    X, y = load_breast_cancer(return_X_y=True)
    data = lgb.Dataset(X, y)

    params_cpu = {"verbosity": -1, "num_leaves": 31, "objective": "binary", "device": "cpu"}
    cpu_bst = lgb.train(params_cpu, data, num_boost_round=10)
    cpu_score = log_loss(y, cpu_bst.predict(X))

    params_gpu = params_cpu.copy()
    params_gpu["device"] = "gpu"
    # Double-precision floats are only supported on x86_64 with PoCL
    params_gpu["gpu_use_dp"] = platform.machine() == "x86_64"
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    rel = 1e-6 if params_gpu["gpu_use_dp"] else 1e-4
    assert cpu_score == pytest.approx(gpu_score, rel=rel)
    assert gpu_score < 0.242


_REQUIRES_CUDA = pytest.mark.skipif(
    os.environ.get("TASK", "") != "cuda",
    reason="requires CUDA-enabled LightGBM build (set TASK=cuda)",
)


def _tree_depth(node, depth=0):
    if "leaf_value" in node:
        return depth
    return max(
        _tree_depth(node["left_child"], depth + 1),
        _tree_depth(node["right_child"], depth + 1),
    )


def _train_pair(params_overrides, X, y):
    out = {}
    for device_type in ("cpu", "cuda"):
        params = {
            "verbose": -1,
            "deterministic": True,
            "num_threads": 1,
            "seed": 0,
            "feature_pre_filter": False,
            "device_type": device_type,
            "gpu_use_dp": True,
            "force_col_wise": True,
            **params_overrides,
        }
        ds = lgb.Dataset(X, label=y, params={"verbose": -1, "feature_pre_filter": False})
        out[device_type] = lgb.train(params, ds, num_boost_round=1)
    return out


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("max_depth", "num_leaves"),
    [
        (1, 2),
        (1, 7),
        (2, 4),
        (2, 7),
        (2, 31),
        (3, 7),
        (3, 31),
        (5, 31),
    ],
)
def test_cuda_respects_max_depth(max_depth, num_leaves):
    """CUDA tree learner must enforce max_depth, matching CPU.

    Regression test for the bug where CUDABestSplitFinder had no max_depth
    check and CUDATree::Split never updated host-side leaf_depth_, causing
    CUDA to produce trees up to log2(num_leaves) deep regardless of
    max_depth. With max_depth=2 and num_leaves=31, CUDA was producing
    depth-7 trees with all 31 leaves filled.
    """
    rng = np.random.default_rng(0)
    n = 64
    X = rng.standard_normal((n, 4)).astype(np.float64)
    y = (X @ rng.standard_normal(4) + 0.1 * rng.standard_normal(n)).astype(np.float64)

    models = _train_pair(
        {"max_depth": max_depth, "num_leaves": num_leaves, "min_data_in_leaf": 1},
        X,
        y,
    )

    cpu_dump = models["cpu"].dump_model()["tree_info"][0]
    cuda_dump = models["cuda"].dump_model()["tree_info"][0]

    cpu_depth = _tree_depth(cpu_dump["tree_structure"])
    cuda_depth = _tree_depth(cuda_dump["tree_structure"])

    assert cuda_depth <= max_depth, (
        f"CUDA exceeded max_depth={max_depth}: produced depth-{cuda_depth} tree with num_leaves={num_leaves}"
    )
    assert cpu_depth == cuda_depth, (
        f"CPU/CUDA depth mismatch with max_depth={max_depth}, num_leaves={num_leaves}: "
        f"cpu={cpu_depth}, cuda={cuda_depth}"
    )
