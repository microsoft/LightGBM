# coding: utf-8
"""Tests for dual GPU+CPU support."""

import os
import pytest

import lightgbm as lgb
import numpy as np

from sklearn.metrics import log_loss

from .utils import load_breast_cancer


@pytest.mark.skipif(
    os.environ.get("LIGHTGBM_TEST_DUAL_CPU_GPU", None) is None,
    reason="Only run if appropriate env variable is set",
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
    gpu_bst = lgb.train(params_gpu, data, num_boost_round=10)
    gpu_score = log_loss(y, gpu_bst.predict(X))

    np.testing.assert_allclose(cpu_score, gpu_score, rtol=1e-4)
    assert gpu_score < 0.25
