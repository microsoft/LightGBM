import numpy as np
import pytest

import lightgbm


@pytest.fixture(scope="function")
def missing_module_cffi(monkeypatch):
    """Mock 'cffi' not being importable"""
    monkeypatch.setattr(lightgbm.compat, "CFFI_INSTALLED", False)
    monkeypatch.setattr(lightgbm.basic, "CFFI_INSTALLED", False)


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="function")
def rng_fixed_seed():
    return np.random.default_rng(seed=42)
