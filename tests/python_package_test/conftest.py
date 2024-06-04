import numpy as np
import pytest
from packaging.version import Version


@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng()


@pytest.fixture(scope="function")
def rng_fixed_seed():
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="function")
def skip_on_pandas3():
    pd = pytest.importorskip("pandas")
    pd_version = pd.__version__
    if Version(pd_version) >= Version("3.0.dev0"):
        pytest.skip(f"skipping this test on pytest>=3.0 (installed version: '{pd_version}')")
