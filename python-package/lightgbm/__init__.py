# coding: utf-8
"""LightGBM, Light Gradient Boosting Machine.

Contributors: https://github.com/microsoft/LightGBM/graphs/contributors.
"""

import platform

# gcc's libgomp tries to allocate a small amount of aligned static thread-local storage ("TLS")
# when it's dynamically loaded.
#
# If it's not able to find a block of aligned memory large enough, loading fails like this:
#
#   > ../lib/libgomp.so.1: cannot allocate memory in static TLS block
#
# On aarch64 Linux, processes and loaded libraries share the same pool of static TLS,
# which makes such failures more likely on that architecture.
# (ref: https://bugzilla.redhat.com/show_bug.cgi?id=1722181#c6)
#
# Therefore, the later in a process libgomp.so is loaded, the greater the risk that loading
# it will fail in this way... so lightgbm tries to dlopen() it immediately, before any
# other imports or computation.
#
# This should generally be safe to do ... many other dynamically-loaded libraries have fallbacks
# that allow successful loading if there isn't sufficient static TLS available.
#
# libgomp.so absolutely needing it, by design, makes it a special case
# (ref: https://gcc.gcc.gnu.narkive.com/vOXMQqLA/failure-to-dlopen-libgomp-due-to-static-tls-data).
#
# other references:
#
#   * https://github.com/microsoft/LightGBM/pull/6654#issuecomment-2352014275
#   * https://github.com/microsoft/LightGBM/issues/6509
#   * https://maskray.me/blog/2021-02-14-all-about-thread-local-storage
#   * https://bugzilla.redhat.com/show_bug.cgi?id=1722181#c6
#
if platform.system().lower() == "linux" and platform.processor().lower() == "aarch64":
    import ctypes

    try:
        # this issue seems specific to libgomp, so no need to attempt e.g. libomp or libiomp
        _ = ctypes.CDLL("libgomp.so.1", ctypes.RTLD_GLOBAL)
    except:  # noqa: E722
        # this needs to be try-catched, to handle these situations:
        #
        #   * LightGBM built without OpenMP (-DUSE_OPENMP=OFF)
        #   * non-gcc OpenMP used (e.g. clang/libomp, icc/libiomp)
        #   * no file "libgomp.so" available to the linker (e.g. maybe only "libgomp.so.1")
        #
        pass

from pathlib import Path

from .basic import Booster, Dataset, Sequence, register_logger
from .callback import EarlyStopException, early_stopping, log_evaluation, record_evaluation, reset_parameter
from .engine import CVBooster, cv, train

try:
    from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor
except ImportError:
    pass
try:
    from .plotting import create_tree_digraph, plot_importance, plot_metric, plot_split_value_histogram, plot_tree
except ImportError:
    pass
try:
    from .dask import DaskLGBMClassifier, DaskLGBMRanker, DaskLGBMRegressor
except ImportError:
    pass


_version_path = Path(__file__).absolute().parent / "VERSION.txt"
if _version_path.is_file():
    __version__ = _version_path.read_text(encoding="utf-8").strip()

__all__ = [
    "Dataset",
    "Booster",
    "CVBooster",
    "Sequence",
    "register_logger",
    "train",
    "cv",
    "LGBMModel",
    "LGBMRegressor",
    "LGBMClassifier",
    "LGBMRanker",
    "DaskLGBMRegressor",
    "DaskLGBMClassifier",
    "DaskLGBMRanker",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    "early_stopping",
    "EarlyStopException",
    "plot_importance",
    "plot_split_value_histogram",
    "plot_metric",
    "plot_tree",
    "create_tree_digraph",
]
