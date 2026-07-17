# coding: utf-8
import logging
import threading

import numpy as np
import pytest

import lightgbm as lgb
from lightgbm.basic import _DummyLeveledLogger, _log_callback_with_level

from .utils import make_synthetic_regression


def test_register_logger(tmp_path):
    logger = logging.getLogger("LightGBM")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    log_filename = tmp_path / "LightGBM_test_logger.log"
    file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def dummy_metric(_, __):
        logger.debug("In dummy_metric")
        return "dummy_metric", 1, True

    lgb.register_logger(logger)

    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb_train = lgb.Dataset(X, y, categorical_feature=[1])
    lgb_valid = lgb.Dataset(X, y, categorical_feature=[1])  # different object for early-stopping

    eval_records = {}
    callbacks = [lgb.record_evaluation(eval_records), lgb.log_evaluation(2), lgb.early_stopping(10)]
    lgb.train(
        {"objective": "binary", "metric": ["auc", "binary_error"], "verbose": 1},
        lgb_train,
        num_boost_round=10,
        feval=dummy_metric,
        valid_sets=[lgb_valid],
        callbacks=callbacks,
    )

    lgb.plot_metric(eval_records)

    expected_log = r"""
INFO | [LightGBM] [Warning] There are no meaningful features which satisfy the provided configuration. Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing Dataset might resolve this warning.
INFO | [LightGBM] [Info] Number of positive: 2, number of negative: 2
INFO | [LightGBM] [Info] Total Bins 0
INFO | [LightGBM] [Info] Number of data points in the train set: 4, number of used features: 0
INFO | [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | Training until validation scores don't improve for 10 rounds
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [2]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [4]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [6]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [8]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [LightGBM] [Warning] Stopped training because there are no more leaves that meet the split requirements
DEBUG | In dummy_metric
INFO | [10]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
INFO | Did not meet early stopping. Best iteration is:
[1]	valid_0's auc: 0.5	valid_0's binary_error: 0.5	valid_0's dummy_metric: 1
WARNING | More than one metric available, picking one to plot.
""".strip()

    gpu_lines = [
        "INFO | [LightGBM] [Info] This is the GPU trainer",
        "INFO | [LightGBM] [Info] Using GPU Device:",
        "INFO | [LightGBM] [Info] Compiling OpenCL Kernel with 16 bins...",
        "INFO | [LightGBM] [Info] GPU programs have been built",
        "INFO | [LightGBM] [Warning] GPU acceleration is disabled because no non-trivial dense features can be found",
        "INFO | [LightGBM] [Warning] Using sparse features with CUDA is currently not supported.",
        "INFO | [LightGBM] [Warning] CUDA currently requires double precision calculations.",
        "INFO | [LightGBM] [Info] LightGBM using CUDA trainer with DP float!!",
    ]
    cuda_lines = [
        "INFO | [LightGBM] [Warning] Metric auc is not implemented in cuda version. Fall back to evaluation on CPU.",
        "INFO | [LightGBM] [Warning] Metric binary_error is not implemented in cuda version. Fall back to evaluation on CPU.",
    ]
    with open(log_filename, "rt", encoding="utf-8") as f:
        actual_log = f.read().strip()
        actual_log_wo_gpu_stuff = []
        for line in actual_log.split("\n"):
            if not any(line.startswith(gpu_or_cuda_line) for gpu_or_cuda_line in gpu_lines + cuda_lines):
                actual_log_wo_gpu_stuff.append(line)

    assert "\n".join(actual_log_wo_gpu_stuff) == expected_log


def test_register_invalid_logger():
    class LoggerWithoutInfoMethod:
        def warning(self, msg: str) -> None:
            print(msg)

    class LoggerWithoutWarningMethod:
        def info(self, msg: str) -> None:
            print(msg)

    class LoggerWithAttributeNotCallable:
        def __init__(self):
            self.info = 1
            self.warning = 2

    expected_error_message = "Logger must provide 'info' and 'warning' method"

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithoutInfoMethod())

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithoutWarningMethod())

    with pytest.raises(TypeError, match=expected_error_message):
        lgb.register_logger(LoggerWithAttributeNotCallable())


def test_register_custom_logger():
    logged_messages = []

    class CustomLogger:
        def custom_info(self, msg: str) -> None:
            logged_messages.append(msg)

        def custom_warning(self, msg: str) -> None:
            logged_messages.append(msg)

    custom_logger = CustomLogger()
    lgb.register_logger(custom_logger, info_method_name="custom_info", warning_method_name="custom_warning")

    lgb.basic._log_info("info message")
    lgb.basic._log_warning("warning message")

    expected_log = ["info message", "warning message"]
    assert logged_messages == expected_log

    logged_messages = []
    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb_data = lgb.Dataset(X, y, categorical_feature=[1])
    lgb.train(
        {"objective": "binary", "metric": "auc"},
        lgb_data,
        num_boost_round=10,
        valid_sets=[lgb_data],
    )
    assert logged_messages, "custom logger was not called"


@pytest.fixture
def _leveled_logger_cleanup():
    """Register the leveled callback and unregister it after the test."""
    lgb.register_leveled_logger(_DummyLeveledLogger())

    yield

    lgb.unregister_leveled_logger()


def test_register_leveled_logger_invalid():
    class NoDebug:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    class NoInfo:
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            pass

    class NoWarning:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def error(self, msg):
            pass

    class NoError:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    class NotCallable:
        def __init__(self):
            self.debug = self.info = self.warning = self.error = 1

    with pytest.raises(TypeError, match="Logger must provide 'debug' method"):
        lgb.register_leveled_logger(NoDebug())
    with pytest.raises(TypeError, match="Logger must provide 'info' method"):
        lgb.register_leveled_logger(NoInfo())
    with pytest.raises(TypeError, match="Logger must provide 'warning' method"):
        lgb.register_leveled_logger(NoWarning())
    with pytest.raises(TypeError, match="Logger must provide 'error' method"):
        lgb.register_leveled_logger(NoError())
    with pytest.raises(TypeError, match="Logger must provide"):
        lgb.register_leveled_logger(NotCallable())


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_log_callback_with_level_unit():
    captured: dict = {"debug": [], "info": [], "warning": [], "error": []}

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            captured["debug"].append(msg)

        def info(self, msg: str) -> None:
            captured["info"].append(msg)

        def warning(self, msg: str) -> None:
            captured["warning"].append(msg)

        def error(self, msg: str) -> None:
            captured["error"].append(msg)

    lgb.register_leveled_logger(CapturingLogger())
    _log_callback_with_level(-1, b"fatal message")  # C_API_LOG_LEVEL_FATAL
    _log_callback_with_level(0, b"warning message")  # C_API_LOG_LEVEL_WARNING
    _log_callback_with_level(1, b"info message")  # C_API_LOG_LEVEL_INFO
    _log_callback_with_level(2, b"debug message")  # C_API_LOG_LEVEL_DEBUG

    assert captured["error"] == ["fatal message"]
    assert captured["warning"] == ["warning message"]
    assert captured["info"] == ["info message"]
    assert captured["debug"] == ["debug message"]


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_register_leveled_logger_routing():
    info_messages: list = []
    warning_messages: list = []

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            info_messages.append(msg)

        def warning(self, msg: str) -> None:
            warning_messages.append(msg)

        def error(self, msg: str) -> None:
            pass

    lgb.register_leveled_logger(CapturingLogger())

    X = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 4], [1, 2, 3]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    lgb.train(
        {"objective": "binary", "verbose": 1},
        lgb.Dataset(X, y, categorical_feature=[1]),
        num_boost_round=2,
    )

    # this dataset deterministically emits these messages on the registering (main) thread
    assert any("number of used features" in m for m in info_messages), (
        "Expected native Info message about used features"
    )
    assert any("meaningful features" in m for m in warning_messages), (
        "Expected native Warning message about meaningful features"
    )
    assert any("no more leaves" in m for m in warning_messages), "Expected native Warning message about no more leaves"

    # messages arrive whole — no empty or whitespace-only 3-chunk artifacts
    assert all(m.strip() for m in info_messages), "Chunk artifact in info_messages"
    assert all(m.strip() for m in warning_messages), "Chunk artifact in warning_messages"


def test_unregister_leveled_logger(capfd):
    """Unregister must clear the native callback slot, not just Python-side state.

    C++ writes the "[LightGBM] [Fatal] " stderr fallback only when the thread's callback slot
    is null (Log::Fatal in include/LightGBM/utils/log.h); a still-live callback would deliver
    a raw, prefix-free message to a Python logger instead. capfd (not capsys) is needed
    because that fallback is written by native code directly to the stderr fd.
    """

    def trigger_fatal():
        with pytest.raises(lgb.basic.LightGBMError, match="Model file"):
            lgb.Booster(model_str="not_a_valid_model")

    captured_a: list = []
    captured_b: list = []

    class LoggerA:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            captured_a.append(msg)

    class LoggerB:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            captured_b.append(msg)

    try:
        lgb.register_leveled_logger(LoggerA())
        lgb.unregister_leveled_logger()
        capfd.readouterr()  # clear anything captured so far

        trigger_fatal()
        assert "[LightGBM] [Fatal]" in capfd.readouterr().err, (
            "Native callback slot should be truly null after unregister, routing the fatal to stderr"
        )
        assert captured_a == [], "Logger A should not receive any message after unregister"

        # Idempotency — second unregister should not raise
        lgb.unregister_leveled_logger()

        # Re-register with B; use a real native message so the C++ layer is exercised,
        # not just the Python-side thunk.
        lgb.register_leveled_logger(LoggerB())
        trigger_fatal()
        assert captured_b, "Logger B should receive the native Fatal message"
        assert captured_a == [], "Logger A should still not receive anything"

    finally:
        lgb.unregister_leveled_logger()


def test_leveled_logger_per_thread_routing():
    """Registration and routing are independent per thread, for both register and unregister.

    The overlap step (main thread triggers a fatal *while* the worker is still registered) is
    load-bearing: without it, a shared-global save/restore implementation would also pass,
    since the worker's register->fatal->unregister would complete before the main thread
    logged again.
    """
    captured_main: list = []
    captured_worker: list = []
    worker_error: list = []
    worker_registered = threading.Event()
    main_triggered_during_overlap = threading.Event()

    class LoggerMain:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            captured_main.append(msg)

    class LoggerWorker:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            captured_worker.append(msg)

    def trigger_fatal():
        with pytest.raises(lgb.basic.LightGBMError, match="Model file"):
            lgb.Booster(model_str="not_a_valid_model")

    def worker():
        try:
            lgb.register_leveled_logger(LoggerWorker())
            trigger_fatal()  # worker thread's own message
            # Signal before waiting: moving this set() into `finally` would make this wait()
            # and the main thread's worker_registered.wait() block on each other every run.
            worker_registered.set()
            main_triggered_during_overlap.wait(timeout=10)  # stay registered while main fires below
        except BaseException as exc:  # noqa: BLE001 — surface any worker-thread failure to the main thread
            worker_error.append(exc)
            worker_registered.set()  # unblock the main thread's wait even if we failed early
        finally:
            lgb.unregister_leveled_logger()

    try:
        lgb.register_leveled_logger(LoggerMain())
        trigger_fatal()  # main thread's message, before the worker exists

        t = threading.Thread(target=worker)
        t.start()
        assert worker_registered.wait(timeout=10), "Worker thread did not register in time"

        try:
            # Overlap window: the worker is still registered on its own thread,
            # yet this must route to LoggerMain.
            trigger_fatal()
        finally:
            # Release the worker even if the trigger above failed, so it isn't
            # stranded for the full timeout.
            main_triggered_during_overlap.set()

        t.join(timeout=10)
        assert not t.is_alive(), "Worker thread did not finish in time"
        assert not worker_error, f"Worker thread raised: {worker_error}"

        trigger_fatal()  # main thread again, after the worker unregistered independently

        assert len(captured_main) == 3, "Main thread's own registration must be untouched by the worker"
        assert len(captured_worker) == 1, "Worker thread's own logger must receive its own message"
    finally:
        main_triggered_during_overlap.set()  # release the worker if we failed before reaching it above
        lgb.unregister_leveled_logger()


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_fatal_through_leveled_callback():
    """Test that C++ Log::Fatal() routes through the leveled callback end-to-end."""
    captured_errors: list = []

    class CapturingLogger:
        def debug(self, msg):
            pass

        def info(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            captured_errors.append(msg)

    lgb.register_leveled_logger(CapturingLogger())

    # match="Model file" stays valid across loader validation-order changes while still
    # scoping the assertion to the model-load failure path
    with pytest.raises(lgb.basic.LightGBMError, match="Model file"):
        lgb.Booster(model_str="not_a_valid_model")

    # Fatal message was routed through the leveled callback to logger.error()
    assert captured_errors, "No Fatal-level messages received through leveled callback"
    # Single-call delivery: no empty/whitespace chunk artifacts
    assert all(m.strip() for m in captured_errors), "Chunk artifact in error messages"
    # Leveled path sends raw message — no [LightGBM] [Fatal] prefix
    assert all("[LightGBM] [Fatal]" not in m for m in captured_errors), (
        "Leveled callback should receive raw message without prefix"
    )


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_log_callback_with_level_unknown_level_falls_back_to_info():
    captured: dict = {"debug": [], "info": [], "warning": [], "error": []}

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            captured["debug"].append(msg)

        def info(self, msg: str) -> None:
            captured["info"].append(msg)

        def warning(self, msg: str) -> None:
            captured["warning"].append(msg)

        def error(self, msg: str) -> None:
            captured["error"].append(msg)

    lgb.register_leveled_logger(CapturingLogger())
    _log_callback_with_level(99, b"weird future level")
    _log_callback_with_level(-5, b"weird negative level")

    assert captured["info"] == ["weird future level", "weird negative level"]
    assert captured["debug"] == captured["warning"] == captured["error"] == []


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_log_callback_with_level_swallows_logger_exceptions(capfd):
    """Logger exceptions are swallowed rather than propagating into C.

    The report goes to stderr rather than warnings.warn so suppression still works
    under `python -W error`.
    """

    class BoomLogger:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            raise ValueError("boom in info")

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    lgb.register_leveled_logger(BoomLogger())
    capfd.readouterr()  # clear anything captured so far

    _log_callback_with_level(1, b"trigger")  # C_API_LOG_LEVEL_INFO — must not raise

    err = capfd.readouterr().err
    assert "leveled logger raised an exception and was suppressed" in err
    assert "boom in info" in err


def test_register_leveled_logger_symbol_missing_degrades_gracefully(monkeypatch):
    """If lib_lightgbm lacks LGBM_RegisterLogCallbackWithLevel, warn and no-op.

    The symbol check happens before any state mutation, so this thread keeps its previous
    routing state (the dummy logger, if none was registered).
    """
    real_lib = lgb.basic._LIB

    class _LibWithoutLeveledSymbol:
        """Forward everything to the real _LIB except the leveled-logging symbol."""

        def __getattr__(self, name):
            if name == "LGBM_RegisterLogCallbackWithLevel":
                raise AttributeError(name)
            return getattr(real_lib, name)

    monkeypatch.setattr(lgb.basic, "_LIB", _LibWithoutLeveledSymbol())

    class CapturingLogger:
        def debug(self, msg: str) -> None:
            pass

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            pass

        def error(self, msg: str) -> None:
            pass

    with pytest.warns(UserWarning, match="does not support leveled logging"):
        lgb.register_leveled_logger(CapturingLogger())

    assert isinstance(lgb.basic._LEVELED_LOG_STATE.logger, lgb.basic._DummyLeveledLogger), (
        "Routing state must be untouched when the native symbol is missing"
    )


@pytest.mark.usefixtures("_leveled_logger_cleanup")
def test_leveled_logger_debug_requires_raised_verbosity():
    """Debug messages reach the callback only when verbosity is raised; C++ filters them otherwise.

    Needs a dataset that actually splits — the 4-row dataset used elsewhere in this file
    emits no Debug messages even at verbose=2.
    """
    X, y = make_synthetic_regression(n_samples=200)

    debug_messages_default: list = []
    debug_messages_verbose: list = []

    def make_logger(sink):
        class CapturingLogger:
            def debug(self, msg: str) -> None:
                sink.append(msg)

            def info(self, msg: str) -> None:
                pass

            def warning(self, msg: str) -> None:
                pass

            def error(self, msg: str) -> None:
                pass

        return CapturingLogger()

    lgb.register_leveled_logger(make_logger(debug_messages_default))
    lgb.train({"objective": "regression", "verbose": 1}, lgb.Dataset(X, y), num_boost_round=3)
    assert debug_messages_default == [], "Debug messages should be filtered at the default verbosity"

    try:
        lgb.register_leveled_logger(make_logger(debug_messages_verbose))
        lgb.train({"objective": "regression", "verbose": 2}, lgb.Dataset(X, y), num_boost_round=3)
        assert debug_messages_verbose, "Debug messages should be delivered when verbosity is raised"
    finally:
        # verbose=2 raised the native log level on this thread; restore the default
        # so later tests don't receive Debug-level output
        lgb.train({"objective": "regression", "verbose": 1}, lgb.Dataset(X, y), num_boost_round=1)
