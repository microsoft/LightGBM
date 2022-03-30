# coding: utf-8
import pytest

import lightgbm as lgb

from .utils import pickle_obj, unpickle_obj


@pytest.mark.parametrize('serializer', ["pickle", "joblib", "cloudpickle"])
def test_early_stopping_callback_is_picklable(serializer, tmp_path):
    rounds = 5
    callback = lgb.early_stopping(stopping_rounds=rounds)
    tmp_file = tmp_path / "early_stopping.pkl"
    pickle_obj(
        obj=callback,
        filepath=tmp_file,
        serializer=serializer
    )
    callback_from_disk = unpickle_obj(
        filepath=tmp_file,
        serializer=serializer
    )
    assert callback.stopping_rounds == callback_from_disk.stopping_rounds
    assert callback.stopping_rounds == rounds


@pytest.mark.parametrize('serializer', ["pickle", "joblib", "cloudpickle"])
def test_log_evaluation_callback_is_picklable(serializer, tmp_path):
    periods = 42
    callback = lgb.log_evaluation(period=periods)
    tmp_file = tmp_path / "log_evaluation.pkl"
    pickle_obj(
        obj=callback,
        filepath=tmp_file,
        serializer=serializer
    )
    callback_from_disk = unpickle_obj(
        filepath=tmp_file,
        serializer=serializer
    )
    assert callback.period == callback_from_disk.period
    assert callback.period == periods
