# coding: utf-8
import pytest

import lightgbm as lgb

from .utils import pickle_obj, unpickle_obj

SERIALIZERS = ["pickle", "joblib", "cloudpickle"]


def pickle_and_unpickle_object(obj, serializer):
    with lgb.basic._TempFile() as tmp_file:
        pickle_obj(
            obj=obj,
            filepath=tmp_file.name,
            serializer=serializer
        )
        obj_from_disk = unpickle_obj(
            filepath=tmp_file.name,
            serializer=serializer
        )
    return obj_from_disk


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_early_stopping_callback_is_picklable(serializer):
    rounds = 5
    callback = lgb.early_stopping(stopping_rounds=rounds)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 30
    assert callback_from_disk.before_iteration is False
    assert callback.stopping_rounds == callback_from_disk.stopping_rounds
    assert callback.stopping_rounds == rounds


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_log_evaluation_callback_is_picklable(serializer):
    periods = 42
    callback = lgb.log_evaluation(period=periods)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is False
    assert callback.period == callback_from_disk.period
    assert callback.period == periods


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_record_evaluation_callback_is_picklable(serializer):
    results = {}
    callback = lgb.record_evaluation(eval_result=results)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 20
    assert callback_from_disk.before_iteration is False
    assert callback.eval_result == callback_from_disk.eval_result
    assert callback.eval_result is results
