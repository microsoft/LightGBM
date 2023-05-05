# coding: utf-8
import pytest
from sklearn.model_selection import train_test_split
import tqdm

import lightgbm as lgb
import lightgbm.callback

from .utils import SERIALIZERS, load_breast_cancer, pickle_and_unpickle_object


def reset_feature_fraction(boosting_round):
    return 0.6 if boosting_round < 15 else 0.8


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


@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_reset_parameter_callback_is_picklable(serializer):
    params = {
        'bagging_fraction': [0.7] * 5 + [0.6] * 5,
        'feature_fraction': reset_feature_fraction
    }
    callback = lgb.reset_parameter(**params)
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    assert callback_from_disk.order == 10
    assert callback_from_disk.before_iteration is True
    assert callback.kwargs == callback_from_disk.kwargs
    assert callback.kwargs == params

@pytest.mark.parametrize('serializer', SERIALIZERS)
def test_progress_bar_callback_is_picklable(serializer):
    callback = lgb.progress_bar()
    callback_from_disk = pickle_and_unpickle_object(obj=callback, serializer=serializer)
    callback(lightgbm.callback.CallbackEnv(model=None,
                         params={},
                         iteration=0,
                         begin_iteration=0,
                         end_iteration=100,
                         evaluation_result_list=[]))
    assert callback_from_disk.order == 40
    assert callback_from_disk.before_iteration is False

def test_progress_bar_warn_override() -> None:
    with pytest.warns(UserWarning):
        lgb.progress_bar(total=100)

def test_progress_bar_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    callback = lgb.progress_bar()
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5), callback])
    
    assert issubclass(callback.tqdm_cls, tqdm.std.tqdm)
    assert isinstance(callback.pbar, tqdm.std.tqdm)
    assert callback.pbar is not None
    assert callback.pbar.total == gbm.n_estimators
    assert callback.pbar.n == gbm.n_estimators

def test_progress_bar_early_stopping_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    early_stopping_callback = lgb.early_stopping(5)
    callback = lgb.progress_bar(early_stopping_callback=early_stopping_callback)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping_callback, callback])
    
    assert issubclass(callback.tqdm_cls, tqdm.std.tqdm)
    assert isinstance(callback.pbar, tqdm.std.tqdm)
    assert callback.pbar is not None
    assert callback.pbar.total == gbm.n_estimators
    assert callback.pbar.n >= 0
    assert callback.pbar.n <= gbm.n_estimators
