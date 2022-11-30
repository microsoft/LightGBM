# coding: utf-8
import itertools
import math
import re
from functools import partial
from os import getenv
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_svmlight_file, make_blobs, make_multilabel_classification
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor, RegressorChain
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

import lightgbm as lgb
from lightgbm.compat import PANDAS_INSTALLED, pd_DataFrame

from .utils import (load_boston, load_breast_cancer, load_digits, load_iris, load_linnerud, make_ranking,
                    make_synthetic_regression, sklearn_multiclass_custom_objective, softmax)

decreasing_generator = itertools.count(0, -1)
task_to_model_factory = {
    'ranking': lgb.LGBMRanker,
    'classification': lgb.LGBMClassifier,
    'regression': lgb.LGBMRegressor,
}


def _create_data(task):
    if task == 'ranking':
        X, y, g = make_ranking(n_features=4)
        g = np.bincount(g)
    elif task == 'classification':
        X, y = load_iris(return_X_y=True)
        g = None
    elif task == 'regression':
        X, y = make_synthetic_regression()
        g = None
    return X, y, g


class UnpicklableCallback:
    def __reduce__(self):
        raise Exception("This class in not picklable")

    def __call__(self, env):
        env.model.attr_set_inside_callback = env.iteration * 10


def custom_asymmetric_obj(y_true, y_pred):
    residual = (y_true - y_pred).astype(np.float64)
    grad = np.where(residual < 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 10.0, 2.0)
    return grad, hess


def objective_ls(y_true, y_pred):
    grad = (y_pred - y_true)
    hess = np.ones(len(y_true))
    return grad, hess


def logregobj(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess


def custom_dummy_obj(y_true, y_pred):
    return np.ones(y_true.shape), np.ones(y_true.shape)


def constant_metric(y_true, y_pred):
    return 'error', 0, False


def decreasing_metric(y_true, y_pred):
    return ('decreasing_metric', next(decreasing_generator), False)


def mse(y_true, y_pred):
    return 'custom MSE', mean_squared_error(y_true, y_pred), False


def binary_error(y_true, y_pred):
    return np.mean((y_pred > 0.5) != y_true)


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = log_loss(y_test, gbm.predict_proba(X_test))
    assert ret < 0.12
    assert gbm.evals_result_['valid_0']['binary_logloss'][gbm.best_iteration_ - 1] == pytest.approx(ret)


def test_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = mean_squared_error(y_test, gbm.predict(X_test))
    assert ret < 7
    assert gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1] == pytest.approx(ret)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda_exp', reason='Skip due to differences in implementation details of CUDA Experimental version')
def test_multiclass():
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = multi_error(y_test, gbm.predict(X_test))
    assert ret < 0.05
    ret = multi_logloss(y_test, gbm.predict_proba(X_test))
    assert ret < 0.16
    assert gbm.evals_result_['valid_0']['multi_logloss'][gbm.best_iteration_ - 1] == pytest.approx(ret)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda_exp', reason='Skip due to differences in implementation details of CUDA Experimental version')
def test_lambdarank():
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    X_train, y_train = load_svmlight_file(str(rank_example_dir / 'rank.train'))
    X_test, y_test = load_svmlight_file(str(rank_example_dir / 'rank.test'))
    q_train = np.loadtxt(str(rank_example_dir / 'rank.train.query'))
    q_test = np.loadtxt(str(rank_example_dir / 'rank.test.query'))
    gbm = lgb.LGBMRanker(n_estimators=50)
    gbm.fit(
        X_train,
        y_train,
        group=q_train,
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 3],
        callbacks=[
            lgb.early_stopping(10),
            lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))
        ]
    )
    assert gbm.best_iteration_ <= 24
    assert gbm.best_score_['valid_0']['ndcg@1'] > 0.5674
    assert gbm.best_score_['valid_0']['ndcg@3'] > 0.578


def test_xendcg():
    xendcg_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'xendcg'
    X_train, y_train = load_svmlight_file(str(xendcg_example_dir / 'rank.train'))
    X_test, y_test = load_svmlight_file(str(xendcg_example_dir / 'rank.test'))
    q_train = np.loadtxt(str(xendcg_example_dir / 'rank.train.query'))
    q_test = np.loadtxt(str(xendcg_example_dir / 'rank.test.query'))
    gbm = lgb.LGBMRanker(n_estimators=50, objective='rank_xendcg', random_state=5, n_jobs=1)
    gbm.fit(
        X_train,
        y_train,
        group=q_train,
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 3],
        eval_metric='ndcg',
        callbacks=[
            lgb.early_stopping(10),
            lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))
        ]
    )
    assert gbm.best_iteration_ <= 24
    assert gbm.best_score_['valid_0']['ndcg@1'] > 0.6211
    assert gbm.best_score_['valid_0']['ndcg@3'] > 0.6253


def test_eval_at_aliases():
    rank_example_dir = Path(__file__).absolute().parents[2] / 'examples' / 'lambdarank'
    X_train, y_train = load_svmlight_file(str(rank_example_dir / 'rank.train'))
    X_test, y_test = load_svmlight_file(str(rank_example_dir / 'rank.test'))
    q_train = np.loadtxt(str(rank_example_dir / 'rank.train.query'))
    q_test = np.loadtxt(str(rank_example_dir / 'rank.test.query'))
    for alias in lgb.basic._ConfigAliases.get('eval_at'):
        gbm = lgb.LGBMRanker(n_estimators=5, **{alias: [1, 2, 3, 9]})
        with pytest.warns(UserWarning, match=f"Found '{alias}' in params. Will use it instead of 'eval_at' argument"):
            gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)], eval_group=[q_test])
        assert list(gbm.evals_result_['valid_0'].keys()) == ['ndcg@1', 'ndcg@2', 'ndcg@3', 'ndcg@9']


@pytest.mark.parametrize("custom_objective", [True, False])
def test_objective_aliases(custom_objective):
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    if custom_objective:
        obj = custom_dummy_obj
        metric_name = 'l2'  # default one
    else:
        obj = 'mape'
        metric_name = 'mape'
    evals = []
    for alias in lgb.basic._ConfigAliases.get('objective'):
        gbm = lgb.LGBMRegressor(n_estimators=5, **{alias: obj})
        if alias != 'objective':
            with pytest.warns(UserWarning, match=f"Found '{alias}' in params. Will use it instead of 'objective' argument"):
                gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        assert list(gbm.evals_result_['valid_0'].keys()) == [metric_name]
        evals.append(gbm.evals_result_['valid_0'][metric_name])
    evals_t = np.array(evals).T
    for i in range(evals_t.shape[0]):
        np.testing.assert_allclose(evals_t[i], evals_t[i][0])
    # check that really dummy objective was used and estimator didn't learn anything
    if custom_objective:
        np.testing.assert_allclose(evals_t, evals_t[0][0])


def test_regression_with_custom_objective():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=50, verbose=-1, objective=objective_ls)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    ret = mean_squared_error(y_test, gbm.predict(X_test))
    assert ret < 7.0
    assert gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1] == pytest.approx(ret)


def test_binary_classification_with_custom_objective():
    X, y = load_digits(n_class=2, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMClassifier(n_estimators=50, verbose=-1, objective=logregobj)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5)])
    # prediction result is actually not transformed (is raw) due to custom objective
    y_pred_raw = gbm.predict_proba(X_test)
    assert not np.all(y_pred_raw >= 0)
    y_pred = 1.0 / (1.0 + np.exp(-y_pred_raw))
    ret = binary_error(y_test, y_pred)
    assert ret < 0.05


def test_dart():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(boosting_type='dart', n_estimators=50)
    gbm.fit(X_train, y_train)
    score = gbm.score(X_test, y_test)
    assert score >= 0.8
    assert score <= 1.


def test_stacking_classifier():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    classifiers = [('gbm1', lgb.LGBMClassifier(n_estimators=3)),
                   ('gbm2', lgb.LGBMClassifier(n_estimators=3))]
    clf = StackingClassifier(estimators=classifiers,
                             final_estimator=lgb.LGBMClassifier(n_estimators=3),
                             passthrough=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.8
    assert score <= 1.
    assert clf.n_features_in_ == 4  # number of input features
    assert len(clf.named_estimators_['gbm1'].feature_importances_) == 4
    assert clf.named_estimators_['gbm1'].n_features_in_ == clf.named_estimators_['gbm2'].n_features_in_
    assert clf.final_estimator_.n_features_in_ == 10  # number of concatenated features
    assert len(clf.final_estimator_.feature_importances_) == 10
    assert all(clf.named_estimators_['gbm1'].classes_ == clf.named_estimators_['gbm2'].classes_)
    assert all(clf.classes_ == clf.named_estimators_['gbm1'].classes_)


def test_stacking_regressor():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    regressors = [('gbm1', lgb.LGBMRegressor(n_estimators=3)),
                  ('gbm2', lgb.LGBMRegressor(n_estimators=3))]
    reg = StackingRegressor(estimators=regressors,
                            final_estimator=lgb.LGBMRegressor(n_estimators=3),
                            passthrough=True)
    reg.fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.
    assert reg.n_features_in_ == 13  # number of input features
    assert len(reg.named_estimators_['gbm1'].feature_importances_) == 13
    assert reg.named_estimators_['gbm1'].n_features_in_ == reg.named_estimators_['gbm2'].n_features_in_
    assert reg.final_estimator_.n_features_in_ == 15  # number of concatenated features
    assert len(reg.final_estimator_.feature_importances_) == 15


def test_grid_search():
    X, y = load_iris(return_X_y=True)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    params = dict(subsample=0.8,
                  subsample_freq=1)
    grid_params = dict(boosting_type=['rf', 'gbdt'],
                       n_estimators=[4, 6],
                       reg_alpha=[0.01, 0.005])
    evals_result = {}
    fit_params = dict(
        eval_set=[(X_val, y_val)],
        eval_metric=constant_metric,
        callbacks=[
            lgb.early_stopping(2),
            lgb.record_evaluation(evals_result)
        ]
    )
    grid = GridSearchCV(estimator=lgb.LGBMClassifier(**params), param_grid=grid_params, cv=2)
    grid.fit(X_train, y_train, **fit_params)
    score = grid.score(X_test, y_test)  # utilizes GridSearchCV default refit=True
    assert grid.best_params_['boosting_type'] in ['rf', 'gbdt']
    assert grid.best_params_['n_estimators'] in [4, 6]
    assert grid.best_params_['reg_alpha'] in [0.01, 0.005]
    assert grid.best_score_ <= 1.
    assert grid.best_estimator_.best_iteration_ == 1
    assert grid.best_estimator_.best_score_['valid_0']['multi_logloss'] < 0.25
    assert grid.best_estimator_.best_score_['valid_0']['error'] == 0
    assert score >= 0.2
    assert score <= 1.
    assert evals_result == grid.best_estimator_.evals_result_


def test_random_search():
    X, y = load_iris(return_X_y=True)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1,
                                                      random_state=42)
    n_iter = 3  # Number of samples
    params = dict(subsample=0.8,
                  subsample_freq=1)
    param_dist = dict(boosting_type=['rf', 'gbdt'],
                      n_estimators=[np.random.randint(low=3, high=10) for i in range(n_iter)],
                      reg_alpha=[np.random.uniform(low=0.01, high=0.06) for i in range(n_iter)])
    fit_params = dict(eval_set=[(X_val, y_val)],
                      eval_metric=constant_metric,
                      callbacks=[lgb.early_stopping(2)])
    rand = RandomizedSearchCV(estimator=lgb.LGBMClassifier(**params),
                              param_distributions=param_dist, cv=2,
                              n_iter=n_iter, random_state=42)
    rand.fit(X_train, y_train, **fit_params)
    score = rand.score(X_test, y_test)  # utilizes RandomizedSearchCV default refit=True
    assert rand.best_params_['boosting_type'] in ['rf', 'gbdt']
    assert rand.best_params_['n_estimators'] in list(range(3, 10))
    assert rand.best_params_['reg_alpha'] >= 0.01  # Left-closed boundary point
    assert rand.best_params_['reg_alpha'] <= 0.06  # Right-closed boundary point
    assert rand.best_score_ <= 1.
    assert rand.best_estimator_.best_score_['valid_0']['multi_logloss'] < 0.25
    assert rand.best_estimator_.best_score_['valid_0']['error'] == 0
    assert score >= 0.2
    assert score <= 1.


def test_multioutput_classifier():
    n_outputs = 3
    X, y = make_multilabel_classification(n_samples=100, n_features=20,
                                          n_classes=n_outputs, random_state=0)
    y = y.astype(str)  # utilize label encoder at it's max power
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)
    clf = MultiOutputClassifier(estimator=lgb.LGBMClassifier(n_estimators=10))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.
    np.testing.assert_array_equal(np.tile(np.unique(y_train), n_outputs),
                                  np.concatenate(clf.classes_))
    for classifier in clf.estimators_:
        assert isinstance(classifier, lgb.LGBMClassifier)
        assert isinstance(classifier.booster_, lgb.Booster)


def test_multioutput_regressor():
    bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
    X, y = bunch['data'], bunch['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)
    reg = MultiOutputRegressor(estimator=lgb.LGBMRegressor(n_estimators=10))
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    _, score, _ = mse(y_test, y_pred)
    assert score >= 0.2
    assert score <= 120.
    for regressor in reg.estimators_:
        assert isinstance(regressor, lgb.LGBMRegressor)
        assert isinstance(regressor.booster_, lgb.Booster)


def test_classifier_chain():
    n_outputs = 3
    X, y = make_multilabel_classification(n_samples=100, n_features=20,
                                          n_classes=n_outputs, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)
    order = [2, 0, 1]
    clf = ClassifierChain(base_estimator=lgb.LGBMClassifier(n_estimators=10),
                          order=order, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    assert score >= 0.2
    assert score <= 1.
    np.testing.assert_array_equal(np.tile(np.unique(y_train), n_outputs),
                                  np.concatenate(clf.classes_))
    assert order == clf.order_
    for classifier in clf.estimators_:
        assert isinstance(classifier, lgb.LGBMClassifier)
        assert isinstance(classifier.booster_, lgb.Booster)


def test_regressor_chain():
    bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
    X, y = bunch['data'], bunch['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    order = [2, 0, 1]
    reg = RegressorChain(base_estimator=lgb.LGBMRegressor(n_estimators=10), order=order,
                         random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    _, score, _ = mse(y_test, y_pred)
    assert score >= 0.2
    assert score <= 120.
    assert order == reg.order_
    for regressor in reg.estimators_:
        assert isinstance(regressor, lgb.LGBMRegressor)
        assert isinstance(regressor.booster_, lgb.Booster)


def test_clone_and_property():
    X, y = make_synthetic_regression()
    gbm = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    gbm.fit(X, y)

    gbm_clone = clone(gbm)
    assert isinstance(gbm.booster_, lgb.Booster)
    assert isinstance(gbm.feature_importances_, np.ndarray)

    X, y = load_digits(n_class=2, return_X_y=True)
    clf = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
    clf.fit(X, y)
    assert sorted(clf.classes_) == [0, 1]
    assert clf.n_classes_ == 2
    assert isinstance(clf.booster_, lgb.Booster)
    assert isinstance(clf.feature_importances_, np.ndarray)


def test_joblib():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=10, objective=custom_asymmetric_obj,
                            verbose=-1, importance_type='split')
    gbm.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_test, y_test)
        ],
        eval_metric=mse,
        callbacks=[
            lgb.early_stopping(5),
            lgb.reset_parameter(learning_rate=list(np.arange(1, 0, -0.1)))
        ]
    )

    joblib.dump(gbm, 'lgb.pkl')  # test model with custom functions
    gbm_pickle = joblib.load('lgb.pkl')
    assert isinstance(gbm_pickle.booster_, lgb.Booster)
    assert gbm.get_params() == gbm_pickle.get_params()
    np.testing.assert_array_equal(gbm.feature_importances_, gbm_pickle.feature_importances_)
    assert gbm_pickle.learning_rate == pytest.approx(0.1)
    assert callable(gbm_pickle.objective)

    for eval_set in gbm.evals_result_:
        for metric in gbm.evals_result_[eval_set]:
            np.testing.assert_allclose(gbm.evals_result_[eval_set][metric],
                                       gbm_pickle.evals_result_[eval_set][metric])
    pred_origin = gbm.predict(X_test)
    pred_pickle = gbm_pickle.predict(X_test)
    np.testing.assert_allclose(pred_origin, pred_pickle)


def test_non_serializable_objects_in_callbacks(tmp_path):
    unpicklable_callback = UnpicklableCallback()

    with pytest.raises(Exception, match="This class in not picklable"):
        joblib.dump(unpicklable_callback, tmp_path / 'tmp.joblib')

    X, y = make_synthetic_regression()
    gbm = lgb.LGBMRegressor(n_estimators=5)
    gbm.fit(X, y, callbacks=[unpicklable_callback])
    assert gbm.booster_.attr_set_inside_callback == 40


def test_random_state_object():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    state1 = np.random.RandomState(123)
    state2 = np.random.RandomState(123)
    clf1 = lgb.LGBMClassifier(n_estimators=10, subsample=0.5, subsample_freq=1, random_state=state1)
    clf2 = lgb.LGBMClassifier(n_estimators=10, subsample=0.5, subsample_freq=1, random_state=state2)
    # Test if random_state is properly stored
    assert clf1.random_state is state1
    assert clf2.random_state is state2
    # Test if two random states produce identical models
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test, raw_score=True)
    y_pred2 = clf2.predict(X_test, raw_score=True)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_array_equal(clf1.feature_importances_, clf2.feature_importances_)
    df1 = clf1.booster_.model_to_string(num_iteration=0)
    df2 = clf2.booster_.model_to_string(num_iteration=0)
    assert df1 == df2
    # Test if subsequent fits sample from random_state object and produce different models
    clf1.fit(X_train, y_train)
    y_pred1_refit = clf1.predict(X_test, raw_score=True)
    df3 = clf1.booster_.model_to_string(num_iteration=0)
    assert clf1.random_state is state1
    assert clf2.random_state is state2
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred1, y_pred1_refit)
    assert df1 != df3


def test_feature_importances_single_leaf():
    data = load_iris(return_X_y=False)
    clf = lgb.LGBMClassifier(n_estimators=10)
    clf.fit(data.data, data.target)
    importances = clf.feature_importances_
    assert len(importances) == 4


def test_feature_importances_type():
    data = load_iris(return_X_y=False)
    clf = lgb.LGBMClassifier(n_estimators=10)
    clf.fit(data.data, data.target)
    clf.set_params(importance_type='split')
    importances_split = clf.feature_importances_
    clf.set_params(importance_type='gain')
    importances_gain = clf.feature_importances_
    # Test that the largest element is NOT the same, the smallest can be the same, i.e. zero
    importance_split_top1 = sorted(importances_split, reverse=True)[0]
    importance_gain_top1 = sorted(importances_gain, reverse=True)[0]
    assert importance_split_top1 != importance_gain_top1


def test_pandas_categorical():
    pd = pytest.importorskip("pandas")
    np.random.seed(42)  # sometimes there is no difference how cols are treated (cat or not cat)
    X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                      "B": np.random.permutation([1, 2, 3] * 100),  # int
                      "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                      "D": np.random.permutation([True, False] * 150),  # bool
                      "E": pd.Categorical(np.random.permutation(['z', 'y', 'x', 'w', 'v'] * 60),
                                          ordered=True)})  # str and ordered categorical
    y = np.random.permutation([0, 1] * 150)
    X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),  # unseen category
                           "B": np.random.permutation([1, 3] * 30),
                           "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                           "D": np.random.permutation([True, False] * 30),
                           "E": pd.Categorical(np.random.permutation(['z', 'y'] * 30),
                                               ordered=True)})
    np.random.seed()  # reset seed
    cat_cols_actual = ["A", "B", "C", "D"]
    cat_cols_to_store = cat_cols_actual + ["E"]
    X[cat_cols_actual] = X[cat_cols_actual].astype('category')
    X_test[cat_cols_actual] = X_test[cat_cols_actual].astype('category')
    cat_values = [X[col].cat.categories.tolist() for col in cat_cols_to_store]
    gbm0 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y)
    pred0 = gbm0.predict(X_test, raw_score=True)
    pred_prob = gbm0.predict_proba(X_test)[:, 1]
    gbm1 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, pd.Series(y), categorical_feature=[0])
    pred1 = gbm1.predict(X_test, raw_score=True)
    gbm2 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=['A'])
    pred2 = gbm2.predict(X_test, raw_score=True)
    gbm3 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
    pred3 = gbm3.predict(X_test, raw_score=True)
    gbm3.booster_.save_model('categorical.model')
    gbm4 = lgb.Booster(model_file='categorical.model')
    pred4 = gbm4.predict(X_test)
    gbm5 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=['A', 'B', 'C', 'D', 'E'])
    pred5 = gbm5.predict(X_test, raw_score=True)
    gbm6 = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y, categorical_feature=[])
    pred6 = gbm6.predict(X_test, raw_score=True)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred1)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred2)
    np.testing.assert_allclose(pred1, pred2)
    np.testing.assert_allclose(pred0, pred3)
    np.testing.assert_allclose(pred_prob, pred4)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred5)  # ordered cat features aren't treated as cat features by default
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(pred0, pred6)
    assert gbm0.booster_.pandas_categorical == cat_values
    assert gbm1.booster_.pandas_categorical == cat_values
    assert gbm2.booster_.pandas_categorical == cat_values
    assert gbm3.booster_.pandas_categorical == cat_values
    assert gbm4.pandas_categorical == cat_values
    assert gbm5.booster_.pandas_categorical == cat_values
    assert gbm6.booster_.pandas_categorical == cat_values


def test_pandas_sparse():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"A": pd.arrays.SparseArray(np.random.permutation([0, 1, 2] * 100)),
                      "B": pd.arrays.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1, 0.2] * 60)),
                      "C": pd.arrays.SparseArray(np.random.permutation([True, False] * 150))})
    y = pd.Series(pd.arrays.SparseArray(np.random.permutation([0, 1] * 150)))
    X_test = pd.DataFrame({"A": pd.arrays.SparseArray(np.random.permutation([0, 2] * 30)),
                           "B": pd.arrays.SparseArray(np.random.permutation([0.0, 0.1, 0.2, -0.1] * 15)),
                           "C": pd.arrays.SparseArray(np.random.permutation([True, False] * 30))})
    for dtype in pd.concat([X.dtypes, X_test.dtypes, pd.Series(y.dtypes)]):
        assert pd.api.types.is_sparse(dtype)
    gbm = lgb.sklearn.LGBMClassifier(n_estimators=10).fit(X, y)
    pred_sparse = gbm.predict(X_test, raw_score=True)
    if hasattr(X_test, 'sparse'):
        pred_dense = gbm.predict(X_test.sparse.to_dense(), raw_score=True)
    else:
        pred_dense = gbm.predict(X_test.to_dense(), raw_score=True)
    np.testing.assert_allclose(pred_sparse, pred_dense)


def test_predict():
    # With default params
    iris = load_iris(return_X_y=False)
    X_train, X_test, y_train, _ = train_test_split(iris.data, iris.target,
                                                   test_size=0.2, random_state=42)

    gbm = lgb.train({'objective': 'multiclass',
                     'num_class': 3,
                     'verbose': -1},
                    lgb.Dataset(X_train, y_train))
    clf = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train)

    # Tests same probabilities
    res_engine = gbm.predict(X_test)
    res_sklearn = clf.predict_proba(X_test)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same predictions
    res_engine = np.argmax(gbm.predict(X_test), axis=1)
    res_sklearn = clf.predict(X_test)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same raw scores
    res_engine = gbm.predict(X_test, raw_score=True)
    res_sklearn = clf.predict(X_test, raw_score=True)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same leaf indices
    res_engine = gbm.predict(X_test, pred_leaf=True)
    res_sklearn = clf.predict(X_test, pred_leaf=True)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same feature contributions
    res_engine = gbm.predict(X_test, pred_contrib=True)
    res_sklearn = clf.predict(X_test, pred_contrib=True)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests other parameters for the prediction works
    res_engine = gbm.predict(X_test)
    res_sklearn_params = clf.predict_proba(X_test,
                                           pred_early_stop=True,
                                           pred_early_stop_margin=1.0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(res_engine, res_sklearn_params)

    # Tests start_iteration
    # Tests same probabilities, starting from iteration 10
    res_engine = gbm.predict(X_test, start_iteration=10)
    res_sklearn = clf.predict_proba(X_test, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same predictions, starting from iteration 10
    res_engine = np.argmax(gbm.predict(X_test, start_iteration=10), axis=1)
    res_sklearn = clf.predict(X_test, start_iteration=10)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same raw scores, starting from iteration 10
    res_engine = gbm.predict(X_test, raw_score=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, raw_score=True, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests same leaf indices, starting from iteration 10
    res_engine = gbm.predict(X_test, pred_leaf=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, pred_leaf=True, start_iteration=10)
    np.testing.assert_equal(res_engine, res_sklearn)

    # Tests same feature contributions, starting from iteration 10
    res_engine = gbm.predict(X_test, pred_contrib=True, start_iteration=10)
    res_sklearn = clf.predict(X_test, pred_contrib=True, start_iteration=10)
    np.testing.assert_allclose(res_engine, res_sklearn)

    # Tests other parameters for the prediction works, starting from iteration 10
    res_engine = gbm.predict(X_test, start_iteration=10)
    res_sklearn_params = clf.predict_proba(X_test,
                                           pred_early_stop=True,
                                           pred_early_stop_margin=1.0, start_iteration=10)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(res_engine, res_sklearn_params)


def test_predict_with_params_from_init():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    predict_params = {
        'pred_early_stop': True,
        'pred_early_stop_margin': 1.0
    }

    y_preds_no_params = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).predict(
        X_test, raw_score=True)

    y_preds_params_in_predict = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).predict(
        X_test, raw_score=True, **predict_params)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_preds_no_params, y_preds_params_in_predict)

    y_preds_params_in_set_params_before_fit = lgb.LGBMClassifier(verbose=-1).set_params(
        **predict_params).fit(X_train, y_train).predict(X_test, raw_score=True)
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_set_params_before_fit)

    y_preds_params_in_set_params_after_fit = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train).set_params(
        **predict_params).predict(X_test, raw_score=True)
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_set_params_after_fit)

    y_preds_params_in_init = lgb.LGBMClassifier(verbose=-1, **predict_params).fit(X_train, y_train).predict(
        X_test, raw_score=True)
    np.testing.assert_allclose(y_preds_params_in_predict, y_preds_params_in_init)

    # test that params passed in predict have higher priority
    y_preds_params_overwritten = lgb.LGBMClassifier(verbose=-1, **predict_params).fit(X_train, y_train).predict(
        X_test, raw_score=True, pred_early_stop=False)
    np.testing.assert_allclose(y_preds_no_params, y_preds_params_overwritten)


def test_evaluate_train_set():
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    gbm = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    gbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
    assert len(gbm.evals_result_) == 2
    assert 'training' in gbm.evals_result_
    assert len(gbm.evals_result_['training']) == 1
    assert 'l2' in gbm.evals_result_['training']
    assert 'valid_1' in gbm.evals_result_
    assert len(gbm.evals_result_['valid_1']) == 1
    assert 'l2' in gbm.evals_result_['valid_1']


def test_metrics():
    X, y = load_boston(return_X_y=True)
    params = {'n_estimators': 2, 'verbose': -1}
    params_fit = {'X': X, 'y': y, 'eval_set': (X, y)}

    # no custom objective, no custom metric
    # default metric
    gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'l2' in gbm.evals_result_['training']

    # non-default metric
    gbm = lgb.LGBMRegressor(metric='mape', **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'mape' in gbm.evals_result_['training']

    # no metric
    gbm = lgb.LGBMRegressor(metric='None', **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # non-default metric in eval_metric
    gbm = lgb.LGBMRegressor(**params).fit(eval_metric='mape', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default metric with non-default metric in eval_metric
    gbm = lgb.LGBMRegressor(metric='gamma', **params).fit(eval_metric='mape', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'gamma' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default metric with multiple metrics in eval_metric
    gbm = lgb.LGBMRegressor(metric='gamma',
                            **params).fit(eval_metric=['l2', 'mape'], **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'gamma' in gbm.evals_result_['training']
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default metric with multiple metrics in eval_metric for LGBMClassifier
    X_classification, y_classification = load_breast_cancer(return_X_y=True)
    params_classification = {'n_estimators': 2, 'verbose': -1,
                             'objective': 'binary', 'metric': 'binary_logloss'}
    params_fit_classification = {'X': X_classification, 'y': y_classification,
                                 'eval_set': (X_classification, y_classification)}
    gbm = lgb.LGBMClassifier(**params_classification).fit(eval_metric=['fair', 'error'],
                                                          **params_fit_classification)
    assert len(gbm.evals_result_['training']) == 3
    assert 'fair' in gbm.evals_result_['training']
    assert 'binary_error' in gbm.evals_result_['training']
    assert 'binary_logloss' in gbm.evals_result_['training']

    # default metric for non-default objective
    gbm = lgb.LGBMRegressor(objective='regression_l1', **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'l1' in gbm.evals_result_['training']

    # non-default metric for non-default objective
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='mape',
                            **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'mape' in gbm.evals_result_['training']

    # no metric
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='None',
                            **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # non-default metric in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective='regression_l1',
                            **params).fit(eval_metric='mape', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l1' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default metric with non-default metric in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='gamma',
                            **params).fit(eval_metric='mape', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'gamma' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default metric with multiple metrics in eval_metric for non-default objective
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='gamma',
                            **params).fit(eval_metric=['l2', 'mape'], **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'gamma' in gbm.evals_result_['training']
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # custom objective, no custom metric
    # default regression metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'l2' in gbm.evals_result_['training']

    # non-default regression metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric='mape', **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'mape' in gbm.evals_result_['training']

    # multiple regression metrics for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=['l1', 'gamma'],
                            **params).fit(**params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l1' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']

    # no metric
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric='None',
                            **params).fit(**params_fit)
    assert gbm.evals_result_ == {}

    # default regression metric with non-default metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj,
                            **params).fit(eval_metric='mape', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # non-default regression metric with metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric='mape',
                            **params).fit(eval_metric='gamma', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'mape' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']

    # multiple regression metrics with metric in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=['l1', 'gamma'],
                            **params).fit(eval_metric='l2', **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'l1' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']
    assert 'l2' in gbm.evals_result_['training']

    # multiple regression metrics with multiple metrics in eval_metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=['l1', 'gamma'],
                            **params).fit(eval_metric=['l2', 'mape'], **params_fit)
    assert len(gbm.evals_result_['training']) == 4
    assert 'l1' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']

    # no custom objective, custom metric
    # default metric with custom metric
    gbm = lgb.LGBMRegressor(**params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l2' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # non-default metric with custom metric
    gbm = lgb.LGBMRegressor(metric='mape',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'mape' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # multiple metrics with custom metric
    gbm = lgb.LGBMRegressor(metric=['l1', 'gamma'],
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'l1' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # custom metric (disable default metric)
    gbm = lgb.LGBMRegressor(metric='None',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'error' in gbm.evals_result_['training']

    # default metric for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective='regression_l1',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'l1' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # non-default metric for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='mape',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'mape' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # multiple metrics for non-default objective with custom metric
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric=['l1', 'gamma'],
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'l1' in gbm.evals_result_['training']
    assert 'gamma' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # custom metric (disable default metric for non-default objective)
    gbm = lgb.LGBMRegressor(objective='regression_l1', metric='None',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'error' in gbm.evals_result_['training']

    # custom objective, custom metric
    # custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj,
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'error' in gbm.evals_result_['training']

    # non-default regression metric with custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric='mape',
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'mape' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    # multiple regression metrics with custom metric for custom objective
    gbm = lgb.LGBMRegressor(objective=custom_dummy_obj, metric=['l2', 'mape'],
                            **params).fit(eval_metric=constant_metric, **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'l2' in gbm.evals_result_['training']
    assert 'mape' in gbm.evals_result_['training']
    assert 'error' in gbm.evals_result_['training']

    X, y = load_digits(n_class=3, return_X_y=True)
    params_fit = {'X': X, 'y': y, 'eval_set': (X, y)}

    # default metric and invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric='binary_error', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'multi_logloss' in gbm.evals_result_['training']
    assert 'multi_error' in gbm.evals_result_['training']

    # invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric='binary_error', **params_fit)
    assert gbm.objective_ == 'multiclass'
    assert len(gbm.evals_result_['training']) == 2
    assert 'multi_logloss' in gbm.evals_result_['training']
    assert 'multi_error' in gbm.evals_result_['training']

    # default metric for non-default multiclass objective
    # and invalid binary metric is replaced with multiclass alternative
    gbm = lgb.LGBMClassifier(objective='ovr',
                             **params).fit(eval_metric='binary_error', **params_fit)
    assert gbm.objective_ == 'ovr'
    assert len(gbm.evals_result_['training']) == 2
    assert 'multi_logloss' in gbm.evals_result_['training']
    assert 'multi_error' in gbm.evals_result_['training']

    X, y = load_digits(n_class=2, return_X_y=True)
    params_fit = {'X': X, 'y': y, 'eval_set': (X, y)}

    # default metric and invalid multiclass metric is replaced with binary alternative
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric='multi_error', **params_fit)
    assert len(gbm.evals_result_['training']) == 2
    assert 'binary_logloss' in gbm.evals_result_['training']
    assert 'binary_error' in gbm.evals_result_['training']

    # invalid multiclass metric is replaced with binary alternative for custom objective
    gbm = lgb.LGBMClassifier(objective=custom_dummy_obj,
                             **params).fit(eval_metric='multi_logloss', **params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'binary_logloss' in gbm.evals_result_['training']


def test_multiple_eval_metrics():

    X, y = load_breast_cancer(return_X_y=True)

    params = {'n_estimators': 2, 'verbose': -1, 'objective': 'binary', 'metric': 'binary_logloss'}
    params_fit = {'X': X, 'y': y, 'eval_set': (X, y)}

    # Verify that can receive a list of metrics, only callable
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[constant_metric, decreasing_metric], **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'error' in gbm.evals_result_['training']
    assert 'decreasing_metric' in gbm.evals_result_['training']
    assert 'binary_logloss' in gbm.evals_result_['training']

    # Verify that can receive a list of custom and built-in metrics
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[constant_metric, decreasing_metric, 'fair'], **params_fit)
    assert len(gbm.evals_result_['training']) == 4
    assert 'error' in gbm.evals_result_['training']
    assert 'decreasing_metric' in gbm.evals_result_['training']
    assert 'binary_logloss' in gbm.evals_result_['training']
    assert 'fair' in gbm.evals_result_['training']

    # Verify that works as expected when eval_metric is empty
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=[], **params_fit)
    assert len(gbm.evals_result_['training']) == 1
    assert 'binary_logloss' in gbm.evals_result_['training']

    # Verify that can receive a list of metrics, only built-in
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=['fair', 'error'], **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'binary_logloss' in gbm.evals_result_['training']

    # Verify that eval_metric is robust to receiving a list with None
    gbm = lgb.LGBMClassifier(**params).fit(eval_metric=['fair', 'error', None], **params_fit)
    assert len(gbm.evals_result_['training']) == 3
    assert 'binary_logloss' in gbm.evals_result_['training']


def test_nan_handle():
    nrows = 100
    ncols = 10
    X = np.random.randn(nrows, ncols)
    y = np.random.randn(nrows) + np.full(nrows, 1e30)
    weight = np.zeros(nrows)
    params = {'n_estimators': 20, 'verbose': -1}
    params_fit = {'X': X, 'y': y, 'sample_weight': weight, 'eval_set': (X, y),
                  'callbacks': [lgb.early_stopping(5)]}
    gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
    np.testing.assert_allclose(gbm.evals_result_['training']['l2'], np.nan)


@pytest.mark.skipif(getenv('TASK', '') == 'cuda_exp', reason='Skip due to differences in implementation details of CUDA Experimental version')
def test_first_metric_only():

    def fit_and_check(eval_set_names, metric_names, assumed_iteration, first_metric_only):
        params['first_metric_only'] = first_metric_only
        gbm = lgb.LGBMRegressor(**params).fit(**params_fit)
        assert len(gbm.evals_result_) == len(eval_set_names)
        for eval_set_name in eval_set_names:
            assert eval_set_name in gbm.evals_result_
            assert len(gbm.evals_result_[eval_set_name]) == len(metric_names)
            for metric_name in metric_names:
                assert metric_name in gbm.evals_result_[eval_set_name]

                actual = len(gbm.evals_result_[eval_set_name][metric_name])
                expected = assumed_iteration + (params['early_stopping_rounds']
                                                if eval_set_name != 'training'
                                                and assumed_iteration != gbm.n_estimators else 0)
                assert expected == actual
                if eval_set_name != 'training':
                    assert assumed_iteration == gbm.best_iteration_
                else:
                    assert gbm.n_estimators == gbm.best_iteration_

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.5, random_state=72)
    params = {'n_estimators': 30,
              'learning_rate': 0.8,
              'num_leaves': 15,
              'verbose': -1,
              'seed': 123,
              'early_stopping_rounds': 5}  # early stop should be supported via global LightGBM parameter
    params_fit = {'X': X_train,
                  'y': y_train}

    iter_valid1_l1 = 3
    iter_valid1_l2 = 18
    iter_valid2_l1 = 11
    iter_valid2_l2 = 7
    assert len(set([iter_valid1_l1, iter_valid1_l2, iter_valid2_l1, iter_valid2_l2])) == 4
    iter_min_l1 = min([iter_valid1_l1, iter_valid2_l1])
    iter_min_l2 = min([iter_valid1_l2, iter_valid2_l2])
    iter_min = min([iter_min_l1, iter_min_l2])
    iter_min_valid1 = min([iter_valid1_l1, iter_valid1_l2])

    # feval
    params['metric'] = 'None'
    params_fit['eval_metric'] = lambda preds, train_data: [decreasing_metric(preds, train_data),
                                                           constant_metric(preds, train_data)]
    params_fit['eval_set'] = (X_test1, y_test1)
    fit_and_check(['valid_0'], ['decreasing_metric', 'error'], 1, False)
    fit_and_check(['valid_0'], ['decreasing_metric', 'error'], 30, True)
    params_fit['eval_metric'] = lambda preds, train_data: [constant_metric(preds, train_data),
                                                           decreasing_metric(preds, train_data)]
    fit_and_check(['valid_0'], ['decreasing_metric', 'error'], 1, True)

    # single eval_set
    params.pop('metric')
    params_fit.pop('eval_metric')
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, False)
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, True)

    params_fit['eval_metric'] = "l2"
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, False)
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, True)

    params_fit['eval_metric'] = "l1"
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_min_valid1, False)
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_valid1_l1, True)

    params_fit['eval_metric'] = ["l1", "l2"]
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_min_valid1, False)
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_valid1_l1, True)

    params_fit['eval_metric'] = ["l2", "l1"]
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_min_valid1, False)
    fit_and_check(['valid_0'], ['l1', 'l2'], iter_valid1_l2, True)

    params_fit['eval_metric'] = ["l2", "regression", "mse"]  # test aliases
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, False)
    fit_and_check(['valid_0'], ['l2'], iter_valid1_l2, True)

    # two eval_set
    params_fit['eval_set'] = [(X_test1, y_test1), (X_test2, y_test2)]
    params_fit['eval_metric'] = ["l1", "l2"]
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min_l1, True)
    params_fit['eval_metric'] = ["l2", "l1"]
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min_l2, True)

    params_fit['eval_set'] = [(X_test2, y_test2), (X_test1, y_test1)]
    params_fit['eval_metric'] = ["l1", "l2"]
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min, False)
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min_l1, True)
    params_fit['eval_metric'] = ["l2", "l1"]
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min, False)
    fit_and_check(['valid_0', 'valid_1'], ['l1', 'l2'], iter_min_l2, True)


def test_class_weight():
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_str = y_train.astype('str')
    y_test_str = y_test.astype('str')
    gbm = lgb.LGBMClassifier(n_estimators=10, class_weight='balanced', verbose=-1)
    gbm.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test), (X_test, y_test),
                      (X_test, y_test), (X_test, y_test)],
            eval_class_weight=['balanced', None, 'balanced', {1: 10, 4: 20}, {5: 30, 2: 40}])
    for eval_set1, eval_set2 in itertools.combinations(gbm.evals_result_.keys(), 2):
        for metric in gbm.evals_result_[eval_set1]:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_allclose,
                                     gbm.evals_result_[eval_set1][metric],
                                     gbm.evals_result_[eval_set2][metric])
    gbm_str = lgb.LGBMClassifier(n_estimators=10, class_weight='balanced', verbose=-1)
    gbm_str.fit(X_train, y_train_str,
                eval_set=[(X_train, y_train_str), (X_test, y_test_str),
                          (X_test, y_test_str), (X_test, y_test_str), (X_test, y_test_str)],
                eval_class_weight=['balanced', None, 'balanced', {'1': 10, '4': 20}, {'5': 30, '2': 40}])
    for eval_set1, eval_set2 in itertools.combinations(gbm_str.evals_result_.keys(), 2):
        for metric in gbm_str.evals_result_[eval_set1]:
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_allclose,
                                     gbm_str.evals_result_[eval_set1][metric],
                                     gbm_str.evals_result_[eval_set2][metric])
    for eval_set in gbm.evals_result_:
        for metric in gbm.evals_result_[eval_set]:
            np.testing.assert_allclose(gbm.evals_result_[eval_set][metric],
                                       gbm_str.evals_result_[eval_set][metric])


def test_continue_training_with_model():
    X, y = load_digits(n_class=3, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    init_gbm = lgb.LGBMClassifier(n_estimators=5).fit(X_train, y_train, eval_set=(X_test, y_test))
    gbm = lgb.LGBMClassifier(n_estimators=5).fit(X_train, y_train, eval_set=(X_test, y_test),
                                                 init_model=init_gbm)
    assert len(init_gbm.evals_result_['valid_0']['multi_logloss']) == len(gbm.evals_result_['valid_0']['multi_logloss'])
    assert len(init_gbm.evals_result_['valid_0']['multi_logloss']) == 5
    assert gbm.evals_result_['valid_0']['multi_logloss'][-1] < init_gbm.evals_result_['valid_0']['multi_logloss'][-1]


def test_actual_number_of_trees():
    X = [[1, 2, 3], [1, 2, 3]]
    y = [1, 1]
    n_estimators = 5
    gbm = lgb.LGBMRegressor(n_estimators=n_estimators).fit(X, y)
    assert gbm.n_estimators == n_estimators
    assert gbm.n_estimators_ == 1
    assert gbm.n_iter_ == 1
    np.testing.assert_array_equal(gbm.predict(np.array(X) * 10), y)


def test_check_is_fitted():
    X, y = load_digits(n_class=2, return_X_y=True)
    est = lgb.LGBMModel(n_estimators=5, objective="binary")
    clf = lgb.LGBMClassifier(n_estimators=5)
    reg = lgb.LGBMRegressor(n_estimators=5)
    rnk = lgb.LGBMRanker(n_estimators=5)
    models = (est, clf, reg, rnk)
    for model in models:
        with pytest.raises(lgb.compat.LGBMNotFittedError):
            check_is_fitted(model)
    est.fit(X, y)
    clf.fit(X, y)
    reg.fit(X, y)
    rnk.fit(X, y, group=np.ones(X.shape[0]))
    for model in models:
        check_is_fitted(model)


@parametrize_with_checks([lgb.LGBMClassifier(), lgb.LGBMRegressor()])
def test_sklearn_integration(estimator, check):
    estimator.set_params(min_child_samples=1, min_data_in_bin=1)
    check(estimator)


@pytest.mark.parametrize('task', ['classification', 'ranking', 'regression'])
def test_training_succeeds_when_data_is_dataframe_and_label_is_column_array(task):
    pd = pytest.importorskip("pandas")
    X, y, g = _create_data(task)
    X = pd.DataFrame(X)
    y_col_array = y.reshape(-1, 1)
    params = {
        'n_estimators': 1,
        'num_leaves': 3,
        'random_state': 0
    }
    model_factory = task_to_model_factory[task]
    with pytest.warns(UserWarning, match='column-vector'):
        if task == 'ranking':
            model_1d = model_factory(**params).fit(X, y, group=g)
            model_2d = model_factory(**params).fit(X, y_col_array, group=g)
        else:
            model_1d = model_factory(**params).fit(X, y)
            model_2d = model_factory(**params).fit(X, y_col_array)

    preds_1d = model_1d.predict(X)
    preds_2d = model_2d.predict(X)
    np.testing.assert_array_equal(preds_1d, preds_2d)


@pytest.mark.parametrize('use_weight', [True, False])
def test_multiclass_custom_objective(use_weight):
    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    weight = np.full_like(y, 2) if use_weight else None
    params = {'n_estimators': 10, 'num_leaves': 7}
    builtin_obj_model = lgb.LGBMClassifier(**params)
    builtin_obj_model.fit(X, y, sample_weight=weight)
    builtin_obj_preds = builtin_obj_model.predict_proba(X)

    custom_obj_model = lgb.LGBMClassifier(objective=sklearn_multiclass_custom_objective, **params)
    custom_obj_model.fit(X, y, sample_weight=weight)
    custom_obj_preds = softmax(custom_obj_model.predict(X, raw_score=True))

    np.testing.assert_allclose(builtin_obj_preds, custom_obj_preds, rtol=0.01)
    assert not callable(builtin_obj_model.objective_)
    assert callable(custom_obj_model.objective_)


@pytest.mark.parametrize('use_weight', [True, False])
def test_multiclass_custom_eval(use_weight):
    def custom_eval(y_true, y_pred, weight):
        loss = log_loss(y_true, y_pred, sample_weight=weight)
        return 'custom_logloss', loss, False

    centers = [[-4, -4], [4, 4], [-4, 4]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=42)
    train_test_split_func = partial(train_test_split, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split_func(X, y)
    if use_weight:
        weight = np.full_like(y, 2)
        weight_train, weight_valid = train_test_split_func(weight)
    else:
        weight_train = None
        weight_valid = None
    params = {'objective': 'multiclass', 'num_class': 3, 'num_leaves': 7}
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=['train', 'valid'],
        eval_sample_weight=[weight_train, weight_valid],
        eval_metric=custom_eval,
    )
    eval_result = model.evals_result_
    train_ds = (X_train, y_train, weight_train)
    valid_ds = (X_valid, y_valid, weight_valid)
    for key, (X, y_true, weight) in zip(['train', 'valid'], [train_ds, valid_ds]):
        np.testing.assert_allclose(
            eval_result[key]['multi_logloss'], eval_result[key]['custom_logloss']
        )
        y_pred = model.predict_proba(X)
        _, metric_value, _ = custom_eval(y_true, y_pred, weight)
        np.testing.assert_allclose(metric_value, eval_result[key]['custom_logloss'][-1])


def test_negative_n_jobs(tmp_path):
    n_threads = joblib.cpu_count()
    if n_threads <= 1:
        return None
    # 'val_minus_two' here is the expected number of threads for n_jobs=-2
    val_minus_two = n_threads - 1
    X, y = load_breast_cancer(return_X_y=True)
    # Note: according to joblib's formula, a value of n_jobs=-2 means
    # "use all but one thread" (formula: n_cpus + 1 + n_jobs)
    gbm = lgb.LGBMClassifier(n_estimators=2, verbose=-1, n_jobs=-2).fit(X, y)
    gbm.booster_.save_model(tmp_path / "model.txt")
    with open(tmp_path / "model.txt", "r") as f:
        model_txt = f.read()
    assert bool(re.search(rf"\[num_threads: {val_minus_two}\]", model_txt))


def test_default_n_jobs(tmp_path):
    n_cores = joblib.cpu_count(only_physical_cores=True)
    X, y = load_breast_cancer(return_X_y=True)
    gbm = lgb.LGBMClassifier(n_estimators=2, verbose=-1, n_jobs=None).fit(X, y)
    gbm.booster_.save_model(tmp_path / "model.txt")
    with open(tmp_path / "model.txt", "r") as f:
        model_txt = f.read()
    assert bool(re.search(rf"\[num_threads: {n_cores}\]", model_txt))


@pytest.mark.skipif(not PANDAS_INSTALLED, reason='pandas is not installed')
@pytest.mark.parametrize('task', ['classification', 'ranking', 'regression'])
def test_validate_features(task):
    X, y, g = _create_data(task)
    features = ['x1', 'x2', 'x3', 'x4']
    df = pd_DataFrame(X, columns=features)
    model = task_to_model_factory[task](n_estimators=10, num_leaves=15, verbose=-1)
    if task == 'ranking':
        model.fit(df, y, group=g)
    else:
        model.fit(df, y)
    assert model.feature_name_ == features

    # try to predict with a different feature
    df2 = df.rename(columns={'x2': 'z'})
    with pytest.raises(lgb.basic.LightGBMError, match="Expected 'x2' at position 1 but found 'z'"):
        model.predict(df2, validate_features=True)

    # check that disabling the check doesn't raise the error
    model.predict(df2, validate_features=False)
