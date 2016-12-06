import numpy as np
import random
import lightgbm as lgb


rng = np.random.RandomState(2016)

def test_binary_classification():

    from sklearn import datasets, metrics, model_selection

    X, y = datasets.make_classification(n_samples=10000, n_features=100)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMClassifier().fit(x_train, y_train)
    from sklearn.datasets import load_digits
    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMClassifier().fit(x_train, y_train)
    preds = lgb_model.predict(x_test)
    err = sum(1 for i in range(len(preds))
          if int(preds[i] > 0.5) != y_test[i]) / float(len(preds))
    assert err < 0.1

def test_multiclass_classification():
    from sklearn.datasets import load_iris
    from sklearn import datasets, metrics, model_selection

    def check_pred(preds, labels):
        err = sum(1 for i in range(len(preds))
                  if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
        assert err < 0.7


    X, y = datasets.make_classification(n_samples=10000, n_features=100, n_classes=4, n_informative=3)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

    lgb_model = lgb.LGBMClassifier().fit(x_train, y_train)
    preds = lgb_model.predict(x_test)

    check_pred(preds, y_test)

def test_regression():
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import KFold
    from sklearn import datasets, metrics, model_selection

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMRegressor().fit(x_train, y_train)
    preds = lgb_model.predict(x_test)
    assert mean_squared_error(preds, y_test) < 100

def test_regression_with_custom_objective():
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import KFold
    from sklearn import datasets, metrics, model_selection
    def objective_ls(y_true, y_pred):
        grad = (y_pred - y_true)
        hess = np.ones(len(y_true))
        return grad, hess
    boston = load_boston()
    y = boston['target']
    X = boston['data']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMRegressor(objective=objective_ls).fit(x_train, y_train)
    preds = lgb_model.predict(x_test)
    assert mean_squared_error(preds, y_test) < 100


def test_binary_classification_with_custom_objective():

    from sklearn import datasets, metrics, model_selection
    def logregobj(y_true, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        grad = y_pred - y_true
        hess = y_pred * (1.0 - y_pred)
        return grad, hess
    X, y = datasets.make_classification(n_samples=10000, n_features=100)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMClassifier(objective=logregobj).fit(x_train, y_train)
    from sklearn.datasets import load_digits
    digits = load_digits(2)
    y = digits['target']
    X = digits['data']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    lgb_model = lgb.LGBMClassifier(objective=logregobj).fit(x_train, y_train)
    preds = lgb_model.predict(x_test)
    err = sum(1 for i in range(len(preds))
          if int(preds[i] > 0.5) != y_test[i]) / float(len(preds))
    assert err < 0.1

def test_early_stopping():
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import KFold
    from sklearn import datasets, metrics, model_selection
    from sklearn.base import clone

    boston = load_boston()
    y = boston['target']
    X = boston['data']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
    lgb_model = lgb.LGBMRegressor(n_estimators=500) \
            .fit(x_train, y_train, eval_set=[(x_test, y_test)], 
                eval_metric='l2', 
                early_stopping_rounds=10,
                verbose=10)
    lgb_model_clone = clone(lgb_model)
    print(lgb_model.best_iteration)

test_binary_classification()
test_multiclass_classification()
test_regression()
test_regression_with_custom_objective()
test_binary_classification_with_custom_objective()
test_early_stopping()