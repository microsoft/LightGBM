'''
BLUF: The `xentropy` objective does logistic regression and generalizes
to the case where labels are probabilistic (i.e. numbers between 0 and 1).

Details: Both `binary` and `xentropy` minimize the log loss and use `boost_from_average = TRUE`
by default. Possibly the only difference between them is that `binary` may achieve a slight
speed improvement by assuming that the labels are binary instead of probabilistic.
'''

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
import time

##################
## Simulate some binary data with a single categorical and single continuous predictor
np.random.seed(0)
N = 1000
X = pd.DataFrame({
    'continuous': range(N),
    'categorical': np.repeat([0,1,2,3,4], N/5)
})
categorical_effects = [-1, -1, -2, -2, 2]
linear_term = np.array([
    -0.5 + 0.01*X['continuous'][k]
    + categorical_effects[X['categorical'][k]] for k in range(X.shape[0])
]) + np.random.normal(0, 1, X.shape[0])
true_prob = expit(linear_term)
y = np.random.binomial(1, true_prob, size = N)
data_with_binary_labels = lgb.Dataset(X, y)
data_with_probability_labels = lgb.Dataset(X, true_prob)

##################
## Set up a couple of utilities for our experiments
def log_loss(preds, labels):
    return -np.sum(labels*np.log(preds))/len(preds)

def make_params(objective):
    return {
        'objective': objective,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'verbose': -1
    }

def experiment(objective, data, X, labels):
    np.random.seed(0)
    nrounds = 1
    t0 = time.time()
    gbm = lgb.train(make_params(objective), data, num_boost_round=nrounds)
    y_fitted = gbm.predict(X)
    t = time.time() - t0
    return {
        'time': t,
        'correlation': np.corrcoef(y_fitted, labels)[0,1],
        'logloss': log_loss(y_fitted, true_prob)
    }

##################
## Observe the behavior of `binary` and `xentropy` objectives
# With binary labels
print(experiment('binary',   data_with_binary_labels, X, y))
print(experiment('xentropy', data_with_binary_labels, X, y))
# With probabilistic labels:
print(experiment('xentropy', data_with_probability_labels, X, true_prob))
# Trying this throws an error on non-binary values of y:
#   experiment('binary', data_with_probability_labels, X)

# The speed of `binary` is not drastically different than `xentropy`. `xentropy`
#   runs faster than `binary` in many cases, although there are reasons to suspect
#   that `binary` should run faster when the label is an integer instead of a float
bt = min([experiment('binary',   data_with_binary_labels, X, y)['time']
        for k in range(10)])
print('Best `binary` time: ' + str(bt))
xt = min([experiment('xentropy', data_with_binary_labels, X, y)['time']
        for k in range(10)])
print('Best `xentropy` time: ' + str(xt))
