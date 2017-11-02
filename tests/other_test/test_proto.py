# coding: utf-8
# pylint: skip-file
import lightgbm as lgb
import numpy as np
import pandas as pd

np.random.seed(42)

filename = 'gbm.proto'

X_train = pd.DataFrame({
    0: np.random.randint(24, 42, 70),
    1: np.random.rand(70)
})
X_train[0] = X_train[0].astype('category')
y_train = np.random.rand(70)

data = lgb.Dataset(X_train, y_train)

gbm = lgb.train({}, data)

X_test = pd.DataFrame({
    0: np.random.randint(24, 42, 30),
    1: np.random.rand(30)
})
X_test[0] = X_test[0].astype('category')

y_pred = gbm.predict(X_test)

gbm.save_proto(filename)

gbm2 = lgb.Booster(model_file=filename, model_format='proto')

y_pred2 = gbm2.predict(X_test)

np.testing.assert_array_almost_equal(gbm.pandas_categorical, gbm2.pandas_categorical)
np.testing.assert_array_equal(y_pred, y_pred2)
