# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# load or create your dataset
df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=100)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
