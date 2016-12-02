# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
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

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# or you can simply use a tuple of length=2 here
lgb_train = (X_train, y_train)
lgb_eval = (X_test, y_test)

# specify your configurations as a dict
params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'regression',
    'metric' : 'l2',
    'num_leaves' : 31,
    'learning_rate' : 0.05,
    'feature_fraction' : 0.9,
    'bagging_fraction' : 0.8,
    'bagging_freq': 5,
    # 'ndcg_eval_at' : [1, 3, 5, 10],
    # this metric is not needed in this task, show as an example
    'verbose' : 0
}

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_datas=lgb_eval,
                # you can use a list to represent multiple valid_datas/valid_names
                # don't use tuple, tuple is used to represent one dataset
                early_stopping_rounds=10)

# save model to file
gbm.save_model('model.txt')

# load model from file
gbm = lgb.Booster(model_file='model.txt')

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# dump model to json (and save to file)
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

