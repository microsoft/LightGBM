# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# load or create your dataset
df_train = pd.read_csv('../../examples/regression/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../../examples/regression/regression.test', header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,free_raw_data=False)


# specify your configurations as a dict
params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'regression',
    'metric' : {'l2', 'auc'},
    'num_leaves' : 31,
    'learning_rate' : 0.05,
    'feature_fraction' : 0.9,
    'bagging_fraction' : 0.8,
    'bagging_freq': 5,
    'verbose' : 0
}

# train
init_gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5,
                valid_sets=lgb_eval)

print('Start continue train')

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=10,
                init_model=init_gbm)


# save model to file
gbm.save_model('model.txt')

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# dump model to json (and save to file)
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

# feature importances
print('Feature importances:', gbm.feature_importance())
print('Feature importances:', gbm.feature_importance("gain"))

print('Start test cv')

lgb.cv(params,
        lgb_train,
        num_boost_round=100,
        nfold=5,
        verbose_eval=5,
        init_model=init_gbm)
