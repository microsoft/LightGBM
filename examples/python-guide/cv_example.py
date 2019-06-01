#!/usr/bin/env python
# coding: utf-8

# # Binary Classification

# In[1]:


import sys
sys.path = ["/Users/kenichi.matsui/Documents/996_github/LightGBM/python-package/"] + sys.path


# In[2]:


import lightgbm as lgb
import numpy as np
import numpy.random as rd
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

from IPython.display import display


# ## Load data

# In[3]:


rd.seed(123)
print('Loading data...')
# load or create your dataset
df_train = pd.read_csv('../binary_classification/binary.train', header=None, sep='\t')
df_test = pd.read_csv('../binary_classification/binary.test', header=None, sep='\t')

print("df_train.shape:{}, df_test.shape:{}".format(df_train.shape, df_test.shape))
display(df_train.head())

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)


# In[5]:


# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    #'metric': ['binary_logloss','auc', ],
    #'metric': ['auc', 'binary_logloss',],
    #'metric_types': ['auc', 'binary_logloss',],
    'metrics': ['auc', 'binary_logloss',],
    'num_leaves': 8,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

callbacks = [
    lgb.early_stopping(stopping_rounds=100, first_metric_only=True, verbose=True),
]

# Training settings
FOLD_NUM = 5
fold_seed = 71
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)

# Fitting
ret = lgb.cv(params=params,
               train_set=lgb_train,
               folds=folds,
               num_boost_round=5000,
               # early_stopping_rounds=10,
               verbose_eval = 500,
               callbacks=callbacks,

               eval_train_metric=True,
               )

# ret, model = lgb.cv(params=params,
#                train_set=lgb_train,
#                folds=folds,
#                num_boost_round=5000,
#                # early_stopping_rounds=10,
#                verbose_eval = 500,
#                #callbacks=callbacks,
#                eval_train_metric=False,
#                # return_boosters=True
#                )

#print(f"best_iteration : {model.best_iteration}")
df_ret = pd.DataFrame(ret)
df_ret.tail()


# In[6]:


oof = model.get_oof_prediction()


# In[8]:


print(f"roc_auc_score on oof preds: {roc_auc_score(lgb_train.label, oof)}")


# In[11]:


test_preds_list = model.predict(X_test, num_iteration=model.best_iteration)
test_preds_avg = np.array(test_preds_list).mean(axis=0)


# In[14]:


print(f"roc_auc_score on oof preds: {roc_auc_score(y_test, test_preds_avg)}")


# In[ ]:





# # Multi-label classification

# In[35]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[51]:


rd.seed(123)

# Loading Iris Dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split dataset for this demonstration.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                shuffle=True,
                                                random_state=42)

# one hot representation of y_train
max_class_num = y_train.max()+1
y_train_ohe = np.identity(max_class_num)[y_train]
    
# Create LightGBM dataset for train.
lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)

# LightGBM parameter
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric' : ["multi_logloss", "multi_error", ],
    'num_class': 3,
    'verbosity' : -1,
}
callbacks = [
    lgb.early_stopping(stopping_rounds=100, first_metric_only=False, verbose=True),
]

# Training settings
FOLD_NUM = 5
fold_seed = 71
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
eval_train_metric=True
# # Fitting
ret, model = lgb.cv(params=params,
               train_set=lgb_train,
               folds=folds,
               num_boost_round=1000,
               verbose_eval = 30,
               callbacks=callbacks, 
               eval_train_metric=eval_train_metric,
               return_boosters=True
               )

print(f"best_iteration : {model.best_iteration}")
df_ret = pd.DataFrame(ret)
df_ret.tail()


# In[18]:


oof = model.get_oof_prediction()


# In[20]:


print(f"accuracy on oof preds: {accuracy_score(lgb_train.label, np.argmax(oof, axis=1))}")


# In[29]:


test_preds_list = model.predict(X_test, num_iteration=model.best_iteration)
test_preds_avg = np.array(test_preds_list).mean(axis=0)
test_preds = np.argmax(test_preds_avg, axis=1)


# In[30]:


print(f"accuracy on oof preds: {accuracy_score(y_test, test_preds)}")


# In[ ]:





# # Regression

# In[45]:


from sklearn.metrics import mean_absolute_error


# In[60]:


df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)


# In[63]:


lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 5,
    'metric': ['l1', 'l2'],
    'verbose': -1
}

callbacks = [
    lgb.early_stopping(stopping_rounds=100, first_metric_only=False, verbose=True),
]

# Training settings
FOLD_NUM = 5
fold_seed = 71
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=fold_seed)
eval_train_metric=True
# # Fitting
ret, model = lgb.cv(params=params,
               train_set=lgb_train,
               folds=folds,
               num_boost_round=5000,
               verbose_eval = 200,
               callbacks=callbacks, 
               eval_train_metric=eval_train_metric,
               return_boosters=True
               )
print(f"best_iteration : {model.best_iteration}")

df_ret = pd.DataFrame(ret)
df_ret.tail()


# In[43]:


oof = model.get_oof_prediction()


# In[49]:


print(f"mae: {mean_absolute_error(y_train, oof):.5f}")


# In[ ]:


test_preds_list = model.predict(X_test, num_iteration=model.best_iteration)
test_preds_avg = np.array(test_preds_list).mean(axis=0)

