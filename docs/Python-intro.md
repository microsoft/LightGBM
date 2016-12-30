Python Package Introduction
===========================
This document gives a basic walkthrough of LightGBM python package.

***List of other Helpful Links***
* [Python Examples](../examples/python-guide/)
* [Python API Reference](./Python-API.md)
* [Parameters Tuning](./Parameters-tuning.md)

Install
-------
* Install the library first, follow the wiki [here](./Installation-Guide.md).
* In the  `python-package` directory, run
```
python setup.py install
```

* To verify your installation, try to `import lightgbm` in Python.
```
import lightgbm as lgb
```

Data Interface
--------------
The LightGBM python module is able to load data from:
- libsvm/tsv/csv txt format file
- Numpy 2D array, pandas object
- LightGBM binary file

The data is stored in a ```Dataset``` object.

#### To load a libsvm text file or a LightGBM binary file into ```Dataset```:
```python
train_data = lgb.Dataset('train.svm.bin')
```

####  To load a numpy array into ```Dataset```:
```python
data = np.random.rand(500,10) # 500 entities, each contains 10 features
label = np.random.randint(2, size=500) # binary target
train_data = lgb.Dataset( data, label=label)
```
#### To load a scpiy.sparse.csr_matrix array into ```Dataset```:
```python
csr = scipy.sparse.csr_matrix((dat, (row, col)))
train_data = lgb.Dataset(csr)
```
#### Saving ```Dataset``` into a LightGBM binary file will make loading faster:
```python
train_data = lgb.Dataset('train.svm.txt')
train_data.save_binary("train.bin")
```
#### Create validation data
```python
test_data = train_data.create_valid('test.svm')
```

or 

```python
test_data = lgb.Dataset('test.svm', reference=train_data)
```

In LightGBM, the validation data should be aligned with training data.

#### Specific feature names and categorical features

```python
train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
```
LightGBM can use categorical features as input directly. It doesn't need to covert to one-hot coding, and is much faster than one-hot coding (about 8x speed-up). 
**Note:You should convert your categorical features to int type before you construct `Dataset`.**

#### Weights can be set when needed:
```python
w = np.random.rand(500, 1)
train_data = lgb.Dataset(data, label=label, weight=w)
```
or
```python
train_data = lgb.Dataset(data, label=label)
w = np.random.rand(500, 1)
train_data.set_weight(w)
```

And you can use `Dataset.set_init_score()` to set initial score, and `Dataset.set_group()` to set group/query data for ranking tasks.

#### Memory efficent usage

The `Dataset` object in LightGBM is very memory-efficient, due to it only need to save discrete bins.
However, Numpy/Array/Pandas object is memory cost. If you concern about your memory consumption. You can save memory accroding to following:

1. Let ```free_raw_data=True```(default is ```True```) when constructing the ```Dataset```
2. Explicit set ```raw_data=None``` after the ```Dataset``` has been constructed
3. Call ```gc```  

Setting Parameters
------------------
LightGBM can use either a list of pairs or a dictionary to set [parameters](./Parameters.md). For instance:
* Booster parameters
```python
param = {'num_leaves':31, 'num_trees':100, 'objective':'binary' }
param['metric'] = 'auc'
```
* You can also specify multiple eval metrics:
```python
param['metric'] = ['auc', 'binary_logloss']

```

Training
--------

Training a model requires a parameter list and data set.
```python
num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data] )
```
After training, the model can be saved.
```python
bst.save_model('model.txt')
```
The trained model can also be dumped to JSON format
```python
# dump model
json_model = bst.dump_model()
```
A saved model can be loaded as follows:
```python
bst = lgb.Booster(model_file="model.txt") #init model
```

CV
--
Training with 5-fold CV:
```python
num_round = 10
lgb.cv(param, train_data, num_round, nfold=5)
```

Early Stopping
--------------
If you have a validation set, you can use early stopping to find the optimal number of boosting rounds.
Early stopping requires at least one set in `valid_sets`. If there's more than one, it will use all of them.

```python
bst = train(param, train_data, num_round, valid_sets=valid_sets, early_stopping_rounds=10)
bst.save_model('model.txt', num_iteration=bst.best_iteration)
```

The model will train until the validation score stops improving. Validation error needs to improve at least every `early_stopping_rounds` to continue training.

If early stopping occurs, the model will have an additional field: `bst.best_iteration`. Note that `train()` will return a model from the last iteration, not the best one. And you can set `num_iteration=bst.best_iteration` when saving model.

This works with both metrics to minimize (L2, log loss, etc.) and to maximize (NDCG, AUC). Note that if you specify more than one evaluation metric, all of them will be used for early stopping.

Prediction
----------
A model that has been trained or loaded can perform predictions on data sets.
```python
# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
ypred = bst.predict(data)
```

If early stopping is enabled during training, you can get predictions from the best iteration with `bst.best_iteration`:
```python
ypred = bst.predict(data,num_iteration=bst.best_iteration)
```
