# Quick Start

This is a quick start guide for LightGBM of cli version.

Follow the [Installation Guide](./Installation-Guide.md) to install LightGBM first.

***List of other Helpful Links***
* [Parameters](./Parameters.md)
* [Parameters Tuning](./Parameters-tuning.md)
* [Python Package quick start guide](./Python-intro.md)
* [Python API Reference](./Python-API.md)

## Training data format 

LightGBM supports input data file with [CSV](https://en.wikipedia.org/wiki/Comma-separated_values), [TSV](https://en.wikipedia.org/wiki/Tab-separated_values) and [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) formats.

Label is the data of first column, and there is no header in the file.

### Categorical feature support

update 12/5/2016:

LightGBM can use categorical feature directly (without one-hot coding). The experiment on [Expo data](http://stat-computing.org/dataexpo/2009/) shows about 8x speed-up compared with one-hot coding.

For the setting details, please refer to [Parameters](./Parameters.md#io-parameters).

### Weight and query/group data
LightGBM also support weighted training, it needs an additional [weight data](./Parameters.md#weight-data). And it needs an additional [query data](./Parameters.md#query-data) for ranking task.

update 11/3/2016:

1. support input with header now
2. can specific label column, weight column and query/group id column. Both index and column are supported
3. can specific a list of ignored columns

For the detailed usage, please refer to [Configuration](./Parameters.md#io-parameters).

## Parameter quick look

The parameter format is ```key1=value1 key2=value2 ... ``` . And parameters can be in both config file and command line.

Some important parameters:

* ```config```, default=```""```, type=string, alias=```config_file```
  * path of config file
* ```task```, default=```train```, type=enum, options=```train```,```prediction```
  * ```train``` for training
  * ```prediction``` for prediction.
* ```application```, default=```regression```, type=enum, options=```regression```,```binary```,```lambdarank```,```multiclass```, alias=```objective```,```app```
  * ```regression```, regression application
  * ```binary```, binary classification application 
  * ```lambdarank```, lambdarank application
  * ```multiclass```, multi-class classification application, should set ```num_class``` as well
* `boosting`, default=`gbdt`, type=enum, options=`gbdt`,`rf`,`dart`,`goss`, alias=`boost`,`boosting_type`
  * `gbdt`, traditional Gradient Boosting Decision Tree 
  * `rf`, Random Forest
  * `dart`, [Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
  * `goss`, Gradient-based One-Side Sampling
* ```data```, default=```""```, type=string, alias=```train```,```train_data```
  * training data, LightGBM will train from this data
* ```valid```, default=```""```, type=multi-string, alias=```test```,```valid_data```,```test_data```
  * validation/test data, LightGBM will output metrics for these data
  * support multi validation data, separate by ```,```
* ```num_iterations```, default=```100```, type=int, alias=```num_iteration```,```num_tree```,```num_trees```,```num_round```,```num_rounds```
  * number of boosting iterations/trees
* ```learning_rate```, default=```0.1```, type=double, alias=```shrinkage_rate```
  * shrinkage rate
* ```num_leaves```, default=```31```, type=int, alias=```num_leaf```
  * number of leaves in one tree
* ```tree_learner```, default=```serial```, type=enum, options=```serial```,```feature```,```data```
  * ```serial```, single machine tree learner
  * ```feature```, feature parallel tree learner
  * ```data```, data parallel tree learner
  * Refer to [Parallel Learning Guide](./Parallel-Learning-Guide.md) to get more details.
* ```num_threads```, default=OpenMP_default, type=int, alias=```num_thread```,```nthread```
  * Number of threads for LightGBM. 
  * For the best speed, set this to the number of **real CPU cores**, not the number of threads (most CPU using [hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading) to generate 2 threads per CPU core).
  * For parallel learning, should not use full CPU cores since this will cause poor performance for the network.
* ```max_depth```, default=```-1```, type=int
  * Limit the max depth for tree model. This is used to deal with overfit when #data is small. Tree still grow by leaf-wise. 
  * ```< 0``` means no limit 
* ```min_data_in_leaf```, default=```20```, type=int, alias=```min_data_per_leaf``` , ```min_data```
  * Minimal number of data in one leaf. Can use this to deal with over-fit.
* ```min_sum_hessian_in_leaf```, default=```1e-3```, type=double, alias=```min_sum_hessian_per_leaf```, ```min_sum_hessian```, ```min_hessian```
  * Minimal sum hessian in one leaf. Like ```min_data_in_leaf```, can use this to deal with over-fit.

For all parameters, please refer to [Parameters](./Parameters.md).


## Run LightGBM

For Windows:
```
lightgbm.exe config=your_config_file other_args ...
```

For unix:
```
./lightgbm config=your_config_file other_args ...
```

Parameters can be both in the config file and command line, and the parameters in command line have higher priority than in config file.
For example, following command line will keep 'num_trees=10' and ignore same parameter in config file.
```
./lightgbm config=train.conf num_trees=10
```

## Examples

* [Binary Classification](../examples/binary_classification)
* [Regression](../examples/regression)
* [Lambdarank](../examples/lambdarank)
* [Parallel Learning](../examples/parallel_learning)
