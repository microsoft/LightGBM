This is a page contains all parameters in LightGBM.

***List of other Helpful Links***
* [Python API Reference](./Python-API.md)
* [Parameters Tuning](./Parameters-tuning.md)


## Parameter format

The parameter format is ```key1=value1 key2=value2 ... ``` . And parameters can be set both in config file and command line. By using command line, parameters should not have spaces before and after ```=```. By using config files, one line can only contain one parameter. you can use ```#``` to comment. If one parameter appears in both command line and config file, LightGBM will use the parameter in command line.

## Core Parameters

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
* ```boosting```, default=```gbdt```, type=enum, options=```gbdt```,```dart```, alias=```boost```,```boosting_type```
  * ```gbdt```, traditional Gradient Boosting Decision Tree 
  * ```dart```, [Dropouts meet Multiple Additive Regression Trees](https://arxiv.org/abs/1505.01866)
* ```data```, default=```""```, type=string, alias=```train```,```train_data```
  * training data, LightGBM will train from this data
* ```valid```, default=```""```, type=multi-string, alias=```test```,```valid_data```,```test_data```
  * validation/test data, LightGBM will output metrics for these data
  * support multi validation data, separate by ```,```
* ```num_iterations```, default=```10```, type=int, alias=```num_iteration```,```num_tree```,```num_trees```,```num_round```,```num_rounds```
  * number of boosting iterations/trees
* ```learning_rate```, default=```0.1```, type=double, alias=```shrinkage_rate```
  * shrinkage rate
  * in ```dart```, it also affects normalization weights of dropped trees
* ```num_leaves```, default=```127```, type=int, alias=```num_leaf```
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

## Learning control parameters
* ```max_depth```, default=```-1```, type=int
  * Limit the max depth for tree model. This is used to deal with overfit when #data is small. Tree still grow by leaf-wise. 
  * ```< 0``` means no limit 
* ```min_data_in_leaf```, default=```100```, type=int, alias=```min_data_per_leaf``` , ```min_data```
  * Minimal number of data in one leaf. Can use this to deal with over-fit.
* ```min_sum_hessian_in_leaf```, default=```10.0```, type=double, alias=```min_sum_hessian_per_leaf```, ```min_sum_hessian```, ```min_hessian```
  * Minimal sum hessian in one leaf. Like ```min_data_in_leaf```, can use this to deal with over-fit.
* ```feature_fraction```, default=```1.0```, type=double, ```0.0 < feature_fraction < 1.0```, alias=```sub_feature```
  * LightGBM will random select part of features on each iteration if ```feature_fraction``` smaller than ```1.0```. For example, if set to ```0.8```, will select 80% features before training each tree.
  * Can use this to speed up training
  * Can use this to deal with over-fit
* ```feature_fraction_seed```, default=```2```, type=int
  * Random seed for feature fraction.
* ```bagging_fraction```, default=```1.0```, type=double, , ```0.0 < bagging_fraction < 1.0```, alias=```sub_row```
  * Like ```feature_fraction```, but this will random select part of data
  * Can use this to speed up training
  * Can use this to deal with over-fit
  * Note: To enable bagging, should set ```bagging_freq``` to a non zero value as well
* ```bagging_freq```, default=```0```, type=int
  * Frequency for bagging, ```0``` means disable bagging. ```k``` means will perform bagging at every ```k``` iteration.
  * Note: To enable bagging, should set ```bagging_fraction``` as well
* ```bagging_seed``` , default=```3```, type=int
  * Random seed for bagging.
* ```early_stopping_round``` , default=```0```, type=int, alias=```early_stopping_rounds```,```early_stopping```
  * Will stop training if one metric of one validation data doesn't improve in last ```early_stopping_round``` rounds.
* ```lambda_l1``` , default=```0```, type=double
  * l1 regularization 
* ```lambda_l2``` , default=```0```, type=double
  * l2 regularization 
* ```min_gain_to_split``` , default=```0```, type=double
  * The minimal gain to perform split 
* ```drop_rate```, default=```0.1```, type=double
  * only used in ```dart```
* ```skip_drop```, default=```0.5```, type=double
  * only used in ```dart```, probability of skipping drop
* ```max_drop```, default=```50```, type=int
  * only used in ```dart```, max number of dropped trees on one iteration. ```<=0``` means no limit.
* ```uniform_drop```, default=```false```, type=bool
  * only used in ```dart```, true if want to use uniform drop
* ```xgboost_dart_mode```, default=```false```, type=bool
  * only used in ```dart```, true if want to use xgboost dart mode
* ```drop_seed```, default=```4```, type=int
  * only used in ```dart```, used to random seed to choose dropping models.


## IO parameters

* ```max_bin```, default=```255```, type=int
  * max number of bin that feature values will bucket in. Small bin may reduce training accuracy but may increase general power (deal with over-fit).
  * LightGBM will auto compress memory according ```max_bin```. For example, LightGBM will use ```uint8_t``` for feature value if ```max_bin=255```.
* ```data_random_seed```, default=```1```, type=int
  * random seed for data partition in parallel learning(not include feature parallel).
* ```output_model```, default=```LightGBM_model.txt```, type=string, alias=```model_output```,```model_out```
  * file name of output model in training.
* ```input_model```, default=```""```, type=string, alias=```model_input```,```model_in```
  * file name of input model.
  * for prediction task, will prediction data using this model.
  * for train task, will continued train from this model.
* ```output_result```, default=```LightGBM_predict_result.txt```, type=string, alias=```predict_result```,```prediction_result```
  * file name of prediction result in prediction task.
* ```is_pre_partition```, default=```false```, type=bool
  * used for parallel learning(not include feature parallel).
  * ```true``` if training data are pre-partitioned, and different machines using different partition.
* ```is_sparse```, default=```true```, type=bool, alias=```is_enable_sparse```
  * used to enable/disable sparse optimization. Set to ```false``` to disable sparse optimization.
* ```two_round```, default=```false```, type=bool, alias=```two_round_loading```,```use_two_round_loading```
  * by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed. But it may out of memory when the data file is very big.
  * set this to ```true``` if data file is too big to fit in memory.
* ```save_binary```, default=```false```, type=bool, alias=```is_save_binary```,```is_save_binary_file```
  * set this to ```true``` will save the data set(include validation data) to a binary file. Speed up the data loading speed for the next time.
* ```verbosity```, default=```1```, type=int, alias=```verbose```
  * ```<0``` = Fatel, ```=0``` = Error(Warn), ```>0``` = Info
* ```header```, default=```false```, type=bool, alias=```has_header```
  * ```true``` if input data has header
* ```label```, default=```""```, type=string, alias=```label_column```
  * specific the label column
  * Use number for index, e.g. ```label=0``` means column_0 is the label
  * Add a prefix ```name:``` for column name, e.g. ```label=name:is_click```
* ```weight```, default=```""```, type=string, alias=```weight_column```
  * specific the weight column
  * Use number for index, e.g. ```weight=0``` means column_0 is the weight
  * Add a prefix ```name:``` for column name, e.g. ```weight=name:weight```
  * Note: Index start from ```0```. And it doesn't count the label column when passing type is Index. e.g. when label is  column_0, and weight is column_1, the correct parameter is ```weight=0```.
* ```query```, default=```""```, type=string, alias=```query_column```,```group```,```group_column```
  * specific the query/group id column
  * Use number for index, e.g. ```query=0``` means column_0 is the query id
  * Add a prefix ```name:``` for column name, e.g. ```query=name:query_id```
  * Note: Data should group by query_id. Index start from ```0```. And it doesn't count the label column when passing type is Index. e.g. when label is  column_0, and query_id is column_1, the correct parameter is ```query=0```.
* ```ignore_column```, default=```""```, type=string, alias=```ignore_feature```,```blacklist```
  * specific some ignore columns in training
  * Use number for index, e.g. ```ignore_column=0,1,2``` means column_0, column_1 and column_2 will be ignored.
  * Add a prefix ```name:``` for column name, e.g. ```ignore_column=name:c1,c2,c3``` means c1, c2 and c3 will be ignored.
  * Note: Index start from ```0```. And it doesn't count the label column.
* ```categorical_feature```, default=```""```, type=string, alias=```categorical_column```,```cat_feature```,```cat_column```
  * specific categorical features
  * Use number for index, e.g. ```categorical_feature=0,1,2``` means column_0, column_1 and column_2 are categorical features.
  * Add a prefix ```name:``` for column name, e.g. ```categorical_feature=name:c1,c2,c3``` means c1, c2 and c3 are categorical features.
  * Note: Only support categorical with ```int``` type. Index start from ```0```. And it doesn't count the label column.
* ```predict_raw_score```, default=```false```, type=bool, alias=```raw_score```,```is_predict_raw_score```
  * only used in prediction task
  * Set to ```true``` will only predict the raw scores.
  * Set to ```false``` will transformed score
* ```predict_leaf_index ```, default=```false```, type=bool, alias=```leaf_index ```,```is_predict_leaf_index ```
  * only used in prediction task
  * Set to ```true``` to predict with leaf index of all trees
* ```bin_construct_sample_cnt```, default=```50000```, type=int
  * Number of data that sampled to construct histogram bins.
  * Will give better training result when set this larger. But will increase data loading time.
  * Set this to larger value if data is very sparse.
* ```num_iteration_predict```, default=```-1```, type=int
  * only used in prediction task, used to how many trained iterations will be used in prediction. 
  * ```<= 0``` means no limit


## Objective parameters

* ```sigmoid```, default=```1.0```, type=double
  * parameter for sigmoid function. Will be used in binary classification and lambdarank.
* ```scale_pos_weight```, default=```1.0```, type=double
  * weight of positive class in binary classification task
* ```is_unbalance```, default=```false```, type=bool
  * used in binary classification. Set this to ```true``` if training data are unbalance.
* ```max_position```, default=```20```, type=int
  * used in lambdarank, will optimize NDCG at this position.
* ```label_gain```, default=```{0,1,3,7,15,31,63,...}```, type=multi-double
  * used in lambdarank, relevant gain for labels. For example, the gain of label ```2``` is ```3``` if using default label gains.
  * Separate by ```,```
* ```num_class```, default=```1```, type=int, alias=```num_classes```
  * only used in multi-class classification

## Metric parameters

* ```metric```, default={```l2``` for regression}, {```binary_logloss``` for binary classification},{```ndcg``` for lambdarank}, type=multi-enum, options=```l1```,```l2```,```ndcg```,```auc```,```binary_logloss```,```binary_error```
  * ```l1```, absolute loss
  * ```l2```, square loss
  * ```ndcg```, [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)
  * ```auc```, [AUC](https://en.wikipedia.org/wiki/Area_under_the_curve_(pharmacokinetics))
  * ```binary_logloss```, [log loss](https://www.kaggle.com/wiki/LogarithmicLoss)
  * ```binary_error```. For one sample ```0``` for correct classification, ```1``` for error classification.
  * ```multi_logloss```, log loss for mulit-class classification
  * ```multi_error```. error rate for mulit-class classification
  * Support multi metrics, separate by ```,```
* ```metric_freq```, default=```1```, type=int
  * frequency for metric output
* ```is_training_metric```, default=```false```, type=bool
  * set this to true if need to output metric result of training
* ```ndcg_at```, default=```{1,2,3,4,5}```, type=multi-int, alias=```ndcg_eval_at```
  * NDCG evaluation position, separate by ```,```

## Network parameters

Following parameters are used for parallel learning, and only used for base(socket) version. It doesn't need to set them for MPI version. 

* ```num_machines```, default=```1```, type=int, alias=```num_machine```
  * Used for parallel learning, the number of machines for parallel learning application
* ```local_listen_port```, default=```12400```, type=int, alias=```local_port```
  * TCP listen port for local machines.
  * Should allow this port in firewall setting before training.
* ```time_out```, default=```120```, type=int
  * Socket time-out in minutes.
* ```machine_list_file```, default=```""```, type=string
  * File that list machines for this parallel learning application
  * Each line contains one IP and one port for one machine. The format is ```ip port```, separate by space.

## Others

### Continued training with input score
LightGBM support continued train with initial score. It uses an additional file to store these initial score, like the following:

```
0.5
-0.1
0.9
...
```

It means the initial score of first data is ```0.5```, second is ```-0.1```, and so on. The initial score file corresponds with data file line by line, and has per score per line. And if the name of data file is "train.txt", the initial score file should be named as "train.txt.init" and in the same folder as the data file. And LightGBM will auto load initial score file if it exists. 


### Weight data
LightGBM support weighted training. It uses an additional file to store weight data, like the following:

```
1.0
0.5
0.8
...
```

It means the weight of first data is ```1.0```, second is ```0.5```, and so on. The weight file corresponds with data file line by line, and has per weight per line. And if the name of data file is "train.txt", the weight file should be named as "train.txt.weight" and in the same folder as the data file. And LightGBM will auto load weight file if it exists.

update:
You can specific weight column in data file now. Please refer to parameter ```weight``` in above.

### Query data

For LambdaRank learning, it needs query information for training data. LightGBM use an additional file to store query data. Following is an example:

```
27
18
67
...
```

It means first ```27``` lines samples belong one query and next ```18``` lines belong to another, and so on.(**Note: data should order by query**) If name of data file is "train.txt", the query file should be named as "train.txt.query" and in same folder of training data. LightGBM will load the query file automatically if it exists.

update:
You can specific query/group id in data file now. Please refer to parameter ```group``` in above.