#ifndef LIGHTGBM_CONFIG_H_
#define LIGHTGBM_CONFIG_H_

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <LightGBM/meta.h>
#include <LightGBM/export.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>

namespace LightGBM {

/*! \brief Types of tasks */
enum TaskType {
  kTrain, kPredict, kConvertModel, KRefitTree
};
const int kDefaultNumLeaves = 31;

struct Config {
public:
  void ToString() const;
  /*!
  * \brief Get string value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetString(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, std::string* out);

  /*!
  * \brief Get int value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetInt(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, int* out);

  /*!
  * \brief Get double value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetDouble(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, double* out);

  /*!
  * \brief Get bool value by specific name of key
  * \param params Store the key and value for params
  * \param name Name of key
  * \param out Value will assign to out if key exists
  * \return True if key exists
  */
  inline static bool GetBool(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& name, bool* out);

  static void KV2Map(std::unordered_map<std::string, std::string>& params, const char* kv);
  static std::unordered_map<std::string, std::string> Str2Map(const char* parameters);

  #pragma region Parameters
  #pragma region Core Parameters

  // [Doc-Only]
  // alias=config_file
  // desc=path of config file
  // desc=**Note**: Only can be used in CLI version
  std::string config = "";

  // [Doc-Only]
  // type=enum
  // default=train
  // options=train,predict,convert_model,refit
  // alias=task_type
  // desc=``train``, alias=\ ``training``, for training
  // desc=``predict``, alias=\ ``prediction``, ``test``, for prediction
  // desc=``convert_model``, for converting model file into if-else format, see more information in `Convert model parameters <#convert-model-parameters>`__
  // desc=``refit``, alias = \ ``refit_tree``, refit existing models with new data
  // desc=**Note**: Only can be used in CLI version
  TaskType task = TaskType::kTrain;

  // [DOC-Only]
  // type=enum
  // options=regression,regression_l1,huber,fair,poisson,quantile,mape,gammma,tweedie,binary,multiclass,multiclassova,xentropy,xentlambda,lambdarank
  // alias=application,app,objective_type
  // desc=regression application
  // descl2=``regression_l2``, L2 loss, alias=\ ``regression``, ``mean_squared_error``, ``mse``, ``l2_root``, ``root_mean_squared_error``, ``rmse``
  // descl2=``regression_l1``, L1 loss, alias=\ ``mean_absolute_error``, ``mae``
  // descl2=``huber``, `Huber loss`_
  // descl2=``fair``, `Fair loss`_
  // descl2=``poisson``, `Poisson regression`_
  // descl2=``quantile``, `Quantile regression`_
  // descl2=``mape``, `MAPE loss`_, alias=\ ``mean_absolute_percentage_error``
  // descl2=``gamma``, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity, or for any target that might be `gamma-distributed`_
  // descl2=``tweedie``, Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any target that might be `tweedie-distributed`_
  // desc=``binary``, binary `log loss`_ classification application
  // desc=multi-class classification application
  // descl2=``multiclass``, `softmax`_ objective function, alias=\ ``softmax``
  // descl2=``multiclassova``, `One-vs-All`_ binary objective function, alias=\ ``multiclass_ova``, ``ova``, ``ovr``
  // descl2=``num_class`` should be set as well
  // desc=cross-entropy application
  // descl2=``xentropy``, objective function for cross-entropy (with optional linear weights), alias=\ ``cross_entropy``
  // descl2=``xentlambda``, alternative parameterization of cross-entropy, alias=\ ``cross_entropy_lambda``
  // descl2=the label is anything in interval [0, 1]
  // desc=``lambdarank``, `lambdarank`_ application
  // descl2=the label should be ``int`` type in lambdarank tasks, and larger number represent the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)
  // descl2=`label_gain <#objective-parameters>`__ can be used to set the gain(weight) of ``int`` label
  // descl2=all values in ``label`` must be smaller than number of elements in ``label_gain``
  std::string objective = "regression";


  // [DOC-Only]
  // type=enum
  // alias=boosting_type,boost
  // options=gbdt,rf,dart,goss
  // desc=``gbdt``, traditional Gradient Boosting Decision Tree
  // desc=``rf``, Random Forest
  // desc=``dart``, `Dropouts meet Multiple Additive Regression Trees`_
  // desc=``goss``, Gradient - based One - Side Sampling
  std::string boosting = "gbdt";

  // alias=train,train_data,data_filename
  // desc=training data, LightGBM will train from this data
  std::string data = "";

  // alias=test,valid_data,test_data,valid_filenames
  // desc=validation/test data, LightGBM will output metrics for these data
  // desc=support multi validation data, separate by ``,``
  std::vector<std::string> valid;

  // alias=num_iteration,num_tree,num_trees,num_round,num_rounds,num_boost_round,n_estimators
  // check=>=0
  // desc=number of boosting iterations
  // desc=**Note**: for Python/R package,**this parameter is ignored**, use num_boost_round (Python) or nrounds (R) input arguments of train and cv methods instead
  // desc=**Note**: internally,LightGBM constructs num_class * num_iterations trees for multiclass problems
  int num_iterations = 100;

  // alias=shrinkage_rate
  // check=>0
  // desc=shrinkage rate
  // desc=in dart,it also affects on normalization weights of dropped trees
  double learning_rate = 0.1;

  // default=31
  // alias = num_leaf
  // check=>1
  // desc=max number of leaves in one tree
  int num_leaves = kDefaultNumLeaves;

  // [Doc-Only]
  // type=enum
  // options=serial, feature, data, voting
  // alias = tree
  // desc=serial,single machine tree learner
  // desc=feature,alias=feature_parallel,feature parallel tree learner
  // desc=data,alias=data_parallel,data parallel tree learner
  // desc=voting,alias=voting_parallel,voting parallel tree learner
  // desc=refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details
  std::string tree_learner = "serial";

  // default=OpenMP_default
  // alias = num_thread, nthread
  // desc = number of threads for LightGBM
  // desc=for the best speed,set this to the number of **real CPU cores**,
  // not the number of threads(most CPU using `hyper-threading`_ to generate 2 threads per CPU core)
  // desc=do not set it too large if your dataset is small (do not use 64 threads for a dataset with 10,000 rows for instance)
  // desc=be aware a task manager or any similar CPU monitoring tool might report cores not being fully utilized. **This is normal**
  // desc=for parallel learning,should not use full CPU cores since this will cause poor performance for the network
  int num_threads = 0;

  // [DOC-Only]
  // options=cpu,gpu
  // desc = choose device for the tree learning, you can use GPU to achieve the faster learning
  // desc=**Note**: it is recommended to use the smaller max_bin (e.g. 63) to get the better speed up
  // desc=**Note**: for the faster speed,GPU use 32-bit float point to sum up by default,may affect the accuracy for some tasks.
  // desc=You can set gpu_use_dp = true to enable 64 - bit float point, but it will slow down the training
  // desc=**Note**: refer to `Installation Guide <./Installation-Guide.rst#build-gpu-version>`__ to build with GPU
  std::string device_type = "cpu";

  // [DOC-Only]
  // desc=Use this seed to generate seeds for others, e.g. data_random_seed.
  // desc=Will be override if set other seeds as well
  // default=none
  int seed = 0;

  #pragma endregion

  #pragma region Learning Control Parameters

  // desc=limit the max depth for tree model. This is used to deal with over-fitting when #data is small. Tree still grows by leaf-wise
  // desc=< 0 means no limit
  int max_depth = -1;

  // alias = min_data_per_leaf, min_data, min_child_samples
  // check=>=0
  // desc=minimal number of data in one leaf. Can be used to deal with over-fitting
  int min_data_in_leaf = 20;

  // alias=min_sum_hessian_per_leaf,min_sum_hessian,min_hessian,min_child_weight
  // check >=0
  // desc=minimal sum hessian in one leaf. Like min_data_in_leaf,it can be used to deal with over-fitting
  double min_sum_hessian_in_leaf = 1e-3;

  // alias=sub_row,subsample,bagging
  // check=>0
  // check=<=1.0
  // desc = like feature_fraction, but this will randomly select part of data without resampling
  // desc=can be used to speed up training
  // desc=can be used to deal with over-fitting
  // desc=**Note**: To enable bagging,bagging_freq should be set to a non zero value as well
  double bagging_fraction = 1.0;

  // alias=subsample_freq
  // desc=frequency for bagging,0 means disable bagging. k means will perform bagging at every k iteration
  // desc=**Note**: to enable bagging,bagging_fraction should be set as well
  int bagging_freq = 0;

  // alias = bagging_fraction_seed
  // desc = random seed for bagging
  int bagging_seed = 3;


  // alias = sub_feature, colsample_bytree
  // check=>0
  // check=<=1.0
  // desc=LightGBM will randomly select part of features on each iteration if feature_fraction smaller than 1.0. For example, if set to 0.8, will select 80 % features before training each tree
  // desc=can be used to speed up training
  // desc=can be used to deal with over-fitting
  double feature_fraction = 1.0;

  // desc=random seed for feature_fraction
  int feature_fraction_seed = 2;

  // alias=early_stopping_rounds,early_stopping
  // desc=will stop training if one metric of one validation data doesn't improve in last early_stopping_round rounds
  // desc=enable when greater than 0
  int early_stopping_round = 0;

  // alias=max_tree_output,max_leaf_output
  // desc=Used to limit the max output of tree leaves
  // desc=when <= 0,there is not constraint
  // desc=the final max output of leaves is learning_rate*max_delta_step
  double max_delta_step = 0.0;

  // alias=reg_alpha
  // check=>=0
  // desc=L1 regularization
  double lambda_l1 = 0.0;

  // alias = reg_lambda
  // check=>=0
  // desc = L2 regularization
  double lambda_l2 = 0.0;

  // alias=min_gain_to_split
  // desc=the minimal gain to perform split
  double min_gain_to_split = 0.0;

  // check=>=0
  // check=<=1.0
  // desc=only used in dart
  double drop_rate = 0.1;

  // desc=only used in dart,max number of dropped trees on one iteration
  // desc=<=0 means no limit
  int max_drop = 50;

  // check=>=0
  // check=<=1.0
  // desc=only used in dart,probability of skipping drop
  double skip_drop = 0.5;

  // desc=only used in dart,set this to true if want to use xgboost dart mode
  bool xgboost_dart_mode = false;

  // desc=only used in dart,set this to true if want to use uniform drop
  bool uniform_drop = false;

  // desc=only used in dart,random seed to choose dropping models
  int drop_seed = 4;

  // check=>=0
  // check=<=1.0
  // desc=only used in goss,the retain ratio of large gradient data
  double top_rate = 0.2;

  // check=>=0
  // check=<=1.0
  // desc=only used in goss,the retain ratio of small gradient data
  double other_rate = 0.1;

  // check=>1
  // desc=min number of data per categorical group
  int min_data_per_group = 100;

  // check=>0
  // desc=use for the categorical features
  // desc=limit the max threshold points in categorical features
  int max_cat_threshold = 32;

  // check=>=0
  // desc=L2 regularization in categorcial split
  double cat_l2 = 10;

  // check=>=0
  // desc=used for the categorical features
  // desc=this can reduce the effect of noises in categorical features,especially for categories with few data
  double cat_smooth = 10;
  
  // check=>1
  // desc=when number of categories of one feature smaller than or equal to max_cat_to_onehot,one-vs-other split algorithm will be used
  int max_cat_to_onehot = 4;

  // alias = topk
  // desc=used in `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__
  // desc=set this to larger value for more accurate result,but it will slow down the training speed
  int top_k = 20;

  // type = multi-int
  // alias = mc
  // default=none
  // desc=used for constraints of monotonic features
  // desc=1 means increasing,-1 means decreasing,0 means non-constraint
  // desc=you need to specify all features in order. For example,mc=-1,0,1 means the decreasing for 1st feature,non-constraint for 2nd feature and increasing for the 3rd feature
  std::vector<int8_t> monotone_constraints;
  
  // desc = path to a.json file that specifies splits to force at the top of every decision tree before best - first learning commences
  // desc=.json file can be arbitrarily nested,and each split contains feature,threshold fields,as well as left and right fields representing subsplits.Categorical splits are forced in a one - hot fashion, with left representing the split containing the feature value and right representing other values
  // desc=see `this file <https://github.com/Microsoft/LightGBM/tree/master/examples/binary_classification/forced_splits.json>`__ as an example
  std::string forcedsplits_filename = "";

  #pragma endregion

  #pragma region IO Parameters

  // check=>1
  // desc=max number of bins that feature values will be bucketed in.
  // desc=Small number of bins may reduce training accuracy but may increase general power(deal with over - fitting)
  // desc=LightGBM will auto compress memory according max_bin.
  // desc=For example, LightGBM will use uint8_t for feature value if max_bin = 255
  int max_bin = 255;

  // check=>0
  // desc=min number of data inside one bin,use this to avoid one-data-one-bin (may over-fitting)
  int min_data_in_bin = 3;

  // desc=random seed for data partition in parallel learning (not include feature parallel)
  int data_random_seed = 1;

  // alias=model_output,model_out
  // desc=file name of output model in training
  std::string output_model = "LightGBM_model.txt";

  // alias = model_input, model_in
  // desc=file name of input model
  // desc=for prediction task,this model will be used for prediction data
  // desc=for train task,training will be continued from this model
  std::string input_model = "";

  // alias=predict_result,prediction_result
  // desc=file name of prediction result in prediction task
  std::string output_result = "LightGBM_predict_result.txt";

  // alias = is_pre_partition
  // desc=used for parallel learning (not include feature parallel)
  // desc=true if training data are pre-partitioned,and different machines use different partitions
  bool pre_partition = false;

  // alias = is_sparse, enable_sparse
  // desc = used to enable / disable sparse optimization.Set to false to disable sparse optimization
  bool is_enable_sparse = true;

  // check=>0
  // check=<=1
  // desc=the threshold of zero elements precentage for treating a feature as a sparse feature.
  double sparse_threshold = 0.8;

  // alias=two_round_loading,use_two_round_loading
  // desc = by default, LightGBM will map data file to memory and load features from memory.
  // desc = This will provide faster data loading speed.But it may run out of memory when the data file is very big
  // desc = set this to true if data file is too big to fit in memory
  bool two_round = false;

  // alias = is_save_binary, is_save_binary_file
  // desc = if true LightGBM will save the dataset(include validation data) to a binary file.
  // desc = Speed up the data loading for the next time
  bool save_binary = false;

  // alias=verbose
  // desc= <0 = Fatal, =0 = Error(Warn), >0 = Info
  int verbosity = 1;

  // alias = has_header
  // desc=set this to true if input data has header
  bool header = false;


  // alias=label
  // desc=specify the label column
  // desc=use number for index,e.g. label=0 means column\_0 is the label
  // desc=add a prefix name: for column name,e.g. label=name:is_click
  std::string label_column = "";

  // alias=weight
  // desc=specify the weight column
  // desc=use number for index,e.g. weight=0 means column\_0 is the weight
  // desc=add a prefix name: for column name,e.g. weight=name:weight
  // desc=**Note**: index starts from 0. And it doesn't count the label column when passing type is Index,e.g. when label is column\_0,and weight is column\_1,the correct parameter is weight=0
  std::string weight_column = "";

  // alias = query_column, group, query
  // desc=specify the query/group id column
  // desc=use number for index,e.g. query=0 means column\_0 is the query id
  // desc=add a prefix name: for column name,e.g. query=name:query_id
  // desc=**Note**: data should be grouped by query\_id. Index starts from 0. And it doesn't count the label column when passing type is Index,e.g. when label is column\_0 and query\_id is column\_1,the correct parameter is query=0
  std::string group_column = "";

  // alias = ignore_feature, blacklist
  // desc=specify some ignoring columns in training
  // desc=use number for index,e.g. ignore_column=0,1,2 means column\_0,column\_1 and column\_2 will be ignored
  // desc=add a prefix name: for column name,e.g. ignore_column=name:c1,c2,c3 means c1,c2 and c3 will be ignored
  // desc=**Note**: works only in case of loading data directly from file
  // desc=**Note**: index starts from 0. And it doesn't count the label column
  std::string ignore_column = "";

  // alias=categorical_column,cat_feature,cat_column
  // desc=specify categorical features
  // desc=use number for index,e.g. categorical_feature=0,1,2 means column\_0,column\_1 and column\_2 are categorical features
  // desc=add a prefix name: for column name,e.g. categorical_feature=name:c1,c2,c3 means c1,c2 and c3 are categorical features
  // desc=**Note**: only supports categorical with int type. Index starts from 0. And it doesn't count the label column
  // desc=**Note**: the negative values will be treated as **missing values**
  std::string categorical_feature = "";

  // alias=raw_score,is_predict_raw_score,predict_rawscore
  // desc=only used in prediction task
  // desc=set to true to predict only the raw scores
  // desc=set to false to predict transformed scores
  bool predict_raw_score = false;

  // alias=leaf_index,is_predict_leaf_index
  // desc=only used in prediction task
  // desc=set to true to predict with leaf index of all trees
  bool predict_leaf_index = false;

  // alias=contrib,is_predict_contrib
  // desc=only used in prediction task
  // desc=set to true to estimate `SHAP values`_,which represent how each feature contributs to each prediction.
  // desc=Produces number of features + 1 values where the last value is the expected value of the model output over the training data
  bool predict_contrib = false;

  // desc=only used in prediction task
  // desc=use to specify how many trained iterations will be used in prediction
  // desc=<= 0 means no limit
  int num_iteration_predict = -1;

  // desc=if true will use early-stopping to speed up the prediction. May affect the accuracy
  bool pred_early_stop = false;
  
  // desc=the frequency of checking early-stopping prediction
  int pred_early_stop_freq = 10;

  // desc = the threshold of margin in early - stopping prediction
  double pred_early_stop_margin = 10.0;

  // alias=subsample_for_bin
  // check=>0
  // desc=number of data that sampled to construct histogram bins
  // desc=will give better training result when set this larger,but will increase data loading time
  // desc=set this to larger value if data is very sparse
  int bin_construct_sample_cnt = 200000;

  // desc=set to false to disable the special handle of missing value
  bool use_missing = true;

  // desc=set to true to treat all zero as missing values (including the unshown values in libsvm/sparse matrics)
  // desc=set to false to use na to represent missing values
  bool zero_as_missing = false;

  // desc = path to training initial score file, "" will use train_data_file + .init(if exists)
  std::string initscore_filename = "";
  
  // desc=path to validation initial score file,"" will use valid_data_file + .init (if exists)
  // desc=separate by ,for multi-validation data
  std::vector<std::string> valid_data_initscores;
  
  // desc=max cache size(unit:MB) for historical histogram. < 0 means no limit
  double histogram_pool_size = -1.0;

  // check=>2
  // alias=num_classes
  // desc=need to specify this in multi-class classification
  int num_class = 1;

  // desc=set to true to enable auto loading from previous saved binary datasets
  // desc=set to false will ignore the binary datasets
  bool enable_load_from_binary_file = true;

  // desc=set to false to disable Exclusive Feature Bundling (EFB)
  bool enable_bundle = true;
  
  // check=>=0
  // max conflict rate for bundles in EFB
  double max_conflict_rate = 0.0;

  // desc= frequency of saving model file snapshot
  // desc= set to positive numbers will enable this function
  int snapshot_freq = -1;

  // desc=only cpp is supported yet
  // desc=if convert_model_language is set when task is set to train,the model will also be converted
  std::string convert_model_language = "";

  // desc=output file name of converted model
  std::string convert_model = "gbdt_prediction.cpp";
  #pragma endregion


  #pragma region Objective Parameters


  // desc=parameter for sigmoid function. Will be used in binary and multiclassova classification and in lambdarank
  double sigmoid = 1.0;

  // desc=parameter for `Huber loss`_ and `Quantile regression`_. Will be used in regression task
  double alpha = 0.9;

  // desc=parameter for `Fair loss`_. Will be used in regression task
  double fair_c = 1.0;

  // desc=parameter for `Poisson regression`_ to safeguard optimization
  double poisson_max_delta_step = 0.7;

  // desc=only used in regression task
  // desc=adjust initial score to the mean of labels for faster convergence
  bool boost_from_average = true;

  // alias=unbalanced_sets
  // desc=used in binary classification
  // desc=set this to true if training data are unbalance
  bool is_unbalance = false;

  // check=>0
  // desc=weight of positive class in binary classification task
  double scale_pos_weight = 1.0;

  // desc=only used in regression, usually works better for the large-range of labels
  // desc=will fit sqrt(label) instead and prediction result will be also automatically converted to pow2(prediction)
  bool reg_sqrt = false;

  // desc=only used in tweedie regression
  // desc=controls the variance of the tweedie distribution
  // desc=set closer to 2 to shift towards a gamma distribution
  // desc=set closer to 1 to shift towards a poisson distribution
  double tweedie_variance_power = 1.5;

  // default = 0, 1, 3, 7, 15, 31, 63, ..., 2 ^ 30 - 1
  // desc=used in lambdarank
  // desc=relevant gain for labels. For example,the gain of label 2 is 3 if using default label gains
  // desc=separate by ,
  std::vector<double> label_gain;

  // check=>0
  // desc=used in lambdarank
  // desc=will optimize `NDCG`_ at this position
  int max_position = 20;

  #pragma endregion

  #pragma region Metric Parameters
  
  // [Doc-Only]
  // default=''
  // type=multi-enum
  // desc=metric to be evaluated on the evaluation sets **in addition** to what is provided in the training arguments
  // descl2='' (empty string or not specific),metric corresponding to specified objective will be used (this is possible only for pre - defined objective functions, otherwise no evaluation metric will be added)
  // descl2='None' (string,**not** a None value),no metric registered,alias=na
  // descl2=l1,absolute loss,alias=mean_absolute_error,mae,regression_l1
  // descl2=l2,square loss,alias=mean_squared_error,mse,regression_l2,regression
  // descl2=l2_root,root square loss,alias=root_mean_squared_error,rmse
  // descl2=quantile,`Quantile regression`_
  // descl2=mape,`MAPE loss`_,alias=mean_absolute_percentage_error
  // descl2=huber,`Huber loss`_
  // descl2=fair,`Fair loss`_
  // descl2=poisson,negative log-likelihood for `Poisson regression`_
  // descl2=gamma,negative log-likelihood for Gamma regression
  // descl2=gamma_deviance,residual deviance for Gamma regression
  // descl2=tweedie,negative log-likelihood for Tweedie regression
  // descl2=ndcg,`NDCG`_
  // descl2=map,`MAP`_,alias=mean_average_precision
  // descl2=auc,`AUC`_
  // descl2=binary_logloss,`log loss`_,alias=binary
  // descl2=binary_error,for one sample: 0 for correct classification,1 for error classification
  // descl2=multi_logloss,log loss for mulit-class classification,alias=multiclass,softmax,multiclassova,multiclass_ova,ova,ovr
  // descl2=multi_error,error rate for mulit-class classification
  // descl2=xentropy,cross-entropy (with optional linear weights),alias=cross_entropy
  // descl2=xentlambda,"intensity-weighted" cross-entropy,alias=cross_entropy_lambda
  // descl2=kldiv,`Kullback-Leibler divergence`_,alias=kullback_leibler
  // desc=support multiple metrics,separated by ,
  std::vector<std::string> metric_types;

  // check=>0
  // alias = output_freq
  // desc = frequency for metric output
  int metric_freq = 1;

  // alias=training_metric,is_training_metric,train_metric
  // desc=set this to true if you need to output metric result over training dataset
  bool is_provide_training_metric = false;

  // default=1,2,3,4,5
  // alias=ndcg_eval_at,ndcg_at
  // desc=`NDCG`_ evaluation positions,separated by ,
  std::vector<int> eval_at;

  #pragma endregion

  #pragma region Network Parameter

  // alias=num_machine
  // desc=used for parallel learning,the number of machines for parallel learning application
  // desc=need to set this in both socket and mpi versions
  int num_machines = 1;

  // alias = local_port
  // desc=TCP listen port for local machines
  // desc=you should allow this port in firewall settings before training
  int local_listen_port = 12400;

  // desc=socket time-out in minutes
  int time_out = 120;  // in minutes

  // alias=mlist
  // desc=file that lists machines for this parallel learning application
  // desc=each line contains one IP and one port for one machine. The format is ip port,separate by space
  std::string machine_list_filename = "";

  // desc=list of machines, format: ip1:port1,ip2:port2
  std::string machines = "";

  #pragma endregion

  #pragma region GPU Parameters

  // desc=OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform
  // desc=default value is -1,means the system-wide default platform
  int gpu_platform_id = -1;

  // desc=OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID
  // desc=default value is -1,means the default device in the selected platform
  int gpu_device_id = -1;

  // desc=set to true to use double precision math on GPU (default using single precision)
  bool gpu_use_dp = false;

  #pragma endregion

  #pragma endregion

  bool is_parallel = false;
  bool is_parallel_find_bin = false;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params);
  static std::unordered_map<std::string, std::string> alias_table;
  static std::unordered_set<std::string> parameter_set;
private:
  void CheckParamConflict();
  void GetMembersFromString(const std::unordered_map<std::string, std::string>& params);
};

inline bool Config::GetString(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, std::string* out) {
  if (params.count(name) > 0) {
    *out = params.at(name);
    return true;
  }
  return false;
}

inline bool Config::GetInt(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, int* out) {
  if (params.count(name) > 0) {
    if (!Common::AtoiAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type int, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetDouble(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, double* out) {
  if (params.count(name) > 0) {
    if (!Common::AtofAndCheck(params.at(name).c_str(), out)) {
      Log::Fatal("Parameter %s should be of type double, got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

inline bool Config::GetBool(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, bool* out) {
  if (params.count(name) > 0) {
    std::string value = params.at(name);
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("false") || value == std::string("-")) {
      *out = false;
    } else if (value == std::string("true") || value == std::string("+")) {
      *out = true;
    } else {
      Log::Fatal("Parameter %s should be \"true\"/\"+\" or \"false\"/\"-\", got \"%s\"",
                 name.c_str(), params.at(name).c_str());
    }
    return true;
  }
  return false;
}

struct ParameterAlias {
  static void KeyAliasTransform(std::unordered_map<std::string, std::string>* params) {
    std::unordered_map<std::string, std::string> tmp_map;
    for (const auto& pair : *params) {
      auto alias = Config::alias_table.find(pair.first);
      if (alias != Config::alias_table.end()) { // found alias
        auto alias_set = tmp_map.find(alias->second);
        if (alias_set != tmp_map.end()) { // alias already set
                                          // set priority by length & alphabetically to ensure reproducible behavior
          if (alias_set->second.size() < pair.first.size() ||
            (alias_set->second.size() == pair.first.size() && alias_set->second < pair.first)) {
            Log::Warning("%s is set with %s=%s, %s=%s will be ignored. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), params->at(alias_set->second).c_str());
          } else {
            Log::Warning("%s is set with %s=%s, will be overridden by %s=%s. Current value: %s=%s",
                         alias->second.c_str(), alias_set->second.c_str(), params->at(alias_set->second).c_str(),
                         pair.first.c_str(), pair.second.c_str(), alias->second.c_str(), pair.second.c_str());
            tmp_map[alias->second] = pair.first;
          }
        } else { // alias not set
          tmp_map.emplace(alias->second, pair.first);
        }
      } else if (Config::parameter_set.find(pair.first) == Config::parameter_set.end()) {
        Log::Warning("Unknown parameter: %s", pair.first.c_str());
      }
    }
    for (const auto& pair : tmp_map) {
      auto alias = params->find(pair.first);
      if (alias == params->end()) { // not find
        params->emplace(pair.first, params->at(pair.second));
        params->erase(pair.second);
      } else {
        Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
                     pair.first.c_str(), alias->second.c_str(), pair.second.c_str(), params->at(pair.second).c_str(),
                     pair.first.c_str(), alias->second.c_str());
      }
    }
  }
};

}   // namespace LightGBM

#endif   // LightGBM_CONFIG_H_
