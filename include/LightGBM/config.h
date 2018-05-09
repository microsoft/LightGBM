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

const std::string kDefaultTreeLearnerType = "serial";
const std::string kDefaultDevice = "cpu";
const std::string kDefaultBoostingType = "gbdt";
const std::string kDefaultObjectiveType = "regression";
const int kDefaultNumLeaves = 31;

/*!
* \brief The interface for Config
*/
struct ConfigBase {
public:
  /*! \brief virtual destructor */
  virtual ~ConfigBase() {}

  /*!
  * \brief Set current config object by params
  * \param params Store the key and value for params
  */
  virtual void Set(
    const std::unordered_map<std::string, std::string>& params) = 0;

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
};

/*! \brief Types of tasks */
enum TaskType {
  kTrain, kPredict, kConvertModel, KRefitTree
};

/*! \brief Config for input and output files */
struct IOConfig: public ConfigBase {
public:
  int max_bin = 255;
  int num_class = 1;
  int data_random_seed = 1;
  std::string data_filename = "";
  std::string initscore_filename = "";
  std::vector<std::string> valid_data_filenames;
  std::vector<std::string> valid_data_initscores;
  int snapshot_freq = -1;
  std::string output_model = "LightGBM_model.txt";
  std::string output_result = "LightGBM_predict_result.txt";
  std::string convert_model = "gbdt_prediction.cpp";
  std::string input_model = "";

  int verbosity = 1;
  int num_iteration_predict = -1;
  bool is_pre_partition = false;
  bool is_enable_sparse = true;
  /*! \brief The threshold of zero elements precentage for treating a feature as a sparse feature.
   *  Default is 0.8, where a feature is treated as a sparse feature when there are over 80% zeros.
   *  When setting to 1.0, all features are processed as dense features.
   */
  double sparse_threshold = 0.8;
  bool use_two_round_loading = false;
  bool is_save_binary_file = false;
  bool enable_load_from_binary_file = true;
  int bin_construct_sample_cnt = 200000;
  bool is_predict_leaf_index = false;
  bool is_predict_contrib = false;
  bool is_predict_raw_score = false;
  int min_data_in_leaf = 20;
  int min_data_in_bin = 3;
  double max_conflict_rate = 0.0;
  bool enable_bundle = true;
  bool has_header = false;
  std::vector<int8_t> monotone_constraints;
  /*! \brief Index or column name of label, default is the first column
   * And add an prefix "name:" while using column name */
  std::string label_column = "";
  /*! \brief Index or column name of weight, < 0 means not used
  * And add an prefix "name:" while using column name
  * Note: when using Index, it doesn't count the label index */
  std::string weight_column = "";
  /*! \brief Index or column name of group/query id, < 0 means not used
  * And add an prefix "name:" while using column name
  * Note: when using Index, it doesn't count the label index */
  std::string group_column = "";
  /*! \brief ignored features, separate by ','
  * And add an prefix "name:" while using column name
  * Note: when using Index, it doesn't count the label index */
  std::string ignore_column = "";
  /*! \brief specific categorical columns, Note:only support for integer type categorical
  * And add an prefix "name:" while using column name
  * Note: when using Index, it doesn't count the label index */
  std::string categorical_column = "";
  std::string device_type = kDefaultDevice;

  /*! \brief Set to true if want to use early stop for the prediction */
  bool pred_early_stop = false;
  /*! \brief Frequency of checking the pred_early_stop */
  int pred_early_stop_freq = 10;
  /*! \brief Threshold of margin of pred_early_stop */
  double pred_early_stop_margin = 10.0;
  bool zero_as_missing = false;
  bool use_missing = true;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;
};

/*! \brief Config for objective function */
struct ObjectiveConfig: public ConfigBase {
public:
  virtual ~ObjectiveConfig() {}
  double sigmoid = 1.0;
  double fair_c = 1.0;
  double poisson_max_delta_step = 0.7;
  // for lambdarank
  std::vector<double> label_gain;
  // for lambdarank
  int max_position = 20;
  // for binary
  bool is_unbalance = false;
  // for multiclass
  int num_class = 1;
  // Balancing of positive and negative weights
  double scale_pos_weight = 1.0;
  // True will sqrt fit the sqrt(label)
  bool reg_sqrt = false;
  double alpha = 0.9;
  double tweedie_variance_power = 1.5;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;
};

/*! \brief Config for metrics interface*/
struct MetricConfig: public ConfigBase {
public:
  virtual ~MetricConfig() {}
  int num_class = 1;
  double sigmoid = 1.0;
  double fair_c = 1.0;
  double alpha = 0.9;
  double tweedie_variance_power = 1.5;
  std::vector<double> label_gain;
  std::vector<int> eval_at;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;
};


/*! \brief Config for tree model */
struct TreeConfig: public ConfigBase {
public:
  int min_data_in_leaf = 20;
  double min_sum_hessian_in_leaf = 1e-3;
  double max_delta_step = 0.0;
  double lambda_l1 = 0.0;
  double lambda_l2 = 0.0;
  double min_gain_to_split = 0.0;
  // should > 1
  int num_leaves = kDefaultNumLeaves;
  int feature_fraction_seed = 2;
  double feature_fraction = 1.0;
  // max cache size(unit:MB) for historical histogram. < 0 means no limit
  double histogram_pool_size = -1.0;
  // max depth of tree model.
  // Still grow tree by leaf-wise, but limit the max depth to avoid over-fitting
  // And the max leaves will be min(num_leaves, pow(2, max_depth))
  // max_depth < 0 means no limit
  int max_depth = -1;
  int top_k = 20;
  /*! \brief OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform.
   *  Default value is -1, using the system-wide default platform
   */
  int gpu_platform_id = -1;
  /*! \brief OpenCL device ID in the specified platform. Each GPU in the selected platform has a
   *  unique device ID. Default value is -1, using the default device in the selected platform
   */
  int gpu_device_id = -1;
  /*! \brief Set to true to use double precision math on GPU (default using single precision) */
  bool gpu_use_dp = false;
  int min_data_per_group = 100;
  int max_cat_threshold = 32;
  double cat_l2 = 10;
  double cat_smooth = 10;
  int max_cat_to_onehot = 4;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;
};

/*! \brief Config for Boosting */
struct BoostingConfig: public ConfigBase {
public:
  virtual ~BoostingConfig() {}
  int output_freq = 1;
  bool is_provide_training_metric = false;
  int num_iterations = 100;
  double learning_rate = 0.1;
  double bagging_fraction = 1.0;
  int bagging_seed = 3;
  int bagging_freq = 0;
  int early_stopping_round = 0;
  int num_class = 1;
  double drop_rate = 0.1;
  int max_drop = 50;
  double skip_drop = 0.5;
  bool xgboost_dart_mode = false;
  bool uniform_drop = false;
  int drop_seed = 4;
  double top_rate = 0.2;
  double other_rate = 0.1;
  // only used for the regression. Will boost from the average labels.
  bool boost_from_average = true;
  std::string tree_learner_type = kDefaultTreeLearnerType;
  std::string device_type = kDefaultDevice;
  TreeConfig tree_config;
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;

  /* filename of forced splits */
  std::string forcedsplits_filename = "";
};

/*! \brief Config for Network */
struct NetworkConfig: public ConfigBase {
public:
  int num_machines = 1;
  int local_listen_port = 12400;
  int time_out = 120;  // in minutes
  std::string machine_list_filename = "";
  std::string machines = "";
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;
};


/*! \brief Overall config, all configs will put on this class */
struct OverallConfig: public ConfigBase {
public:
  TaskType task_type = TaskType::kTrain;
  NetworkConfig network_config;
  int seed = 0;
  int num_threads = 0;
  bool is_parallel = false;
  bool is_parallel_find_bin = false;
  IOConfig io_config;
  std::string boosting_type = kDefaultBoostingType;
  BoostingConfig boosting_config;
  std::string objective_type =  kDefaultObjectiveType;
  ObjectiveConfig objective_config;
  std::vector<std::string> metric_types;
  MetricConfig metric_config;
  std::string convert_model_language = "";
  LIGHTGBM_EXPORT void Set(const std::unordered_map<std::string, std::string>& params) override;

private:
  void CheckParamConflict();
};


inline bool ConfigBase::GetString(
  const std::unordered_map<std::string, std::string>& params,
  const std::string& name, std::string* out) {
  if (params.count(name) > 0) {
    *out = params.at(name);
    return true;
  }
  return false;
}

inline bool ConfigBase::GetInt(
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

inline bool ConfigBase::GetDouble(
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

inline bool ConfigBase::GetBool(
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
    const std::unordered_map<std::string, std::string> alias_table(
    {
      { "config", "config_file" },
      { "nthread", "num_threads" },
      { "num_thread", "num_threads" },
      { "random_seed", "seed" },
      { "boosting", "boosting_type" },
      { "boost", "boosting_type" },
      { "application", "objective" },
      { "app", "objective" },
      { "train_data", "data" },
      { "train", "data" },
      { "model_output", "output_model" },
      { "model_out", "output_model" },
      { "model_input", "input_model" },
      { "model_in", "input_model" },
      { "predict_result", "output_result" },
      { "prediction_result", "output_result" },
      { "valid", "valid_data" },
      { "test_data", "valid_data" },
      { "test", "valid_data" },
      { "is_sparse", "is_enable_sparse" },
      { "enable_sparse", "is_enable_sparse" },
      { "pre_partition", "is_pre_partition" },
      { "training_metric", "is_training_metric" },
      { "train_metric", "is_training_metric" },
      { "ndcg_at", "ndcg_eval_at" },
      { "eval_at", "ndcg_eval_at" },
      { "min_data_per_leaf", "min_data_in_leaf" },
      { "min_data", "min_data_in_leaf" },
      { "min_child_samples", "min_data_in_leaf" },
      { "min_sum_hessian_per_leaf", "min_sum_hessian_in_leaf" },
      { "min_sum_hessian", "min_sum_hessian_in_leaf" },
      { "min_hessian", "min_sum_hessian_in_leaf" },
      { "min_child_weight", "min_sum_hessian_in_leaf" },
      { "num_leaf", "num_leaves" },
      { "sub_feature", "feature_fraction" },
      { "colsample_bytree", "feature_fraction" },
      { "num_iteration", "num_iterations" },
      { "num_tree", "num_iterations" },
      { "num_round", "num_iterations" },
      { "num_trees", "num_iterations" },
      { "num_rounds", "num_iterations" },
      { "num_boost_round", "num_iterations" },
      { "n_estimators", "num_iterations"},
      { "sub_row", "bagging_fraction" },
      { "subsample", "bagging_fraction" },
      { "subsample_freq", "bagging_freq" },
      { "shrinkage_rate", "learning_rate" },
      { "tree", "tree_learner" },
      { "num_machine", "num_machines" },
      { "local_port", "local_listen_port" },
      { "two_round_loading", "use_two_round_loading"},
      { "two_round", "use_two_round_loading" },
      { "mlist", "machine_list_file" },
      { "is_save_binary", "is_save_binary_file" },
      { "save_binary", "is_save_binary_file" },
      { "early_stopping_rounds", "early_stopping_round"},
      { "early_stopping", "early_stopping_round"},
      { "verbosity", "verbose" },
      { "header", "has_header" },
      { "label", "label_column" },
      { "weight", "weight_column" },
      { "group", "group_column" },
      { "query", "group_column" },
      { "query_column", "group_column" },
      { "ignore_feature", "ignore_column" },
      { "blacklist", "ignore_column" },
      { "categorical_feature", "categorical_column" },
      { "cat_column", "categorical_column" },
      { "cat_feature", "categorical_column" },
      { "predict_raw_score", "is_predict_raw_score" },
      { "raw_score", "is_predict_raw_score" },
      { "leaf_index", "is_predict_leaf_index" },
      { "predict_leaf_index", "is_predict_leaf_index" },
      { "contrib", "is_predict_contrib" },
      { "predict_contrib", "is_predict_contrib" },
      { "min_split_gain", "min_gain_to_split" },
      { "topk", "top_k" },
      { "reg_alpha", "lambda_l1" },
      { "reg_lambda", "lambda_l2" },
      { "num_classes", "num_class" },
      { "unbalanced_sets", "is_unbalance" },
      { "bagging_fraction_seed", "bagging_seed" },
      { "workers", "machines" },
      { "nodes", "machines" },
      { "subsample_for_bin", "bin_construct_sample_cnt" },
      { "metric_freq", "output_freq" },
      { "mc", "monotone_constraints" },
      { "max_tree_output", "max_delta_step" },
      { "max_leaf_output", "max_delta_step" }
    });
    const std::unordered_set<std::string> parameter_set({
      "config", "config_file", "task", "device",
      "num_threads", "seed", "boosting_type", "objective", "data",
      "output_model", "input_model", "output_result", "valid_data",
      "is_enable_sparse", "is_pre_partition", "is_training_metric",
      "ndcg_eval_at", "min_data_in_leaf", "min_sum_hessian_in_leaf",
      "num_leaves", "feature_fraction", "num_iterations",
      "bagging_fraction", "bagging_freq", "learning_rate", "tree_learner",
      "num_machines", "local_listen_port", "use_two_round_loading",
      "machine_list_file", "is_save_binary_file", "early_stopping_round",
      "verbose", "has_header", "label_column", "weight_column", "group_column",
      "ignore_column", "categorical_column", "is_predict_raw_score",
      "is_predict_leaf_index", "min_gain_to_split", "top_k",
      "lambda_l1", "lambda_l2", "num_class", "is_unbalance",
      "max_depth", "max_bin", "bagging_seed",
      "drop_rate", "skip_drop", "max_drop", "uniform_drop",
      "xgboost_dart_mode", "drop_seed", "top_rate", "other_rate",
      "min_data_in_bin", "data_random_seed", "bin_construct_sample_cnt",
      "num_iteration_predict", "pred_early_stop", "pred_early_stop_freq",
      "pred_early_stop_margin", "use_missing", "sigmoid",
      "fair_c", "poission_max_delta_step", "scale_pos_weight",
      "boost_from_average", "max_position", "label_gain",
      "metric", "output_freq", "time_out",
      "gpu_platform_id", "gpu_device_id", "gpu_use_dp",
      "convert_model", "convert_model_language",
      "feature_fraction_seed", "enable_bundle", "data_filename", "valid_data_filenames",
      "snapshot_freq", "verbosity", "sparse_threshold", "enable_load_from_binary_file",
      "max_conflict_rate", "poisson_max_delta_step",
      "histogram_pool_size", "is_provide_training_metric", "machine_list_filename", "machines",
      "zero_as_missing", "init_score_file", "valid_init_score_file", "is_predict_contrib",
      "max_cat_threshold",  "cat_smooth", "min_data_per_group", "cat_l2", "max_cat_to_onehot",
      "alpha", "reg_sqrt", "tweedie_variance_power", "monotone_constraints", "max_delta_step",
      "forced_splits"
    });
    std::unordered_map<std::string, std::string> tmp_map;
    for (const auto& pair : *params) {
      auto alias = alias_table.find(pair.first);
      if (alias != alias_table.end()) { // found alias
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
      } else if (parameter_set.find(pair.first) == parameter_set.end()) {
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
