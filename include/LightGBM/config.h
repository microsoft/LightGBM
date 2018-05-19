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

  TaskType task_type = TaskType::kTrain;
  std::string device_type = "cpu";
  std::string boosting_type = "gbdt";
  std::string objective_type = "regression";
  std::vector<std::string> metric_types;

  int seed = 0;
  int num_threads = 0;
  int max_bin = 255;
  int num_class = 1;
  int data_random_seed = 1;
  std::string data_filename = "";
  std::string initscore_filename = "";
  std::vector<std::string> valid_data_filenames;
  std::vector<std::string> valid_data_initscores;
  int snapshot_freq = -1;
  std::string convert_model_language = "";
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

  /*! \brief Set to true if want to use early stop for the prediction */
  bool pred_early_stop = false;
  /*! \brief Frequency of checking the pred_early_stop */
  int pred_early_stop_freq = 10;
  /*! \brief Threshold of margin of pred_early_stop */
  double pred_early_stop_margin = 10.0;
  bool zero_as_missing = false;
  bool use_missing = true;
  double sigmoid = 1.0;
  double fair_c = 1.0;
  double poisson_max_delta_step = 0.7;
  // for lambdarank
  std::vector<double> label_gain;
  // for lambdarank
  int max_position = 20;
  // for binary
  bool is_unbalance = false;
  // Balancing of positive and negative weights
  double scale_pos_weight = 1.0;
  // True will sqrt fit the sqrt(label)
  bool reg_sqrt = false;
  double alpha = 0.9;
  double tweedie_variance_power = 1.5;
  std::vector<int> eval_at;
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
  int output_freq = 1;
  bool is_provide_training_metric = false;
  int num_iterations = 100;
  double learning_rate = 0.1;
  double bagging_fraction = 1.0;
  int bagging_seed = 3;
  int bagging_freq = 0;
  int early_stopping_round = 0;
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
  std::string tree_learner_type = "serial";
  /* filename of forced splits */
  std::string forcedsplits_filename = "";

  int num_machines = 1;
  int local_listen_port = 12400;
  int time_out = 120;  // in minutes
  std::string machine_list_filename = "";
  std::string machines = "";

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
