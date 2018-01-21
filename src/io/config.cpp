#include <LightGBM/config.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/utils/log.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <limits>

namespace LightGBM {

void ConfigBase::KV2Map(std::unordered_map<std::string, std::string>& params, const char* kv) {
  std::vector<std::string> tmp_strs = Common::Split(kv, '=');
  if (tmp_strs.size() == 2) {
    std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
    std::string value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
    if (key.size() > 0) {
      auto value_search = params.find(key);
      if (value_search == params.end()) { // not set
        params.emplace(key, value);
      } else {
        Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s.",
          key.c_str(), value_search->second.c_str(), key.c_str(), value.c_str(),
          key.c_str(), value_search->second.c_str());
      }
    }
  } else {
    Log::Warning("Unknown parameter %s", kv);
  }
}

std::unordered_map<std::string, std::string> ConfigBase::Str2Map(const char* parameters) {
  std::unordered_map<std::string, std::string> params;
  auto args = Common::Split(parameters, " \t\n\r");
  for (auto arg : args) {
    KV2Map(params, Common::Trim(arg).c_str());
  }
  ParameterAlias::KeyAliasTransform(&params);
  return params;
}

void GetBoostingType(const std::unordered_map<std::string, std::string>& params, std::string* boosting_type) {
  std::string value;
  if (ConfigBase::GetString(params, "boosting_type", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("gbdt") || value == std::string("gbrt")) {
      *boosting_type = "gbdt";
    } else if (value == std::string("dart")) {
      *boosting_type = "dart";
    } else if (value == std::string("goss")) {
      *boosting_type = "goss";
    } else if (value == std::string("rf") || value == std::string("randomforest")) {
      *boosting_type = "rf";
    } else {
      Log::Fatal("Unknown boosting type %s", value.c_str());
    }
  }
}

void GetObjectiveType(const std::unordered_map<std::string, std::string>& params, std::string* objective_type) {
  std::string value;
  if (ConfigBase::GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    *objective_type = value;
  }
}

void GetMetricType(const std::unordered_map<std::string, std::string>& params, std::vector<std::string>* metric_types) {
  std::string value;
  if (ConfigBase::GetString(params, "metric", &value)) {
    // clear old metrics
    metric_types->clear();
    // to lower
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    // split
    std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
    // remove duplicate
    std::unordered_set<std::string> metric_sets;
    for (auto& metric : metrics) {
      std::transform(metric.begin(), metric.end(), metric.begin(), Common::tolower);
      if (metric_sets.count(metric) <= 0) {
        metric_sets.insert(metric);
      }
    }
    for (auto& metric : metric_sets) {
      metric_types->push_back(metric);
    }
    metric_types->shrink_to_fit();
  }
}

void GetTaskType(const std::unordered_map<std::string, std::string>& params, TaskType* task_type) {
  std::string value;
  if (ConfigBase::GetString(params, "task", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("train") || value == std::string("training")) {
      *task_type = TaskType::kTrain;
    } else if (value == std::string("predict") || value == std::string("prediction")
               || value == std::string("test")) {
      *task_type = TaskType::kPredict;
    } else if (value == std::string("convert_model")) {
      *task_type = TaskType::kConvertModel;
    } else if (value == std::string("refit") || value == std::string("refit_tree")) {
      *task_type = TaskType::KRefitTree;
    } else {
      Log::Fatal("Unknown task type %s", value.c_str());
    }
  }
}

void GetDeviceType(const std::unordered_map<std::string, std::string>& params, std::string* device_type) {
  std::string value;
  if (ConfigBase::GetString(params, "device", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("cpu")) {
      *device_type = "cpu";
    } else if (value == std::string("gpu")) {
      *device_type = "gpu";
    } else {
      Log::Fatal("Unknown device type %s", value.c_str());
    }
  }
}

void GetTreeLearnerType(const std::unordered_map<std::string, std::string>& params, std::string* tree_learner_type) {
  std::string value;
  if (ConfigBase::GetString(params, "tree_learner", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("serial")) {
      *tree_learner_type = "serial";
    } else if (value == std::string("feature") || value == std::string("feature_parallel")) {
      *tree_learner_type = "feature";
    } else if (value == std::string("data") || value == std::string("data_parallel")) {
      *tree_learner_type = "data";
    } else if (value == std::string("voting") || value == std::string("voting_parallel")) {
      *tree_learner_type = "voting";
    } else {
      Log::Fatal("Unknown tree learner type %s", value.c_str());
    }
  }
}

void OverallConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  // load main config types
  GetInt(params, "num_threads", &num_threads);
  GetString(params, "convert_model_language", &convert_model_language);

  // generate seeds by seed.
  if (GetInt(params, "seed", &seed)) {
    Random rand(seed);
    int int_max = std::numeric_limits<short>::max();
    io_config.data_random_seed = static_cast<int>(rand.NextShort(0, int_max));
    boosting_config.bagging_seed = static_cast<int>(rand.NextShort(0, int_max));
    boosting_config.drop_seed = static_cast<int>(rand.NextShort(0, int_max));
    boosting_config.tree_config.feature_fraction_seed = static_cast<int>(rand.NextShort(0, int_max));
  }
  GetTaskType(params, &task_type);
  GetBoostingType(params, &boosting_type);

  GetMetricType(params, &metric_types);

  // sub-config setup
  network_config.Set(params);
  io_config.Set(params);

  boosting_config.Set(params);
  GetObjectiveType(params, &objective_type);
  objective_config.Set(params);
  metric_config.Set(params);

  // check for conflicts
  CheckParamConflict();

  if (io_config.verbosity == 1) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Info);
  } else if (io_config.verbosity == 0) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
  } else if (io_config.verbosity >= 2) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Debug);
  } else {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
  }
}

void OverallConfig::CheckParamConflict() {
  // check if objective_type, metric_type, and num_class match
  bool objective_type_multiclass = (objective_type == std::string("multiclass")
                                    || objective_type == std::string("multiclassova"));
  int num_class_check = boosting_config.num_class;
  if (objective_type_multiclass) {
    if (num_class_check <= 1) {
      Log::Fatal("Number of classes should be specified and greater than 1 for multiclass training");
    }
  } else {
    if (task_type == TaskType::kTrain && num_class_check != 1) {
      Log::Fatal("Number of classes must be 1 for non-multiclass training");
    }
  }
  if (boosting_config.is_provide_training_metric || !io_config.valid_data_filenames.empty()) {
    for (std::string metric_type : metric_types) {
      bool metric_type_multiclass = (metric_type == std::string("multi_logloss")
                                     || metric_type == std::string("multi_error"));
      if ((objective_type_multiclass && !metric_type_multiclass)
        || (!objective_type_multiclass && metric_type_multiclass)) {
        Log::Fatal("Objective and metrics don't match");
      }
    }
  }

  if (network_config.num_machines > 1) {
    is_parallel = true;
  } else {
    is_parallel = false;
    boosting_config.tree_learner_type = "serial";
  }

  bool is_single_tree_learner = boosting_config.tree_learner_type == std::string("serial");

  if (is_single_tree_learner) {
    is_parallel = false;
    network_config.num_machines = 1;
  }

  if (is_single_tree_learner || boosting_config.tree_learner_type == std::string("feature")) {
    is_parallel_find_bin = false;
  } else if (boosting_config.tree_learner_type == std::string("data")
             || boosting_config.tree_learner_type == std::string("voting")) {
    is_parallel_find_bin = true;
    if (boosting_config.tree_config.histogram_pool_size >= 0
        && boosting_config.tree_learner_type == std::string("data")) {
      Log::Warning("Histogram LRU queue was enabled (histogram_pool_size=%f). Will disable this to reduce communication costs"
        , boosting_config.tree_config.histogram_pool_size);
      // Change pool size to -1 (no limit) when using data parallel to reduce communication costs
      boosting_config.tree_config.histogram_pool_size = -1;
    }
  }
  // Check max_depth and num_leaves
  if (boosting_config.tree_config.max_depth > 0) {
    int full_num_leaves = static_cast<int>(std::pow(2, boosting_config.tree_config.max_depth));
    if (full_num_leaves > boosting_config.tree_config.num_leaves 
        && boosting_config.tree_config.num_leaves == kDefaultNumLeaves) {
      Log::Warning("Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves.");
    }
  }
}

void IOConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "max_bin", &max_bin);
  CHECK(max_bin > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetInt(params, "data_random_seed", &data_random_seed);
  GetString(params, "data", &data_filename);
  GetString(params, "init_score_file", &initscore_filename);
  GetInt(params, "verbose", &verbosity);
  GetInt(params, "num_iteration_predict", &num_iteration_predict);
  GetInt(params, "bin_construct_sample_cnt", &bin_construct_sample_cnt);
  CHECK(bin_construct_sample_cnt > 0);
  GetBool(params, "is_pre_partition", &is_pre_partition);
  GetBool(params, "is_enable_sparse", &is_enable_sparse);
  GetDouble(params, "sparse_threshold", &sparse_threshold);
  GetBool(params, "use_two_round_loading", &use_two_round_loading);
  GetBool(params, "is_save_binary_file", &is_save_binary_file);
  GetBool(params, "enable_load_from_binary_file", &enable_load_from_binary_file);
  GetBool(params, "is_predict_raw_score", &is_predict_raw_score);
  GetBool(params, "is_predict_leaf_index", &is_predict_leaf_index);
  GetBool(params, "is_predict_contrib", &is_predict_contrib);
  GetInt(params, "snapshot_freq", &snapshot_freq);
  GetString(params, "output_model", &output_model);
  GetString(params, "input_model", &input_model);
  GetString(params, "convert_model", &convert_model);
  GetString(params, "output_result", &output_result);
  std::string tmp_str = "";
  if (GetString(params, "valid_data", &tmp_str)) {
    valid_data_filenames = Common::Split(tmp_str.c_str(), ',');
  }
  if (GetString(params, "valid_init_score_file", &tmp_str)) {
    valid_data_initscores = Common::Split(tmp_str.c_str(), ',');
  } else {
    valid_data_initscores = std::vector<std::string>(valid_data_filenames.size(), "");
  }
  CHECK(valid_data_filenames.size() == valid_data_initscores.size());
  GetBool(params, "has_header", &has_header);
  GetString(params, "label_column", &label_column);
  GetString(params, "weight_column", &weight_column);
  GetString(params, "group_column", &group_column);
  GetString(params, "ignore_column", &ignore_column);
  GetString(params, "categorical_column", &categorical_column);
  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  GetInt(params, "min_data_in_bin", &min_data_in_bin);
  CHECK(min_data_in_bin > 0);
  CHECK(min_data_in_leaf >= 0);
  GetDouble(params, "max_conflict_rate", &max_conflict_rate);
  CHECK(max_conflict_rate >= 0);
  GetBool(params, "enable_bundle", &enable_bundle);
  GetBool(params, "pred_early_stop", &pred_early_stop);
  GetInt(params, "pred_early_stop_freq", &pred_early_stop_freq);
  GetDouble(params, "pred_early_stop_margin", &pred_early_stop_margin);
  GetBool(params, "use_missing", &use_missing);
  GetBool(params, "zero_as_missing", &zero_as_missing);
  GetDeviceType(params, &device_type);
}

void ObjectiveConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetBool(params, "is_unbalance", &is_unbalance);
  GetDouble(params, "sigmoid", &sigmoid);
  CHECK(sigmoid > 0);
  GetDouble(params, "fair_c", &fair_c);
  CHECK(fair_c > 0);
  GetDouble(params, "poisson_max_delta_step", &poisson_max_delta_step);
  CHECK(poisson_max_delta_step > 0);
  GetInt(params, "max_position", &max_position);
  CHECK(max_position > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetDouble(params, "scale_pos_weight", &scale_pos_weight);
  CHECK(scale_pos_weight > 0);
  GetDouble(params, "alpha", &alpha);
  CHECK(alpha > 0 && alpha < 1);
  GetBool(params, "reg_sqrt", &reg_sqrt);
  GetDouble(params, "tweedie_variance_power", &tweedie_variance_power);
  CHECK(tweedie_variance_power >= 1 && tweedie_variance_power < 2);
  std::string tmp_str = "";
  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToArray<double>(tmp_str, ',');
  } else {
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    const int max_label = 31;
    label_gain.push_back(0.0f);
    for (int i = 1; i < max_label; ++i) {
      label_gain.push_back(static_cast<double>((1 << i) - 1));
    }
  }
  label_gain.shrink_to_fit();
}


void MetricConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetDouble(params, "sigmoid", &sigmoid);
  CHECK(sigmoid > 0);
  GetDouble(params, "fair_c", &fair_c);
  CHECK(fair_c > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetDouble(params, "alpha", &alpha);
  CHECK(alpha > 0 && alpha < 1);
  GetDouble(params, "tweedie_variance_power", &tweedie_variance_power);
  CHECK(tweedie_variance_power >= 1 && tweedie_variance_power < 2);
  std::string tmp_str = "";
  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToArray<double>(tmp_str, ',');
  } else {
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    const int max_label = 31;
    label_gain.push_back(0.0f);
    for (int i = 1; i < max_label; ++i) {
      label_gain.push_back(static_cast<double>((1 << i) - 1));
    }
  }
  label_gain.shrink_to_fit();
  if (GetString(params, "ndcg_eval_at", &tmp_str)) {
    eval_at = Common::StringToArray<int>(tmp_str, ',');
    std::sort(eval_at.begin(), eval_at.end());
    for (size_t i = 0; i < eval_at.size(); ++i) {
      CHECK(eval_at[i] > 0);
    }
  } else {
    // default eval ndcg @[1-5]
    for (int i = 1; i <= 5; ++i) {
      eval_at.push_back(i);
    }
  }
  eval_at.shrink_to_fit();
}


void TreeConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  GetDouble(params, "min_sum_hessian_in_leaf", &min_sum_hessian_in_leaf);
  CHECK(min_data_in_leaf > 0);
  CHECK(min_sum_hessian_in_leaf >= 0);
  GetDouble(params, "lambda_l1", &lambda_l1);
  CHECK(lambda_l1 >= 0.0f);
  GetDouble(params, "lambda_l2", &lambda_l2);
  CHECK(lambda_l2 >= 0.0f);
  GetDouble(params, "min_gain_to_split", &min_gain_to_split);
  CHECK(min_gain_to_split >= 0.0f);
  GetInt(params, "num_leaves", &num_leaves);
  CHECK(num_leaves > 1);
  GetInt(params, "feature_fraction_seed", &feature_fraction_seed);
  GetDouble(params, "feature_fraction", &feature_fraction);
  CHECK(feature_fraction > 0.0f && feature_fraction <= 1.0f);
  GetDouble(params, "histogram_pool_size", &histogram_pool_size);
  GetInt(params, "max_depth", &max_depth);
  GetInt(params, "top_k", &top_k);
  CHECK(top_k > 0);
  GetInt(params, "gpu_platform_id", &gpu_platform_id);
  GetInt(params, "gpu_device_id", &gpu_device_id);
  GetBool(params, "gpu_use_dp", &gpu_use_dp);
  GetInt(params, "max_cat_threshold", &max_cat_threshold);
  GetDouble(params, "cat_l2", &cat_l2);
  GetDouble(params, "cat_smooth", &cat_smooth);
  GetInt(params, "min_data_per_group", &min_data_per_group);
  GetInt(params, "max_cat_to_onehot", &max_cat_to_onehot);
  CHECK(max_cat_threshold > 0);
  CHECK(cat_l2 >= 0.0f);
  CHECK(cat_smooth >= 1);
  CHECK(min_data_per_group > 0);
  CHECK(max_cat_to_onehot > 0);
}

void BoostingConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "num_iterations", &num_iterations);
  CHECK(num_iterations >= 0);
  GetInt(params, "bagging_seed", &bagging_seed);
  GetInt(params, "bagging_freq", &bagging_freq);
  CHECK(bagging_freq >= 0);
  GetDouble(params, "bagging_fraction", &bagging_fraction);
  CHECK(bagging_fraction > 0.0f && bagging_fraction <= 1.0f);
  GetDouble(params, "learning_rate", &learning_rate);
  CHECK(learning_rate > 0.0f);
  GetInt(params, "early_stopping_round", &early_stopping_round);
  CHECK(early_stopping_round >= 0);
  GetInt(params, "output_freq", &output_freq);
  CHECK(output_freq >= 0);
  GetBool(params, "is_training_metric", &is_provide_training_metric);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class > 0);
  GetInt(params, "drop_seed", &drop_seed);
  GetDouble(params, "drop_rate", &drop_rate);
  GetDouble(params, "skip_drop", &skip_drop);
  CHECK(drop_rate <= 1.0 && drop_rate >= 0.0);
  CHECK(skip_drop <= 1.0 && skip_drop >= 0.0);
  GetInt(params, "max_drop", &max_drop);
  CHECK(max_drop > 0);
  GetBool(params, "xgboost_dart_mode", &xgboost_dart_mode);
  GetBool(params, "uniform_drop", &uniform_drop);
  GetDouble(params, "top_rate", &top_rate);
  GetDouble(params, "other_rate", &other_rate);
  CHECK(top_rate > 0);
  CHECK(other_rate > 0);
  CHECK(top_rate + other_rate <= 1.0);
  GetBool(params, "boost_from_average", &boost_from_average);
  GetDeviceType(params, &device_type);
  GetTreeLearnerType(params, &tree_learner_type);
  tree_config.Set(params);
}

void NetworkConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "num_machines", &num_machines);
  CHECK(num_machines >= 1);
  GetInt(params, "local_listen_port", &local_listen_port);
  CHECK(local_listen_port > 0);
  GetInt(params, "time_out", &time_out);
  CHECK(time_out > 0);
  GetString(params, "machine_list_file", &machine_list_filename);
  GetString(params, "machines", &machines);
}

}  // namespace LightGBM
