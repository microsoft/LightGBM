#include <LightGBM/config.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

namespace LightGBM {

void OverallConfig::LoadFromString(const char* str) {
  std::unordered_map<std::string, std::string> params;
  auto args = Common::Split(str, " \t\n\r");
  for (auto arg : args) {
    std::vector<std::string> tmp_strs = Common::Split(arg.c_str(), '=');
    if (tmp_strs.size() == 2) {
      std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
      std::string value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
      if (key.size() <= 0) {
        continue;
      }
      params[key] = value;
    } else {
      Log::Error("Unknown parameter %s", arg.c_str());
    }
  }
  ParameterAlias::KeyAliasTransform(&params);
  Set(params);
}

void OverallConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  // load main config types
  GetInt(params, "num_threads", &num_threads);
  GetTaskType(params);
  
  GetBool(params, "predict_leaf_index", &predict_leaf_index);

  GetBoostingType(params);
  GetObjectiveType(params);
  GetMetricType(params);

  // construct boosting configs
  if (boosting_type == BoostingType::kGBDT) {
    boosting_config = new GBDTConfig();
  }

  // sub-config setup
  network_config.Set(params);
  io_config.Set(params);

  boosting_config->Set(params);
  objective_config.Set(params);
  metric_config.Set(params);
  // check for conflicts
  CheckParamConflict();

  if (io_config.verbosity == 1) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Info);
  }
  else if (io_config.verbosity == 0) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Error);
  }
  else if (io_config.verbosity >= 2) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Debug);
  }
  else {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
  }
}

void OverallConfig::GetBoostingType(const std::unordered_map<std::string, std::string>& params) {
  std::string value;
  if (GetString(params, "boosting_type", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == std::string("gbdt") || value == std::string("gbrt")) {
      boosting_type = BoostingType::kGBDT;
    } else {
      Log::Fatal("Boosting type %s error", value.c_str());
    }
  }
}

void OverallConfig::GetObjectiveType(const std::unordered_map<std::string, std::string>& params) {
  std::string value;
  if (GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    objective_type = value;
  }
}

void OverallConfig::GetMetricType(const std::unordered_map<std::string, std::string>& params) {
  std::string value;
  if (GetString(params, "metric", &value)) {
    // clear old metrics
    metric_types.clear();
    // to lower
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    // split
    std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
    // remove dumplicate
    std::unordered_map<std::string, int> metric_maps;
    for (auto& metric : metrics) {
      std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);
      if (metric_maps.count(metric) <= 0) {
        metric_maps[metric] = 1;
      }
    }
    for (auto& pair : metric_maps) {
      std::string sub_metric_str = pair.first;
      metric_types.push_back(sub_metric_str);
    }
  }
}


void OverallConfig::GetTaskType(const std::unordered_map<std::string, std::string>& params) {
  std::string value;
  if (GetString(params, "task", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == std::string("train") || value == std::string("training")) {
      task_type = TaskType::kTrain;
    } else if (value == std::string("predict") || value == std::string("prediction")
      || value == std::string("test")) {
      task_type = TaskType::kPredict;
    } else {
      Log::Fatal("Task type error");
    }
  }
}

void OverallConfig::CheckParamConflict() {
  GBDTConfig* gbdt_config = dynamic_cast<GBDTConfig*>(boosting_config);  
    
  // check if objective_type, metric_type, and num_class match
  bool objective_type_multiclass = (objective_type == std::string("multiclass"));
  int num_class_check = gbdt_config->num_class;
  if (objective_type_multiclass){
      if (num_class_check <= 1){
          Log::Fatal("You should specify number of class(>=2) for multiclass training.");
      }
  }
  else {
      if (task_type == TaskType::kTrain && num_class_check != 1){
          Log::Fatal("Number of class must be 1 for non-multiclass training.");
      }      
  }
  for (std::string metric_type : metric_types){
        bool metric_type_multiclass = ( metric_type == std::string("multi_logloss") || metric_type == std::string("multi_error"));
        if ((objective_type_multiclass && !metric_type_multiclass) 
            || (!objective_type_multiclass && metric_type_multiclass)){
            Log::Fatal("Objective and metrics don't match.");    
        }
  }  

  if (network_config.num_machines > 1) {
    is_parallel = true;
  } else {
    is_parallel = false;
    gbdt_config->tree_learner_type = TreeLearnerType::kSerialTreeLearner;
  }

  if (gbdt_config->tree_learner_type == TreeLearnerType::kSerialTreeLearner) {
    is_parallel = false;
    network_config.num_machines = 1;
  }

  if (gbdt_config->tree_learner_type == TreeLearnerType::kSerialTreeLearner ||
    gbdt_config->tree_learner_type == TreeLearnerType::kFeatureParallelTreelearner) {
    is_parallel_find_bin = false;
  } else if (gbdt_config->tree_learner_type == TreeLearnerType::kDataParallelTreeLearner) {
    is_parallel_find_bin = true;
    if (gbdt_config->tree_config.histogram_pool_size >= 0) {
      Log::Error("Histogram LRU queue was enabled (histogram_pool_size=%f). Will disable this for reducing communication cost."
                 , gbdt_config->tree_config.histogram_pool_size);
      // Change pool size to -1(not limit) when using data parallel for reducing communication cost
      gbdt_config->tree_config.histogram_pool_size = -1;
    }

  }
}

void IOConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "max_bin", &max_bin);
  CHECK(max_bin > 0);
  GetInt(params, "data_random_seed", &data_random_seed);

  if (!GetString(params, "data", &data_filename)) {
    Log::Fatal("No training/prediction data, application quit");
  }
  GetInt(params, "verbose", &verbosity);
  GetInt(params, "num_model_predict", &num_model_predict);
  GetInt(params, "bin_construct_sample_cnt", &bin_construct_sample_cnt);
  GetBool(params, "is_pre_partition", &is_pre_partition);
  GetBool(params, "is_enable_sparse", &is_enable_sparse);
  GetBool(params, "use_two_round_loading", &use_two_round_loading);
  GetBool(params, "is_save_binary_file", &is_save_binary_file);
  GetBool(params, "enable_load_from_binary_file", &enable_load_from_binary_file);
  GetBool(params, "is_sigmoid", &is_sigmoid);
  GetString(params, "output_model", &output_model);
  GetString(params, "input_model", &input_model);
  GetString(params, "output_result", &output_result);
  GetString(params, "input_init_score", &input_init_score);
  std::string tmp_str = "";
  if (GetString(params, "valid_data", &tmp_str)) {
    valid_data_filenames = Common::Split(tmp_str.c_str(), ',');
  }
  GetBool(params, "has_header", &has_header);
  GetString(params, "label_column", &label_column);
  GetString(params, "weight_column", &weight_column);
  GetString(params, "group_column", &group_column);
  GetString(params, "ignore_column", &ignore_column);
}


void ObjectiveConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetBool(params, "is_unbalance", &is_unbalance);
  GetFloat(params, "sigmoid", &sigmoid);
  GetInt(params, "max_position", &max_position);
  CHECK(max_position > 0);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class >= 1);
  std::string tmp_str = "";
  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToFloatArray(tmp_str, ',');
  } else {
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    const int max_label = 31;
    label_gain.push_back(0.0f);
    for (int i = 1; i < max_label; ++i) {
      label_gain.push_back(static_cast<float>((1 << i) - 1));
    }
  }
}


void MetricConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetFloat(params, "sigmoid", &sigmoid);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class >= 1);
  std::string tmp_str = "";
  if (GetString(params, "label_gain", &tmp_str)) {
    label_gain = Common::StringToFloatArray(tmp_str, ',');
  } else {
    // label_gain = 2^i - 1, may overflow, so we use 31 here
    const int max_label = 31;
    label_gain.push_back(0.0f);
    for (int i = 1; i < max_label; ++i) {
      label_gain.push_back(static_cast<float>((1 << i) - 1));
    }
  }
  if (GetString(params, "ndcg_eval_at", &tmp_str)) {
    eval_at = Common::StringToIntArray(tmp_str, ',');
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
}


void TreeConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "min_data_in_leaf", &min_data_in_leaf);
  GetFloat(params, "min_sum_hessian_in_leaf", &min_sum_hessian_in_leaf);
  CHECK(min_sum_hessian_in_leaf > 1.0f || min_data_in_leaf > 0);
  GetInt(params, "num_leaves", &num_leaves);
  CHECK(num_leaves > 1);
  GetInt(params, "feature_fraction_seed", &feature_fraction_seed);
  GetFloat(params, "feature_fraction", &feature_fraction);
  CHECK(feature_fraction > 0.0f && feature_fraction <= 1.0f);
  GetFloat(params, "histogram_pool_size", &histogram_pool_size);
  GetInt(params, "max_depth", &max_depth);
  CHECK(max_depth > 1 || max_depth < 0);
}


void BoostingConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  GetInt(params, "num_iterations", &num_iterations);
  CHECK(num_iterations >= 0);
  GetInt(params, "bagging_seed", &bagging_seed);
  GetInt(params, "bagging_freq", &bagging_freq);
  CHECK(bagging_freq >= 0);
  GetFloat(params, "bagging_fraction", &bagging_fraction);
  CHECK(bagging_fraction > 0.0f && bagging_fraction <= 1.0f);
  GetFloat(params, "learning_rate", &learning_rate);
  CHECK(learning_rate > 0.0f);
  GetInt(params, "early_stopping_round", &early_stopping_round);
  CHECK(early_stopping_round >= 0);
  GetInt(params, "metric_freq", &output_freq);
  CHECK(output_freq >= 0);
  GetBool(params, "is_training_metric", &is_provide_training_metric);
  GetInt(params, "num_class", &num_class);
  CHECK(num_class >= 1);
}

void GBDTConfig::GetTreeLearnerType(const std::unordered_map<std::string, std::string>& params) {
  std::string value;
  if (GetString(params, "tree_learner", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == std::string("serial")) {
      tree_learner_type = TreeLearnerType::kSerialTreeLearner;
    } else if (value == std::string("feature") || value == std::string("feature_parallel")) {
      tree_learner_type = TreeLearnerType::kFeatureParallelTreelearner;
    } else if (value == std::string("data") || value == std::string("data_parallel")) {
      tree_learner_type = TreeLearnerType::kDataParallelTreeLearner;
    }
    else {
      Log::Fatal("Tree learner type error");
    }
  }
}

void GBDTConfig::Set(const std::unordered_map<std::string, std::string>& params) {
  BoostingConfig::Set(params);
  GetTreeLearnerType(params);
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
}

}  // namespace LightGBM
