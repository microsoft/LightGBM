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

void Config::KV2Map(std::unordered_map<std::string, std::string>& params, const char* kv) {
  std::vector<std::string> tmp_strs = Common::Split(kv, '=');
  if (tmp_strs.size() == 2) {
    std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
    std::string value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
    if (key.size() > 0) {
      auto value_search = params.find(key);
      if (value_search == params.end()) { // not set
        params.emplace(key, value);
      } else {
        Log::Warning("%s is set=%s, %s=%s will be ignored. Current value: %s=%s",
          key.c_str(), value_search->second.c_str(), key.c_str(), value.c_str(),
          key.c_str(), value_search->second.c_str());
      }
    }
  } else {
    Log::Warning("Unknown parameter %s", kv);
  }
}

std::unordered_map<std::string, std::string> Config::Str2Map(const char* parameters) {
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
  if (Config::GetString(params, "boosting_type", &value)) {
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
  if (Config::GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    *objective_type = value;
  }
}

void GetMetricType(const std::unordered_map<std::string, std::string>& params, std::vector<std::string>* metric_types) {
  std::string value;
  if (Config::GetString(params, "metric", &value)) {
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
  // add names of objective function if not providing metric
  if (metric_types->empty() && value.size() == 0) {
    if (Config::GetString(params, "objective", &value)) {
      std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
      metric_types->push_back(value);
    }
  }
}

void GetTaskType(const std::unordered_map<std::string, std::string>& params, TaskType* task_type) {
  std::string value;
  if (Config::GetString(params, "task", &value)) {
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
  if (Config::GetString(params, "device", &value)) {
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
  if (Config::GetString(params, "tree_learner", &value)) {
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

void Config::Set(const std::unordered_map<std::string, std::string>& params) {

  // generate seeds by seed.
  if (GetInt(params, "seed", &seed)) {
    Random rand(seed);
    int int_max = std::numeric_limits<short>::max();
    data_random_seed = static_cast<int>(rand.NextShort(0, int_max));
    bagging_seed = static_cast<int>(rand.NextShort(0, int_max));
    drop_seed = static_cast<int>(rand.NextShort(0, int_max));
    feature_fraction_seed = static_cast<int>(rand.NextShort(0, int_max));
  }

  GetTaskType(params, &task_type);
  GetBoostingType(params, &boosting_type);
  GetMetricType(params, &metric_types);
  GetObjectiveType(params, &objective_type);
  GetDeviceType(params, &device_type);
  GetTreeLearnerType(params, &tree_learner_type);

  GetMembersFromString(params);

  // check for conflicts
  CheckParamConflict();

  if (verbosity == 1) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Info);
  } else if (verbosity == 0) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Warning);
  } else if (verbosity >= 2) {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Debug);
  } else {
    LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
  }
}

bool CheckMultiClassObjective(const std::string& objective_type) {
  return (objective_type == std::string("multiclass")
          || objective_type == std::string("multiclassova")
          || objective_type == std::string("softmax")
          || objective_type == std::string("multiclass_ova")
          || objective_type == std::string("ova")
          || objective_type == std::string("ovr"));
}

void Config::CheckParamConflict() {
  // check if objective_type, metric_type, and num_class match
  int num_class_check = num_class;
  bool objective_custom = objective_type == std::string("none") || objective_type == std::string("null") || objective_type == std::string("custom");
  bool objective_type_multiclass = CheckMultiClassObjective(objective_type) || (objective_custom && num_class_check > 1);
  
  if (objective_type_multiclass) {
    if (num_class_check <= 1) {
      Log::Fatal("Number of classes should be specified and greater than 1 for multiclass training");
    }
  } else {
    if (task_type == TaskType::kTrain && num_class_check != 1) {
      Log::Fatal("Number of classes must be 1 for non-multiclass training");
    }
  }
  if (is_provide_training_metric || !valid_data_filenames.empty()) {
    for (std::string metric_type : metric_types) {
      bool metric_type_multiclass = (CheckMultiClassObjective(metric_type) 
                                     || metric_type == std::string("multi_logloss")
                                     || metric_type == std::string("multi_error"));
      if ((objective_type_multiclass && !metric_type_multiclass)
        || (!objective_type_multiclass && metric_type_multiclass)) {
        Log::Fatal("Objective and metrics don't match");
      }
    }
  }

  if (num_machines > 1) {
    is_parallel = true;
  } else {
    is_parallel = false;
    tree_learner_type = "serial";
  }

  bool is_single_tree_learner = tree_learner_type == std::string("serial");

  if (is_single_tree_learner) {
    is_parallel = false;
    num_machines = 1;
  }

  if (is_single_tree_learner || tree_learner_type == std::string("feature")) {
    is_parallel_find_bin = false;
  } else if (tree_learner_type == std::string("data")
             || tree_learner_type == std::string("voting")) {
    is_parallel_find_bin = true;
    if (histogram_pool_size >= 0
        && tree_learner_type == std::string("data")) {
      Log::Warning("Histogram LRU queue was enabled (histogram_pool_size=%f).\n"
                   "Will disable this to reduce communication costs",
                   histogram_pool_size);
      // Change pool size to -1 (no limit) when using data parallel to reduce communication costs
      histogram_pool_size = -1;
    }
  }
  // Check max_depth and num_leaves
  if (max_depth > 0) {
    int full_num_leaves = static_cast<int>(std::pow(2, max_depth));
    if (full_num_leaves > num_leaves 
        && num_leaves == kDefaultNumLeaves) {
      Log::Warning("Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves");
    }
  }
}

}  // namespace LightGBM
