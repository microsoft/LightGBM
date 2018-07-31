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

void GetBoostingType(const std::unordered_map<std::string, std::string>& params, std::string* boosting) {
  std::string value;
  if (Config::GetString(params, "boosting", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("gbdt") || value == std::string("gbrt")) {
      *boosting = "gbdt";
    } else if (value == std::string("dart")) {
      *boosting = "dart";
    } else if (value == std::string("goss")) {
      *boosting = "goss";
    } else if (value == std::string("rf") || value == std::string("random_forest")) {
      *boosting = "rf";
    } else {
      Log::Fatal("Unknown boosting type %s", value.c_str());
    }
  }
}

void GetObjectiveType(const std::unordered_map<std::string, std::string>& params, std::string* objective) {
  std::string value;
  if (Config::GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    *objective = value;
  }
}

void GetMetricType(const std::unordered_map<std::string, std::string>& params, std::vector<std::string>* metric) {
  std::string value;
  if (Config::GetString(params, "metric", &value)) {
    // clear old metrics
    metric->clear();
    // to lower
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    // split
    std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
    // remove duplicate
    std::unordered_set<std::string> metric_sets;
    for (auto& met : metrics) {
      std::transform(met.begin(), met.end(), met.begin(), Common::tolower);
      if (metric_sets.count(met) <= 0) {
        metric_sets.insert(met);
      }
    }
    for (auto& met : metric_sets) {
      metric->push_back(met);
    }
    metric->shrink_to_fit();
  }
  // add names of objective function if not providing metric
  if (metric->empty() && value.size() == 0) {
    if (Config::GetString(params, "objective", &value)) {
      std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
      metric->push_back(value);
    }
  }
}

void GetTaskType(const std::unordered_map<std::string, std::string>& params, TaskType* task) {
  std::string value;
  if (Config::GetString(params, "task", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("train") || value == std::string("training")) {
      *task = TaskType::kTrain;
    } else if (value == std::string("predict") || value == std::string("prediction")
               || value == std::string("test")) {
      *task = TaskType::kPredict;
    } else if (value == std::string("convert_model")) {
      *task = TaskType::kConvertModel;
    } else if (value == std::string("refit") || value == std::string("refit_tree")) {
      *task = TaskType::KRefitTree;
    } else {
      Log::Fatal("Unknown task type %s", value.c_str());
    }
  }
}

void GetDeviceType(const std::unordered_map<std::string, std::string>& params, std::string* device_type) {
  std::string value;
  if (Config::GetString(params, "device_type", &value)) {
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

void GetTreeLearnerType(const std::unordered_map<std::string, std::string>& params, std::string* tree_learner) {
  std::string value;
  if (Config::GetString(params, "tree_learner", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("serial")) {
      *tree_learner = "serial";
    } else if (value == std::string("feature") || value == std::string("feature_parallel")) {
      *tree_learner = "feature";
    } else if (value == std::string("data") || value == std::string("data_parallel")) {
      *tree_learner = "data";
    } else if (value == std::string("voting") || value == std::string("voting_parallel")) {
      *tree_learner = "voting";
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

  GetTaskType(params, &task);
  GetBoostingType(params, &boosting);
  GetMetricType(params, &metric);
  GetObjectiveType(params, &objective);
  GetDeviceType(params, &device_type);
  GetTreeLearnerType(params, &tree_learner);

  GetMembersFromString(params);

  if (valid_data_initscores.size() == 0 && valid.size() > 0) {
    valid_data_initscores = std::vector<std::string>(valid.size(), "");
  }
  CHECK(valid.size() == valid_data_initscores.size());

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

bool CheckMultiClassObjective(const std::string& objective) {
  return (objective == std::string("multiclass")
          || objective == std::string("multiclassova")
          || objective == std::string("softmax")
          || objective == std::string("multiclass_ova")
          || objective == std::string("ova")
          || objective == std::string("ovr"));
}

void Config::CheckParamConflict() {
  // check if objective, metric, and num_class match
  int num_class_check = num_class;
  bool objective_custom = objective == std::string("none") || objective == std::string("null") 
                                       || objective == std::string("custom") || objective == std::string("na");
  bool objective_type_multiclass = CheckMultiClassObjective(objective) || (objective_custom && num_class_check > 1);
  
  if (objective_type_multiclass) {
    if (num_class_check <= 1) {
      Log::Fatal("Number of classes should be specified and greater than 1 for multiclass training");
    }
  } else {
    if (task == TaskType::kTrain && num_class_check != 1) {
      Log::Fatal("Number of classes must be 1 for non-multiclass training");
    }
  }
  for (std::string metric_type : metric) {
    bool metric_custom_or_none = metric_type == std::string("none") || metric_type == std::string("null") 
                                 || metric_type == std::string("custom") || metric_type == std::string("na");
    bool metric_type_multiclass = (CheckMultiClassObjective(metric_type)
                                   || metric_type == std::string("multi_logloss")
                                   || metric_type == std::string("multi_error")
                                   || (metric_custom_or_none && num_class_check > 1));
    if ((objective_type_multiclass && !metric_type_multiclass)
        || (!objective_type_multiclass && metric_type_multiclass)) {
      Log::Fatal("Multiclass objective and metrics don't match");
    }
  }

  if (num_machines > 1) {
    is_parallel = true;
  } else {
    is_parallel = false;
    tree_learner = "serial";
  }

  bool is_single_tree_learner = tree_learner == std::string("serial");

  if (is_single_tree_learner) {
    is_parallel = false;
    num_machines = 1;
  }

  if (is_single_tree_learner || tree_learner == std::string("feature")) {
    is_parallel_find_bin = false;
  } else if (tree_learner == std::string("data")
             || tree_learner == std::string("voting")) {
    is_parallel_find_bin = true;
    if (histogram_pool_size >= 0
        && tree_learner == std::string("data")) {
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

std::string Config::ToString() const {
  std::stringstream str_buf;
  str_buf << "[boosting: " << boosting << "]\n";
  str_buf << "[objective: " << objective << "]\n";
  str_buf << "[metric: " << Common::Join(metric, ",") << "]\n";
  str_buf << "[tree_learner: " << tree_learner << "]\n";
  str_buf << "[device_type: " << device_type << "]\n";
  str_buf << SaveMembersToString();
  return str_buf.str();
}

}  // namespace LightGBM
