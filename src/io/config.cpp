/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/config.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/random.h>

#include <limits>

namespace LightGBM {

void Config::KV2Map(std::unordered_map<std::string, std::string>* params, const char* kv) {
  std::vector<std::string> tmp_strs = Common::Split(kv, '=');
  if (tmp_strs.size() == 2 || tmp_strs.size() == 1) {
    std::string key = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[0]));
    std::string value = "";
    if (tmp_strs.size() == 2) {
      value = Common::RemoveQuotationSymbol(Common::Trim(tmp_strs[1]));
    }
    if (!Common::CheckASCII(key) || !Common::CheckASCII(value)) {
      Log::Fatal("Do not support non-ascii characters in config.");
    }
    if (key.size() > 0) {
      auto value_search = params->find(key);
      if (value_search == params->end()) {  // not set
        params->emplace(key, value);
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
    KV2Map(&params, Common::Trim(arg).c_str());
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

std::string ParseObjectiveAlias(const std::string& type) {
  if (type == std::string("regression") || type == std::string("regression_l2")
    || type == std::string("mean_squared_error") || type == std::string("mse") || type == std::string("l2")
    || type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return "regression";
  } else if (type == std::string("regression_l1") || type == std::string("mean_absolute_error")
    || type == std::string("l1") || type == std::string("mae")) {
    return "regression_l1";
  } else if (type == std::string("multiclass") || type == std::string("softmax")) {
    return "multiclass";
  } else if (type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return "multiclassova";
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return "cross_entropy";
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return "cross_entropy_lambda";
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return "mape";
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
    return "custom";
  }
  return type;
}

std::string ParseMetricAlias(const std::string& type) {
  if (type == std::string("regression") || type == std::string("regression_l2") || type == std::string("l2") || type == std::string("mean_squared_error") || type == std::string("mse")) {
    return "l2";
  } else if (type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return "rmse";
  } else if (type == std::string("regression_l1") || type == std::string("l1") || type == std::string("mean_absolute_error") || type == std::string("mae")) {
    return "l1";
  } else if (type == std::string("binary_logloss") || type == std::string("binary")) {
    return "binary_logloss";
  } else if (type == std::string("ndcg") || type == std::string("lambdarank")) {
    return "ndcg";
  } else if (type == std::string("map") || type == std::string("mean_average_precision")) {
    return "map";
  } else if (type == std::string("multi_logloss") || type == std::string("multiclass") || type == std::string("softmax") || type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return "multi_logloss";
  } else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return "cross_entropy";
  } else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return "cross_entropy_lambda";
  } else if (type == std::string("kldiv") || type == std::string("kullback_leibler")) {
    return "kullback_leibler";
  } else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return "mape";
  } else if (type == std::string("none") || type == std::string("null") || type == std::string("custom") || type == std::string("na")) {
    return "custom";
  }
  return type;
}

void ParseMetrics(const std::string& value, std::vector<std::string>* out_metric) {
  std::unordered_set<std::string> metric_sets;
  out_metric->clear();
  std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
  for (auto& met : metrics) {
    auto type = ParseMetricAlias(met);
    if (metric_sets.count(type) <= 0) {
      out_metric->push_back(type);
      metric_sets.insert(type);
    }
  }
}

void GetObjectiveType(const std::unordered_map<std::string, std::string>& params, std::string* objective) {
  std::string value;
  if (Config::GetString(params, "objective", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    *objective = ParseObjectiveAlias(value);
  }
}

void GetMetricType(const std::unordered_map<std::string, std::string>& params, std::vector<std::string>* metric) {
  std::string value;
  if (Config::GetString(params, "metric", &value)) {
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    ParseMetrics(value, metric);
  }
  // add names of objective function if not providing metric
  if (metric->empty() && value.size() == 0) {
    if (Config::GetString(params, "objective", &value)) {
      std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
      ParseMetrics(value, metric);
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
    int int_max = std::numeric_limits<int16_t>::max();
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

  // sort eval_at
  std::sort(eval_at.begin(), eval_at.end());

  if (valid_data_initscores.size() == 0 && valid.size() > 0) {
    valid_data_initscores = std::vector<std::string>(valid.size(), "");
  }
  CHECK(valid.size() == valid_data_initscores.size());

  if (valid_data_initscores.empty()) {
    std::vector<std::string> new_valid;
    for (size_t i = 0; i < valid.size(); ++i) {
      if (valid[i] != data) {
        // Only push the non-training data
        new_valid.push_back(valid[i]);
      } else {
        is_provide_training_metric = true;
      }
    }
    valid = new_valid;
  }

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
  return (objective == std::string("multiclass") || objective == std::string("multiclassova"));
}

void Config::CheckParamConflict() {
  // check if objective, metric, and num_class match
  int num_class_check = num_class;
  bool objective_type_multiclass = CheckMultiClassObjective(objective) || (objective == std::string("custom") && num_class_check > 1);

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
    bool metric_type_multiclass = (CheckMultiClassObjective(metric_type)
                                   || metric_type == std::string("multi_logloss")
                                   || metric_type == std::string("multi_error")
                                   || (metric_type == std::string("custom") && num_class_check > 1));
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
    double full_num_leaves = std::pow(2, max_depth);
    if (full_num_leaves > num_leaves
        && num_leaves == kDefaultNumLeaves) {
      Log::Warning("Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves");
    }

    if (full_num_leaves < num_leaves) {
      // Fits in an int, and is more restrictive than the current num_leaves
      num_leaves = static_cast<int>(full_num_leaves);
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
