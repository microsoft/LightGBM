#include <LightGBM/sample_strategy.h>
#include "goss.hpp"
#include "bagging.hpp"

namespace LightGBM {

SampleStrategy* SampleStrategy::CreateSampleStrategy(const Config* config, const Dataset* train_data, const ObjectiveFunction* objective_function, int num_tree_per_iteration) {
  bool use_goss_as_boosting = config->boosting == std::string("goss");
  bool use_goss_as_strategy = config->data_sample_strategy == std::string("goss");
  if (use_goss_as_boosting) {
    Log::Warning("Setting goss as `boosting_type` is NOT suggested. Please set `data_sample_strategy = goss` in your config file.");
    if (use_goss_as_strategy) {
      Log::Warning("Both `boosting_type` and `data_sample_strategy` are set as GOSS. Only one time of sampling will be conducted. Please check and modify your config file.");
    }
  }
  if (use_goss_as_boosting || use_goss_as_strategy) {
    return new GOSS(config, train_data, num_tree_per_iteration);
  } else {
    return new BAGGING(config, train_data, objective_function, num_tree_per_iteration);
  }
}

} // namespace LightGBM