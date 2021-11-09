#include <LightGBM/sample_strategy.h>
#include "goss1.hpp"

namespace LightGBM {

SampleStrategy* SampleStrategy::CreateSampleStrategy(const Config* config, const Dataset* train_data, int num_tree_per_iteration) {
  bool use_goss_as_boosting = config->boosting == std::string("goss");
  bool use_goss_as_strategy = config->data_sample_strategy == std::string("goss");
  if (use_goss_as_boosting || use_goss_as_strategy) {
      return new GOSS1(config, train_data, num_tree_per_iteration);
  } else if (config->data_sample_strategy == std::string("bagging")) {
      return nullptr;
  }
}

} // namespace LightGBM