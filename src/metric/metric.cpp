#include <LightGBM/metric.h>
#include "regression_metric.hpp"
#include "binary_metric.hpp"
#include "rank_metric.hpp"
#include "multiclass_metric.hpp"

namespace LightGBM {

Metric* Metric::CreateMetric(const std::string& type, const MetricConfig& config) {
  if (type == "l2") {
    return new L2Metric(config);
  } else if (type == "l1") {
    return new L1Metric(config);
  } else if (type == "binary_logloss") {
    return new BinaryLoglossMetric(config);
  } else if (type == "binary_error") {
    return new BinaryErrorMetric(config);
  } else if (type == "auc") {
    return new AUCMetric(config);
  } else if (type == "ndcg") {
    return new NDCGMetric(config);
  } else if (type == "multi_logloss"){
    return new MultiLoglossMetric(config);
  } else if (type == "multi_error"){
    return new MultiErrorMetric(config);
  }
  return nullptr;
}

}  // namespace LightGBM
