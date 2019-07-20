/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/metric.h>

#include "binary_metric.hpp"
#include "map_metric.hpp"
#include "multiclass_metric.hpp"
#include "rank_metric.hpp"
#include "regression_metric.hpp"
#include "xentropy_metric.hpp"

namespace LightGBM {

Metric* Metric::CreateMetric(const std::string& type, const Config& config) {
  if (type == "l2") {
    return new L2Metric(config);
  }
  else if (type == std::string("rmse")) {
    return new RMSEMetric(config);
  }
  else if (type == std::string("l1")) {
    return new L1Metric(config);
  }
  else if (type == std::string("quantile")) {
    return new QuantileMetric(config);
  }
  else if (type == std::string("huber")) {
    return new HuberLossMetric(config);
  }
  else if (type == std::string("fair")) {
    return new FairLossMetric(config);
  }
  else if (type == std::string("poisson")) {
    return new PoissonMetric(config);
  }
  else if (type == std::string("binary_logloss")) {
    return new BinaryLoglossMetric(config);
  }
  else if (type == std::string("binary_error")) {
    return new BinaryErrorMetric(config);
  }
  else if (type == std::string("auc")) {
    return new AUCMetric(config);
  }
  else if (type == std::string("ndcg")) {
    return new NDCGMetric(config);
  }
  else if (type == std::string("map")) {
    return new MapMetric(config);
  }
  else if (type == std::string("multi_logloss")) {
    return new MultiSoftmaxLoglossMetric(config);
  }
  else if (type == std::string("multi_error")) {
    return new MultiErrorMetric(config);
  }
  else if (type == std::string("cross_entropy")) {
    return new CrossEntropyMetric(config);
  }
  else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambdaMetric(config);
  }
  else if (type == std::string("kullback_leibler")) {
    return new KullbackLeiblerDivergence(config);
  }
  else if (type == std::string("mape")) {
    return new MAPEMetric(config);
  }
  else if (type == std::string("gamma")) {
    return new GammaMetric(config);
  }
  else if (type == std::string("gamma_deviance")) {
    return new GammaDevianceMetric(config);
  }
  else if (type == std::string("tweedie")) {
    return new TweedieMetric(config);
  }
  return nullptr;
}

std::string GetMetricType(const std::string& type) {
  if (type == std::string("regression") || type == std::string("regression_l2") || type == std::string("l2") || type == std::string("mean_squared_error") || type == std::string("mse")) {
    return "l2";
  }
  else if (type == std::string("l2_root") || type == std::string("root_mean_squared_error") || type == std::string("rmse")) {
    return "rmse";
  }
  else if (type == std::string("regression_l1") || type == std::string("l1") || type == std::string("mean_absolute_error") || type == std::string("mae")) {
    return "l1";
  }
  else if (type == std::string("binary_logloss") || type == std::string("binary")) {
    return "binary_logloss";
  }
  else if (type == std::string("ndcg") || type == std::string("lambdarank")) {
    return "ndcg";
  }
  else if (type == std::string("map") || type == std::string("mean_average_precision")) {
    return "map";
  }
  else if (type == std::string("multi_logloss") || type == std::string("multiclass") || type == std::string("softmax") || type == std::string("multiclassova") || type == std::string("multiclass_ova") || type == std::string("ova") || type == std::string("ovr")) {
    return "multi_logloss";
  }
  else if (type == std::string("xentropy") || type == std::string("cross_entropy")) {
    return "cross_entropy";
  }
  else if (type == std::string("xentlambda") || type == std::string("cross_entropy_lambda")) {
    return "cross_entropy_lambda";
  }
  else if (type == std::string("kldiv") || type == std::string("kullback_leibler")) {
    return "kullback_leibler";
  }
  else if (type == std::string("mean_absolute_percentage_error") || type == std::string("mape")) {
    return "mape";
  }
  return type;
}

void Metric::ParseMetrics(const std::string& value, std::vector<std::string>* metric) {
  std::unordered_set<std::string> metric_sets;
  metric->clear();
  std::vector<std::string> metrics = Common::Split(value.c_str(), ',');
  for (auto& met : metrics) {
    auto type = GetMetricType(met);
    if (metric_sets.count(type) <= 0) {
      metric->push_back(type);
      metric_sets.insert(type);
    }
  }
}

}  // namespace LightGBM
