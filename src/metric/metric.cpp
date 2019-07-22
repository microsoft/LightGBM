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
  if (type == std::string("l2")) {
    return new L2Metric(config);
  } else if (type == std::string("rmse")) {
    return new RMSEMetric(config);
  } else if (type == std::string("l1")) {
    return new L1Metric(config);
  } else if (type == std::string("quantile")) {
    return new QuantileMetric(config);
  } else if (type == std::string("huber")) {
    return new HuberLossMetric(config);
  } else if (type == std::string("fair")) {
    return new FairLossMetric(config);
  } else if (type == std::string("poisson")) {
    return new PoissonMetric(config);
  } else if (type == std::string("binary_logloss")) {
    return new BinaryLoglossMetric(config);
  } else if (type == std::string("binary_error")) {
    return new BinaryErrorMetric(config);
  } else if (type == std::string("auc")) {
    return new AUCMetric(config);
  } else if (type == std::string("ndcg")) {
    return new NDCGMetric(config);
  } else if (type == std::string("map")) {
    return new MapMetric(config);
  } else if (type == std::string("multi_logloss")) {
    return new MultiSoftmaxLoglossMetric(config);
  } else if (type == std::string("multi_error")) {
    return new MultiErrorMetric(config);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropyMetric(config);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambdaMetric(config);
  } else if (type == std::string("kullback_leibler")) {
    return new KullbackLeiblerDivergence(config);
  } else if (type == std::string("mape")) {
    return new MAPEMetric(config);
  } else if (type == std::string("gamma")) {
    return new GammaMetric(config);
  } else if (type == std::string("gamma_deviance")) {
    return new GammaDevianceMetric(config);
  } else if (type == std::string("tweedie")) {
    return new TweedieMetric(config);
  }
  return nullptr;
}

}  // namespace LightGBM
