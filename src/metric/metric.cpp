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

#include "cuda/cuda_binary_metric.hpp"
#include "cuda/cuda_regression_metric.hpp"
#include "cuda/cuda_multiclass_metric.hpp"
#include "cuda/cuda_xentropy_metric.hpp"

namespace LightGBM {

Metric* Metric::CreateMetric(const std::string& type, const Config& config) {
  if (config.device_type == std::string("cuda")) {
    if (type == std::string("l2")) {
      return new CUDAL2Metric(config);
    } else if (type == std::string("rmse")) {
      return new CUDARMSEMetric(config);
    } else if (type == std::string("l1")) {
      return new CUDAL1Metric(config);
    } else if (type == std::string("quantile")) {
      return new CUDAQuantileMetric(config);
    } else if (type == std::string("huber")) {
      return new CUDAHuberLossMetric(config);
    } else if (type == std::string("fair")) {
      return new CUDAFairLossMetric(config);
    } else if (type == std::string("poisson")) {
      return new CUDAPoissonMetric(config);
    } else if (type == std::string("binary_logloss")) {
      return new CUDABinaryLoglossMetric(config);
    } else if (type == std::string("binary_error")) {
      return new CUDABinaryErrorMetric(config);
    } else if (type == std::string("auc")) {
      return new CUDAAUCMetric(config);
    } else if (type == std::string("average_precision")) {
      return new CUDAAveragePrecisionMetric(config);
    } else if (type == std::string("multi_logloss")) {
      return new CUDAMultiSoftmaxLoglossMetric(config);
    } else if (type == std::string("multi_error")) {
      return new CUDAMultiErrorMetric(config);
    } else if (type == std::string("cross_entropy")) {
      return new CUDACrossEntropyMetric(config);
    } else if (type == std::string("cross_entropy_lambda")) {
      return new CUDACrossEntropyLambdaMetric(config);
    } else if (type == std::string("kullback_leibler")) {
      return new CUDAKullbackLeiblerDivergence(config);
    } else if (type == std::string("mape")) {
      return new CUDAMAPEMetric(config);
    } else if (type == std::string("gamma")) {
      return new CUDAGammaMetric(config);
    } else if (type == std::string("gamma_deviance")) {
      return new CUDAGammaDevianceMetric(config);
    } else if (type == std::string("tweedie")) {
      return new CUDATweedieMetric(config);
    }
  } else {
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
    } else if (type == std::string("average_precision")) {
      return new AveragePrecisionMetric(config);
    } else if (type == std::string("auc_mu")) {
      return new AucMuMetric(config);
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
  }
  Log::Fatal("Unknown metric type name: %s", type.c_str());
  return nullptr;
}

}  // namespace LightGBM
