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
  #ifdef USE_CUDA_EXP
  if (config.device_type == std::string("cuda_exp")) {
    if (type == std::string("l2")) {
      Log::Warning("Metric l2 is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new L2Metric(config);
    } else if (type == std::string("rmse")) {
      Log::Warning("Metric rmse is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new RMSEMetric(config);
    } else if (type == std::string("l1")) {
      Log::Warning("Metric l1 is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new L1Metric(config);
    } else if (type == std::string("quantile")) {
      Log::Warning("Metric quantile is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new QuantileMetric(config);
    } else if (type == std::string("huber")) {
      Log::Warning("Metric huber is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new HuberLossMetric(config);
    } else if (type == std::string("fair")) {
      Log::Warning("Metric fair is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new FairLossMetric(config);
    } else if (type == std::string("poisson")) {
      Log::Warning("Metric poisson is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new PoissonMetric(config);
    } else if (type == std::string("binary_logloss")) {
      Log::Warning("Metric binary_logloss is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new BinaryLoglossMetric(config);
    } else if (type == std::string("binary_error")) {
      Log::Warning("Metric binary_error is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new BinaryErrorMetric(config);
    } else if (type == std::string("auc")) {
      Log::Warning("Metric auc is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new AUCMetric(config);
    } else if (type == std::string("average_precision")) {
      Log::Warning("Metric average_precision is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new AveragePrecisionMetric(config);
    } else if (type == std::string("auc_mu")) {
      Log::Warning("Metric auc_mu is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new AucMuMetric(config);
    } else if (type == std::string("ndcg")) {
      Log::Warning("Metric ndcg is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new NDCGMetric(config);
    } else if (type == std::string("map")) {
      Log::Warning("Metric map is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new MapMetric(config);
    } else if (type == std::string("multi_logloss")) {
      Log::Warning("Metric multi_logloss is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new MultiSoftmaxLoglossMetric(config);
    } else if (type == std::string("multi_error")) {
      Log::Warning("Metric multi_error is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new MultiErrorMetric(config);
    } else if (type == std::string("cross_entropy")) {
      Log::Warning("Metric cross_entropy is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new CrossEntropyMetric(config);
    } else if (type == std::string("cross_entropy_lambda")) {
      Log::Warning("Metric cross_entropy_lambda is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new CrossEntropyLambdaMetric(config);
    } else if (type == std::string("kullback_leibler")) {
      Log::Warning("Metric kullback_leibler is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new KullbackLeiblerDivergence(config);
    } else if (type == std::string("mape")) {
      Log::Warning("Metric mape is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new MAPEMetric(config);
    } else if (type == std::string("gamma")) {
      Log::Warning("Metric gamma is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new GammaMetric(config);
    } else if (type == std::string("gamma_deviance")) {
      Log::Warning("Metric gamma_deviance is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new GammaDevianceMetric(config);
    } else if (type == std::string("tweedie")) {
      Log::Warning("Metric tweedie is not implemented in cuda_exp version. Fall back to evaluation on CPU.");
      return new TweedieMetric(config);
    }
  } else {
  #endif  // USE_CUDA_EXP
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
  #ifdef USE_CUDA_EXP
  }
  #endif  // USE_CUDA_EXP
  return nullptr;
}

}  // namespace LightGBM
