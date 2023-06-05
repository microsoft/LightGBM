/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.h>

#include <vector>

#include "cuda_pointwise_metric.hpp"
#include "../regression_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDARegressionMetricInterface: public CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC> {
 public:
  explicit CUDARegressionMetricInterface(const Config& config):
    CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>(config) {}

  virtual ~CUDARegressionMetricInterface() {}

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;
};

class CUDARMSEMetric: public CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric> {
 public:
  explicit CUDARMSEMetric(const Config& config);

  virtual ~CUDARMSEMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double /*alpha*/) {
    return (score - label) * (score - label);
  }
};

class CUDAL2Metric : public CUDARegressionMetricInterface<L2Metric, CUDAL2Metric> {
 public:
  explicit CUDAL2Metric(const Config& config);

  virtual ~CUDAL2Metric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double /*alpha*/) {
    return (score - label) * (score - label);
  }
};

class CUDAQuantileMetric : public CUDARegressionMetricInterface<QuantileMetric, CUDAQuantileMetric> {
 public:
  explicit CUDAQuantileMetric(const Config& config);

  virtual ~CUDAQuantileMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score, double alpha) {
    double delta = label - score;
    if (delta < 0) {
      return (alpha - 1.0f) * delta;
    } else {
      return alpha * delta;
    }
  }

  double GetParamFromConfig() const override {
    return alpha_;
  }

 private:
  const double alpha_;
};

class CUDAL1Metric : public CUDARegressionMetricInterface<L1Metric, CUDAL1Metric> {
 public:
  explicit CUDAL1Metric(const Config& config);

  virtual ~CUDAL1Metric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double /*alpha*/) {
    return std::fabs(score - label);
  }
};

class CUDAHuberLossMetric : public CUDARegressionMetricInterface<HuberLossMetric, CUDAHuberLossMetric> {
 public:
  explicit CUDAHuberLossMetric(const Config& config);

  virtual ~CUDAHuberLossMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double alpha) {
    const double diff = score - label;
    if (std::abs(diff) <= alpha) {
      return 0.5f * diff * diff;
    } else {
      return alpha * (std::abs(diff) - 0.5f * alpha);
    }
  }

  double GetParamFromConfig() const override {
    return alpha_;
  }
 private:
  const double alpha_;
};

class CUDAFairLossMetric : public CUDARegressionMetricInterface<FairLossMetric, CUDAFairLossMetric> {
 public:
  explicit CUDAFairLossMetric(const Config& config);

  virtual ~CUDAFairLossMetric() {}

  __device__ inline static double MetricOnPointCUDA(label_t label, double score,  double fair_c) {
    const double x = std::fabs(score - label);
    const double c =  fair_c;
    return c * x - c * c * std::log1p(x / c);
  }

  double GetParamFromConfig() const override {
    return fair_c_;
  }

 private:
  const double fair_c_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_REGRESSION_METRIC_HPP_
