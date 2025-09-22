/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_

#if defined(USE_CUDA) || defined(USE_ROCM)

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.hu>

#include <vector>

#include "cuda_regression_metric.hpp"
#include "../binary_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDABinaryMetricInterface: public CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC> {
 public:
  explicit CUDABinaryMetricInterface(const Config& config): CUDAPointwiseMetricInterface<HOST_METRIC, CUDA_METRIC>(config) {}

  virtual ~CUDABinaryMetricInterface() {}

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;
};

class CUDABinaryLoglossMetric: public CUDABinaryMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric> {
 public:
  explicit CUDABinaryLoglossMetric(const Config& config);

  virtual ~CUDABinaryLoglossMetric() {}

  __device__ static double MetricOnPointCUDA(label_t label, double score, const double /*param*/) {
    // score should have been converted to probability
    if (label <= 0) {
      if (1.0f - score > kEpsilon) {
        return -log(1.0f - score);
      }
    } else {
      if (score > kEpsilon) {
        return -log(score);
      }
    }
    return -log(kEpsilon);
  }
};

}  // namespace LightGBM

#endif  // USE_CUDA || USE_ROCM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
