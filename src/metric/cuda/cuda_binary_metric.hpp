/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.h>

#include <vector>

#include "cuda_regression_metric.hpp"
#include "../binary_metric.hpp"

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDABinaryMetricInterface: public CUDAMetricInterface<HOST_METRIC> {
 public:
  explicit CUDABinaryMetricInterface(const Config& config): CUDAMetricInterface<HOST_METRIC>(config) {}

  virtual ~CUDABinaryMetricInterface() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const override;

 protected:
  double LaunchEvalKernel(const double* score_convert) const;

  CUDAVector<double> score_convert_buffer_;
  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
};

class CUDABinaryLoglossMetric: public CUDABinaryMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric> {
 public:
  explicit CUDABinaryLoglossMetric(const Config& config): CUDABinaryMetricInterface<BinaryLoglossMetric, CUDABinaryLoglossMetric>(config) {}

  virtual ~CUDABinaryLoglossMetric() {}

  __device__ static double MetricOnPointCUDA(label_t label, double score) {
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

#endif  // USE_CUDA_EXP

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_BINARY_METRIC_HPP_
