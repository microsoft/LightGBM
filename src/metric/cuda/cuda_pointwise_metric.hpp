/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_METRIC_CUDA_CUDA_POINTWISE_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_POINTWISE_METRIC_HPP_

#if defined(USE_CUDA) || defined(USE_ROCM)

#include <LightGBM/cuda/cuda_metric.hpp>
#include <LightGBM/cuda/cuda_utils.hu>

#include <vector>

#define NUM_DATA_PER_EVAL_THREAD (1024)

namespace LightGBM {

template <typename HOST_METRIC, typename CUDA_METRIC>
class CUDAPointwiseMetricInterface: public CUDAMetricInterface<HOST_METRIC> {
 public:
  explicit CUDAPointwiseMetricInterface(const Config& config): CUDAMetricInterface<HOST_METRIC>(config), num_class_(config.num_class) {}

  virtual ~CUDAPointwiseMetricInterface() {}

  void Init(const Metadata& metadata, data_size_t num_data) override;

 protected:
  void LaunchEvalKernel(const double* score_convert, double* sum_loss, double* sum_weight) const;

  virtual double GetParamFromConfig() const { return 0.0; }

  mutable CUDAVector<double> score_convert_buffer_;
  CUDAVector<double> reduce_block_buffer_;
  CUDAVector<double> reduce_block_buffer_inner_;
  const int num_class_;
};

}  // namespace LightGBM

#endif  // USE_CUDA || USE_ROCM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_POINTWISE_METRIC_HPP_
