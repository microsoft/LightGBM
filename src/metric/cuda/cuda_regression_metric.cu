/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_algorithms.hpp>

#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename CUDA_METRIC, bool USE_WEIGHTS>
__global__ void EvalKernel(const data_size_t num_data, const label_t* labels, const label_t* weights,
                                 const double* scores, double* reduce_block_buffer) {
  __shared__ double shared_mem_buffer[32];
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_metric = 0.0;
  if (index < num_data) {
    point_metric = CUDA_METRIC::MetricOnPointCUDA(labels[index], scores[index]);
  }
  const double block_sum_point_metric = ShuffleReduceSum<double>(point_metric, shared_mem_buffer, NUM_DATA_PER_EVAL_THREAD);
  reduce_block_buffer[blockIdx.x] = block_sum_point_metric;
  if (USE_WEIGHTS) {
    double weight = 0.0;
    if (index < num_data) {
      weight = static_cast<double>(weights[index]);
      const double block_sum_weight = ShuffleReduceSum<double>(weight, shared_mem_buffer, NUM_DATA_PER_EVAL_THREAD);
      reduce_block_buffer[blockIdx.x + blockDim.x] = block_sum_weight;
    }
  }
}

template <typename HOST_METRIC, typename CUDA_METRIC>
double CUDARegressionMetricInterface<HOST_METRIC, CUDA_METRIC>::LaunchEvalKernel(const double* score) const {
  const int num_blocks = (this->num_data_ + NUM_DATA_PER_EVAL_THREAD - 1) / NUM_DATA_PER_EVAL_THREAD;
  if (this->cuda_weights_ != nullptr) {
    EvalKernel<CUDA_METRIC, true><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, reduce_block_buffer_.RawData());
  } else {
    EvalKernel<CUDA_METRIC, false><<<num_blocks, NUM_DATA_PER_EVAL_THREAD>>>(
      this->num_data_, this->cuda_labels_, this->cuda_weights_, score, reduce_block_buffer_.RawData());
  }
  ShuffleReduceSumGlobal<double, double>(reduce_block_buffer_.RawData(), num_blocks, reduce_block_buffer_inner_.RawData());
  double sum_loss = 0.0;
  CopyFromCUDADeviceToHost<double>(&sum_loss, reduce_block_buffer_inner_.RawData(), 1, __FILE__, __LINE__);
  double sum_weight = static_cast<double>(this->num_data_);
  if (this->cuda_weights_ != nullptr) {
    ShuffleReduceSumGlobal<double, double>(reduce_block_buffer_.RawData() + num_blocks, num_blocks, reduce_block_buffer_inner_.RawData());
    CopyFromCUDADeviceToHost<double>(&sum_weight, reduce_block_buffer_inner_.RawData(), 1, __FILE__, __LINE__);
  }
  return this->AverageLoss(sum_loss, sum_weight);
}

template double CUDARegressionMetricInterface<RMSEMetric, CUDARMSEMetric>::LaunchEvalKernel(const double* score) const;

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
