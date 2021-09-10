/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#include <LightGBM/cuda/cuda_algorithms.hpp>
#include "cuda_regression_metric.hpp"

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator, bool USE_WEIGHT>
__global__ void EvalKernel_RegressionPointWiseLoss(const double* score,
                            const label_t* label,
                            const label_t* weights,
                            const data_size_t num_data,
                            const double sum_weight,
                            double* cuda_sum_loss_buffer,
                            const double alpha,
                            const double fair_c,
                            const double tweedie_variance_power) {
  // assert that warpSize == 32 and maximum number of threads per block is 1024
  __shared__ double shared_buffer[32];
  const int data_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const double pointwise_loss = data_index < num_data ?
    (USE_WEIGHT ? CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index], alpha, fair_c, tweedie_variance_power) * weights[data_index] :
                  CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index], alpha, fair_c, tweedie_variance_power)) :
                  0.0f;
  const double loss = ShuffleReduceSum<double>(pointwise_loss, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = loss;
  }
}

template <typename CUDAPointWiseLossCalculator>
__global__ void ReduceLossKernel_Regression(const double* cuda_sum_loss_buffer, const data_size_t num_blocks, double* out_loss) {
  __shared__ double shared_buffer[32];
  double thread_sum_loss = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_loss += cuda_sum_loss_buffer[block_index];
  }
  const double sum_loss = ShuffleReduceSum<double>(thread_sum_loss, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    *out_loss = sum_loss;
  }
}

template <typename CUDAPointWiseLossCalculator>
void CUDARegressionMetric<CUDAPointWiseLossCalculator>::LaunchEvalKernelInner(const double* score) const {
  const data_size_t num_blocks = (RegressionMetric<CUDAPointWiseLossCalculator>::num_data_ + EVAL_BLOCK_SIZE_REGRESSION_METRIC - 1) / EVAL_BLOCK_SIZE_REGRESSION_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_RegressionPointWiseLoss<CUDAPointWiseLossCalculator, false><<<num_blocks, EVAL_BLOCK_SIZE_REGRESSION_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_,
      this->config_.alpha,
      this->config_.fair_c,
      this->config_.tweedie_variance_power);
  } else {
    EvalKernel_RegressionPointWiseLoss<CUDAPointWiseLossCalculator, true><<<num_blocks, EVAL_BLOCK_SIZE_REGRESSION_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_,
      this->config_.alpha,
      this->config_.fair_c,
      this->config_.tweedie_variance_power);
  }
  ReduceLossKernel_Regression<CUDAPointWiseLossCalculator><<<1, EVAL_BLOCK_SIZE_REGRESSION_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <>
void CUDARegressionMetric<CUDARMSEMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAL2Metric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAL1Metric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAQuantileMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAHuberLossMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAFairLossMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAPoissonMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAMAPEMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAGammaMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDAGammaDevianceMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDARegressionMetric<CUDATweedieMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

}  // namespace LightGBM
