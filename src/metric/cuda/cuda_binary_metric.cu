/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include "cuda_binary_metric.hpp"

#include <chrono>

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator, bool USE_WEIGHT>
__global__ void EvalKernel_BinaryPointWiseLoss(const double* score,
                           const label_t* label,
                           const label_t* weights,
                           const data_size_t num_data,
                           const double sum_weight,
                           double* cuda_sum_loss_buffer) {
  // assert that warpSize == 32 and maximum number of threads per block is 1024
  __shared__ double shared_buffer[32];
  const int data_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const double pointwise_loss = data_index < num_data ?
    (USE_WEIGHT ? CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index]) * weights[data_index] :
                  CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index])) :
                  0.0f;
  const double loss = ShuffleReduceSum<double>(pointwise_loss, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = loss;
  }
}

__global__ void ReduceLossKernel(const double* cuda_sum_loss_buffer, const data_size_t num_blocks, double* out_loss) {
  __shared__ double shared_buffer[32];
  double thread_sum_loss = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_loss += cuda_sum_loss_buffer[block_index];
  }
  const double sum_loss = ShuffleReduceSum<double>(thread_sum_loss, shared_buffer, static_cast<size_t>(num_blocks));
  if (threadIdx.x == 0) {
    *out_loss = sum_loss;
  }
}

template <typename CUDAPointWiseLossCalculator>
void CUDABinaryMetric<CUDAPointWiseLossCalculator>::LaunchEvalKernelInner(const double* score) const {
  const data_size_t num_blocks = (BinaryMetric<CUDAPointWiseLossCalculator>::num_data_ + EVAL_BLOCK_SIZE_BINARY_METRIC - 1) / EVAL_BLOCK_SIZE_BINARY_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_BinaryPointWiseLoss<CUDAPointWiseLossCalculator, false><<<num_blocks, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_);
  } else {
    EvalKernel_BinaryPointWiseLoss<CUDAPointWiseLossCalculator, true><<<num_blocks, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_);
  }
  ReduceLossKernel<<<1, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <>
void CUDABinaryMetric<CUDABinaryLoglossMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDABinaryMetric<CUDABinaryErrorMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

void CUDAAUCMetric::LaunchEvalKernel(const double* score) const {
  BitonicArgSortGlobal<double, data_size_t, false>(score, cuda_indices_buffer_, static_cast<size_t>(num_data_));
  if (cuda_weights_ == nullptr) {
    GlobalGenAUCPosNegSum<false, false>(cuda_label_, cuda_weights_, cuda_indices_buffer_, cuda_sum_pos_buffer_, cuda_block_sum_pos_buffer_, num_data_);
  } else {
    GlobalGenAUCPosNegSum<true, false>(cuda_label_, cuda_weights_, cuda_indices_buffer_, cuda_sum_pos_buffer_, cuda_block_sum_pos_buffer_, num_data_);
    Log::Fatal("CUDA AUC with weights is not supported.");
  }
  GloblGenAUCMark(score, cuda_indices_buffer_, cuda_threshold_mark_, cuda_block_threshold_mark_buffer_, cuda_block_mark_first_zero_, num_data_);
  GlobalCalcAUC(cuda_sum_pos_buffer_, cuda_threshold_mark_, num_data_, cuda_block_sum_pos_buffer_);
  double total_area = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&total_area, cuda_block_sum_pos_buffer_, 1, __FILE__, __LINE__);
}

}  // namespace LightGBM
