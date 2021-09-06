/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_xentropy_metric.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

__device__ inline static double XentLossCUDA(label_t label, double prob) {
  const double log_arg_epsilon = 1.0e-12;
  double a = label;
  if (prob > log_arg_epsilon) {
    a *= log(prob);
  } else {
    a *= log(log_arg_epsilon);
  }
  double b = 1.0f - label;
  if (1.0f - prob > log_arg_epsilon) {
    b *= log(1.0f - prob);
  } else {
    b *= log(log_arg_epsilon);
  }
  return - (a + b);
}

__device__ inline static double XentLambdaLossCUDA(label_t label, label_t weight, double hhat) {
  return XentLossCUDA(label, 1.0f - exp(-weight * hhat));
}

template <bool USE_WEIGHT>
__global__ void EvalKernel_CrossEntropy(
  const double* score,
  const label_t* cuda_label,
  const label_t* cuda_weights,
  const data_size_t num_data,
  double* cuda_sum_loss_buffer) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_loss = 0.0f;
  __shared__ double shared_mem_buffer[32];
  if (data_index < num_data) {
    const label_t label = cuda_label[data_index];
    if (!USE_WEIGHT) {
      point_loss = XentLossCUDA(label, score[data_index]);
    } else {
      const label_t weight = cuda_weights[data_index];
      point_loss = XentLossCUDA(label, score[data_index]) * weight;
    }
  }
  const double block_sum_loss = ShuffleReduceSum<double>(point_loss, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = block_sum_loss;
  }
}

__global__ void ReduceLossKernel_CrossEntropy(const double* cuda_sum_loss_buffer, const data_size_t num_blocks, double* out_loss) {
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

void CUDACrossEntropyMetric::LaunchEvalKernel(const double* score) const {
  const data_size_t num_blocks = (num_data_ + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_CrossEntropy<false><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  } else {
    EvalKernel_CrossEntropy<true><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  }
  ReduceLossKernel_CrossEntropy<<<1, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <bool USE_WEIGHT>
__global__ void EvalKernel_CrossEntropyLambda(
  const double* score,
  const label_t* cuda_label,
  const label_t* cuda_weights,
  const data_size_t num_data,
  double* cuda_sum_loss_buffer) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_loss = 0.0f;
  __shared__ double shared_mem_buffer[32];
  if (data_index < num_data) {
    const label_t label = cuda_label[data_index];
    if (!USE_WEIGHT) {
      point_loss = XentLambdaLossCUDA(label, 1.0f, score[data_index]);
    } else {
      const label_t weight = cuda_weights[data_index];
      point_loss = XentLambdaLossCUDA(label, weight, score[data_index]);
    }
  }
  const double block_sum_loss = ShuffleReduceSum<double>(point_loss, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = block_sum_loss;
  }
}

void CUDACrossEntropyLambdaMetric::LaunchEvalKernel(const double* score) const {
  const data_size_t num_blocks = (num_data_ + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_CrossEntropyLambda<false><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  } else {
    EvalKernel_CrossEntropyLambda<true><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  }
  ReduceLossKernel_CrossEntropy<<<1, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <bool USE_WEIGHT>
__global__ void EvalKernel_KullbackLeiblerDivergence(
  const double* score,
  const label_t* cuda_label,
  const label_t* cuda_weights,
  const data_size_t num_data,
  double* cuda_sum_loss_buffer) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double point_loss = 0.0f;
  __shared__ double shared_mem_buffer[32];
  if (data_index < num_data) {
    const label_t label = cuda_label[data_index];
    if (!USE_WEIGHT) {
      point_loss = XentLossCUDA(label, score[data_index]);
    } else {
      const label_t weight = cuda_weights[data_index];
      point_loss = XentLossCUDA(label, score[data_index]) * weight;
    }
  }
  const double block_sum_loss = ShuffleReduceSum<double>(point_loss, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = block_sum_loss;
  }
}

void CUDAKullbackLeiblerDivergence::LaunchEvalKernel(const double* score) const {
  const data_size_t num_blocks = (num_data_ + EVAL_BLOCK_SIZE_XENTROPY_METRIC - 1) / EVAL_BLOCK_SIZE_XENTROPY_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_KullbackLeiblerDivergence<false><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  } else {
    EvalKernel_KullbackLeiblerDivergence<true><<<num_blocks, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(score, cuda_label_, cuda_weights_, num_data_, cuda_sum_loss_buffer_);
  }
  ReduceLossKernel_CrossEntropy<<<1, EVAL_BLOCK_SIZE_XENTROPY_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

}  // namespace LightGBM
