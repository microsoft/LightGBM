/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */


#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

template <bool USE_INDICES>
__global__ void CUDAInitValuesKernel1(const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t num_data, const data_size_t* cuda_bagging_data_indices,
  double* cuda_sum_of_gradients, double* cuda_sum_of_hessians) {
  __shared__ double shared_mem_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double gradient = 0.0f;
  double hessian = 0.0f;
  if (data_index < num_data) {
    gradient = USE_INDICES ? cuda_gradients[cuda_bagging_data_indices[data_index]] : cuda_gradients[data_index];
    hessian = USE_INDICES ? cuda_hessians[cuda_bagging_data_indices[data_index]] : cuda_hessians[data_index];
  }
  const double block_sum_gradient = ShuffleReduceSum<double>(gradient, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double block_sum_hessian = ShuffleReduceSum<double>(hessian, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_gradients[blockIdx.x] += block_sum_gradient;
    cuda_sum_of_hessians[blockIdx.x] += block_sum_hessian;
  }
}

__global__ void CUDAInitValuesKernel2(
  const double lambda_l1,
  const double lambda_l2,
  const int num_blocks_to_reduce,
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  __shared__ double shared_mem_buffer[32];
  double thread_sum_of_gradients = 0.0f;
  double thread_sum_of_hessians = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks_to_reduce; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_of_gradients += cuda_sum_of_gradients[block_index];
    thread_sum_of_hessians += cuda_sum_of_hessians[block_index];
  }
  const double sum_of_gradients = ShuffleReduceSum<double>(thread_sum_of_gradients, shared_mem_buffer, blockDim.x);
  __syncthreads();
  const double sum_of_hessians = ShuffleReduceSum<double>(thread_sum_of_hessians, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_of_hessians[0] = sum_of_hessians;
    cuda_struct->leaf_index = 0;
    cuda_struct->sum_of_gradients = sum_of_gradients;
    cuda_struct->sum_of_hessians = sum_of_hessians;
    cuda_struct->num_data_in_leaf = num_data;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      cuda_struct->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
    cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
  }
}

__global__ void InitValuesEmptyKernel(CUDALeafSplitsStruct* cuda_struct) {
  cuda_struct->leaf_index = -1;
  cuda_struct->sum_of_gradients = 0.0f;
  cuda_struct->sum_of_hessians = 0.0f;
  cuda_struct->num_data_in_leaf = 0;
  cuda_struct->gain = 0.0f;
  cuda_struct->leaf_value = 0.0f;
  cuda_struct->data_indices_in_leaf = nullptr;
  cuda_struct->hist_in_leaf = nullptr;
}

void CUDALeafSplits::LaunchInitValuesEmptyKernel() {
  InitValuesEmptyKernel<<<1, 1>>>(cuda_struct_);
}

void CUDALeafSplits::LaunchInitValuesKernal(
  const double lambda_l1, const double lambda_l2,
  const data_size_t* cuda_bagging_data_indices,
  const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf) {
  if (cuda_bagging_data_indices == nullptr) {
    CUDAInitValuesKernel1<false><<<num_blocks_init_from_gradients_, NUM_THRADS_PER_BLOCK_LEAF_SPLITS>>>(
      cuda_gradients_, cuda_hessians_, num_used_indices, nullptr, cuda_sum_of_gradients_buffer_,
      cuda_sum_of_hessians_buffer_);
  } else {
    CUDAInitValuesKernel1<true><<<num_blocks_init_from_gradients_, NUM_THRADS_PER_BLOCK_LEAF_SPLITS>>>(
      cuda_gradients_, cuda_hessians_, num_used_indices, cuda_bagging_data_indices, cuda_sum_of_gradients_buffer_,
      cuda_sum_of_hessians_buffer_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CUDAInitValuesKernel2<<<1, NUM_THRADS_PER_BLOCK_LEAF_SPLITS>>>(
    lambda_l1, lambda_l2,
    num_blocks_init_from_gradients_,
    cuda_sum_of_gradients_buffer_,
    cuda_sum_of_hessians_buffer_,
    num_used_indices,
    cuda_data_indices_in_leaf,
    cuda_hist_in_leaf,
    cuda_struct_);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
