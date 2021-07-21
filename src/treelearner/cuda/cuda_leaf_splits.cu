/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

__global__ void CUDAInitValuesKernel1(const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* cuda_num_data, double* cuda_sum_of_gradients, double* cuda_sum_of_hessians) {
  __shared__ score_t shared_gradients[NUM_THRADS_PER_BLOCK_LEAF_SPLITS];
  __shared__ score_t shared_hessians[NUM_THRADS_PER_BLOCK_LEAF_SPLITS];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = (blockIdx.x * blockDim.x + tid) * NUM_DATA_THREAD_ADD_LEAF_SPLITS;
  const unsigned int num_data_ref = static_cast<unsigned int>(*cuda_num_data);
  shared_gradients[tid] = 0.0f;
  shared_hessians[tid] = 0.0f;
  __syncthreads();
  for (unsigned int j = 0; j < NUM_DATA_THREAD_ADD_LEAF_SPLITS; ++j) {
    if (i + j < num_data_ref) {
      shared_gradients[tid] += cuda_gradients[i + j];
      shared_hessians[tid] += cuda_hessians[i + j];
    }
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < NUM_THRADS_PER_BLOCK_LEAF_SPLITS) {
      shared_gradients[tid] += shared_gradients[tid + s];
      shared_hessians[tid] += shared_hessians[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    cuda_sum_of_gradients[blockIdx.x] += shared_gradients[0];
    cuda_sum_of_hessians[blockIdx.x] += shared_hessians[0];
  }
}

__global__ void CUDAInitValuesKernel2(
  double* cuda_sum_of_gradients,
  double* cuda_sum_of_hessians,
  const data_size_t num_data,
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf,
  CUDALeafSplitsStruct* cuda_struct) {
  double sum_of_gradients = 0.0f;
  double sum_of_hessians = 0.0f;
  for (unsigned int i = 0; i < gridDim.x; ++i) {
    sum_of_gradients += cuda_sum_of_gradients[i];
    sum_of_hessians += cuda_sum_of_hessians[i];
  }
  cuda_sum_of_gradients[0] = sum_of_gradients;
  cuda_sum_of_hessians[0] = sum_of_hessians;
  cuda_struct->leaf_index = 0;
  cuda_struct->sum_of_gradients = sum_of_gradients;
  cuda_struct->sum_of_hessians = sum_of_hessians;
  cuda_struct->num_data_in_leaf = num_data;
  cuda_struct->gain = kMinScore;
  cuda_struct->leaf_value = 0.0f;
  cuda_struct->data_indices_in_leaf = cuda_data_indices_in_leaf;
  cuda_struct->hist_in_leaf = cuda_hist_in_leaf;
}

__global__ void InitValuesEmptyKernel(CUDALeafSplitsStruct* cuda_struct) {
  cuda_struct->leaf_index = -1;
  cuda_struct->sum_of_gradients = 0.0f;
  cuda_struct->sum_of_hessians = 0.0f;
  cuda_struct->num_data_in_leaf = 0;
  cuda_struct->gain = kMinScore;
  cuda_struct->leaf_value = 0.0f;
  cuda_struct->data_indices_in_leaf = nullptr;
  cuda_struct->hist_in_leaf = nullptr;
}

void CUDALeafSplits::LaunchInitValuesEmptyKernel() {
  InitValuesEmptyKernel<<<1, 1>>>(cuda_struct_);
}

void CUDALeafSplits::LaunchInitValuesKernal(
  const data_size_t* cuda_data_indices_in_leaf,
  hist_t* cuda_hist_in_leaf) {
  CUDAInitValuesKernel1<<<num_blocks_init_from_gradients_, NUM_THRADS_PER_BLOCK_LEAF_SPLITS>>>(
    cuda_gradients_, cuda_hessians_, cuda_num_data_, cuda_sum_of_gradients_buffer_,
    cuda_sum_of_hessians_buffer_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  CUDAInitValuesKernel2<<<1, 1>>>(
    cuda_sum_of_gradients_buffer_,
    cuda_sum_of_hessians_buffer_,
    num_data_,
    cuda_data_indices_in_leaf,
    cuda_hist_in_leaf,
    cuda_struct_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
