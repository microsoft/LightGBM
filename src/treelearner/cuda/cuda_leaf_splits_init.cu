/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits_init.hpp"

namespace LightGBM {

__global__ void CUDALeafSplitsInitKernel1(const float* cuda_gradients, const float* cuda_hessians,
  const data_size_t* num_data, double* grad_sum_out, double* hess_sum_out) {
  __shared__ float shared_gradients[NUM_THRADS_PER_BLOCK_SPLITS_INIT];
  __shared__ float shared_hessians[NUM_THRADS_PER_BLOCK_SPLITS_INIT];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = (blockIdx.x * blockDim.x + tid) * NUM_DATA_THREAD_ADD;
  const unsigned int num_data_ref = static_cast<unsigned int>(*num_data);
  shared_gradients[tid] = 0.0f;
  shared_hessians[tid] = 0.0f;
  __syncthreads();
  for (unsigned int j = 0; j < NUM_DATA_THREAD_ADD; ++j) {
    if (i + j < num_data_ref) {
      shared_gradients[tid] += cuda_gradients[i + j];
      shared_hessians[tid] += cuda_hessians[i + j];
    }
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < NUM_THRADS_PER_BLOCK_SPLITS_INIT) {
      shared_gradients[tid] += shared_gradients[tid + s];
      shared_hessians[tid] += shared_hessians[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_sum_out[blockIdx.x] += shared_gradients[0];
    hess_sum_out[blockIdx.x] += shared_hessians[0];
  }
}

__global__ void CUDALeafSplitsInitKernel2(double* grad_sum_out, double* hess_sum_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (unsigned int i = 1; i < gridDim.x; ++i) {
      grad_sum_out[0] += grad_sum_out[i];
      hess_sum_out[0] += hess_sum_out[i];
    }
  }
}

void CUDALeafSplitsInit::LaunchLeafSplitsInit() {
  CUDALeafSplitsInitKernel1<<<num_blocks_, NUM_THRADS_PER_BLOCK_SPLITS_INIT>>>(
    cuda_gradients_, cuda_hessians_, cuda_num_data_, smaller_leaf_sum_gradients_,
    smaller_leaf_sum_hessians_);
  SynchronizeCUDADevice();
  CUDALeafSplitsInitKernel2<<<num_blocks_, NUM_THRADS_PER_BLOCK_SPLITS_INIT>>>(
    smaller_leaf_sum_gradients_, smaller_leaf_sum_hessians_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
