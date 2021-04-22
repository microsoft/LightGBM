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
  __shared__ float shared_gradients[INIT_SUM_BLOCK_SIZE];
  __shared__ float shared_hessians[INIT_SUM_BLOCK_SIZE];
  const unsigned int tid = threadIdx.x;
  const unsigned int i = blockIdx.x * blockDim.x + tid;
  if (i < static_cast<unsigned int>(*num_data)) {
    shared_gradients[tid] = cuda_gradients[i];
    shared_hessians[tid] = cuda_hessians[i];
  } else {
    shared_gradients[tid] = 0.0f;
    shared_hessians[tid] = 0.0f;
  }
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < INIT_SUM_BLOCK_SIZE) {
      shared_gradients[tid] += shared_gradients[tid + s];
      shared_hessians[tid] += shared_hessians[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_sum_out[blockIdx.x] = shared_gradients[0];
    hess_sum_out[blockIdx.x] = shared_hessians[0];
  }
}

__global__ void CUDALeafSplitsInitKernel2(const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* num_data, double* grad_sum_out, double* hess_sum_out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (unsigned int i = 1; i < gridDim.x; ++i) {
      grad_sum_out[0] += grad_sum_out[i];
      hess_sum_out[0] += hess_sum_out[i];
    }
  }
}

void CUDALeafSplitsInit::LaunchLeafSplitsInit(const int num_blocks, const int init_sum_block_size,
  const score_t* cuda_gradients, const score_t* cuda_hessians, const data_size_t* num_data,
  double* smaller_leaf_sum_gradients, double* smaller_leaf_sum_hessians) {
  CUDALeafSplitsInitKernel1<<<num_blocks, init_sum_block_size>>>(
    cuda_gradients, cuda_hessians, num_data, smaller_leaf_sum_gradients,
    smaller_leaf_sum_hessians);
  CUDALeafSplitsInitKernel2<<<num_blocks, init_sum_block_size>>>(
    cuda_gradients, cuda_hessians, num_data, smaller_leaf_sum_gradients,
    smaller_leaf_sum_hessians);
}

}  // namespace LightGBM

#endif  // USE_CUDA
