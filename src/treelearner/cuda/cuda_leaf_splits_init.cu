/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits_init.hu"

namespace LightGBM {

__global__ void CUDALeafSplitsInitKernel1(const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t num_data, double* grad_sum_out, double* hess_sum_out) {
  extern __shared__ score_t shared_gradients[blockDim.x];
  extern __shared__ score_t shared_hessians[blockDim.x];
  double sum_gradient = 0.0f;
  double sum_hessian = 0.0f;
  const unsigned int tid = threadIdx.x;
  const unsigned i = blockIdx.x * blockDim.x + tid;
  if (i < static_cast<unsigned int>(num_data)) {
    shared_gradients[tid] = cuda_gradients[i];
    shared_hessians[tid] = cuda_hessians[i];
  }
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
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
  const data_size_t num_data, double* grad_sum_out, double* hess_sum_out) {
  if (threadIdx.x == 0) {
    for (unsigned int i = 1; i < blockDim.x; ++i) {
      grad_sum_out[0] += grad_sum_out[i];
      hess_sum_out[0] += hess_sum_out[i];
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
