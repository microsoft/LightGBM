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

__global__ void CUDAInitValuesKernel2(double* cuda_sum_of_gradients, double* cuda_sum_of_hessians) {
  if (blockIdx.x == 0) {
    double sum_of_gradients = 0.0f;
    double sum_of_hessians = 0.0f;
    for (unsigned int i = 1; i < gridDim.x; ++i) {
      sum_of_gradients += cuda_sum_of_gradients[i];
      sum_of_hessians += cuda_sum_of_hessians[i];
    }
    //printf("sum_of_gradients = %f\n", sum_of_gradients);
    //printf("sum_of_hessians = %f\n", sum_of_hessians);
    cuda_sum_of_gradients[0] += sum_of_gradients;
    cuda_sum_of_hessians[0] += sum_of_hessians;
  }
}

void CUDALeafSplits::LaunchInitValuesKernal() {
  auto start = std::chrono::steady_clock::now();
  CUDAInitValuesKernel1<<<num_blocks_init_from_gradients_, NUM_THRADS_PER_BLOCK_LEAF_SPLITS>>>(
    cuda_gradients_, cuda_hessians_, cuda_num_data_, cuda_sum_of_gradients_,
    cuda_sum_of_hessians_);
  CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_num_data_in_leaf_, cuda_num_data_, 1);
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  auto duration = static_cast<std::chrono::duration<double>>(end - start);
  //Log::Warning("CUDAInitValuesKernel1 duration = %f", duration.count());
  start = std::chrono::steady_clock::now();
  CUDAInitValuesKernel2<<<num_blocks_init_from_gradients_, 1>>>(
    cuda_sum_of_gradients_, cuda_sum_of_hessians_);
  SynchronizeCUDADevice();
  end = std::chrono::steady_clock::now();
  duration = static_cast<std::chrono::duration<double>>(end - start);
  //Log::Warning("cuda_sum_of_gradients_ = %f, cuda_sum_of_hessians_ = %f", *cuda_sum_of_gradients_, *cuda_sum_of_hessians_);
  //Log::Warning("CUDAInitValuesKernel2 duration = %f", duration.count());
}

}  // namespace LightGBM

#endif  // USE_CUDA
