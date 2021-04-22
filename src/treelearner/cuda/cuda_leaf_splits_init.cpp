/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits_init.hpp"

#include <chrono>

namespace LightGBM {

CUDALeafSplitsInit::CUDALeafSplitsInit(const score_t* cuda_gradients,
  const score_t* cuda_hessians, const data_size_t num_data):
cuda_gradients_(cuda_gradients), cuda_hessians_(cuda_hessians), num_data_(num_data) {}

void CUDALeafSplitsInit::Init() {
  num_blocks_ = (num_data_ + INIT_SUM_BLOCK_SIZE - 1) / INIT_SUM_BLOCK_SIZE;

  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  void* smaller_leaf_sum_gradients_ptr = nullptr;
  void* smaller_leaf_sum_hessians_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&smaller_leaf_sum_gradients_ptr, num_blocks_ * sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMalloc(&smaller_leaf_sum_hessians_ptr, num_blocks_ * sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMemset(smaller_leaf_sum_gradients_ptr, 0, num_blocks_ * sizeof(double)));
  CUDASUCCESS_OR_FATAL(cudaMemset(smaller_leaf_sum_hessians_ptr, 0, num_blocks_ * sizeof(double)));
  smaller_leaf_sum_gradients_ = reinterpret_cast<double*>(smaller_leaf_sum_gradients_ptr);
  smaller_leaf_sum_hessians_ = reinterpret_cast<double*>(smaller_leaf_sum_hessians_ptr);

  void* cuda_num_data_ptr = nullptr;
  CUDASUCCESS_OR_FATAL(cudaMalloc(&cuda_num_data_ptr, sizeof(int)));
  cuda_num_data_ = reinterpret_cast<data_size_t*>(cuda_num_data_ptr);
  const void* num_data_ptr = reinterpret_cast<const void*>(&num_data_);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(cuda_num_data_ptr, num_data_ptr, sizeof(int),  cudaMemcpyHostToDevice));
}

void CUDALeafSplitsInit::Compute() {
  LaunchLeafSplitsInit(num_blocks_, INIT_SUM_BLOCK_SIZE,
    cuda_gradients_, cuda_hessians_, cuda_num_data_,
    smaller_leaf_sum_gradients_, smaller_leaf_sum_hessians_);
  Log::Warning(cudaGetErrorName(cudaGetLastError()));
  CUDASUCCESS_OR_FATAL(cudaDeviceSynchronize());

  const void* smaller_leaf_sum_gradients_ptr = reinterpret_cast<const void*>(smaller_leaf_sum_gradients_);
  const void* smaller_leaf_sum_hessians_ptr = reinterpret_cast<const void*>(smaller_leaf_sum_hessians_);
  void* host_smaller_leaf_sum_gradients_ptr = reinterpret_cast<void*>(&host_smaller_leaf_sum_gradients_);
  void* host_smaller_leaf_sum_hessians_ptr = reinterpret_cast<void*>(&host_smaller_leaf_sum_hessians_);
  CUDASUCCESS_OR_FATAL(cudaMemcpy(host_smaller_leaf_sum_gradients_ptr, smaller_leaf_sum_gradients_ptr, sizeof(double),  cudaMemcpyDeviceToHost));
  CUDASUCCESS_OR_FATAL(cudaMemcpy(host_smaller_leaf_sum_hessians_ptr, smaller_leaf_sum_hessians_ptr, sizeof(double),  cudaMemcpyDeviceToHost));
  Log::Warning("host_smaller_leaf_sum_gradients_ = %f", host_smaller_leaf_sum_gradients_);
  Log::Warning("host_smaller_leaf_sum_hessians_ = %f", host_smaller_leaf_sum_hessians_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
