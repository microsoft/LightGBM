/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_centralized_info.hpp"

namespace LightGBM {

CUDACentralizedInfo::CUDACentralizedInfo(const data_size_t num_data, const int num_leaves, const int num_features):
num_data_(num_data), num_leaves_(num_leaves), num_features_(num_features) {}

void CUDACentralizedInfo::Init() {
  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_num_data_, &num_data_, 1);
  InitCUDAMemoryFromHostMemory<int>(&cuda_num_leaves_, &num_leaves_, 1);
  InitCUDAMemoryFromHostMemory<int>(&cuda_num_features_, &num_features_, 1);

  AllocateCUDAMemory<score_t>(static_cast<size_t>(num_data_), &cuda_gradients_);
  AllocateCUDAMemory<score_t>(static_cast<size_t>(num_data_), &cuda_hessians_);
}

void CUDACentralizedInfo::BeforeTrain(const score_t* gradients, const score_t* hessians) {
  CopyFromHostToCUDADevice<score_t>(cuda_gradients_, gradients, static_cast<size_t>(num_data_));
  CopyFromHostToCUDADevice<score_t>(cuda_hessians_, hessians, static_cast<size_t>(num_data_));
}

}  // namespace LightGBM

#endif  // USE_CUDA
