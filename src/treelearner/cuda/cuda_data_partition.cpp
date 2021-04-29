/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

namespace LightGBM {

CUDADataPartition::CUDADataPartition(const data_size_t num_data, const int num_leaves,
  const data_size_t* cuda_num_data, const int* cuda_num_leaves):
  num_data_(num_data), num_leaves_(num_leaves) {
  cuda_num_data_ = cuda_num_data;
  cuda_num_leaves_ = cuda_num_leaves;
}

void CUDADataPartition::Init() {
  // allocate CUDA memory
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_data_indices_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_) + 1, &cuda_leaf_num_data_offsets_);
}

void CUDADataPartition::BeforeTrain(const data_size_t* data_indices) {
  if (data_indices == nullptr) {
    // no bagging
    LaunchFillDataIndicesBeforeTrain();
    SetCUDAMemory<data_size_t>(cuda_leaf_num_data_offsets_, 0, static_cast<size_t>(num_leaves_) + 1);
    SynchronizeCUDADevice();
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_num_data_offsets_ + 1, cuda_num_data_, 1);
    SynchronizeCUDADevice();
  } else {
    Log::Fatal("bagging is not supported by GPU");
  }
}

void CUDADataPartition::Split(const int* /*leaf_id*/,
  const int* /*best_split_feature*/,
  const int* /*best_split_threshold*/) {}

Tree* CUDADataPartition::GetCPUTree() {}


}  // namespace LightGBM

#endif  // USE_CUDA
