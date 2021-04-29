/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_data_splitter.hpp"

namespace LightGBM {

CUDADataSplitter::CUDADataSplitter(const data_size_t num_data, const int max_num_leaves):
  num_data_(num_data), max_num_leaves_(max_num_leaves) {}

void CUDADataSplitter::Init() {
  // allocate GPU memory
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_data_indices_);
  
  AllocateCUDAMemory<int>(static_cast<size_t>(max_num_leaves_), &cuda_leaf_num_data_offsets_);

  AllocateCUDAMemory<data_size_t>(1, &cuda_num_data_);
  CopyFromHostToCUDADevice<data_size_t>(cuda_num_data_, &num_data_, 1);

  AllocateCUDAMemory<int>(1, &cuda_max_num_leaves_);
  CopyFromHostToCUDADevice<int>(cuda_max_num_leaves_, &max_num_leaves_, 1);

  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_leaves_), &cuda_leaf_num_data_offsets_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_leaves_), &cuda_leaf_num_data_);
}

void CUDADataSplitter::BeforeTrain(const data_size_t* data_indices) {
  if (data_indices == nullptr) {
    // no bagging
    LaunchFillDataIndicesBeforeTrain();
    SynchronizeCUDADevice();
    data_indices_.resize(num_data_);
    CopyFromCUDADeviceToHost<data_size_t>(data_indices_.data(), cuda_data_indices_, static_cast<size_t>(num_data_));
    /*for (int i = 0; i < 100; ++i) {
      Log::Warning("data_indices_[%d] = %d", i, data_indices_[i]);
      Log::Warning("data_indices_[end - %d] = %d", i, data_indices_[num_data_ - 1 - i]);
    }*/
    SetCUDAMemory<data_size_t>(cuda_leaf_num_data_offsets_, 0, max_num_leaves_);
    SetCUDAMemory<data_size_t>(cuda_leaf_num_data_, 0, max_num_leaves_);
    //Log::Warning("num_data_ = %d", num_data_);
    CopyFromHostToCUDADevice(cuda_leaf_num_data_, &num_data_, 1);
    data_size_t root_leaf_num_data = 0;
    CopyFromCUDADeviceToHost<data_size_t>(&root_leaf_num_data, cuda_leaf_num_data_, 1);
    //Log::Warning("root_leaf_num_data = %d", root_leaf_num_data);
  } else {
    Log::Fatal("bagging is not supported by GPU");
  }
}

void CUDADataSplitter::Split(const int* /*leaf_id*/,
  const int* /*best_split_feature*/,
  const int* /*best_split_threshold*/) {}

Tree* CUDADataSplitter::GetCPUTree() {}


}  // namespace LightGBM
