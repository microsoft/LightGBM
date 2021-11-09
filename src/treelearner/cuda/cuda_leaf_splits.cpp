/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"

namespace LightGBM {

CUDALeafSplits::CUDALeafSplits(const data_size_t num_data):
num_data_(num_data) {
  cuda_struct_ = nullptr;
  cuda_sum_of_gradients_buffer_ = nullptr;
  cuda_sum_of_hessians_buffer_ = nullptr;
}

CUDALeafSplits::~CUDALeafSplits() {
  DeallocateCUDAMemory<CUDALeafSplitsStruct>(&cuda_struct_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_sum_of_gradients_buffer_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_sum_of_hessians_buffer_, __FILE__, __LINE__);
}

void CUDALeafSplits::Init() {
  num_blocks_init_from_gradients_ = (num_data_ + NUM_THRADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THRADS_PER_BLOCK_LEAF_SPLITS;

  // allocate more memory for sum reduction in CUDA
  // only the first element records the final sum
  AllocateCUDAMemory<double>(&cuda_sum_of_gradients_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  AllocateCUDAMemory<double>(&cuda_sum_of_hessians_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);

  AllocateCUDAMemory<CUDALeafSplitsStruct>(&cuda_struct_, 1, __FILE__, __LINE__);
}

void CUDALeafSplits::InitValues() {
  LaunchInitValuesEmptyKernel();
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::InitValues(
  const double lambda_l1, const double lambda_l2,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* cuda_bagging_data_indices, const data_size_t* cuda_data_indices_in_leaf,
  const data_size_t num_used_indices, hist_t* cuda_hist_in_leaf, double* root_sum_hessians) {
  cuda_gradients_ = cuda_gradients;
  cuda_hessians_ = cuda_hessians;
  SetCUDAMemory<double>(cuda_sum_of_gradients_buffer_, 0, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  SetCUDAMemory<double>(cuda_sum_of_hessians_buffer_, 0, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  LaunchInitValuesKernal(lambda_l1, lambda_l2, cuda_bagging_data_indices, cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf);
  CopyFromCUDADeviceToHost<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_, 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDALeafSplits::Resize(const data_size_t num_data) {
  if (num_data > num_data_) {
    DeallocateCUDAMemory<double>(&cuda_sum_of_gradients_buffer_, __FILE__, __LINE__);
    DeallocateCUDAMemory<double>(&cuda_sum_of_hessians_buffer_, __FILE__, __LINE__);
    num_blocks_init_from_gradients_ = (num_data + NUM_THRADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THRADS_PER_BLOCK_LEAF_SPLITS;
    AllocateCUDAMemory<double>(&cuda_sum_of_gradients_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);
    AllocateCUDAMemory<double>(&cuda_sum_of_hessians_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  } else {
    num_blocks_init_from_gradients_ = (num_data + NUM_THRADS_PER_BLOCK_LEAF_SPLITS - 1) / NUM_THRADS_PER_BLOCK_LEAF_SPLITS;
  }
  num_data_ = num_data;
}

}  // namespace LightGBM

#endif  // USE_CUDA
