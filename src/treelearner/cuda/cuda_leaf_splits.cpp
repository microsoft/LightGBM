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
}

void CUDALeafSplits::Init() {
  num_blocks_init_from_gradients_ = (num_data_ + INIT_SUM_BLOCK_SIZE_LEAF_SPLITS - 1) / INIT_SUM_BLOCK_SIZE_LEAF_SPLITS;

  // allocate more memory for sum reduction in CUDA
  // only the first element records the final sum
  AllocateCUDAMemoryOuter<double>(&cuda_sum_of_gradients_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_sum_of_hessians_buffer_, num_blocks_init_from_gradients_, __FILE__, __LINE__);

  AllocateCUDAMemoryOuter<CUDALeafSplitsStruct>(&cuda_struct_, 1, __FILE__, __LINE__);
}

void CUDALeafSplits::InitValues() {
  LaunchInitValuesEmptyKernel();
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDALeafSplits::InitValues(
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
  hist_t* cuda_hist_in_leaf, double* root_sum_hessians) {
  cuda_gradients_ = cuda_gradients;
  cuda_hessians_ = cuda_hessians;
  SetCUDAMemoryOuter<double>(cuda_sum_of_gradients_buffer_, 0, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_sum_of_hessians_buffer_, 0, num_blocks_init_from_gradients_, __FILE__, __LINE__);
  LaunchInitValuesKernal(cuda_data_indices_in_leaf, num_used_indices, cuda_hist_in_leaf);
  CopyFromCUDADeviceToHostOuter<double>(root_sum_hessians, cuda_sum_of_hessians_buffer_, 1, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
