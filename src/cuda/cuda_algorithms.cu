/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
 
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

template <typename T>
__global__ void BlockReduceMax(T* block_buffer, const data_size_t num_blocks) {
  __shared__ T shared_buffer[32];
  T thread_max = 0;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    const T value = block_buffer[block_index];
    if (value > thread_max) {
      thread_max = value;
    }
  }
  thread_max = ShuffleReduceMax<T>(thread_max, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[0] = thread_max;
  }
}

}  // namespace LightGBM
