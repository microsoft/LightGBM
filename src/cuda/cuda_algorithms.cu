/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

template <typename T>
__global__ void ShufflePrefixSumGlobalKernel(T* values, size_t len, T* block_prefix_sum_buffer) {
  __shared__ T shared_mem_buffer[32];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  T value = 0;
  if (index < len) {
    value = values[index];
  }
  const T prefix_sum_value = ShufflePrefixSum<T>(value, shared_mem_buffer);
  values[index] = prefix_sum_value;
  if (threadIdx.x == blockDim.x - 1) {
    block_prefix_sum_buffer[blockIdx.x] = prefix_sum_value;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalReduceBlockKernel(T* block_prefix_sum_buffer, int num_blocks) {
  __shared__ T shared_mem_buffer[32];
  const int num_blocks_per_thread = (num_blocks + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 2) / (GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1);
  int thread_block_start = threadIdx.x == 0 ? 0 : (threadIdx.x - 1) * num_blocks_per_thread;
  int thread_block_end = threadIdx.x == 0 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  T base = 0;
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    base += block_prefix_sum_buffer[block_index];
  }
  base = ShufflePrefixSum<T>(base, shared_mem_buffer);
  thread_block_start = threadIdx.x == blockDim.x - 1 ? 0 : threadIdx.x * num_blocks_per_thread;
  thread_block_end = threadIdx.x == blockDim.x - 1 ? 0 : min(thread_block_start + num_blocks_per_thread, num_blocks);
  for (int block_index = thread_block_start + 1; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += block_prefix_sum_buffer[block_index - 1];
  }
  for (int block_index = thread_block_start; block_index < thread_block_end; ++block_index) {
    block_prefix_sum_buffer[block_index] += base;
  }
}

template <typename T>
__global__ void ShufflePrefixSumGlobalAddBase(size_t len, const T* block_prefix_sum_buffer, T* values) {
  const T base = blockIdx.x == 0 ? 0 : block_prefix_sum_buffer[blockIdx.x - 1];
  const size_t index = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (index < len) {
    values[index] += base;
  }
}

template <typename T>
void ShufflePrefixSumGlobalInner(T* values, size_t len, T* block_prefix_sum_buffer) {
  const int num_blocks = (static_cast<int>(len) + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ShufflePrefixSumGlobalKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, len, block_prefix_sum_buffer);
  ShufflePrefixSumGlobalReduceBlockKernel<<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_prefix_sum_buffer, num_blocks);
  ShufflePrefixSumGlobalAddBase<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(len, block_prefix_sum_buffer, values);
}

template <>
void ShufflePrefixSumGlobal(uint16_t* values, size_t len, uint16_t* block_prefix_sum_buffer) {
  ShufflePrefixSumGlobalInner<uint16_t>(values, len, block_prefix_sum_buffer);
}

template <>
void ShufflePrefixSumGlobal(uint32_t* values, size_t len, uint32_t* block_prefix_sum_buffer) {
  ShufflePrefixSumGlobalInner<uint32_t>(values, len, block_prefix_sum_buffer);
}

template <>
void ShufflePrefixSumGlobal(uint64_t* values, size_t len, uint64_t* block_prefix_sum_buffer) {
  ShufflePrefixSumGlobalInner<uint64_t>(values, len, block_prefix_sum_buffer);
}

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
