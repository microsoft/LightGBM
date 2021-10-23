/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
#define LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/utils/log.h>

#include <algorithm>

#define NUM_BANKS_DATA_PARTITION (16)
#define LOG_NUM_BANKS_DATA_PARTITION (4)
#define GLOBAL_PREFIX_SUM_BLOCK_SIZE (1024)

#define BITONIC_SORT_NUM_ELEMENTS (1024)
#define BITONIC_SORT_DEPTH (11)
#define BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE (10)

#define CONFLICT_FREE_INDEX(n) \
  ((n) + ((n) >> LOG_NUM_BANKS_DATA_PARTITION)) \

namespace LightGBM {

template <typename T>
__device__ __forceinline__ T ShufflePrefixSum(T value, T* shared_mem_buffer) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
    const T other_value = __shfl_up_sync(mask, value, offset);
    if (warpLane >= offset) {
      value += other_value;
    }
  }
  if (warpLane == warpSize - 1) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  if (warpID == 0) {
    T warp_sum = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    for (uint32_t offset = 1; offset < warpSize; offset <<= 1) {
      const T other_warp_sum = __shfl_up_sync(mask, warp_sum, offset);
      if (warpLane >= offset) {
        warp_sum += other_warp_sum;
      }
    }
    shared_mem_buffer[warpLane] = warp_sum;
  }
  __syncthreads();
  const T warp_base = warpID == 0 ? 0 : shared_mem_buffer[warpID - 1];
  return warp_base + value;
}

template <typename T>
void ShufflePrefixSumGlobal(T* values, size_t len, T* block_prefix_sum_buffer);

template <typename T>
__device__ __forceinline__ T ShuffleReduceSumWarp(T value, const data_size_t len) {
  if (len > 0) {
    // TODO(shiyu1994): check how mask works
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greather than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceSum(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceSumWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceSumWarp<T>(value, num_warp);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T GlobalMemoryPrefixSum(T* array, const size_t len) {
  const size_t num_values_per_thread = (len + blockDim.x - 1) / blockDim.x;
  const size_t start = threadIdx.x * num_values_per_thread;
  const size_t end = min(start + num_values_per_thread, len);
  T thread_sum = 0;
  for (size_t index = start; index < end; ++index) {
    thread_sum += array[index];
  }
  __shared__ T shared_mem[32];
  const T thread_base = ShuffleReduceSum<T>(thread_sum, shared_mem);
  
}

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
