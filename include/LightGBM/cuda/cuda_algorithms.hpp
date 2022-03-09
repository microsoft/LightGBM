/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
#define LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_

#ifdef USE_CUDA_EXP

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
__device__ __forceinline__ T ShufflePrefixSumExclusive(T value, T* shared_mem_buffer) {
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
  const T inclusive_result = warp_base + value;
  if (threadIdx.x % warpSize == warpSize - 1) {
    shared_mem_buffer[warpLane] = inclusive_result;
  }
  __syncthreads();
  T exclusive_result = __shfl_up_sync(mask, inclusive_result, 1);
  if (threadIdx.x == 0) {
    exclusive_result = 0;
  } else if (threadIdx.x % warpSize == 0) {
    exclusive_result = shared_mem_buffer[warpLane - 1];
  }
  return exclusive_result;
}

template <typename T>
void ShufflePrefixSumGlobal(T* values, size_t len, T* block_prefix_sum_buffer);

template <typename T>
__device__ __forceinline__ T ShuffleReduceSumWarp(T value, const data_size_t len) {
  if (len > 0) {
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
__device__ __forceinline__ T ShuffleReduceMaxWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      value = max(value, __shfl_down_sync(mask, value, offset));
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greather than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceMax(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMaxWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : 0);
    value = ShuffleReduceMaxWarp<T>(value, num_warp);
  }
  return value;
}

// calculate prefix sum values within an 1-dimensional block in global memory, exclusively
template <typename T>
__device__ __forceinline__ void GlobalMemoryPrefixSum(T* array, const size_t len) {
  const size_t num_values_per_thread = (len + blockDim.x - 1) / blockDim.x;
  const size_t start = threadIdx.x * num_values_per_thread;
  const size_t end = min(start + num_values_per_thread, len);
  T thread_sum = 0;
  for (size_t index = start; index < end; ++index) {
    thread_sum += array[index];
  }
  __shared__ T shared_mem[32];
  const T thread_base = ShufflePrefixSumExclusive<T>(thread_sum, shared_mem);
  if (start < end) {
    array[start] += thread_base;
  }
  for (size_t index = start + 1; index < end; ++index) {
    array[index] += array[index - 1];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ __forceinline__ void BitonicArgSort_1024(const VAL_T* scores, INDEX_T* indices, const INDEX_T num_items) {
  INDEX_T depth = 1;
  INDEX_T num_items_aligend = 1;
  INDEX_T num_items_ref = num_items - 1;
  while (num_items_ref > 0) {
    num_items_ref >>= 1;
    num_items_aligend <<= 1;
    ++depth;
  }
  for (INDEX_T outer_depth = depth - 1; outer_depth >= 1; --outer_depth) {
    const INDEX_T outer_segment_length = 1 << (depth - outer_depth);
    const INDEX_T outer_segment_index = threadIdx.x / outer_segment_length;
    const bool ascending = ASCENDING ? (outer_segment_index % 2 == 0) : (outer_segment_index % 2 > 0);
    for (INDEX_T inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const INDEX_T segment_length = 1 << (depth - inner_depth);
      const INDEX_T half_segment_length = segment_length >> 1;
      const INDEX_T half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligend) {
        if (half_segment_index % 2 == 0) {
          const INDEX_T index_to_compare = threadIdx.x + half_segment_length;
          if ((scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) == ascending) {
            const INDEX_T index = indices[threadIdx.x];
            indices[threadIdx.x] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
      }
      __syncthreads();
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, uint32_t BLOCK_DIM, uint32_t MAX_DEPTH>
__device__ void BitonicArgSortDevice(const VAL_T* values, INDEX_T* indices, const int len) {
  __shared__ VAL_T shared_values[BLOCK_DIM];
  __shared__ INDEX_T shared_indices[BLOCK_DIM];
  int len_to_shift = len - 1;
  int max_depth = 1;
  while (len_to_shift > 0) {
    len_to_shift >>= 1;
    ++max_depth;
  }
  const int num_blocks = (len + static_cast<int>(BLOCK_DIM) - 1) / static_cast<int>(BLOCK_DIM);
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
    if (this_index < len) {
      shared_values[threadIdx.x] = values[this_index];
      shared_indices[threadIdx.x] = this_index;
    } else {
      shared_indices[threadIdx.x] = len;
    }
    __syncthreads();
    for (int depth = max_depth - 1; depth > max_depth - static_cast<int>(MAX_DEPTH); --depth) {
      const int segment_length = (1 << (max_depth - depth));
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      {
        const int half_segment_length = (segment_length >> 1);
        const int half_segment_index = this_index / half_segment_length;
        const int num_total_segment = (len + segment_length - 1) / segment_length;
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = static_cast<int>(threadIdx.x) + half_segment_length - offset;
            const INDEX_T this_data_index = shared_indices[threadIdx.x];
            const INDEX_T other_data_index = shared_indices[other_index];
            const VAL_T this_value = shared_values[threadIdx.x];
            const VAL_T other_value = shared_values[other_index];
            if (other_data_index < len && (this_value > other_value) == ascending) {
              shared_indices[threadIdx.x] = other_data_index;
              shared_indices[other_index] = this_data_index;
              shared_values[threadIdx.x] = other_value;
              shared_values[other_index] = this_value;
            }
          }
        }
        __syncthreads();
      }
      for (int inner_depth = depth + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
    }
    if (this_index < len) {
      indices[this_index] = shared_indices[threadIdx.x];
    }
    __syncthreads();
  }
  for (int depth = max_depth - static_cast<int>(MAX_DEPTH); depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    {
      const int num_total_segment = (len + segment_length - 1) / segment_length;
      const int half_segment_length = (segment_length >> 1);
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = this_index + half_segment_length - offset;
            if (other_index < len) {
              const INDEX_T this_data_index = indices[this_index];
              const INDEX_T other_data_index = indices[other_index];
              const VAL_T this_value = values[this_data_index];
              const VAL_T other_value = values[other_data_index];
              if ((this_value > other_value) == ascending) {
                indices[this_index] = other_data_index;
                indices[other_index] = this_data_index;
              }
            }
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - static_cast<int>(MAX_DEPTH); ++inner_depth) {
      const int half_segment_length = (1 << (max_depth - inner_depth - 1));
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        if (half_segment_index % 2 == 0) {
          const int other_index = this_index + half_segment_length;
          if (other_index < len) {
            const INDEX_T this_data_index = indices[this_index];
            const INDEX_T other_data_index = indices[other_index];
            const VAL_T this_value = values[this_data_index];
            const VAL_T other_value = values[other_data_index];
            if ((this_value > other_value) == ascending) {
              indices[this_index] = other_data_index;
              indices[other_index] = this_data_index;
            }
          }
        }
        __syncthreads();
      }
    }
    for (int block_index = 0; block_index < num_blocks; ++block_index) {
      const int this_index = block_index * static_cast<int>(BLOCK_DIM) + static_cast<int>(threadIdx.x);
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      if (this_index < len) {
        const INDEX_T index = indices[this_index];
        shared_values[threadIdx.x] = values[index];
        shared_indices[threadIdx.x] = index;
      } else {
        shared_indices[threadIdx.x] = len;
      }
      __syncthreads();
      for (int inner_depth = max_depth - static_cast<int>(MAX_DEPTH) + 1; inner_depth < max_depth; ++inner_depth) {
        const int half_segment_length = (1 << (max_depth - inner_depth - 1));
        const int half_segment_index = this_index / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const int other_index = static_cast<int>(threadIdx.x) + half_segment_length;
          const INDEX_T this_data_index = shared_indices[threadIdx.x];
          const INDEX_T other_data_index = shared_indices[other_index];
          const VAL_T this_value = shared_values[threadIdx.x];
          const VAL_T other_value = shared_values[other_index];
          if (other_data_index < len && (this_value > other_value) == ascending) {
            shared_indices[threadIdx.x] = other_data_index;
            shared_indices[other_index] = this_data_index;
            shared_values[threadIdx.x] = other_value;
            shared_values[other_index] = this_value;
          }
        }
        __syncthreads();
      }
      if (this_index < len) {
        indices[this_index] = shared_indices[threadIdx.x];
      }
      __syncthreads();
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
