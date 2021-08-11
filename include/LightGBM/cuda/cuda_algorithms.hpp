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

#define NUM_BANKS_DATA_PARTITION (16)
#define LOG_NUM_BANKS_DATA_PARTITION (4)

#define CONFLICT_FREE_INDEX(n) \
  ((n) + ((n) >> LOG_NUM_BANKS_DATA_PARTITION)) \

namespace LightGBM {

#define ReduceSumInner(values, n) \
  const unsigned int thread_index = threadIdx.x; \
  for (size_t s = 1; s < n; s <<= 1) { \
    if (thread_index % (s << 1) == 0 && (thread_index + s) < n) { \
      values[thread_index] += values[thread_index + s]; \
    } \
    __syncthreads(); \
  }


#define ReduceSumConflictFreeInner(values, n) \
  const unsigned int thread_index = threadIdx.x; \
  for (size_t s = 1; s < n; s <<= 1) { \
    if (thread_index % (s << 1) == 0 && (thread_index + s) < n) { \
      values[CONFLICT_FREE_INDEX(thread_index)] += values[CONFLICT_FREE_INDEX(thread_index + s)]; \
    } \
    __syncthreads(); \
  } \


#define PrefixSumInner(elements, n, type) \
  size_t offset = 1; \
  unsigned int threadIdx_x = threadIdx.x; \
  const size_t conflict_free_n_minus_1 = CONFLICT_FREE_INDEX(n - 1); \
  const type last_element = elements[conflict_free_n_minus_1]; \
  __syncthreads(); \
  for (int d = (n >> 1); d > 0; d >>= 1) { \
    if (threadIdx_x < d) { \
      const size_t src_pos = offset * (2 * threadIdx_x + 1) - 1; \
      const size_t dst_pos = offset * (2 * threadIdx_x + 2) - 1; \
      elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)]; \
    } \
    offset <<= 1; \
    __syncthreads(); \
  } \
  if (threadIdx_x == 0) { \
    elements[conflict_free_n_minus_1] = 0; \
  } \
  __syncthreads(); \
  for (int d = 1; d < n; d <<= 1) { \
    offset >>= 1; \
    if (threadIdx_x < d) { \
      const size_t dst_pos = offset * (2 * threadIdx_x + 2) - 1; \
      const size_t src_pos = offset * (2 * threadIdx_x + 1) - 1; \
      const size_t conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos); \
      const size_t conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos); \
      const type src_val = elements[conflict_free_src_pos]; \
      elements[conflict_free_src_pos] = elements[conflict_free_dst_pos]; \
      elements[conflict_free_dst_pos] += src_val; \
    } \
    __syncthreads(); \
  } \
  if (threadIdx_x == 0) { \
    elements[CONFLICT_FREE_INDEX(n)] = elements[conflict_free_n_minus_1] + last_element; \
  } \

template <typename T>
__device__ void ReduceSum(T* values, size_t n) {
  ReduceSumInner(values, n);
}

template <typename T>
__device__ void ReduceSumConflictFree(T* values, size_t n) {
  ReduceSumConflictFreeInner(values, n);
}

template <typename T>
__device__ void ReduceMax(T* values, size_t n);

template <typename T>
__device__ void PrefixSum(T* values, size_t n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = static_cast<int>(threadIdx.x);
  const T last_element = values[n - 1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      values[dst_pos] += values[src_pos];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    values[n - 1] = 0; 
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const T src_val = values[src_pos];
      values[src_pos] = values[dst_pos];
      values[dst_pos] += src_val;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    values[n] = values[n - 1] + last_element;
  }
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void PrefixSumConflictFree(T* values, size_t n) {
  size_t offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  const size_t conflict_free_n_minus_1 = CONFLICT_FREE_INDEX(n - 1);
  const T last_element = values[conflict_free_n_minus_1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const size_t src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const size_t dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      values[CONFLICT_FREE_INDEX(dst_pos)] += values[CONFLICT_FREE_INDEX(src_pos)];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    values[conflict_free_n_minus_1] = 0;
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const size_t dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const size_t src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const size_t conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
      const size_t conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
      const T src_val = values[conflict_free_src_pos];
      values[conflict_free_src_pos] = values[conflict_free_dst_pos];
      values[conflict_free_dst_pos] += src_val;
    }
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    values[CONFLICT_FREE_INDEX(n)] = values[conflict_free_n_minus_1] + last_element;
  }
}

template <typename VAL_T, bool ASCENDING>
void BitonicSortGlobal(VAL_T* values, const size_t len);

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
void BitonicArgSortGlobal(const VAL_T* values, INDEX_T* indices, const size_t len);

void BitonicArgSortItemsGlobal(const double* values,
                               const int num_queries,
                               const data_size_t* cuda_query_boundaries,
                               data_size_t* out_indices);

__device__ __forceinline__ void BitonicArgSort_1024(const score_t* scores, uint16_t* indices, const uint16_t num_items) {
  uint16_t depth = 1;
  uint16_t num_items_aligend = 1;
  uint16_t num_items_ref = num_items - 1;
  while (num_items_ref > 0) {
    num_items_ref >>= 1;
    num_items_aligend <<= 1;
    ++depth;
  }
  for (uint16_t outer_depth = depth - 1; outer_depth >= 1; --outer_depth) {
    const uint16_t outer_segment_length = 1 << (depth - outer_depth);
    const uint16_t outer_segment_index = threadIdx.x / outer_segment_length;
    const bool ascending = (outer_segment_index % 2 > 0);
    for (uint16_t inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const uint16_t segment_length = 1 << (depth - inner_depth);
      const uint16_t half_segment_length = segment_length >> 1;
      const uint16_t half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligend) {
        if (half_segment_index % 2 == 0) {
          const uint16_t index_to_compare = threadIdx.x + half_segment_length;
          if ((scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) == ascending) {
            const uint16_t index = indices[threadIdx.x];
            indices[threadIdx.x] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void BitonicArgSort_2048(const score_t* scores, uint16_t* indices) {
  for (uint16_t base = 0; base < 2048; base += 1024) {
    for (uint16_t outer_depth = 10; outer_depth >= 1; --outer_depth) {
      const uint16_t outer_segment_length = 1 << (11 - outer_depth);
      const uint16_t outer_segment_index = threadIdx.x / outer_segment_length;
      const bool ascending = (base == 0) ? (outer_segment_index % 2 > 0) : (outer_segment_index % 2 == 0);
      for (uint16_t inner_depth = outer_depth; inner_depth < 11; ++inner_depth) {
        const uint16_t segment_length = 1 << (11 - inner_depth);
        const uint16_t half_segment_length = segment_length >> 1;
        const uint16_t half_segment_index = threadIdx.x / half_segment_length;
        if (half_segment_index % 2 == 0) {
          const uint16_t index_to_compare = threadIdx.x + half_segment_length + base;
          if ((scores[indices[threadIdx.x + base]] > scores[indices[index_to_compare]]) == ascending) {
            const uint16_t index = indices[threadIdx.x + base];
            indices[threadIdx.x + base] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
        __syncthreads();
      }
    }
  }
  const unsigned int index_to_compare = threadIdx.x + 1024;
  if (scores[indices[index_to_compare]] > scores[indices[threadIdx.x]]) {
    const uint16_t temp_index = indices[index_to_compare];
    indices[index_to_compare] = indices[threadIdx.x];
    indices[threadIdx.x] = temp_index;
  }
  __syncthreads();
  for (uint16_t base = 0; base < 2048; base += 1024) {
    for (uint16_t inner_depth = 1; inner_depth < 11; ++inner_depth) {
      const uint16_t segment_length = 1 << (11 - inner_depth);
      const uint16_t half_segment_length = segment_length >> 1;
      const uint16_t half_segment_index = threadIdx.x / half_segment_length;
      if (half_segment_index % 2 == 0) {
        const uint16_t index_to_compare = threadIdx.x + half_segment_length + base;
        if (scores[indices[threadIdx.x + base]] < scores[indices[index_to_compare]]) {
          const uint16_t index = indices[threadIdx.x + base];
          indices[threadIdx.x + base] = indices[index_to_compare];
          indices[index_to_compare] = index;
        }
      }
      __syncthreads();
    }
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
