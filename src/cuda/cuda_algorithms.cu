/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
 
#include <LightGBM/cuda/cuda_algorithms.hpp>

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


template <>
__device__ void ReduceSumConflictFree<uint16_t>(uint16_t* values, size_t n) {
  ReduceSumConflictFreeInner(values, n);
}

template <>
__device__ void PrefixSumConflictFree<uint16_t>(uint16_t* values, size_t n) {
  PrefixSumInner(values, n, uint16_t);
}

template <>
__device__ void PrefixSumConflictFree<uint32_t>(uint32_t* values, size_t n) {
  PrefixSumInner(values, n, uint32_t);
}

template <>
__device__ void PrefixSumConflictFree<hist_t>(hist_t* values, size_t n) {
  PrefixSumInner(values, n, hist_t);
}

template <>
__device__ void ReduceSum<hist_t>(hist_t* values, size_t n) {
  ReduceSumInner(values, n);
}

}  // namespace LightGBM
