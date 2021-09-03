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
#define GLOBAL_PREFIX_SUM_BLOCK_SIZE (1024)

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

template <typename T>
__device__ void ReduceSum(T* values, size_t n) {
  ReduceSumInner(values, n);
}

template <typename T>
__device__ void ReduceSumConflictFree(T* values, size_t n) {
  ReduceSumConflictFreeInner(values, n);
}

template <typename VAL_T, typename REDUCE_T>
void ReduceSumGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer);

template <typename VAL_T, typename REDUCE_T>
void ReduceMaxGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer);

template <typename VAL_T, typename REDUCE_T>
void ReduceMinGlobal(const VAL_T* values, size_t n, REDUCE_T* block_buffer);

template <typename T>
__device__ void ReduceMax(T* values, size_t n);

template <typename T>
void GlobalInclusivePrefixSum(T* values, T* block_buffer, size_t n);

template <bool USE_WEIGHT, bool IS_POS>
void GlobalGenAUCPosNegSum(const label_t* labels,
                        const label_t* weights,
                        const data_size_t* sorted_indices,
                        double* sum_pos_buffer,
                        double* block_sum_pos_buffer,
                        const data_size_t num_data);

void GloblGenAUCMark(const double* scores,
                     const data_size_t* sorted_indices,
                     data_size_t* mark_buffer,
                     data_size_t* block_mark_buffer,
                     uint16_t* block_mark_first_zero,
                     const data_size_t num_data);

template <bool USE_WEIGHT>
void GlobalCalcAUC(const double* sum_pos_buffer,
                   const double* sum_neg_buffer,
                   const data_size_t* mark_buffer,
                   const data_size_t num_data,
                   double* block_buffer);

template <bool USE_WEIGHT>
void GlobalCalcAveragePrecision(const double* sum_pos_buffer,
                                const double* sum_neg_buffer,
                                const data_size_t* mark_buffer,
                                const data_size_t num_data,
                                double* block_buffer);

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
__device__ __forceinline__ T ShuffleReduceMaxWarp(T value, const data_size_t len) {
  if (len > 0) {
    // TODO(shiyu1994): check how mask works
    const uint32_t mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) { 
      const T other_value = __shfl_down_sync(mask, value, offset);
      value = (other_value > value) ? other_value : value;
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
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : shared_mem_buffer[0]);
    value = ShuffleReduceMaxWarp<T>(value, num_warp);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T ShuffleReduceMinWarp(T value, const data_size_t len) {
  if (len > 0) {
    const uint32_t mask = (0xffffffff >> (warpSize - len));
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) { 
      const T other_value = __shfl_down_sync(mask, value, offset);
      value = (other_value < value) ? other_value : value;
    }
  }
  return value;
}

// reduce values from an 1-dimensional block (block size must be no greather than 1024)
template <typename T>
__device__ __forceinline__ T ShuffleReduceMin(T value, T* shared_mem_buffer, const size_t len) {
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const data_size_t warp_len = min(static_cast<data_size_t>(warpSize), static_cast<data_size_t>(len) - static_cast<data_size_t>(warpID * warpSize));
  value = ShuffleReduceMinWarp<T>(value, warp_len);
  if (warpLane == 0) {
    shared_mem_buffer[warpID] = value;
  }
  __syncthreads();
  const data_size_t num_warp = static_cast<data_size_t>((len + warpSize - 1) / warpSize);
  if (warpID == 0) {
    value = (warpLane < num_warp ? shared_mem_buffer[warpLane] : shared_mem_buffer[0]);
    value = ShuffleReduceMinWarp<T>(value, num_warp);
  }
  return value;
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ void BitonicArgSortDevice(const VAL_T* values, INDEX_T* indices, const int len);

template <typename VAL_T, typename REDUCE_VAL_T, typename INDEX_T>
__device__ void PrefixSumDevice(const VAL_T* in_values,
                                const INDEX_T* sorted_indices,
                                REDUCE_VAL_T* out_values,
                                const INDEX_T num_data) {
  __shared__ REDUCE_VAL_T shared_buffer[1025];
  const INDEX_T num_data_per_thread = (num_data + static_cast<INDEX_T>(blockDim.x) - 1) / static_cast<INDEX_T>(blockDim.x);
  const INDEX_T start = num_data_per_thread * static_cast<INDEX_T>(threadIdx.x);
  const INDEX_T end = min(start + num_data_per_thread, num_data);
  REDUCE_VAL_T thread_sum = 0;
  for (INDEX_T index = start; index < end; ++index) {
    thread_sum += static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  shared_buffer[threadIdx.x] = thread_sum;
  __syncthreads();
  PrefixSum<REDUCE_VAL_T>(shared_buffer, blockDim.x);
  const REDUCE_VAL_T thread_base = shared_buffer[threadIdx.x];
  for (INDEX_T index = start; index < end; ++index) {
    out_values[index] = thread_base + static_cast<REDUCE_VAL_T>(in_values[sorted_indices[index]]);
  }
  __syncthreads();
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename REDUCE_WEIGHT_T, bool ASCENDING, bool USE_WEIGHT>
__device__ VAL_T PercentileDevice(const VAL_T* values,
                                  const WEIGHT_T* weights,
                                  INDEX_T* indices,
                                  REDUCE_WEIGHT_T* weights_prefix_sum,
                                  const double alpha,
                                  const INDEX_T len);

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename WEIGHT_REDUCE_T, bool ASCENDING, bool USE_WEIGHT>
__global__ void PercentileGlobalKernel(const VAL_T* values,
                                       const WEIGHT_T* weights,
                                       const INDEX_T* sorted_indices,
                                       const WEIGHT_REDUCE_T* weights_prefix_sum,
                                       const double alpha,
                                       const INDEX_T len,
                                       VAL_T* out_value) {
  if (!USE_WEIGHT) {
    const double float_pos = (1.0f - alpha) * len;
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);
    if (pos < 1) {
      *out_value = values[sorted_indices[0]];
    } else if (pos >= len) {
      *out_value = values[sorted_indices[len - 1]];
    } else {
      const double bias = float_pos - static_cast<double>(pos);
      const VAL_T v1 = values[sorted_indices[pos - 1]];
      const VAL_T v2 = values[sorted_indices[pos]];
      *out_value = static_cast<VAL_T>(v1 - (v1 - v2) * bias);
    }
  } else {
    const WEIGHT_REDUCE_T threshold = weights_prefix_sum[len - 1] * (1.0f - alpha);
    __shared__ INDEX_T pos;
    if (threadIdx.x == 0) {
      pos = len;
    }
    __syncthreads();
    for (INDEX_T index = static_cast<INDEX_T>(threadIdx.x); index < len; index += static_cast<INDEX_T>(blockDim.x)) {
      if (weights_prefix_sum[index] > threshold && (index == 0 || weights_prefix_sum[index - 1] <= threshold)) {
        pos = index;
      }
    }
    __syncthreads();
    pos = min(pos, len - 1);
    if (pos == 0 || pos == len - 1) {
      *out_value = values[pos];
    }
    const VAL_T v1 = values[sorted_indices[pos - 1]];
    const VAL_T v2 = values[sorted_indices[pos]];
    *out_value = static_cast<VAL_T>(v1 - (v1 - v2) * (threshold - weights_prefix_sum[pos - 1]) / (weights_prefix_sum[pos] - weights_prefix_sum[pos - 1]));
  }
}

template <typename VAL_T, typename REDUCE_T, typename INDEX_T>
void GlobalInclusiveArgPrefixSum(const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, size_t n);

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename WEIGHT_REDUCE_T, bool ASCENDING, bool USE_WEIGHT>
void PercentileGlobal(const VAL_T* values,
                      const WEIGHT_T* weights,
                      INDEX_T* indices,
                      WEIGHT_REDUCE_T* weights_prefix_sum,
                      WEIGHT_REDUCE_T* weights_prefix_sum_buffer,
                      const double alpha,
                      const INDEX_T len,
                      VAL_T* cuda_out_value) {
  if (len <= 1) {
    CopyFromCUDADeviceToCUDADeviceOuter<VAL_T>(cuda_out_value, values, 1, __FILE__, __LINE__);
  }
  BitonicArgSortGlobal<VAL_T, INDEX_T, ASCENDING>(values, indices, len);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  if (USE_WEIGHT) {
    Log::Warning("before prefix sum");
    GlobalInclusiveArgPrefixSum<WEIGHT_T, WEIGHT_REDUCE_T, INDEX_T>(indices, weights, weights_prefix_sum, weights_prefix_sum_buffer, static_cast<size_t>(len));
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    Log::Warning("after prefix sum");
  PercentileGlobalKernel<VAL_T, INDEX_T, WEIGHT_T, WEIGHT_REDUCE_T, ASCENDING, USE_WEIGHT><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, weights, indices, weights_prefix_sum, alpha, len, cuda_out_value);  
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  Log::Warning("after percentile");
}

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_ALGORITHMS_HPP_
