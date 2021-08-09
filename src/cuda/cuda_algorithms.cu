/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
 
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

#define BITONIC_SORT_NUM_ELEMENTS (1024)
#define BITONIC_SORT_DEPTH (11)
#define BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE (10)

template <typename T, bool ASCENDING>
__global__ void BitonicSortGlobalKernel(T* values, const int num_total_data) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const int low = static_cast<int>(blockIdx.x * BITONIC_SORT_NUM_ELEMENTS);
  const bool outer_ascending = ASCENDING ? (blockIdx.x % 2 == 0) : (blockIdx.x % 2 == 1);
  T* values_pointer = values + low;
  const int num_data = min(BITONIC_SORT_NUM_ELEMENTS, num_total_data - low);
  __shared__ T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  if (thread_index < num_data) {
    shared_values[thread_index] = values_pointer[thread_index];
  }
  __syncthreads();
  for (int depth = BITONIC_SORT_DEPTH - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (BITONIC_SORT_DEPTH - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = outer_ascending ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    const int num_total_segment = (num_data + segment_length - 1) / segment_length;
    {
      const int inner_depth = depth;
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == ASCENDING) ?
        (num_total_segment * segment_length - num_data) : 0;
      const int segment_start = segment_index * segment_length;
      if (inner_segment_index_half % 2 == 0) {
        if (thread_index >= offset + segment_start) {
          const int index_to_compare = thread_index + inner_segment_length_half - offset;
          if (index_to_compare < num_data && (shared_values[thread_index] > shared_values[index_to_compare]) == ascending) {
            const T tmp = shared_values[thread_index];
            shared_values[thread_index] = shared_values[index_to_compare];
            shared_values[index_to_compare] = tmp;
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      if (inner_segment_index_half % 2 == 0) {
        const int index_to_compare = thread_index + inner_segment_length_half;
        if (index_to_compare < num_data && (shared_values[thread_index] > shared_values[index_to_compare]) == ascending) {
          const T tmp = shared_values[thread_index];
          shared_values[thread_index] = shared_values[index_to_compare];
          shared_values[index_to_compare] = tmp;
        }
      }
      __syncthreads();
    }
  }
  if (thread_index < num_data) {
    values_pointer[thread_index] = shared_values[thread_index];
  }
}

template <typename VAL_T, bool ASCENDING>
__global__ void BitonicSortMergeKernel(VAL_T* values, const int segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  const int offset = static_cast<int>(blockIdx.x * blockDim.x);
  const int local_len = min(BITONIC_SORT_NUM_ELEMENTS, len - offset);
  if (thread_index < len) {
    shared_values[threadIdx.x] = values[thread_index];
  }
  __syncthreads();
  int half_segment_length = BITONIC_SORT_NUM_ELEMENTS / 2;
  while (half_segment_length >= 1) {
    const int half_segment_index = static_cast<int>(threadIdx.x) / half_segment_length;
    if (half_segment_index % 2 == 0) {
      const int index_to_compare = static_cast<int>(threadIdx.x) + half_segment_length;
      if (index_to_compare < local_len && ((shared_values[threadIdx.x] > shared_values[index_to_compare]) == ascending)) {
        const VAL_T tmp = shared_values[index_to_compare];
        shared_values[index_to_compare] = shared_values[threadIdx.x];
        shared_values[threadIdx.x] = tmp;
      }
    }
    __syncthreads();
    half_segment_length >>= 1;
  }
  if (thread_index < len) {
    values[thread_index] = shared_values[threadIdx.x];
  }
}

template <typename VAL_T, bool ASCENDING, bool BEGIN>
__global__ void BitonicCompareKernel(VAL_T* values, const int half_segment_length, const int outer_segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / outer_segment_length;
  const int half_segment_index = thread_index / half_segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  if (half_segment_index % 2 == 0) {
    const int num_total_segment = (len + outer_segment_length - 1) / outer_segment_length;
    if (BEGIN && (half_segment_index >> 1) == num_total_segment - 1 && ascending == ASCENDING) {
      const int offset = num_total_segment * outer_segment_length - len;
      const int segment_start = segment_index * outer_segment_length;
      if (thread_index >= offset + segment_start) {
        const int index_to_compare = thread_index + half_segment_length - offset;
        if (index_to_compare < len && (values[thread_index] > values[index_to_compare]) == ascending) {
          const VAL_T tmp = values[index_to_compare];
          values[index_to_compare] = values[thread_index];
          values[thread_index] = tmp;
        }
      }
    } else {
      const int index_to_compare = thread_index + half_segment_length;
      if (index_to_compare < len) {
        if ((values[thread_index] > values[index_to_compare]) == ascending) {
          const VAL_T tmp = values[index_to_compare];
          values[index_to_compare] = values[thread_index];
          values[thread_index] = tmp;
        }
      }
    }
  }
}

template <typename VAL_T, bool ASCENDING>
void BitonicSortGlobalHelper(VAL_T* values, const size_t len) {
  int max_depth = 1;
  int len_to_shift = static_cast<int>(len) - 1;
  while (len_to_shift > 0) {
    ++max_depth;
    len_to_shift >>= 1;
  }
  const int num_blocks = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicSortGlobalKernel<VAL_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, static_cast<int>(len));
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicCompareKernel<VAL_T, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicCompareKernel<VAL_T, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    BitonicSortMergeKernel<VAL_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, segment_length, static_cast<int>(len));
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  }
}

template <>
void BitonicSortGlobal<int, true>(int* values, const size_t len) {
  BitonicSortGlobalHelper<int, true>(values, len);
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortGlobalKernel(const VAL_T* values, INDEX_T* indices, const int num_total_data) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const int low = static_cast<int>(blockIdx.x * BITONIC_SORT_NUM_ELEMENTS);
  const bool outer_ascending = ASCENDING ? (blockIdx.x % 2 == 0) : (blockIdx.x % 2 == 1);
  const VAL_T* values_pointer = values + low;
  INDEX_T* indices_pointer = indices + low;
  const int num_data = min(BITONIC_SORT_NUM_ELEMENTS, num_total_data - low);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  if (thread_index < num_data) {
    shared_values[thread_index] = values_pointer[thread_index];
    shared_indices[thread_index] = indices_pointer[thread_index];
  }
  __syncthreads();
  for (int depth = BITONIC_SORT_DEPTH - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (BITONIC_SORT_DEPTH - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = outer_ascending ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    const int num_total_segment = (num_data + segment_length - 1) / segment_length;
    {
      const int inner_depth = depth;
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == ASCENDING) ?
        (num_total_segment * segment_length - num_data) : 0;
      const int segment_start = segment_index * segment_length;
      if (inner_segment_index_half % 2 == 0) {
        if (thread_index >= offset + segment_start) {
          const int index_to_compare = thread_index + inner_segment_length_half - offset;
          const INDEX_T this_index = shared_indices[thread_index];
          const INDEX_T other_index = shared_indices[index_to_compare];
          if (index_to_compare < num_data && (shared_values[this_index] > shared_values[other_index]) == ascending) {
            shared_indices[thread_index] = other_index;
            shared_indices[index_to_compare] = this_index;
          }
        }
      }
      __syncthreads();
    }
    for (int inner_depth = depth + 1; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
      const int inner_segment_length_half = 1 << (BITONIC_SORT_DEPTH - 1 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      if (inner_segment_index_half % 2 == 0) {
        const int index_to_compare = thread_index + inner_segment_length_half;
        const INDEX_T this_index = shared_indices[thread_index];
        const INDEX_T other_index = shared_indices[thread_index];
        if (index_to_compare < num_data && (shared_values[this_index] > shared_values[other_index]) == ascending) {
          shared_indices[thread_index] = other_index;
          shared_indices[index_to_compare] = this_index;
        }
      }
      __syncthreads();
    }
  }
  if (thread_index < num_data) {
    indices_pointer[thread_index] = shared_indices[thread_index];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__global__ void BitonicArgSortMergeKernel(const VAL_T* values, INDEX_T* indices, const int segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  const int offset = static_cast<int>(blockIdx.x * blockDim.x);
  const int local_len = min(BITONIC_SORT_NUM_ELEMENTS, len - offset);
  if (thread_index < len) {
    shared_values[threadIdx.x] = values[thread_index];
    shared_indices[threadIdx.x] = indices[thread_index];
  }
  __syncthreads();
  int half_segment_length = BITONIC_SORT_NUM_ELEMENTS / 2;
  while (half_segment_length >= 1) {
    const int half_segment_index = static_cast<int>(threadIdx.x) / half_segment_length;
    if (half_segment_index % 2 == 0) {
      const int index_to_compare = static_cast<int>(threadIdx.x) + half_segment_length;
      const INDEX_T this_index = shared_indices[thread_index];
      const INDEX_T other_index = shared_indices[index_to_compare];
      if (index_to_compare < local_len && ((shared_values[this_index] > shared_values[other_index]) == ascending)) {
        shared_indices[thread_index] = other_index;
        shared_indices[index_to_compare] = this_index;
      }
    }
    __syncthreads();
    half_segment_length >>= 1;
  }
  if (thread_index < len) {
    indices[thread_index] = shared_indices[threadIdx.x];
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING, bool BEGIN>
__global__ void BitonicArgCompareKernel(const VAL_T* values, INDEX_T* indices, const int half_segment_length, const int outer_segment_length, const int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int segment_index = thread_index / outer_segment_length;
  const int half_segment_index = thread_index / half_segment_length;
  const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
  if (half_segment_index % 2 == 0) {
    const int num_total_segment = (len + outer_segment_length - 1) / outer_segment_length;
    if (BEGIN && (half_segment_index >> 1) == num_total_segment - 1 && ascending == ASCENDING) {
      const int offset = num_total_segment * outer_segment_length - len;
      const int segment_start = segment_index * outer_segment_length;
      if (thread_index >= offset + segment_start) {
        const int index_to_compare = thread_index + half_segment_length - offset;
        const INDEX_T this_index = indices[thread_index];
        const INDEX_T other_index = indices[index_to_compare];
        if (index_to_compare < len && (values[this_index] > values[other_index]) == ascending) {
          indices[thread_index] = other_index;
          indices[index_to_compare] = this_index;
        }
      }
    } else {
      const int index_to_compare = thread_index + half_segment_length;
      const INDEX_T this_index = indices[thread_index];
      const INDEX_T other_index = indices[index_to_compare];
      if (index_to_compare < len) {
        if ((values[this_index] > values[other_index]) == ascending) {
          indices[thread_index] = other_index;
          indices[index_to_compare] = this_index;
        }
      }
    }
  }
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
void BitonicArgSortGlobalHelper(const VAL_T* values, INDEX_T* indices, const size_t len) {
  int max_depth = 1;
  int len_to_shift = static_cast<int>(len) - 1;
  while (len_to_shift > 0) {
    ++max_depth;
    len_to_shift >>= 1;
  }
  const int num_blocks = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicArgSortGlobalKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, indices, static_cast<int>(len));
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicArgCompareKernel<VAL_T, INDEX_T, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        values, indices, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      half_segment_length >>= 1;
    }
    BitonicArgSortMergeKernel<VAL_T, INDEX_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
      values, indices, segment_length, static_cast<int>(len));
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  }
}

template <>
void BitonicArgSortGlobal<double, int, true>(const double* values, int* indices, const size_t len) {
  BitonicArgSortGlobalHelper<double, int, true>(values, indices, len);
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ void BitonicArgSortDevice(const VAL_T* values, INDEX_T* indices, const int len) {
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS];
  int len_to_shift = len - 1;
  int max_depth = 1;
  while (len_to_shift > 0) {
    len_to_shift >>= 1;
    ++max_depth;
  }
  const int num_blocks = (len + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS + static_cast<int>(threadIdx.x);
    if (this_index < len) {
      shared_values[threadIdx.x] = values[this_index];
      shared_indices[threadIdx.x] = this_index;
    } else {
      shared_indices[threadIdx.x] = len;
    }
    __syncthreads();
    const int num_data_in_block = min(BITONIC_SORT_NUM_ELEMENTS, len - block_index * BITONIC_SORT_NUM_ELEMENTS);
    for (int depth = max_depth - 1; depth > max_depth - 11; --depth) {
      const int segment_length = (1 << (max_depth - depth));
      const int segment_index = this_index / segment_length;
      const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
      {
        const int half_segment_length = (segment_length >> 1);
        const int half_segment_index = this_index / half_segment_length;
        const int num_total_segment = (num_data_in_block + segment_length - 1) / segment_length;
        const int offset = (segment_index == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - num_data_in_block) : 0; 
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (static_cast<int>(threadIdx.x) >= offset + segment_start) {
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
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    {
      const int num_total_segment = (len + segment_length - 1) / segment_length;
      const int half_segment_length = (segment_length >> 1);
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS + static_cast<int>(threadIdx.x);
        const int segment_index = this_index / segment_length;
        const int half_segment_index = this_index / half_segment_length;
        const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
        const int offset = ((half_segment_index >> 1) == num_total_segment - 1 && ascending == ASCENDING) ?
          (num_total_segment * segment_length - len) : 0;
        if (half_segment_index % 2 == 0) {
          const int segment_start = segment_index * segment_length;
          if (this_index >= offset + segment_start) {
            const int other_index = this_index + half_segment_length - offset;
            const INDEX_T this_data_index = indices[this_index];
            const INDEX_T other_data_index = indices[other_index];
            const VAL_T this_value = values[this_index];
            const VAL_T other_value = values[other_index];
            if ((this_value > other_value) == ascending) {
              indices[this_index] = other_data_index;
              indices[other_index] = this_data_index;
            }
          }
        }
      }
    }
    for (int inner_depth = depth + 1; inner_depth < 11; ++inner_depth) {
      const int half_segment_length = (1 << (max_depth - inner_depth - 1));
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS + static_cast<int>(threadIdx.x);
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
      const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS + static_cast<int>(threadIdx.x);
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
      for (int inner_depth = 11; inner_depth < max_depth; ++inner_depth) {
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
  }
}

__global__ void BitonicArgSortItemsGlobalKernel(const double* scores,
                                                const int num_queries,
                                                const data_size_t* cuda_query_boundaries,
                                                data_size_t* out_indices) {
  for (int query_index = 0; query_index < num_queries; query_index += BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE) {
    const data_size_t query_item_start = cuda_query_boundaries[query_index];
    const data_size_t query_item_end = cuda_query_boundaries[query_index];
    const data_size_t num_items_in_query = query_item_end - query_item_start;
    BitonicArgSortDevice<double, data_size_t, false>(scores + query_item_start,
                                                     out_indices + query_item_start,
                                                     num_items_in_query);
  }
}

void BitonicArgSortItemsGlobal(
  const double* scores,
  const int num_queries,
  const data_size_t* cuda_query_boundaries,
  data_size_t* out_indices) {
  const int num_blocks = (num_queries + BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE - 1) / BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE;
  BitonicArgSortItemsGlobalKernel<<<num_blocks, BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE>>>(
    scores, num_queries, cuda_query_boundaries, out_indices);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM
