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
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == outer_ascending) ?
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

template <>
void BitonicSortGlobal<double, true>(double* values, const size_t len) {
  BitonicSortGlobalHelper<double, true>(values, len);
}

template <>
void BitonicSortGlobal<double, false>(double* values, const size_t len) {
  BitonicSortGlobalHelper<double, false>(values, len);
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
    shared_indices[thread_index] = static_cast<INDEX_T>(thread_index + blockIdx.x * blockDim.x);
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
      const int offset = ((inner_segment_index_half >> 1) == num_total_segment - 1 && ascending == outer_ascending) ?
        (num_total_segment * segment_length - num_data) : 0;
      const int segment_start = segment_index * segment_length;
      if (inner_segment_index_half % 2 == 0) {
        if (thread_index >= offset + segment_start) {
          const int index_to_compare = thread_index + inner_segment_length_half - offset;
          const INDEX_T this_index = shared_indices[thread_index];
          const INDEX_T other_index = shared_indices[index_to_compare];
          const VAL_T this_value = shared_values[thread_index];
          const VAL_T other_value = shared_values[index_to_compare];
          if (index_to_compare < num_data && (this_value > other_value) == ascending) {
            shared_indices[thread_index] = other_index;
            shared_indices[index_to_compare] = this_index;
            shared_values[thread_index] = other_value;
            shared_values[index_to_compare] = this_value;
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
        const INDEX_T other_index = shared_indices[index_to_compare];
        const VAL_T this_value = shared_values[thread_index];
        const VAL_T other_value = shared_values[index_to_compare];
        if (index_to_compare < num_data && (this_value > other_value) == ascending) {
          shared_indices[thread_index] = other_index;
          shared_indices[index_to_compare] = this_index;
          shared_values[thread_index] = other_value;
          shared_values[index_to_compare] = this_value;
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
    const INDEX_T index = indices[thread_index];
    shared_values[threadIdx.x] = values[index];
    shared_indices[threadIdx.x] = index;
  }
  __syncthreads();
  int half_segment_length = BITONIC_SORT_NUM_ELEMENTS / 2;
  while (half_segment_length >= 1) {
    const int half_segment_index = static_cast<int>(threadIdx.x) / half_segment_length;
    if (half_segment_index % 2 == 0) {
      const int index_to_compare = static_cast<int>(threadIdx.x) + half_segment_length;
      const INDEX_T this_index = shared_indices[threadIdx.x];
      const INDEX_T other_index = shared_indices[index_to_compare];
      const VAL_T this_value = shared_values[threadIdx.x];
      const VAL_T other_value = shared_values[index_to_compare];
      if (index_to_compare < local_len && ((this_value > other_value) == ascending)) {
        shared_indices[threadIdx.x] = other_index;
        shared_indices[index_to_compare] = this_index;
        shared_values[threadIdx.x] = other_value;
        shared_values[index_to_compare] = this_value;
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
        if (index_to_compare < len) {
          const INDEX_T this_index = indices[thread_index];
          const INDEX_T other_index = indices[index_to_compare];
          if ((values[this_index] > values[other_index]) == ascending) {
            indices[thread_index] = other_index;
            indices[index_to_compare] = this_index;
          }
        }
      }
    } else {
      const int index_to_compare = thread_index + half_segment_length;
      if (index_to_compare < len) {
        const INDEX_T this_index = indices[thread_index];
        const INDEX_T other_index = indices[index_to_compare];
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
void BitonicArgSortGlobal<double, data_size_t, false>(const double* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<double, data_size_t, false>(values, indices, len);
}

template <>
void BitonicArgSortGlobal<double, data_size_t, true>(const double* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<double, data_size_t, true>(values, indices, len);
}

template <>
void BitonicArgSortGlobal<label_t, data_size_t, false>(const label_t* values, data_size_t* indices, const size_t len) {
  BitonicArgSortGlobalHelper<label_t, data_size_t, false>(values, indices, len);
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
    for (int depth = max_depth - 1; depth > max_depth - 11; --depth) {
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
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
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
      for (int inner_depth = max_depth - 10; inner_depth < max_depth; ++inner_depth) {
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

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ void BitonicArgSortDevice512(const VAL_T* values, INDEX_T* indices, const int len) {
  __shared__ VAL_T shared_values[BITONIC_SORT_NUM_ELEMENTS / 2];
  __shared__ INDEX_T shared_indices[BITONIC_SORT_NUM_ELEMENTS / 2];
  int len_to_shift = len - 1;
  int max_depth = 1;
  while (len_to_shift > 0) {
    len_to_shift >>= 1;
    ++max_depth;
  }
  const int num_blocks = (len + (BITONIC_SORT_NUM_ELEMENTS / 2) - 1) / (BITONIC_SORT_NUM_ELEMENTS / 2);
  for (int block_index = 0; block_index < num_blocks; ++block_index) {
    const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS / 2 + static_cast<int>(threadIdx.x);
    if (this_index < len) {
      shared_values[threadIdx.x] = values[this_index];
      shared_indices[threadIdx.x] = this_index;
    } else {
      shared_indices[threadIdx.x] = len;
    }
    __syncthreads();
    for (int depth = max_depth - 1; depth > max_depth - 10; --depth) {
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
  for (int depth = max_depth - 10; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    {
      const int num_total_segment = (len + segment_length - 1) / segment_length;
      const int half_segment_length = (segment_length >> 1);
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS / 2 + static_cast<int>(threadIdx.x);
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
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 10; ++inner_depth) {
      const int half_segment_length = (1 << (max_depth - inner_depth - 1));
      for (int block_index = 0; block_index < num_blocks; ++block_index) {
        const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS / 2 + static_cast<int>(threadIdx.x);
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
      const int this_index = block_index * BITONIC_SORT_NUM_ELEMENTS / 2 + static_cast<int>(threadIdx.x);
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
      for (int inner_depth = max_depth - 9; inner_depth < max_depth; ++inner_depth) {
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

__global__ void BitonicArgSortItemsGlobalKernel(const double* scores,
                                                const int num_queries,
                                                const data_size_t* cuda_query_boundaries,
                                                data_size_t* out_indices) {
  const int query_index_start = static_cast<int>(blockIdx.x) * BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE;
  const int query_index_end = min(query_index_start + BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE, num_queries);
  for (int query_index = query_index_start; query_index < query_index_end; ++query_index) {
    const data_size_t query_item_start = cuda_query_boundaries[query_index];
    const data_size_t query_item_end = cuda_query_boundaries[query_index + 1];
    const data_size_t num_items_in_query = query_item_end - query_item_start;
    BitonicArgSortDevice<double, data_size_t, false>(scores + query_item_start,
                                                     out_indices + query_item_start,
                                                     num_items_in_query);
    __syncthreads();
  }
}

void BitonicArgSortItemsGlobal(
  const double* scores,
  const int num_queries,
  const data_size_t* cuda_query_boundaries,
  data_size_t* out_indices) {
  const int num_blocks = (num_queries + BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE - 1) / BITONIC_SORT_QUERY_ITEM_BLOCK_SIZE;
  BitonicArgSortItemsGlobalKernel<<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
    scores, num_queries, cuda_query_boundaries, out_indices);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

__device__ void PrefixSumZeroOut(data_size_t* values, size_t n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = static_cast<int>(threadIdx.x);
  const data_size_t last_element = values[n - 1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      if (values[dst_pos] != 0) {
        values[dst_pos] += values[src_pos];
      }
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
      const data_size_t src_val = values[src_pos];
      values[src_pos] = values[dst_pos];
      if (src_val != 0) {
        values[dst_pos] += src_val;
      } else {
        values[dst_pos] = 0;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (last_element != 0) {
      values[n] = values[n - 1] + last_element;
    } else {
      values[n] = 0;
    }
  }
  __syncthreads();
}

__device__ void PrefixSumZeroOut(data_size_t* values, bool* is_all_non_zero, size_t n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = static_cast<int>(threadIdx.x);
  const data_size_t last_element = values[n - 1];
  const bool last_is_all_non_zero = is_all_non_zero[n - 1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      if (is_all_non_zero[dst_pos]) {
        values[dst_pos] += values[src_pos];
        is_all_non_zero[dst_pos] &= is_all_non_zero[src_pos];
      }
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    values[n - 1] = 0; 
    is_all_non_zero[n - 1] = true;
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const data_size_t src_val = values[src_pos];
      const bool src_is_all_non_zero = is_all_non_zero[src_pos];
      values[src_pos] = values[dst_pos];
      is_all_non_zero[src_pos] = is_all_non_zero[dst_pos];
      if (src_is_all_non_zero) {
        values[dst_pos] += src_val;
      } else {
        values[dst_pos] = src_val;
      }
      is_all_non_zero[dst_pos] &= src_is_all_non_zero;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (last_is_all_non_zero) {
      values[n] = values[n - 1] + last_element;
      is_all_non_zero[n] = is_all_non_zero[n - 1];
    } else {
      values[n] = last_element;
      is_all_non_zero[n] = last_is_all_non_zero;
    }
  }
  __syncthreads();
}

template <typename T>
__global__ void GlobalInclusivePrefixSumKernel(T* values, T* block_buffer, data_size_t num_data) {
  __shared__ T shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  shared_buffer[threadIdx.x] = (data_index < num_data ? values[data_index] : 0);
  __syncthreads();
  PrefixSum<T>(shared_buffer, blockDim.x);
  if (data_index < num_data) {
    values[data_index] = shared_buffer[threadIdx.x + 1];
  }
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x + 1] = shared_buffer[blockDim.x];
  }
}

template <typename INDEX_T, typename VAL_T, typename REDUCE_T>
__global__ void GlobalInclusiveArgPrefixSumKernel(
  const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, data_size_t num_data) {
  __shared__ REDUCE_T shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    if (sorted_indices[data_index] >= num_data || sorted_indices[data_index] < 0) {
      printf("error find sorted_indices[%d] = %d\n", data_index, sorted_indices[data_index]);
    }
  }
  shared_buffer[threadIdx.x] = (data_index < num_data ? in_values[sorted_indices[data_index]] : 0);
  __syncthreads();
  PrefixSum<REDUCE_T>(shared_buffer, blockDim.x);
  if (data_index < num_data) {
    out_values[data_index] = shared_buffer[threadIdx.x + 1];
  }
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x + 1] = shared_buffer[blockDim.x];
  }
}

template <typename T>
__global__ void GlobalInclusivePrefixSumReduceBlockKernel(T* block_buffer, data_size_t num_blocks) {
  __shared__ T shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  T thread_sum = 0;
  const data_size_t num_blocks_per_thread = (num_blocks + static_cast<data_size_t>(blockDim.x)) / static_cast<data_size_t>(blockDim.x);
  const data_size_t thread_start_block_index = static_cast<data_size_t>(threadIdx.x) * num_blocks_per_thread;
  const data_size_t thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks + 1);
  for (data_size_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    thread_sum += block_buffer[block_index];
  }
  shared_buffer[threadIdx.x] = thread_sum;
  __syncthreads();
  PrefixSum<T>(shared_buffer, blockDim.x);
  const T thread_sum_base = shared_buffer[threadIdx.x];
  for (data_size_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_buffer[block_index] += thread_sum_base;
  }
}

__global__ void GlobalInclusivePrefixSumReduceBlockZeroOutKernel(data_size_t* block_buffer, const uint16_t* block_mark_first_zero, data_size_t num_blocks) {
  __shared__ data_size_t shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  __shared__ bool is_all_non_zero[GLOBAL_PREFIX_SUM_BLOCK_SIZE];
  data_size_t thread_sum = 0;
  const data_size_t num_blocks_per_thread = (num_blocks + static_cast<data_size_t>(blockDim.x) - 1) / static_cast<data_size_t>(blockDim.x);
  const data_size_t thread_start_block_index = static_cast<data_size_t>(threadIdx.x) * num_blocks_per_thread;
  const data_size_t thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks);
  bool thread_is_all_non_zero = true;
  data_size_t first_with_zero_block = thread_end_block_index;
  for (data_size_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    const uint16_t mark_first_zero = block_mark_first_zero[block_index];
    const data_size_t block_buffer_value = block_buffer[block_index];
    if (mark_first_zero == GLOBAL_PREFIX_SUM_BLOCK_SIZE) {
      thread_sum += block_buffer_value;
    } else {
      thread_is_all_non_zero = false;
      thread_sum = block_buffer_value;
      if (first_with_zero_block == thread_end_block_index) {
        first_with_zero_block = block_index;
      }
    }
  }
  is_all_non_zero[threadIdx.x] = thread_is_all_non_zero;
  shared_buffer[threadIdx.x] = thread_sum;
  __syncthreads();
  PrefixSumZeroOut(shared_buffer, is_all_non_zero, blockDim.x);
  data_size_t thread_sum_base = shared_buffer[threadIdx.x];
  for (data_size_t block_index = thread_start_block_index; block_index < first_with_zero_block; ++block_index) {
    block_buffer[block_index] += thread_sum_base;
  }
}

template <typename T>
__global__ void GlobalInclusivePrefixSumAddBlockBaseKernel(const T* block_buffer, T* values, data_size_t num_data) {
  const T block_sum_base = block_buffer[blockIdx.x];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    values[data_index] += block_sum_base;
  }
}

__global__ void GlobalInclusivePrefixSumAddBlockBaseGenAUCMarkKernel(
  const data_size_t* block_buffer,
  data_size_t* values,
  const uint16_t* block_first_zero,
  data_size_t num_data) {
  const data_size_t block_sum_base = (blockIdx.x == 0 ? 0 : block_buffer[blockIdx.x - 1]);
  const uint16_t first_zero = block_first_zero[blockIdx.x];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data && threadIdx.x < first_zero) {
    values[data_index] += block_sum_base;
  }
}

template <typename T>
void GlobalInclusivePrefixSum(T* values, T* block_buffer, size_t n) {
  const data_size_t num_data = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalInclusivePrefixSumKernel<T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    values, block_buffer, num_data);
  GlobalInclusivePrefixSumReduceBlockKernel<T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseKernel<T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, values, num_data);
}

template <typename VAL_T, typename REDUCE_T, typename INDEX_T>
void GlobalInclusiveArgPrefixSumInner(const INDEX_T* sorted_indices, const VAL_T* in_values, REDUCE_T* out_values, REDUCE_T* block_buffer, size_t n) {
  const data_size_t num_data = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalInclusiveArgPrefixSumKernel<INDEX_T, VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    sorted_indices, in_values, out_values, block_buffer, num_data);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  GlobalInclusivePrefixSumReduceBlockKernel<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, num_blocks);
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  GlobalInclusivePrefixSumAddBlockBaseKernel<REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, out_values, num_data);
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

template <>
void GlobalInclusiveArgPrefixSum<label_t, double, data_size_t>(const data_size_t* sorted_indices, const label_t* in_values, double* out_values, double* block_buffer, size_t n) {
  GlobalInclusiveArgPrefixSumInner<label_t, double, data_size_t>(sorted_indices, in_values, out_values, block_buffer, n);
}

__global__ void GlobalGenAUCMarkKernel(const double* scores,
                                       const data_size_t* sorted_indices,
                                       data_size_t* mark_buffer,
                                       data_size_t* block_mark_buffer,
                                       uint16_t* block_mark_first_zero,
                                       data_size_t num_data) {
  __shared__ data_size_t shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  __shared__ uint16_t shuffle_reduce_shared_buffer[32];
  __shared__ bool is_all_non_zero[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    if (data_index > 0) {
      shared_buffer[threadIdx.x] = static_cast<data_size_t>(scores[sorted_indices[data_index]] == scores[sorted_indices[data_index - 1]]);
    } else {
      shared_buffer[threadIdx.x] = 0;
    }
  } else {
    shared_buffer[threadIdx.x] = 0;
  }
  is_all_non_zero[threadIdx.x] = static_cast<bool>(shared_buffer[threadIdx.x]);
  __syncthreads();
  uint16_t block_first_zero = (shared_buffer[threadIdx.x] == 0 ? threadIdx.x : blockDim.x);
  PrefixSumZeroOut(shared_buffer, is_all_non_zero, blockDim.x);
  block_first_zero = ShuffleReduceMin<uint16_t>(block_first_zero, shuffle_reduce_shared_buffer, blockDim.x);
  if (data_index < num_data) {
    mark_buffer[data_index] = shared_buffer[threadIdx.x + 1];
  }
  if (threadIdx.x == 0) {
    block_mark_buffer[blockIdx.x] = shared_buffer[blockDim.x];
    block_mark_first_zero[blockIdx.x] = block_first_zero;
  }
}

void GloblGenAUCMark(const double* scores,
  const data_size_t* sorted_indices,
  data_size_t* mark_buffer,
  data_size_t* block_mark_buffer,
  uint16_t* block_mark_first_zero,
  const data_size_t num_data) {
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalGenAUCMarkKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(scores, sorted_indices, mark_buffer, block_mark_buffer, block_mark_first_zero, num_data);
  GlobalInclusivePrefixSumReduceBlockZeroOutKernel<<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_mark_buffer, block_mark_first_zero, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseGenAUCMarkKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_mark_buffer, mark_buffer, block_mark_first_zero, num_data);
}

template <bool USE_WEIGHT, bool IS_POS>
__global__ void GlobalGenAUCPosSumKernel(
  const label_t* labels,
  const label_t* weights,
  const data_size_t* sorted_indices,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer,
  const data_size_t num_data) {
  __shared__ double shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const double pos = IS_POS ?
    (USE_WEIGHT ?
      (data_index < num_data ? static_cast<double>(labels[sorted_indices[data_index]] > 0) * weights[sorted_indices[data_index]] : 0.0f) :
      (data_index < num_data ? static_cast<double>(labels[sorted_indices[data_index]] > 0) : 0.0f)) :
    (USE_WEIGHT ?
      (data_index < num_data ? static_cast<double>(labels[sorted_indices[data_index]] <= 0) * weights[sorted_indices[data_index]] : 0.0f) :
      (data_index < num_data ? static_cast<double>(labels[sorted_indices[data_index]] <= 0) : 0.0f));

  shared_buffer[threadIdx.x] = pos;
  __syncthreads();
  PrefixSum<double>(shared_buffer, blockDim.x);
  if (data_index < num_data) {
    sum_pos_buffer[data_index] = shared_buffer[threadIdx.x + 1];
  }
  if (threadIdx.x == 0) {
    block_sum_pos_buffer[blockIdx.x + 1] = shared_buffer[blockDim.x];
  }
}

template <bool USE_WEIGHT, bool IS_POS>
void GlobalGenAUCPosNegSumInner(const label_t* labels,
  const label_t* weights,
  const data_size_t* sorted_indices,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer,
  const data_size_t num_data) {
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalGenAUCPosSumKernel<USE_WEIGHT, IS_POS><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(labels, weights, sorted_indices, sum_pos_buffer, block_sum_pos_buffer, num_data);
  GlobalInclusivePrefixSumReduceBlockKernel<double><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_sum_pos_buffer, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseKernel<double><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_sum_pos_buffer, sum_pos_buffer, num_data);
}

template <>
void GlobalGenAUCPosNegSum<false, true>(const label_t* labels,
  const label_t* weights,
  const data_size_t* sorted_indices,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer,
  const data_size_t num_data) {
  GlobalGenAUCPosNegSumInner<false, true>(labels, weights, sorted_indices, sum_pos_buffer, block_sum_pos_buffer, num_data);
}

template <>
void GlobalGenAUCPosNegSum<true, false>(const label_t* labels,
  const label_t* weights,
  const data_size_t* sorted_indices,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer,
  const data_size_t num_data) {
  GlobalGenAUCPosNegSumInner<true, false>(labels, weights, sorted_indices, sum_pos_buffer, block_sum_pos_buffer, num_data);
}

template <>
void GlobalGenAUCPosNegSum<true, true>(const label_t* labels,
  const label_t* weights,
  const data_size_t* sorted_indices,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer,
  const data_size_t num_data) {
  GlobalGenAUCPosNegSumInner<true, true>(labels, weights, sorted_indices, sum_pos_buffer, block_sum_pos_buffer, num_data);
}

template <bool USE_WEIGHT>
__global__ void GlobalCalcAUCKernel(
  const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  __shared__ double shared_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double area = 0.0f;
  if (data_index < num_data) {
    if (data_index == num_data - 1 || mark_buffer[data_index + 1] == 0) {
      const data_size_t prev_data_index = data_index - mark_buffer[data_index] - 1;
      const double prev_sum_pos = (prev_data_index < 0 ? 0.0f : sum_pos_buffer[prev_data_index]);
      if (USE_WEIGHT) {
        const double prev_sum_neg = (prev_data_index < 0 ? 0.0f : sum_neg_buffer[prev_data_index]);
        const double cur_pos = sum_pos_buffer[data_index] - prev_sum_pos;
        const double cur_neg = sum_neg_buffer[data_index] - prev_sum_neg;
        area = cur_neg * (cur_pos * 0.5f + prev_sum_pos);
      } else {
        const double cur_pos = sum_pos_buffer[data_index] - prev_sum_pos;
        const double cur_neg = static_cast<double>(data_index - prev_data_index) - cur_pos;
        area = cur_neg * (cur_pos * 0.5f + prev_sum_pos);
      }
    }
  }
  area = ShuffleReduceSum<double>(area, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = area;
  }
}

template <typename T>
__global__ void BlockReduceSum(T* block_buffer, const data_size_t num_blocks) {
  __shared__ T shared_buffer[32];
  T thread_sum = 0;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    thread_sum += block_buffer[block_index];
  }
  thread_sum = ShuffleReduceSum<T>(thread_sum, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[0] = thread_sum;
  }
}

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

template <typename T>
__global__ void BlockReduceMin(T* block_buffer, const data_size_t num_blocks) {
  __shared__ T shared_buffer[32];
  T thread_min = 0;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    const T value = block_buffer[block_index];
    if (value < thread_min) {
      thread_min = value;
    }
  }
  thread_min = ShuffleReduceMin<T>(thread_min, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[0] = thread_min;
  }
}

template <bool USE_WEIGHT>
void GlobalCalcAUCInner(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalCalcAUCKernel<USE_WEIGHT><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
  BlockReduceSum<double><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template <>
void GlobalCalcAUC<false>(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  GlobalCalcAUCInner<false>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
}

template <>
void GlobalCalcAUC<true>(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  GlobalCalcAUCInner<true>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
}

template <bool USE_WEIGHT>
__global__ void GlobalCalcAveragePrecisionKernel(
  const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  __shared__ double shared_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double area = 0.0f;
  if (data_index < num_data) {
    if (data_index == num_data - 1 || mark_buffer[data_index + 1] == 0) {
      const data_size_t prev_data_index = data_index - mark_buffer[data_index] - 1;
      const double prev_sum_pos = (prev_data_index < 0 ? 0.0f : sum_pos_buffer[prev_data_index]);
      if (USE_WEIGHT) {
        const double prev_sum_neg = (prev_data_index < 0 ? 0.0f : sum_neg_buffer[prev_data_index]);
        const double cur_pos = sum_pos_buffer[data_index] - prev_sum_pos;
        const double cur_neg = sum_neg_buffer[data_index] - prev_sum_neg;
        area = cur_pos * (cur_pos + prev_sum_pos) / (prev_sum_neg + prev_sum_pos + cur_pos + cur_neg);
      } else {
        const double cur_pos = sum_pos_buffer[data_index] - prev_sum_pos;
        const double cur_neg = static_cast<double>(data_index - prev_data_index) - cur_pos;
        area = cur_pos * (cur_pos + prev_sum_pos) / static_cast<double>(data_index + 1);
      }
    }
  }
  area = ShuffleReduceSum<double>(area, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = area;
  }
}

template <bool USE_WEIGHT>
void GlobalCalcAveragePrecisionInner(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalCalcAveragePrecisionKernel<USE_WEIGHT><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
  BlockReduceSum<double><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template <>
void GlobalCalcAveragePrecision<false>(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  GlobalCalcAveragePrecisionInner<false>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
}

template <>
void GlobalCalcAveragePrecision<true>(const double* sum_pos_buffer,
  const double* sum_neg_buffer,
  const data_size_t* mark_buffer,
  const data_size_t num_data,
  double* block_buffer) {
  GlobalCalcAveragePrecisionInner<true>(sum_pos_buffer, sum_neg_buffer, mark_buffer, num_data, block_buffer);
}

template <typename VAL_T, typename REDUCE_T>
__global__ void ReduceSumGlobalKernel(const VAL_T* values, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x); 
  const REDUCE_T value = (data_index < num_value ? static_cast<REDUCE_T>(values[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceSum<REDUCE_T>(value, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ReduceSumGlobalInner(const VAL_T* values, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ReduceSumGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, num_value, block_buffer);
  BlockReduceSum<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template <>
void ReduceSumGlobal<label_t, double>(const label_t* values, size_t n, double* block_buffer) {
  ReduceSumGlobalInner(values, n, block_buffer);
}

template <typename VAL_T, typename REDUCE_T>
__global__ void ReduceMaxGlobalKernel(const VAL_T* values, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x); 
  const REDUCE_T value = (data_index < num_value ? static_cast<REDUCE_T>(values[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceMax<REDUCE_T>(value, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ReduceMaxGlobalInner(const VAL_T* values, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ReduceMaxGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, num_value, block_buffer);
  BlockReduceMax<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template <>
void ReduceMaxGlobal<label_t, double>(const label_t* values, size_t n, double* block_buffer) {
  ReduceMaxGlobalInner(values, n, block_buffer);
}

template <typename VAL_T, typename REDUCE_T>
__global__ void ReduceMinGlobalKernel(const VAL_T* values, const data_size_t num_value, REDUCE_T* block_buffer) {
  __shared__ REDUCE_T shared_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x); 
  const REDUCE_T value = (data_index < num_value ? static_cast<REDUCE_T>(values[data_index]) : 0.0f);
  const REDUCE_T reduce_value = ShuffleReduceMin<REDUCE_T>(value, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = reduce_value;
  }
}

template <typename VAL_T, typename REDUCE_T>
void ReduceMinGlobalInner(const VAL_T* values, size_t n, REDUCE_T* block_buffer) {
  const data_size_t num_value = static_cast<data_size_t>(n);
  const data_size_t num_blocks = (num_value + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  ReduceMinGlobalKernel<VAL_T, REDUCE_T><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(values, num_value, block_buffer);
  BlockReduceMin<REDUCE_T><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(block_buffer, num_blocks);
}

template <>
void ReduceMinGlobal<label_t, double>(const label_t* values, size_t n, double* block_buffer) {
  ReduceMinGlobalInner(values, n, block_buffer);
}

template <typename VAL_T, typename INDEX_T, typename WEIGHT_T, typename REDUCE_WEIGHT_T, bool ASCENDING, bool USE_WEIGHT>
__device__ VAL_T PercentileDeviceInner(const VAL_T* values,
                                       const WEIGHT_T* weights,
                                       INDEX_T* indices,
                                       REDUCE_WEIGHT_T* weights_prefix_sum,
                                       const double alpha,
                                       const INDEX_T len) {
  if (len <= 1) {
    return values[0];
  }
  BitonicArgSortDevice512<VAL_T, INDEX_T, ASCENDING>(values, indices, len);
  if (!USE_WEIGHT) {
    const double float_pos = (1.0f - alpha) * len;
    const INDEX_T pos = static_cast<INDEX_T>(float_pos);
    if (pos < 1) {
      return values[indices[0]];
    } else if (pos >= len) {
      return values[indices[len - 1]];
    } else {
      const double bias = float_pos - pos;
      const VAL_T v1 = values[indices[pos - 1]];
      const VAL_T v2 = values[indices[pos]];
      return static_cast<VAL_T>(v1 - (v1 - v2) * bias);
    }
  } else {
    PrefixSumDevice<WEIGHT_T, REDUCE_WEIGHT_T, INDEX_T>(weights, indices, weights_prefix_sum, len);
    const REDUCE_WEIGHT_T threshold = weights_prefix_sum[len - 1] * (1.0f - alpha);
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
      return values[pos];
    }
    const VAL_T v1 = values[indices[pos - 1]];
    const VAL_T v2 = values[indices[pos]];
    return static_cast<VAL_T>(v1 - (v1 - v2) * (threshold - weights_prefix_sum[pos - 1]) / (weights_prefix_sum[pos] - weights_prefix_sum[pos - 1]));
  }
}

template <>
__device__ double PercentileDevice<double, data_size_t, label_t, double, false, true>(
                                  const double* values,
                                  const label_t* weights,
                                  data_size_t* indices,
                                  double* weights_prefix_sum,
                                  const double alpha,
                                  const data_size_t len) {
  return PercentileDeviceInner<double, data_size_t, label_t, double, false, true>(values, weights, indices, weights_prefix_sum, alpha, len);
}

template <>
__device__ double PercentileDevice<double, data_size_t, label_t, double, false, false>(
                                  const double* values,
                                  const label_t* weights,
                                  data_size_t* indices,
                                  double* weights_prefix_sum,
                                  const double alpha,
                                  const data_size_t len) {
  return PercentileDeviceInner<double, data_size_t, label_t, double, false, false>(values, weights, indices, weights_prefix_sum, alpha, len);
}

}  // namespace LightGBM
