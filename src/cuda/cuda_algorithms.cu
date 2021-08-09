/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
 
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

#define QUICKSORT_MAX_DEPTH (12)
#define BITONIC_SORT_NUM_ELEMENTS (1024)
#define BITONIC_SORT_DEPTH (11)

template <typename T, bool ASCENDING>
__global__ void BitonicSort(T* values, const int low, const int high) {
  const int thread_index = static_cast<int>(threadIdx.x);
  T* values_pointer = values + low;
  const int num_data = high - low;
  __shared__ T shared_values[BITONIC_SORT_NUM_ELEMENTS];
  if (thread_index < num_data) {
    shared_values[thread_index] = values_pointer[thread_index];
  }
  __syncthreads();
  for (int depth = BITONIC_SORT_DEPTH - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (BITONIC_SORT_DEPTH - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    for (int inner_depth = depth; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
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

template <typename T, bool ASCENDING>
__global__ void BitonicSortForMergeSort(T* values, const int num_total_data) {
  const int thread_index = static_cast<int>(threadIdx.x);
  const int low = static_cast<int>(blockIdx.x * BITONIC_SORT_NUM_ELEMENTS);
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
    const bool ascending = ASCENDING ? (segment_index % 2 == 0) : (segment_index % 2 == 1);
    for (int inner_depth = depth; inner_depth < BITONIC_SORT_DEPTH; ++inner_depth) {
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

template <typename T, bool ASCENDING>
__global__ void CUDAQuickSortHelper(T* values, const int low, const int high, const int depth) {
  if (high - low <= BITONIC_SORT_NUM_ELEMENTS) {
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
    BitonicSort<T, ASCENDING><<<1, BITONIC_SORT_NUM_ELEMENTS, 0, cuda_stream>>>(values, low, high);
    cudaStreamDestroy(cuda_stream);
    return;
  }
  int i = low - 1;
  int j = high - 1;
  int p = i;
  int q = j;
  const T pivot = values[high - 1];
  while (i < j) {
    if (ASCENDING) {
      while (values[++i] < pivot);
    } else {
      while (values[++i] > pivot);
    }
    if (ASCENDING) {
      while (j > low && values[--j] > pivot);
    } else {
      while (j > low && values[--j] < pivot);
    }
    if (i < j) {
      const T tmp = values[j];
      values[j] = values[i];
      values[i] = tmp;
      if (values[i] == pivot) {
        ++p;
        const T tmp = values[i];
        values[i] = values[p];
        values[p] = tmp;
      }
      if (values[j] == pivot) {
        --q;
        const T tmp = values[j];
        values[j] = values[q];
        values[q] = tmp;
      }
    }
  }
  values[high - 1] = values[i];
  values[i] = pivot;
  j = i - 1;
  i = i + 1;
  for (int k = low; k <= p; ++k, --j) {
    const T tmp = values[k];
    values[k] = values[j];
    values[j] = tmp;
  }
  for (int k = high - 2; k >= q; --k, ++i) {
    const T tmp = values[k];
    values[k] = values[i];
    values[i] = tmp;
  }
  if (j > low) {
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
    CUDAQuickSortHelper<T, ASCENDING><<<1, 1>>>(values, low, j + 1, depth + 1);
    cudaStreamDestroy(cuda_stream);
  }
  if (i + 1 < high) {
    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
    CUDAQuickSortHelper<T, ASCENDING><<<1, 1>>>(values, i, high, depth + 1);
    cudaStreamDestroy(cuda_stream);
  }
}

template <>
void CUDAQuickSort<int>(int* values, const size_t n) {
  CUDAQuickSortHelper<int, true><<<1, 1>>>(values, 0, static_cast<int>(n), 0);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

template <typename T, bool ASCENDING>
__global__ void CUDAMergeKernel(T* values, T* buffer, int block_size, int len) {
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int low = block_size * 2 * thread_index;
  T* values_first_part = values + low;
  int num_data_first_part = min(block_size, len - low);
  T* values_second_part = values + low + block_size;
  int num_data_second_part = min(block_size, len - low - block_size);
  T* buffer_pointer = buffer + low;
  int first_part_index = 0;
  int second_part_index = 0;
  int buffer_index = 0;
  while (first_part_index < num_data_first_part && second_part_index < num_data_second_part) {
    if (ASCENDING) {
      if (values_first_part[first_part_index] > values_second_part[second_part_index]) {
        buffer_pointer[buffer_index++] = values_second_part[second_part_index++];
      } else {
        buffer_pointer[buffer_index++] = values_first_part[first_part_index++];
      }
    } else {
      if (values_first_part[first_part_index] < values_second_part[second_part_index]) {
        buffer_pointer[buffer_index++] = values_second_part[second_part_index++];
      } else {
        buffer_pointer[buffer_index++] = values_first_part[first_part_index++];
      }
    }
  }
  while (first_part_index < num_data_first_part) {
    buffer_pointer[buffer_index++] = values_first_part[first_part_index++];
  }
  for (int data_index = 0; data_index < buffer_index; ++data_index) {
    values_first_part[data_index] = buffer_pointer[data_index];
  }
}

template <>
void CUDAMergeSort<int>(int* values, const size_t n) {
  const int bitonic_num_blocks = (static_cast<int>(n) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  auto start = std::chrono::steady_clock::now();
  BitonicSortForMergeSort<int, true><<<bitonic_num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, n);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  auto end = std::chrono::steady_clock::now();
  auto duration = static_cast<std::chrono::duration<double>>(end - start);
  Log::Warning("bitonic sort time = %f", duration.count());
  int num_blocks_to_merge = bitonic_num_blocks;
  int* buffer = nullptr;
  AllocateCUDAMemoryOuter<int>(&buffer, n, __FILE__, __LINE__);
  int block_size = BITONIC_SORT_NUM_ELEMENTS;
  start = std::chrono::steady_clock::now();
  while (num_blocks_to_merge > 1) {
    num_blocks_to_merge = (num_blocks_to_merge + 1) / 2;
    const int block_dim = 32;
    const int num_kernel_blocks = (num_blocks_to_merge + block_dim - 1) / block_dim;
    CUDAMergeKernel<int, true><<<num_kernel_blocks, block_dim>>>(values, buffer, block_size, static_cast<int>(n));
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    block_size <<= 1;
  }
  end = std::chrono::steady_clock::now();
  duration = static_cast<std::chrono::duration<double>>(end - start);
  Log::Warning("merge time = %f", duration.count());
}

template <typename VAL_T, typename INDEX_T, bool ASCENDING>
__device__ void BitonicArgSort_1024(const VAL_T* values, INDEX_T* indices, const size_t len) {
  const int thread_index = static_cast<int>(threadIdx.x);
  for (int depth = 9; depth >= 1; --depth) {
    const int segment_length = 1 << (10 - depth);
    const int segment_index = thread_index / segment_length;
    const bool ascending = ASCENDING ? (segment_index % 2 == 1) : (segment_index % 2 == 0);
    for (int inner_depth = depth; inner_depth < 10; ++inner_depth) {
      const int inner_segment_length_half = 1 << (9 - inner_depth);
      const int inner_segment_index_half = thread_index / inner_segment_length_half;
      if (inner_segment_index_half % 2 == 0) {
        const int index_to_compare = thread_index + inner_segment_length_half;
        const INDEX_T this_index = indices[thread_index];
        const INDEX_T other_index = indices[index_to_compare];
        if ((values[this_index] > values[other_index]) == ascending && (index_to_compare < static_cast<INDEX_T>(len))) {
          indices[thread_index] = other_index;
          indices[index_to_compare] = this_index;
        }
      }
      __syncthreads();
    }
  }
}
   
template <typename VAL_T, typename INDEX_T>
__device__ void BitonicArgSort(const VAL_T* values, INDEX_T* indices, size_t len) {
  const int num_segments = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  int max_depth = 1;
  int num_segments_to_move = num_segments - 1;
  while (num_segments_to_move > 0) {
    ++max_depth;
  }
  for (int depth = max_depth - 1; depth >= 1; --depth) {
    const int segment_length = 1 << (max_depth - depth);
    const int num_segments_in_level = 1 << depth;
    for (int segment_index = 0; segment_index < num_segments_in_level; ++segment_index) {
      const bool ascending = (segment_index % 2 == 0);
      const size_t segment_start = segment_index * segment_length * BITONIC_SORT_NUM_ELEMENTS;
      //const size_t segment_end = min(segment_start + )
    }
  }
}

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
  /*std::vector<VAL_T> tmp_result(len);
  CopyFromCUDADeviceToHostOuter<VAL_T>(tmp_result.data(), values, len, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  Log::Warning("=============================== before sorting ===============================");
  for (size_t i = 0; i < len; ++i) {
    Log::Warning("tmp_result[%d] = %d", i, tmp_result[i]);
  }*/
  const int num_blocks = (static_cast<int>(len) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicSortGlobalKernel<VAL_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, static_cast<int>(len));
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  /*CopyFromCUDADeviceToHostOuter<VAL_T>(tmp_result.data(), values, len, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  Log::Warning("=============================== after block sort stage ===============================");
  for (size_t i = 0; i < len; ++i) {
    Log::Warning("tmp_result[%d] = %d", i, tmp_result[i]);
  }*/
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicCompareKernel<VAL_T, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      /*CopyFromCUDADeviceToHostOuter<VAL_T>(tmp_result.data(), values, len, __FILE__, __LINE__);
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      Log::Warning("=============================== after compare stage depth %d inner depth = %d ===============================", depth, depth);
      for (size_t i = 0; i < len; ++i) {
        Log::Warning("tmp_result[%d] = %d", i, tmp_result[i]);
      }*/
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicCompareKernel<VAL_T, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, half_segment_length, segment_length, static_cast<int>(len));
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      /*CopyFromCUDADeviceToHostOuter<VAL_T>(tmp_result.data(), values, len, __FILE__, __LINE__);
      SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
      Log::Warning("=============================== after compare stage depth %d inner depth = %d ===============================", depth, inner_depth);
      for (size_t i = 0; i < len; ++i) {
        Log::Warning("tmp_result[%d] = %d", i, tmp_result[i]);
      }*/
      half_segment_length >>= 1;
    }
    BitonicSortMergeKernel<VAL_T, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(values, segment_length, static_cast<int>(len));
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    /*CopyFromCUDADeviceToHostOuter<VAL_T>(tmp_result.data(), values, len, __FILE__, __LINE__);
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    Log::Warning("=============================== after merge stage depth %d ===============================", depth);
    for (size_t i = 0; i < len; ++i) {
      Log::Warning("tmp_result[%d] = %d", i, tmp_result[i]);
    }*/
  }
}

template <>
void BitonicSortGlobal<int, true>(int* values, const size_t len) {
  BitonicSortGlobalHelper<int, true>(values, len);
}

}  // namespace LightGBM
