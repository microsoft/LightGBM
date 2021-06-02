/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"
#include <LightGBM/tree.h>

namespace LightGBM {

#define CONFLICT_FREE_INDEX(n) \
  ((n) + ((n) >> LOG_NUM_BANKS_DATA_PARTITION)) \

__device__ void PrefixSum(uint32_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_n_minus_1 = CONFLICT_FREE_INDEX(n - 1);
  const uint32_t last_element = elements[conflict_free_n_minus_1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[conflict_free_n_minus_1] = 0; 
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
      const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
      const uint32_t src_val = elements[conflict_free_src_pos];
      elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
      elements[conflict_free_dst_pos] += src_val;
    }
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[CONFLICT_FREE_INDEX(n)] = elements[conflict_free_n_minus_1] + last_element;
  }
}

__device__ void PrefixSum(uint16_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_n_minus_1 = CONFLICT_FREE_INDEX(n - 1);
  const uint16_t last_element = elements[conflict_free_n_minus_1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[conflict_free_n_minus_1] = 0; 
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
      const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
      const uint16_t src_val = elements[conflict_free_src_pos];
      elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
      elements[conflict_free_dst_pos] += src_val;
    }
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[CONFLICT_FREE_INDEX(n)] = elements[conflict_free_n_minus_1] + last_element;
  }
}

__device__ void ReduceSum(uint16_t* array, const size_t size) {
  const unsigned int threadIdx_x = threadIdx.x;
  for (int s = 1; s < size; s <<= 1) {
    if (threadIdx_x % (2 * s) == 0 && (threadIdx_x + s) < size) {
      array[CONFLICT_FREE_INDEX(threadIdx_x)] += array[CONFLICT_FREE_INDEX(threadIdx_x + s)];
    }
    __syncthreads();
  }
}

__device__ void ReduceSum(double* array, const size_t size) {
  const unsigned int threadIdx_x = threadIdx.x;
  for (int s = 1; s < size; s <<= 1) {
    if (threadIdx_x % (2 * s) == 0 && (threadIdx_x + s) < size) {
      array[threadIdx_x] += array[threadIdx_x + s];
    }
    __syncthreads();
  }
}

__global__ void FillDataIndicesBeforeTrainKernel(const data_size_t* cuda_num_data,
  data_size_t* data_indices) {
  const data_size_t num_data_ref = *cuda_num_data;
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data_ref) {
    data_indices[data_index] = data_index;
  }
}

void CUDADataPartition::LaunchFillDataIndicesBeforeTrain() {
  const int num_blocks = (num_data_ + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(cuda_num_data_, cuda_data_indices_); 
}

__device__ void PrepareOffset(const data_size_t num_data_in_leaf_ref, const uint8_t* split_to_left_bit_vector,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition,
  uint16_t* thread_to_left_offset_cnt) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x / 2;
  __syncthreads();
  ReduceSum(thread_to_left_offset_cnt, split_indices_block_size_data_partition);
  __syncthreads();
  if (threadIdx_x == 0) {
    const data_size_t num_data_in_block = (blockIdx.x + 1) * blockDim_x * 2 <= num_data_in_leaf_ref ? static_cast<data_size_t>(blockDim_x * 2) :
      num_data_in_leaf_ref - static_cast<data_size_t>(blockIdx.x * blockDim_x * 2);
    if (num_data_in_block > 0) {
      const data_size_t data_to_left = static_cast<data_size_t>(thread_to_left_offset_cnt[0]);
      block_to_left_offset_buffer[blockIdx.x + 1] = data_to_left;
      block_to_right_offset_buffer[blockIdx.x + 1] = num_data_in_block - data_to_left;
    } else {
      block_to_left_offset_buffer[blockIdx.x + 1] = 0;
      block_to_right_offset_buffer[blockIdx.x + 1] = 0;
    }
  }
}

// missing_is_zero = 0, missing_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel0_1_2_3(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t /*split_missing_default_to_left*/,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 0, missing_is_na = 1, mfb_is_zero = 0, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel4(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == t_zero_bin) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 0, missing_is_na = 1, mfb_is_zero = 0, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel5(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 0, missing_is_na = 1, mfb_is_zero = 1, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel6(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 0, missing_is_na = 1, mfb_is_zero = 1, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel7(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 0, mfb_is_zero = 0, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel8(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == t_zero_bin) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 0, mfb_is_zero = 0, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel9(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == t_zero_bin) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 0, mfb_is_zero = 1, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel10(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 0, mfb_is_zero = 1, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel11(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 1, mfb_is_zero = 0, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel12(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == t_zero_bin || bin == max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 1, mfb_is_zero = 0, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel13(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == t_zero_bin) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 1, mfb_is_zero = 1, mfb_is_na = 0, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel14(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin == max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

// missing_is_zero = 1, missing_is_na = 1, mfb_is_zero = 1, mfb_is_na = 1, min_bin_ref < max_bin_ref
__global__ void GenDataToLeftBitVectorKernel15(const int best_split_feature_ref, const data_size_t cuda_leaf_data_start,
  const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
  const uint32_t th, const int num_features_ref, const uint8_t* cuda_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t /*split_default_to_left*/, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + cuda_leaf_data_start;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(cuda_data[global_data_index]);
    if (bin < min_bin_ref || bin > max_bin_ref) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin > th) {
      cuda_data_to_left[local_data_index] = 0;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
    } else {
      cuda_data_to_left[local_data_index] = 1;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

__global__ void GenDataToLeftBitVectorKernel(const int* leaf_index, const data_size_t* cuda_leaf_data_start,
  const data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices, const int* best_split_feature,
  const uint32_t* best_split_threshold, const int* cuda_num_features, const uint8_t* cuda_data,
  const uint32_t* default_bin, const uint32_t* most_freq_bin, const uint8_t* default_left,
  const uint32_t* min_bin, const uint32_t* max_bin, const uint8_t* missing_is_zero, const uint8_t* missing_is_na,
  const uint8_t* mfb_is_zero, const uint8_t* mfb_is_na,
  uint8_t* cuda_data_to_left) {
  const int leaf_index_ref = *leaf_index;
  const int best_split_feature_ref = best_split_feature[leaf_index_ref];
  const int num_features_ref = *cuda_num_features;
  const uint32_t best_split_threshold_ref = best_split_threshold[leaf_index_ref];
  const uint8_t default_left_ref = default_left[leaf_index_ref];
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[leaf_index_ref];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[leaf_index_ref];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const unsigned int global_feature_value_index = global_data_index * num_features_ref + best_split_feature_ref;
    const uint32_t default_bin_ref = default_bin[best_split_feature_ref];
    const uint32_t most_freq_bin_ref = most_freq_bin[best_split_feature_ref];
    const uint32_t max_bin_ref = max_bin[best_split_feature_ref];
    const uint32_t min_bin_ref = min_bin[best_split_feature_ref];
    const uint8_t missing_is_zero_ref = missing_is_zero[best_split_feature_ref];
    const uint8_t missing_is_na_ref = missing_is_na[best_split_feature_ref];
    const uint8_t mfb_is_zero_ref = mfb_is_zero[best_split_feature_ref];
    const uint8_t mfb_is_na_ref = mfb_is_na[best_split_feature_ref];
    uint32_t th = best_split_threshold_ref + min_bin_ref;
    uint32_t t_zero_bin = min_bin_ref + default_bin_ref;
    if (most_freq_bin_ref == 0) {
      --th;
      --t_zero_bin;
    }
    uint8_t split_default_to_left = 0;
    uint8_t split_missing_default_to_left = 0;
    if (most_freq_bin_ref <= best_split_threshold_ref) {
      split_default_to_left = 1;
    }
    if (missing_is_zero_ref || missing_is_na_ref) {
      if (default_left_ref) {
        split_missing_default_to_left = 1;
      }
    }
    if (local_data_index < static_cast<unsigned int>(num_data_in_leaf)) {
      const uint32_t bin = static_cast<uint32_t>(cuda_data[global_feature_value_index]);
      if (min_bin_ref < max_bin_ref) {
        if ((missing_is_zero_ref && !mfb_is_zero_ref && bin == t_zero_bin) ||
          (missing_is_na_ref && !mfb_is_na_ref && bin == max_bin_ref)) {
          cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        } else if (bin < min_bin_ref || bin > max_bin_ref) {
          if ((missing_is_na_ref && mfb_is_na_ref) || (missing_is_zero_ref && mfb_is_zero_ref)) {
            cuda_data_to_left[local_data_index] = split_missing_default_to_left;
          } else {
            cuda_data_to_left[local_data_index] = split_default_to_left;
          }
        } else if (bin > th) {
          cuda_data_to_left[local_data_index] = 0;
        } else {
          cuda_data_to_left[local_data_index] = 1;
        }
      } else {
        if (missing_is_zero_ref && !mfb_is_zero_ref && bin == t_zero_bin) {
          cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        } else if (bin != max_bin_ref) {
          if ((missing_is_na_ref && mfb_is_na_ref) || (missing_is_zero_ref && mfb_is_zero_ref)) {
            cuda_data_to_left[local_data_index] = split_missing_default_to_left;
          } else {
            cuda_data_to_left[local_data_index] = split_default_to_left;
          }
        } else {
          if (missing_is_na_ref && !mfb_is_na_ref) {
            cuda_data_to_left[local_data_index] = split_missing_default_to_left;
          } else {
            cuda_data_to_left[local_data_index] = split_default_to_left;
          }
        }
      }
    }
  }
}

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel2(const data_size_t num_data_in_leaf,
  const int split_feature_index, const uint32_t split_threshold,
  const uint8_t split_default_left, const data_size_t leaf_data_start) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  const uint8_t missing_is_zero = feature_missing_is_zero_[split_feature_index];
  const uint8_t missing_is_na = feature_missing_is_na_[split_feature_index];
  const uint8_t mfb_is_zero = feature_mfb_is_zero_[split_feature_index];
  const uint8_t mfb_is_na = feature_mfb_is_na_[split_feature_index];
  const uint32_t default_bin = feature_default_bins_[split_feature_index];
  const uint32_t most_freq_bin = feature_most_freq_bins_[split_feature_index];
  const uint32_t min_bin = feature_min_bins_[split_feature_index];
  const uint32_t max_bin = feature_max_bins_[split_feature_index];

  uint32_t th = split_threshold + min_bin;
  uint32_t t_zero_bin = min_bin + default_bin;
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;  
  }
  uint8_t split_default_to_left = 0;
  uint8_t split_missing_default_to_left = 0;
  if (most_freq_bin <= split_threshold) {
    split_default_to_left = 1;
  }
  if (missing_is_zero || missing_is_na) {
    if (split_default_left) {
      split_missing_default_to_left = 1;
    }
  }
  const uint8_t* cuda_data_col_wise_ptr = cuda_data_col_wise_ + split_feature_index * num_data_;
  if (min_bin < max_bin) {
    if (!missing_is_zero && !missing_is_na) {
      GenDataToLeftBitVectorKernel0_1_2_3<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
        split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
        th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
        cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
        split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
        split_indices_block_size_data_partition_aligned);
    } else {
      if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel4<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel5<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel6<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel7<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel8<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel9<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel10<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel11<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel12<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel13<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
        GenDataToLeftBitVectorKernel14<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
        GenDataToLeftBitVectorKernel15<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
          split_feature_index, leaf_data_start, num_data_in_leaf, cuda_data_indices_,
          th, num_features_, /* TODO(shiyu1994): the case when num_features != num_groups*/
          cuda_data_col_wise_ptr, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,
          split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
          split_indices_block_size_data_partition_aligned);
      }
    }
  } else {
    Log::Fatal("Unsupported for max_bin == min_bin");
  }

  SynchronizeCUDADevice();
}

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel(const int* leaf_index, const data_size_t num_data_in_leaf, const int* best_split_feature,
  const uint32_t* best_split_threshold, const uint8_t* best_split_default_left) {
  const int num_blocks = std::max(80, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  GenDataToLeftBitVectorKernel<<<num_blocks, split_indices_block_size_data_partition_aligned>>>(
    leaf_index, cuda_leaf_data_start_, cuda_leaf_num_data_,
    cuda_data_indices_, best_split_feature, best_split_threshold,
    cuda_num_features_, cuda_data_,
    cuda_feature_default_bins_, cuda_feature_most_freq_bins_, best_split_default_left,
    cuda_feature_min_bins_, cuda_feature_max_bins_, cuda_feature_missing_is_zero_, cuda_feature_missing_is_na_,
    cuda_feature_mfb_is_zero_, cuda_feature_mfb_is_na_,
    cuda_data_to_left_);
  SynchronizeCUDADevice();
}

__global__ void AggregateBlockOffsetKernel(const int* leaf_index, data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  int* cuda_cur_num_leaves,
  const int* best_split_feature, const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left, const double* best_split_gain,
  const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
  const double* best_left_gain, const double* best_left_leaf_value,
  const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
  const double* best_right_gain, const double* best_right_leaf_value,
  // for leaf splits information update
  int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
  double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
  double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
  const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** smaller_leaf_cuda_hist_pointer_pointer,
  int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
  double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
  double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
  const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** larger_leaf_cuda_hist_pointer_pointer,
  const int* cuda_num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool, const int split_indices_block_size_data_partition,

  int* tree_split_leaf_index, int* tree_inner_feature_index, uint32_t* tree_threshold,
  double* tree_left_output, double* tree_right_output, data_size_t* tree_left_count, data_size_t* tree_right_count,
  double* tree_left_sum_hessian, double* tree_right_sum_hessian, double* tree_gain, uint8_t* tree_default_left,
  double* data_partition_leaf_output) {
  __shared__ uint32_t block_to_left_offset[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint32_t block_to_right_offset[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  const int leaf_index_ref = *leaf_index;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[leaf_index_ref];
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_threadIdx_x = CONFLICT_FREE_INDEX(threadIdx_x);
  const unsigned int conflict_free_threadIdx_x_plus_blockDim_x = CONFLICT_FREE_INDEX(threadIdx_x + blockDim_x);
  const uint32_t num_blocks = (num_data_in_leaf + split_indices_block_size_data_partition - 1) / split_indices_block_size_data_partition;
  const uint32_t num_aggregate_blocks = (num_blocks + split_indices_block_size_data_partition - 1) / split_indices_block_size_data_partition;
  uint32_t left_prev_sum = 0;
  for (uint32_t block_id = 0; block_id < num_aggregate_blocks; ++block_id) {
    const unsigned int read_index = block_id * blockDim_x * 2 + threadIdx_x;
    if (read_index < num_blocks) {
      block_to_left_offset[conflict_free_threadIdx_x] = block_to_left_offset_buffer[read_index + 1];
    } else {
      block_to_left_offset[conflict_free_threadIdx_x] = 0;
    }
    const unsigned int read_index_plus_blockDim_x = read_index + blockDim_x;
    if (read_index_plus_blockDim_x < num_blocks) {
      block_to_left_offset[conflict_free_threadIdx_x_plus_blockDim_x] = block_to_left_offset_buffer[read_index_plus_blockDim_x + 1];
    } else {
      block_to_left_offset[conflict_free_threadIdx_x_plus_blockDim_x] = 0;
    }
    if (threadIdx_x == 0) {
      block_to_left_offset[0] += left_prev_sum;
    }
    __syncthreads();
    PrefixSum(block_to_left_offset, split_indices_block_size_data_partition);
    __syncthreads();
    if (threadIdx_x == 0) {
      left_prev_sum = block_to_left_offset[CONFLICT_FREE_INDEX(split_indices_block_size_data_partition)];
    }
    if (read_index < num_blocks) {
      const unsigned int conflict_free_threadIdx_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + 1);
      block_to_left_offset_buffer[read_index + 1] = block_to_left_offset[conflict_free_threadIdx_x_plus_1];
    }
    if (read_index_plus_blockDim_x < num_blocks) {
      const unsigned int conflict_free_threadIdx_x_plus_1_plus_blockDim_x = CONFLICT_FREE_INDEX(threadIdx_x + 1 + blockDim_x);
      block_to_left_offset_buffer[read_index_plus_blockDim_x + 1] = block_to_left_offset[conflict_free_threadIdx_x_plus_1_plus_blockDim_x];
    }
    __syncthreads();
  }
  const unsigned int to_left_total_cnt = block_to_left_offset_buffer[num_blocks];
  uint32_t right_prev_sum = to_left_total_cnt;
  for (uint32_t block_id = 0; block_id < num_aggregate_blocks; ++block_id) {
    const unsigned int read_index = block_id * blockDim_x * 2 + threadIdx_x;
    if (read_index < num_blocks) {
      block_to_right_offset[conflict_free_threadIdx_x] = block_to_right_offset_buffer[read_index + 1];
    } else {
      block_to_right_offset[conflict_free_threadIdx_x] = 0;
    }
    const unsigned int read_index_plus_blockDim_x = read_index + blockDim_x;
    if (read_index_plus_blockDim_x < num_blocks) {
      block_to_right_offset[conflict_free_threadIdx_x_plus_blockDim_x] = block_to_right_offset_buffer[read_index_plus_blockDim_x + 1];
    } else {
      block_to_right_offset[conflict_free_threadIdx_x_plus_blockDim_x] = 0;
    }
    if (threadIdx_x == 0) {
      block_to_right_offset[0] += right_prev_sum;
    }
    __syncthreads();
    PrefixSum(block_to_right_offset, split_indices_block_size_data_partition);
    __syncthreads();
    if (threadIdx_x == 0) {
      right_prev_sum = block_to_right_offset[CONFLICT_FREE_INDEX(split_indices_block_size_data_partition)];
    }
    if (read_index < num_blocks) {
      const unsigned int conflict_free_threadIdx_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + 1);
      block_to_right_offset_buffer[read_index + 1] = block_to_right_offset[conflict_free_threadIdx_x_plus_1];
    }
    if (read_index_plus_blockDim_x < num_blocks) {
      const unsigned int conflict_free_threadIdx_x_plus_1_plus_blockDim_x = CONFLICT_FREE_INDEX(threadIdx_x + 1 + blockDim_x);
      block_to_right_offset_buffer[read_index_plus_blockDim_x + 1] = block_to_right_offset[conflict_free_threadIdx_x_plus_1_plus_blockDim_x];
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ++(*cuda_cur_num_leaves);
    const int cur_max_leaf_index = (*cuda_cur_num_leaves) - 1;
    /*printf("leaf_index_ref = %d, cuda_cur_num_leaves = %d, cur_max_leaf_index = %d\n",
      leaf_index_ref, *cuda_cur_num_leaves, cur_max_leaf_index);*/
    block_to_left_offset_buffer[0] = 0;
    const unsigned int to_left_total_cnt = block_to_left_offset_buffer[num_blocks];
    block_to_right_offset_buffer[0] = to_left_total_cnt;
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[leaf_index_ref];
    cuda_leaf_data_end[leaf_index_ref] = cuda_leaf_data_start[leaf_index_ref] + static_cast<data_size_t>(to_left_total_cnt);
    cuda_leaf_num_data[leaf_index_ref] = static_cast<data_size_t>(to_left_total_cnt);
    cuda_leaf_data_start[cur_max_leaf_index] = cuda_leaf_data_end[leaf_index_ref];
    cuda_leaf_data_end[cur_max_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[cur_max_leaf_index] = block_to_right_offset_buffer[num_blocks] - to_left_total_cnt;
  }
}

__global__ void SplitTreeStructureKernel(const int* leaf_index, data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  int* cuda_cur_num_leaves,
  const int* best_split_feature, const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left, const double* best_split_gain,
  const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
  const double* best_left_gain, const double* best_left_leaf_value,
  const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
  const double* best_right_gain, const double* best_right_leaf_value,
  // for leaf splits information update
  int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
  double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
  double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
  const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** smaller_leaf_cuda_hist_pointer_pointer,
  int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
  double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
  double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
  const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** larger_leaf_cuda_hist_pointer_pointer,
  const int* cuda_num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool, const int split_indices_block_size_data_partition,

  int* tree_split_leaf_index, int* tree_inner_feature_index, uint32_t* tree_threshold,
  double* tree_left_output, double* tree_right_output, data_size_t* tree_left_count, data_size_t* tree_right_count,
  double* tree_left_sum_hessian, double* tree_right_sum_hessian, double* tree_gain, uint8_t* tree_default_left,
  double* data_partition_leaf_output,
  int* cuda_split_info_buffer) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int leaf_index_ref = *leaf_index;
    const int cur_max_leaf_index = (*cuda_cur_num_leaves) - 1;
    const unsigned int to_left_total_cnt = cuda_leaf_num_data[leaf_index_ref];
    const int cuda_num_total_bin_ref = *cuda_num_total_bin;

    tree_split_leaf_index[cur_max_leaf_index - 1] = leaf_index_ref;
    tree_inner_feature_index[cur_max_leaf_index - 1] = best_split_feature[leaf_index_ref];
    tree_threshold[cur_max_leaf_index - 1] = best_split_threshold[leaf_index_ref];
    tree_left_output[cur_max_leaf_index - 1] = best_left_leaf_value[leaf_index_ref];
    tree_right_output[cur_max_leaf_index - 1] = best_right_leaf_value[leaf_index_ref];
    tree_left_count[cur_max_leaf_index - 1] = best_left_count[leaf_index_ref];
    tree_right_count[cur_max_leaf_index - 1] = best_right_count[leaf_index_ref];
    tree_left_sum_hessian[cur_max_leaf_index - 1] = best_left_sum_hessians[leaf_index_ref];
    tree_right_sum_hessian[cur_max_leaf_index - 1] = best_right_sum_hessians[leaf_index_ref];
    tree_gain[cur_max_leaf_index - 1] = best_split_gain[leaf_index_ref];
    tree_default_left[cur_max_leaf_index - 1] = best_split_default_left[leaf_index_ref];
    data_partition_leaf_output[leaf_index_ref] = best_left_leaf_value[leaf_index_ref];
    data_partition_leaf_output[cur_max_leaf_index] = best_right_leaf_value[leaf_index_ref];

    cuda_split_info_buffer[0] = leaf_index_ref;
    cuda_split_info_buffer[1] = cuda_leaf_num_data[leaf_index_ref];
    cuda_split_info_buffer[2] = cuda_leaf_data_start[leaf_index_ref];
    cuda_split_info_buffer[3] = cur_max_leaf_index;
    cuda_split_info_buffer[4] = cuda_leaf_num_data[cur_max_leaf_index];
    cuda_split_info_buffer[5] = cuda_leaf_data_start[cur_max_leaf_index];

    /*if (cuda_leaf_num_data[leaf_index_ref] <= 0) {
      printf("error !!! leaf %d has count %d\n", leaf_index_ref, cuda_leaf_num_data[leaf_index_ref]);
    }

    if (cuda_leaf_num_data[cur_max_leaf_index] <= 0) {
      printf("error !!! leaf %d has count %d\n", cur_max_leaf_index, cuda_leaf_num_data[cur_max_leaf_index]);
    }

    printf("splitting %d into %d with num data %d and %d with num data %d\n",
        leaf_index_ref, leaf_index_ref, cuda_leaf_num_data[leaf_index_ref],
        cur_max_leaf_index, cuda_leaf_num_data[cur_max_leaf_index]);*/

    if (cuda_leaf_num_data[leaf_index_ref] < cuda_leaf_num_data[cur_max_leaf_index]) {
      *smaller_leaf_cuda_leaf_index_pointer = leaf_index_ref;
      *smaller_leaf_cuda_sum_of_gradients_pointer = best_left_sum_gradients[leaf_index_ref];
      *smaller_leaf_cuda_sum_of_hessians_pointer = best_left_sum_hessians[leaf_index_ref];
      *smaller_leaf_cuda_num_data_in_leaf_pointer = to_left_total_cnt;//best_left_count[leaf_index_ref];
      *smaller_leaf_cuda_gain_pointer = best_left_gain[leaf_index_ref];
      *smaller_leaf_cuda_leaf_value_pointer = best_left_leaf_value[leaf_index_ref];
      *smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer = cuda_data_indices + cuda_leaf_data_start[leaf_index_ref];

      *larger_leaf_cuda_leaf_index_pointer = cur_max_leaf_index;
      *larger_leaf_cuda_sum_of_gradients_pointer = best_right_sum_gradients[leaf_index_ref];
      *larger_leaf_cuda_sum_of_hessians_pointer = best_right_sum_hessians[leaf_index_ref];
      *larger_leaf_cuda_num_data_in_leaf_pointer = cuda_leaf_num_data[cur_max_leaf_index];//best_right_count[leaf_index_ref];
      *larger_leaf_cuda_gain_pointer = best_right_gain[leaf_index_ref];
      *larger_leaf_cuda_leaf_value_pointer = best_right_leaf_value[leaf_index_ref];
      *larger_leaf_cuda_data_indices_in_leaf_pointer_pointer = cuda_data_indices + cuda_leaf_data_start[cur_max_leaf_index];

      hist_t* parent_hist_ptr = cuda_hist_pool[leaf_index_ref];
      cuda_hist_pool[cur_max_leaf_index] = parent_hist_ptr;
      cuda_hist_pool[leaf_index_ref] = cuda_hist + 2 * cur_max_leaf_index * cuda_num_total_bin_ref;
      *smaller_leaf_cuda_hist_pointer_pointer = cuda_hist_pool[leaf_index_ref];
      *larger_leaf_cuda_hist_pointer_pointer = cuda_hist_pool[cur_max_leaf_index];
      cuda_split_info_buffer[6] = leaf_index_ref;
      cuda_split_info_buffer[7] = cur_max_leaf_index;
    } else {
      *larger_leaf_cuda_leaf_index_pointer = leaf_index_ref;
      *larger_leaf_cuda_sum_of_gradients_pointer = best_left_sum_gradients[leaf_index_ref];
      *larger_leaf_cuda_sum_of_hessians_pointer = best_left_sum_hessians[leaf_index_ref];
      *larger_leaf_cuda_num_data_in_leaf_pointer = to_left_total_cnt;//best_left_count[leaf_index_ref];
      *larger_leaf_cuda_gain_pointer = best_left_gain[leaf_index_ref];
      *larger_leaf_cuda_leaf_value_pointer = best_left_leaf_value[leaf_index_ref];
      *larger_leaf_cuda_data_indices_in_leaf_pointer_pointer = cuda_data_indices + cuda_leaf_data_start[leaf_index_ref];

      *smaller_leaf_cuda_leaf_index_pointer = cur_max_leaf_index;
      *smaller_leaf_cuda_sum_of_gradients_pointer = best_right_sum_gradients[leaf_index_ref];
      *smaller_leaf_cuda_sum_of_hessians_pointer = best_right_sum_hessians[leaf_index_ref];
      *smaller_leaf_cuda_num_data_in_leaf_pointer = cuda_leaf_num_data[cur_max_leaf_index];//best_right_count[leaf_index_ref];
      *smaller_leaf_cuda_gain_pointer = best_right_gain[leaf_index_ref];
      *smaller_leaf_cuda_leaf_value_pointer = best_right_leaf_value[leaf_index_ref];
      *smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer = cuda_data_indices + cuda_leaf_data_start[cur_max_leaf_index];

      cuda_hist_pool[cur_max_leaf_index] = cuda_hist + 2 * cur_max_leaf_index * cuda_num_total_bin_ref;
      *smaller_leaf_cuda_hist_pointer_pointer = cuda_hist_pool[cur_max_leaf_index];
      *larger_leaf_cuda_hist_pointer_pointer = cuda_hist_pool[leaf_index_ref];
      cuda_split_info_buffer[6] = cur_max_leaf_index;
      cuda_split_info_buffer[7] = leaf_index_ref;
    }
  }
}

__global__ void SplitInnerKernel(const int* leaf_index, const int* cuda_cur_num_leaves,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices, const uint8_t* split_to_left_bit_vector,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  data_size_t* out_data_indices_in_leaf, const int split_indices_block_size_data_partition) {
  __shared__ uint8_t thread_split_to_left_bit_vector[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  __shared__ uint16_t thread_to_left_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint16_t thread_to_right_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  const int leaf_index_ref = *leaf_index;
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[leaf_index_ref];
  const data_size_t num_data_in_leaf_ref = cuda_leaf_num_data[leaf_index_ref] + cuda_leaf_num_data[(*cuda_cur_num_leaves) - 1];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int conflict_free_threadIdx_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + 1);
  const unsigned int global_thread_index = blockIdx.x * blockDim_x * 2 + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_thread_index];
    thread_split_to_left_bit_vector[threadIdx_x] = bit;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_1] = bit;
  } else {
    thread_split_to_left_bit_vector[threadIdx_x] = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_1] = 0;
  }
  const unsigned int conflict_free_threadIdx_x_plus_blockDim_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + blockDim_x + 1);
  const unsigned int global_thread_index_plus_blockDim_x = global_thread_index + blockDim_x;
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_thread_index_plus_blockDim_x];
    thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] = bit;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = bit;
  } else {
    thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = 0;
  }
  __syncthreads();
  const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
  if (threadIdx_x == 0) {
    thread_to_left_pos[0] = 0;
  }
  __syncthreads();
  PrefixSum(thread_to_left_pos, split_indices_block_size_data_partition);
  __syncthreads();
  if (threadIdx_x > 0) {
    thread_to_right_pos[threadIdx_x] = (threadIdx_x - thread_to_left_pos[conflict_free_threadIdx_x_plus_1]);
  } else {
    thread_to_right_pos[threadIdx_x] = 0;
  }
  thread_to_right_pos[threadIdx_x + blockDim_x] = (threadIdx_x + blockDim_x - thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]);
  __syncthreads();
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + to_right_block_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    if (thread_split_to_left_bit_vector[threadIdx_x] == 1) {
      left_out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      right_out_data_indices_in_leaf[thread_to_right_pos[threadIdx_x]] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    if (thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] == 1) {
      left_out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    } else {
      right_out_data_indices_in_leaf[thread_to_right_pos[threadIdx_x + blockDim_x]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    }
  }
}

__global__ void CopyDataIndicesKernel(const int* leaf_index,
  const int* cuda_cur_num_leaves,
  const data_size_t* cuda_leaf_data_start,
  const data_size_t* cuda_leaf_num_data,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const int leaf_index_ref = *leaf_index;
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[leaf_index_ref];
  const data_size_t num_data_in_leaf_ref = cuda_leaf_num_data[leaf_index_ref] + cuda_leaf_num_data[(*cuda_cur_num_leaves) - 1];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    cuda_data_indices_in_leaf[global_thread_index] = out_data_indices_in_leaf[global_thread_index];
  }
}

void CUDADataPartition::LaunchSplitInnerKernel(const int* leaf_index, const data_size_t num_data_in_leaf,
  const int* best_split_feature, const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left, const double* best_split_gain,
  const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
  const double* best_left_gain, const double* best_left_leaf_value,
  const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
  const double* best_right_gain, const double* best_right_leaf_value,
  // for leaf splits information update
  int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
  double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
  double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
  const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** smaller_leaf_cuda_hist_pointer_pointer,
  int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
  double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
  double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
  const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** larger_leaf_cuda_hist_pointer_pointer,
  std::vector<data_size_t>* cpu_leaf_num_data, std::vector<data_size_t>* cpu_leaf_data_start,
  int* smaller_leaf_index, int* larger_leaf_index) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  global_timer.Start("CUDADataPartition::AggregateBlockOffsetKernel");
  auto start = std::chrono::steady_clock::now();
  /*int cpu_leaf_index = 0, cpu_cur_num_leaves = 0;
  CopyFromCUDADeviceToHost(&cpu_leaf_index, leaf_index, 1);
  CopyFromCUDADeviceToHost(&cpu_cur_num_leaves, cuda_cur_num_leaves_, 1);
  Log::Warning("cpu_leaf_index = %d, cpu_cur_num_leaves = %d before aggregate", cpu_leaf_index, cpu_cur_num_leaves);*/
  AggregateBlockOffsetKernel<<<1, split_indices_block_size_data_partition_aligned / 2>>>(leaf_index, cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
    cuda_leaf_num_data_, cuda_data_indices_,
    cuda_cur_num_leaves_,
    best_split_feature, best_split_threshold, best_split_default_left, best_split_gain,
    best_left_sum_gradients, best_left_sum_hessians, best_left_count,
    best_left_gain, best_left_leaf_value,
    best_right_sum_gradients, best_right_sum_hessians, best_right_count,
    best_right_gain, best_right_leaf_value,

    smaller_leaf_cuda_leaf_index_pointer, smaller_leaf_cuda_sum_of_gradients_pointer,
    smaller_leaf_cuda_sum_of_hessians_pointer, smaller_leaf_cuda_num_data_in_leaf_pointer,
    smaller_leaf_cuda_gain_pointer, smaller_leaf_cuda_leaf_value_pointer,
    smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    smaller_leaf_cuda_hist_pointer_pointer,
    larger_leaf_cuda_leaf_index_pointer, larger_leaf_cuda_sum_of_gradients_pointer,
    larger_leaf_cuda_sum_of_hessians_pointer, larger_leaf_cuda_num_data_in_leaf_pointer,
    larger_leaf_cuda_gain_pointer, larger_leaf_cuda_leaf_value_pointer,
    larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    larger_leaf_cuda_hist_pointer_pointer,
    cuda_num_total_bin_,
    cuda_hist_,
    cuda_hist_pool_, split_indices_block_size_data_partition_aligned,
  
    tree_split_leaf_index_, tree_inner_feature_index_, tree_threshold_,
    tree_left_output_, tree_right_output_, tree_left_count_, tree_right_count_,
    tree_left_sum_hessian_, tree_right_sum_hessian_, tree_gain_, tree_default_left_,
    data_partition_leaf_output_);
  SynchronizeCUDADevice();
  /*PrintLastCUDAError();
  CopyFromCUDADeviceToHost(&cpu_leaf_index, leaf_index, 1);
  CopyFromCUDADeviceToHost(&cpu_cur_num_leaves, cuda_cur_num_leaves_, 1);
  Log::Warning("cpu_leaf_index = %d, cpu_cur_num_leaves = %d after aggregate", cpu_leaf_index, cpu_cur_num_leaves);*/
  auto end = std::chrono::steady_clock::now();
  auto duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("CUDADataPartition::AggregateBlockOffsetKernel");
  global_timer.Start("CUDADataPartition::SplitInnerKernel");

  SplitInnerKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned / 2, 0, cuda_streams_[1]>>>(
    leaf_index, cuda_cur_num_leaves_, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_data_indices_, cuda_data_to_left_,
    cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
    cuda_out_data_indices_in_leaf_, split_indices_block_size_data_partition_aligned);
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("CUDADataPartition::SplitInnerKernel");
  global_timer.Start("CUDADataPartition::CopyDataIndicesKernel");
  start = std::chrono::steady_clock::now();
  CopyDataIndicesKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[1]>>>(
    leaf_index, cuda_cur_num_leaves_, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_out_data_indices_in_leaf_, cuda_data_indices_);
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("CUDADataPartition::CopyDataIndicesKernel");

  start = std::chrono::steady_clock::now();
  global_timer.Start("CUDADataPartition::SplitTreeStructureKernel");
  SplitTreeStructureKernel<<<1, 1, 0, cuda_streams_[0]>>>(leaf_index, cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
    cuda_leaf_num_data_, cuda_data_indices_,
    cuda_cur_num_leaves_,
    best_split_feature, best_split_threshold, best_split_default_left, best_split_gain,
    best_left_sum_gradients, best_left_sum_hessians, best_left_count,
    best_left_gain, best_left_leaf_value,
    best_right_sum_gradients, best_right_sum_hessians, best_right_count,
    best_right_gain, best_right_leaf_value,

    smaller_leaf_cuda_leaf_index_pointer, smaller_leaf_cuda_sum_of_gradients_pointer,
    smaller_leaf_cuda_sum_of_hessians_pointer, smaller_leaf_cuda_num_data_in_leaf_pointer,
    smaller_leaf_cuda_gain_pointer, smaller_leaf_cuda_leaf_value_pointer,
    smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    smaller_leaf_cuda_hist_pointer_pointer,
    larger_leaf_cuda_leaf_index_pointer, larger_leaf_cuda_sum_of_gradients_pointer,
    larger_leaf_cuda_sum_of_hessians_pointer, larger_leaf_cuda_num_data_in_leaf_pointer,
    larger_leaf_cuda_gain_pointer, larger_leaf_cuda_leaf_value_pointer,
    larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    larger_leaf_cuda_hist_pointer_pointer,
    cuda_num_total_bin_,
    cuda_hist_,
    cuda_hist_pool_, split_indices_block_size_data_partition_aligned,
  
    tree_split_leaf_index_, tree_inner_feature_index_, tree_threshold_,
    tree_left_output_, tree_right_output_, tree_left_count_, tree_right_count_,
    tree_left_sum_hessian_, tree_right_sum_hessian_, tree_gain_, tree_default_left_,
    data_partition_leaf_output_, cuda_split_info_buffer_);
  global_timer.Stop("CUDADataPartition::SplitTreeStructureKernel");
  std::vector<int> cpu_split_info_buffer(8);
  CopyFromCUDADeviceToHostAsync<int>(cpu_split_info_buffer.data(), cuda_split_info_buffer_, 8, cuda_streams_[0]);
  SynchronizeCUDADevice();
  const int left_leaf_index = cpu_split_info_buffer[0];
  const data_size_t left_leaf_num_data = cpu_split_info_buffer[1];
  const data_size_t left_leaf_data_start = cpu_split_info_buffer[2];
  const int right_leaf_index = cpu_split_info_buffer[3];
  const data_size_t right_leaf_num_data = cpu_split_info_buffer[4];
  const data_size_t right_leaf_data_start = cpu_split_info_buffer[5];
  (*cpu_leaf_num_data)[left_leaf_index] = left_leaf_num_data;
  (*cpu_leaf_data_start)[left_leaf_index] = left_leaf_data_start;
  (*cpu_leaf_num_data)[right_leaf_index] = right_leaf_num_data;
  (*cpu_leaf_data_start)[right_leaf_index] = right_leaf_data_start;
  *smaller_leaf_index = cpu_split_info_buffer[6];
  *larger_leaf_index = cpu_split_info_buffer[7];
}

__global__ void PrefixSumKernel(uint32_t* cuda_elements) {
  __shared__ uint32_t elements[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_read_index = blockIdx.x * blockDim.x * 2 + threadIdx_x;
  elements[threadIdx_x] = cuda_elements[global_read_index];
  elements[threadIdx_x + blockDim.x] = cuda_elements[global_read_index + blockDim.x];
  __syncthreads();
  PrefixSum(elements, SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  __syncthreads();
  cuda_elements[global_read_index] = elements[threadIdx_x];
  cuda_elements[global_read_index + blockDim.x] = elements[threadIdx_x + blockDim.x];
}

void CUDADataPartition::LaunchPrefixSumKernel(uint32_t* cuda_elements) {
  PrefixSumKernel<<<1, SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION / 2>>>(cuda_elements);
  SynchronizeCUDADevice();
}

__global__ void AddPredictionToScoreKernel(const double* data_partition_leaf_output,
  const data_size_t* num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const data_size_t* leaf_data_start, const double learning_rate, double* output_score, double* cuda_scores) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const data_size_t num_data = num_data_in_leaf[blockIdx_x];
  const data_size_t* data_indices = data_indices_in_leaf + leaf_data_start[blockIdx_x];
  const double leaf_prediction_value = data_partition_leaf_output[blockIdx_x] * learning_rate;
  for (unsigned int offset = 0; offset < static_cast<unsigned int>(num_data); offset += blockDim_x) {
    const data_size_t inner_data_index = static_cast<data_size_t>(offset + threadIdx_x);
    if (inner_data_index < num_data) {
      const data_size_t data_index = data_indices[inner_data_index];
      cuda_scores[data_index] += leaf_prediction_value;
    }
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double learning_rate, double* cuda_scores) {
  global_timer.Start("CUDADataPartition::AddPredictionToScoreKernel");
  AddPredictionToScoreKernel<<<cur_num_leaves_, 1024>>>(data_partition_leaf_output_,
    cuda_leaf_num_data_, cuda_data_indices_, cuda_leaf_data_start_, learning_rate, train_data_score_tmp_, cuda_scores);
  SynchronizeCUDADevice();
  global_timer.Stop("CUDADataPartition::AddPredictionToScoreKernel");
}

__global__ void CopyColWiseDataKernel(const uint8_t* row_wise_data,
  const data_size_t num_data, const int num_features,
  uint8_t* col_wise_data) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const data_size_t read_offset = data_index * num_features;
    for (int feature_index = 0; feature_index < num_features; ++feature_index) {
      const data_size_t write_pos = feature_index * num_data + data_index;
      col_wise_data[write_pos] = row_wise_data[read_offset + feature_index];
    }
  }
}

void CUDADataPartition::LaunchCopyColWiseDataKernel() {
  const int block_size = 1024;
  const int num_blocks = (num_data_ + block_size - 1) / block_size;
  CopyColWiseDataKernel<<<num_blocks, block_size>>>(cuda_data_, num_data_, num_features_, cuda_data_col_wise_);
}

__global__ void CUDACheckKernel(const data_size_t** data_indices_in_leaf_ptr,
  const data_size_t num_data_in_leaf,
  const score_t* gradients,
  const score_t* hessians,
  double* gradients_sum_buffer,
  double* hessians_sum_buffer) {
  const data_size_t* data_indices_in_leaf = *data_indices_in_leaf_ptr;
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double local_gradients[1024];
  __shared__ double local_hessians[1024];
  if (local_data_index < num_data_in_leaf) {
    const data_size_t global_data_index = data_indices_in_leaf[local_data_index];
    local_gradients[threadIdx.x] = gradients[global_data_index];
    local_hessians[threadIdx.x] = hessians[global_data_index];
  } else {
    local_gradients[threadIdx.x] = 0.0f;
    local_hessians[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  ReduceSum(local_gradients, 1024);
  __syncthreads();
  ReduceSum(local_hessians, 1024);
  __syncthreads();
  if (threadIdx.x == 0) {
    gradients_sum_buffer[blockIdx.x] = local_gradients[0];
    hessians_sum_buffer[blockIdx.x] = local_hessians[0];
  }
}

__global__ void CUDACheckKernel2(
  const int leaf_index,
  const data_size_t* num_data_expected,
  const double* sum_gradients_expected,
  const double* sum_hessians_expected,
  const double* gradients_sum_buffer,
  const double* hessians_sum_buffer,
  const int num_blocks) {
  double sum_gradients = 0.0f;
  double sum_hessians = 0.0f;
  for (int i = 0; i < num_blocks; ++i) {
    sum_gradients += gradients_sum_buffer[i];
    sum_hessians += hessians_sum_buffer[i];
  }
  if (fabs(sum_gradients - *sum_gradients_expected) >= 1.0f) {
    printf("error in leaf_index = %d\n", leaf_index);
    printf("num data expected = %d\n", *num_data_expected);
    printf("error sum_gradients: %f vs %f\n", sum_gradients, *sum_gradients_expected);
  }
  if (fabs(sum_hessians - *sum_hessians_expected) >= 1.0f) {
    printf("error in leaf_index = %d\n", leaf_index);
    printf("num data expected = %d\n", *num_data_expected);
    printf("error sum_hessians: %f vs %f\n", sum_hessians, *sum_hessians_expected);
  }
}

void CUDADataPartition::LaunchCUDACheckKernel(
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const std::vector<data_size_t>& num_data_in_leaf,
  const CUDALeafSplits* smaller_leaf_splits,
  const CUDALeafSplits* larger_leaf_splits,
  const score_t* gradients,
  const score_t* hessians) {
  const data_size_t num_data_in_smaller_leaf = num_data_in_leaf[smaller_leaf_index];
  const int block_dim = 1024;
  const int smaller_num_blocks = (num_data_in_smaller_leaf + block_dim - 1) / block_dim;
  CUDACheckKernel<<<smaller_num_blocks, block_dim>>>(smaller_leaf_splits->cuda_data_indices_in_leaf(),
    num_data_in_smaller_leaf,
    gradients,
    hessians,
    cuda_gradients_sum_buffer_,
    cuda_hessians_sum_buffer_);
  CUDACheckKernel2<<<1, 1>>>(
    smaller_leaf_index,
    smaller_leaf_splits->cuda_num_data_in_leaf(),
    smaller_leaf_splits->cuda_sum_of_gradients(),
    smaller_leaf_splits->cuda_sum_of_hessians(),
    cuda_gradients_sum_buffer_,
    cuda_hessians_sum_buffer_,
    smaller_num_blocks);
  if (larger_leaf_index >= 0) {
    const data_size_t num_data_in_larger_leaf = num_data_in_leaf[larger_leaf_index];
    const int larger_num_blocks = (num_data_in_larger_leaf + block_dim - 1) / block_dim;
    CUDACheckKernel<<<larger_num_blocks, block_dim>>>(larger_leaf_splits->cuda_data_indices_in_leaf(),
      num_data_in_larger_leaf,
      gradients,
      hessians,
      cuda_gradients_sum_buffer_,
      cuda_hessians_sum_buffer_);
    CUDACheckKernel2<<<1, 1>>>(
      larger_leaf_index,
      larger_leaf_splits->cuda_num_data_in_leaf(),
      larger_leaf_splits->cuda_sum_of_gradients(),
      larger_leaf_splits->cuda_sum_of_hessians(),
      cuda_gradients_sum_buffer_,
      cuda_hessians_sum_buffer_,
      larger_num_blocks);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
