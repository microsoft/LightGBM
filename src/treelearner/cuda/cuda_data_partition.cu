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

__device__ void PrefixSum_1024(uint32_t* elements, unsigned int n) {
  unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_n_minus_1 = CONFLICT_FREE_INDEX(n - 1);
  const uint32_t last_element = elements[conflict_free_n_minus_1];
  __syncthreads();

  if (threadIdx_x < 512) {
    const unsigned int src_pos = (2 * threadIdx_x + 1) - 1;
    const unsigned int dst_pos = (2 * threadIdx_x + 2) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();

  if (threadIdx_x < 256) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 1) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 1) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();
  
  if (threadIdx_x < 128) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 2) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 2) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();
  
  if (threadIdx_x < 64) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 3) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 3) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();
  
  if (threadIdx_x < 32) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 4) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 4) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();
  
  if (threadIdx_x < 16) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 5) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 5) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();

  if (threadIdx_x < 8) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 6) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 6) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();

  if (threadIdx_x < 4) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 7) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 7) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();

  if (threadIdx_x < 2) {
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 8) - 1;
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 8) - 1;
    elements[CONFLICT_FREE_INDEX(dst_pos)] += elements[CONFLICT_FREE_INDEX(src_pos)];
  }
  __syncthreads();

  if (threadIdx_x == 0) {
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(1023);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(511);
    elements[conflict_free_dst_pos] += elements[conflict_free_src_pos];
    elements[conflict_free_n_minus_1] = 0; 
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 2) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 8) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 8) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 4) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 7) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 7) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 8) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 6) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 6) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 16) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 5) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 5) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 32) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 4) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 4) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 64) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 3) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 3) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 128) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 2) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 2) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 256) {
    const unsigned int dst_pos = ((2 * threadIdx_x + 2) << 1) - 1;
    const unsigned int src_pos = ((2 * threadIdx_x + 1) << 1) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

  if (threadIdx_x < 512) {
    const unsigned int dst_pos = (2 * threadIdx_x + 2) - 1;
    const unsigned int src_pos = (2 * threadIdx_x + 1) - 1;
    const unsigned int conflict_free_dst_pos = CONFLICT_FREE_INDEX(dst_pos);
    const unsigned int conflict_free_src_pos = CONFLICT_FREE_INDEX(src_pos);
    const uint32_t src_val = elements[conflict_free_src_pos];
    elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
    elements[conflict_free_dst_pos] += src_val;
  }
  __syncthreads();

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
  data_size_t* data_indices, int* cuda_data_index_to_leaf_index) {
  const data_size_t num_data_ref = *cuda_num_data;
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data_ref) {
    data_indices[data_index] = data_index;
    cuda_data_index_to_leaf_index[data_index] = 0;
  }
}

void CUDADataPartition::LaunchFillDataIndicesBeforeTrain() {
  const int num_blocks = (num_data_ + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(cuda_num_data_, cuda_data_indices_, cuda_data_index_to_leaf_index_);
}

__device__ void PrepareOffset(const data_size_t num_data_in_leaf_ref, const uint8_t* split_to_left_bit_vector,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition,
  uint16_t* thread_to_left_offset_cnt) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  __syncthreads();
  ReduceSum(thread_to_left_offset_cnt, split_indices_block_size_data_partition);
  __syncthreads();
  if (threadIdx_x == 0) {
    const data_size_t num_data_in_block = (blockIdx.x + 1) * blockDim_x <= num_data_in_leaf_ref ? static_cast<data_size_t>(blockDim_x) :
      num_data_in_leaf_ref - static_cast<data_size_t>(blockIdx.x * blockDim_x);
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

template <bool MIN_IS_MAX, bool MAX_TO_LEFT, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
__global__ void UpdateDataIndexToLeafIndexKernel(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t th, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index, const int missing_default_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin_ref)) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin < min_bin_ref || bin > max_bin_ref) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else if (bin > th) {
        cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin != max_bin_ref) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          if (!MAX_TO_LEFT) {
            cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
          }
        }
      }
    }
  }
}

#define UpdateDataIndexToLeafIndex_ARGS \
  num_data_in_leaf, data_indices_in_leaf, th, column_data, \
  t_zero_bin, max_bin_ref, min_bin_ref, cuda_data_index_to_leaf_index, left_leaf_index, right_leaf_index, \
  default_leaf_index, missing_default_leaf_index

template <typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t th, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index, const int missing_default_leaf_index,
  const bool missing_is_zero, const bool missing_is_na, const bool mfb_is_zero, const bool mfb_is_na, const bool max_to_left,
  const int num_blocks, const int block_size) {
  if (min_bin_ref < max_bin_ref) {
    if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, false, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, false, true, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, false, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<false, true, true, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    }
  } else {
    if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, false, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, false, true, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, false, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, false, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, false, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, false, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, false, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, true, false, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, true, false, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, true, true, false, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_to_left) {
      UpdateDataIndexToLeafIndexKernel<true, true, true, true, true, true, BIN_TYPE><<<num_blocks, block_size, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_ARGS);
    }
  }
}

// min_bin_ref < max_bin_ref
template <typename BIN_TYPE, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA>
__global__ void GenDataToLeftBitVectorKernel0(const int best_split_feature_ref,
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t th, const int num_features_ref, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index, const int missing_default_leaf_index) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
      (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin_ref)) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if ((bin < min_bin_ref || bin > max_bin_ref)) {
      if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO || MFB_IS_ZERO)) {
        cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
      } else {
        cuda_data_to_left[local_data_index] = split_default_to_left;
        thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
      }
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

// min_bin_ref == max_bin_ref
template <typename BIN_TYPE, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT>
__global__ void GenDataToLeftBitVectorKernel16(const int best_split_feature_ref,
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t th, const int num_features_ref, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t t_zero_bin, const uint32_t most_freq_bin_ref, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
  const uint8_t split_default_to_left, const uint8_t split_missing_default_to_left,
  uint8_t* cuda_data_to_left,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index, const int missing_default_leaf_index) {
  __shared__ uint16_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
      cuda_data_to_left[local_data_index] = split_missing_default_to_left;
      thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
    } else if (bin != max_bin_ref) {
      if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
        cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
      } else {
        cuda_data_to_left[local_data_index] = split_default_to_left;
        thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_default_to_left;
      }
    } else {
      if (MISSING_IS_NA && !MFB_IS_NA) {
        cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = split_missing_default_to_left;
      } else {
        if (MAX_TO_LEFT) {
          cuda_data_to_left[local_data_index] = 1;
          thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 1;
        } else {
          cuda_data_to_left[local_data_index] = 0;
          thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
        }
      }
    }
  } else {
    thread_to_left_offset_cnt[CONFLICT_FREE_INDEX(threadIdx.x)] = 0;
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, cuda_data_to_left, block_to_left_offset_buffer, block_to_right_offset_buffer,
    split_indices_block_size_data_partition, thread_to_left_offset_cnt);
}

#define GenBitVector_ARGS \
  split_feature_index, num_data_in_leaf, data_indices_in_leaf, \
  th, num_features_,  \
  column_data, t_zero_bin, most_freq_bin, max_bin, min_bin, split_default_to_left,  \
  split_missing_default_to_left, cuda_data_to_left_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_, \
  split_indices_block_size_data_partition_aligned, \
  cuda_data_index_to_leaf_index_, left_leaf_index, right_leaf_index, default_leaf_index, missing_default_leaf_index

template <typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelMaxIsMinInner(
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left,
  const int column_index,
  const int num_blocks_final,
  const int split_indices_block_size_data_partition_aligned,
  const int split_feature_index,
  const data_size_t leaf_data_start,
  const data_size_t num_data_in_leaf,
  const uint32_t th,
  const uint32_t t_zero_bin,
  const uint32_t most_freq_bin,
  const uint32_t max_bin,
  const uint32_t min_bin,
  const uint8_t split_default_to_left,
  const uint8_t split_missing_default_to_left,
  const int left_leaf_index,
  const int right_leaf_index,
  const int default_leaf_index,
  const int missing_default_leaf_index) {
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  const data_size_t* data_indices_in_leaf = cuda_data_indices_ + leaf_data_start;
  if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, false, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, false, true, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, false, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel16<BIN_TYPE, true, true, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  }
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelMaxIsNotMinInner(
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const int column_index,
  const int num_blocks_final,
  const int split_indices_block_size_data_partition_aligned,
  const int split_feature_index,
  const data_size_t leaf_data_start,
  const data_size_t num_data_in_leaf,
  const uint32_t th,
  const uint32_t t_zero_bin,
  const uint32_t most_freq_bin,
  const uint32_t max_bin,
  const uint32_t min_bin,
  const uint8_t split_default_to_left,
  const uint8_t split_missing_default_to_left,
  const int left_leaf_index,
  const int right_leaf_index,
  const int default_leaf_index,
  const int missing_default_leaf_index) {
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  const data_size_t* data_indices_in_leaf = cuda_data_indices_ + leaf_data_start;
  if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (!missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, false, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, false, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, false, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, false, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, true, false><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  } else if (missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na) {
    const BIN_TYPE* column_data = reinterpret_cast<const BIN_TYPE*>(column_data_pointer);
    GenDataToLeftBitVectorKernel0<BIN_TYPE, true, true, true, true><<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[0]>>>(GenBitVector_ARGS);
  }
}

#undef GenBitVector_ARGS

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel(const data_size_t num_data_in_leaf,
  const int split_feature_index, const uint32_t split_threshold,
  const uint8_t split_default_left, const data_size_t leaf_data_start,
  const int left_leaf_index, const int right_leaf_index) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  const uint8_t missing_is_zero = cuda_column_data_->feature_missing_is_zero(split_feature_index);
  const uint8_t missing_is_na = cuda_column_data_->feature_missing_is_na(split_feature_index);
  const uint8_t mfb_is_zero = cuda_column_data_->feature_mfb_is_zero(split_feature_index);
  const uint8_t mfb_is_na = cuda_column_data_->feature_mfb_is_na(split_feature_index);
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t min_bin = cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);

  uint32_t th = split_threshold + min_bin;
  uint32_t t_zero_bin = min_bin + default_bin;
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;  
  }
  uint8_t split_default_to_left = 0;
  uint8_t split_missing_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  int missing_default_leaf_index = right_leaf_index;
  if (most_freq_bin <= split_threshold) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (missing_is_zero || missing_is_na) {
    if (split_default_left) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = left_leaf_index;
    }
  }
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);

  const bool max_bin_to_left = (max_bin <= th);

  const data_size_t* data_indices_in_leaf = cuda_data_indices_ + leaf_data_start;

  if (min_bin < max_bin) {
    if (bit_type == 8) {
      LaunchGenDataToLeftBitVectorKernelMaxIsNotMinInner<uint8_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    } else if (bit_type == 16) {
      LaunchGenDataToLeftBitVectorKernelMaxIsNotMinInner<uint16_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    } else if (bit_type == 32) {
      LaunchGenDataToLeftBitVectorKernelMaxIsNotMinInner<uint32_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    }
  } else {
    if (bit_type == 8) {
      LaunchGenDataToLeftBitVectorKernelMaxIsMinInner<uint8_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        max_bin_to_left,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    } else if (bit_type == 16) {
      LaunchGenDataToLeftBitVectorKernelMaxIsMinInner<uint16_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        max_bin_to_left,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    } else if (bit_type == 32) {
      LaunchGenDataToLeftBitVectorKernelMaxIsMinInner<uint32_t>(
        missing_is_zero,
        missing_is_na,
        mfb_is_zero,
        mfb_is_na,
        max_bin_to_left,
        column_index,
        num_blocks_final,
        split_indices_block_size_data_partition_aligned,
        split_feature_index,
        leaf_data_start,
        num_data_in_leaf,
        th,
        t_zero_bin,
        most_freq_bin,
        max_bin,
        min_bin,
        split_default_to_left,
        split_missing_default_to_left,
        left_leaf_index,
        right_leaf_index,
        default_leaf_index,
        missing_default_leaf_index);
    }
  }

  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    LaunchUpdateDataIndexToLeafIndexKernel<uint8_t>(num_data_in_leaf,
      data_indices_in_leaf, th, column_data, t_zero_bin, max_bin, min_bin, cuda_data_index_to_leaf_index_,
      left_leaf_index, right_leaf_index, default_leaf_index, missing_default_leaf_index,
      static_cast<bool>(missing_is_zero),
      static_cast<bool>(missing_is_na),
      static_cast<bool>(mfb_is_zero),
      static_cast<bool>(mfb_is_na),
      max_bin_to_left,
      num_blocks_final,
      split_indices_block_size_data_partition_aligned);
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    LaunchUpdateDataIndexToLeafIndexKernel<uint16_t>(num_data_in_leaf,
      data_indices_in_leaf, th, column_data, t_zero_bin, max_bin, min_bin, cuda_data_index_to_leaf_index_,
      left_leaf_index, right_leaf_index, default_leaf_index, missing_default_leaf_index,
      static_cast<bool>(missing_is_zero),
      static_cast<bool>(missing_is_na),
      static_cast<bool>(mfb_is_zero),
      static_cast<bool>(mfb_is_na),
      max_bin_to_left,
      num_blocks_final,
      split_indices_block_size_data_partition_aligned);
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    LaunchUpdateDataIndexToLeafIndexKernel<uint32_t>(num_data_in_leaf,
      data_indices_in_leaf, th, column_data, t_zero_bin, max_bin, min_bin, cuda_data_index_to_leaf_index_,
      left_leaf_index, right_leaf_index, default_leaf_index, missing_default_leaf_index,
      static_cast<bool>(missing_is_zero),
      static_cast<bool>(missing_is_na),
      static_cast<bool>(mfb_is_zero),
      static_cast<bool>(mfb_is_na),
      max_bin_to_left,
      num_blocks_final,
      split_indices_block_size_data_partition_aligned);
  }
}

__global__ void AggregateBlockOffsetKernel0(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t block_to_left_offset[AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2 +
    (AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint32_t block_to_right_offset[AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2 +
    (AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_threadIdx_x = CONFLICT_FREE_INDEX(threadIdx_x);
  const data_size_t num_blocks_plus_1 = num_blocks + 1;
  const uint32_t num_blocks_per_thread = (num_blocks_plus_1 + blockDim_x - 1) / blockDim_x;
  const uint32_t remain = num_blocks_plus_1 - ((num_blocks_per_thread - 1) * blockDim_x);
  const uint32_t remain_offset = remain * num_blocks_per_thread;
  uint32_t thread_start_block_index = 0;
  uint32_t thread_end_block_index = 0;
  if (threadIdx_x < remain) {
    thread_start_block_index = threadIdx_x * num_blocks_per_thread;
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks_plus_1);
  } else {
    thread_start_block_index = remain_offset + (num_blocks_per_thread - 1) * (threadIdx_x - remain);
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread - 1, num_blocks_plus_1);
  }
  if (threadIdx.x == 0) {
    block_to_right_offset_buffer[0] = 0;
  }
  __syncthreads();
  for (uint32_t block_index = thread_start_block_index + 1; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += block_to_left_offset_buffer[block_index - 1];
    block_to_right_offset_buffer[block_index] += block_to_right_offset_buffer[block_index - 1];
  }
  __syncthreads();
  if (thread_start_block_index < thread_end_block_index) {
    block_to_left_offset[conflict_free_threadIdx_x] = block_to_left_offset_buffer[thread_end_block_index - 1];
    block_to_right_offset[conflict_free_threadIdx_x] = block_to_right_offset_buffer[thread_end_block_index - 1];
  } else {
    block_to_left_offset[conflict_free_threadIdx_x] = 0;
    block_to_right_offset[conflict_free_threadIdx_x] = 0;
  }
  __syncthreads();
  PrefixSum_1024(block_to_left_offset, blockDim_x);
  PrefixSum_1024(block_to_right_offset, blockDim_x);
  __syncthreads();
  const uint32_t to_left_total_count = block_to_left_offset[CONFLICT_FREE_INDEX(blockDim_x)];
  const uint32_t to_left_thread_block_offset = block_to_left_offset[conflict_free_threadIdx_x];
  const uint32_t to_right_thread_block_offset = block_to_right_offset[conflict_free_threadIdx_x] + to_left_total_count;
  for (uint32_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += to_left_thread_block_offset;
    block_to_right_offset_buffer[block_index] += to_right_thread_block_offset;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

__global__ void AggregateBlockOffsetKernel1(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks, const data_size_t num_blocks_aligned) {
  __shared__ uint32_t block_to_left_offset[AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2 +
    (AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint32_t block_to_right_offset[AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2 +
    (AGGREGATE_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_threadIdx_x = CONFLICT_FREE_INDEX(threadIdx_x);
  const unsigned int conflict_free_threadIdx_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + 1);
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset[conflict_free_threadIdx_x] = block_to_left_offset_buffer[threadIdx_x + 1];
    block_to_right_offset[conflict_free_threadIdx_x] = block_to_right_offset_buffer[threadIdx_x + 1];
  } else {
    block_to_left_offset[conflict_free_threadIdx_x] = 0;
    block_to_right_offset[conflict_free_threadIdx_x] = 0;
  }
  __syncthreads();
  PrefixSum(block_to_left_offset, num_blocks_aligned);
  PrefixSum(block_to_right_offset, num_blocks_aligned);
  __syncthreads();
  const uint32_t to_left_total_count = block_to_left_offset[CONFLICT_FREE_INDEX(num_blocks_aligned)];
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset_buffer[threadIdx_x + 1] = block_to_left_offset[conflict_free_threadIdx_x_plus_1];
    block_to_right_offset_buffer[threadIdx_x + 1] = block_to_right_offset[conflict_free_threadIdx_x_plus_1] + to_left_total_count;
  }
  if (threadIdx_x == 0) {
    block_to_right_offset_buffer[0] = to_left_total_count;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

__global__ void SplitTreeStructureKernel(const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool, const int split_indices_block_size_data_partition,

  const double* cuda_bin_upper_bounds, const int* cuda_feature_num_bin_offsets,
  double* cuda_leaf_output,
  int* cuda_split_info_buffer) {
  const unsigned int to_left_total_cnt = cuda_leaf_num_data[left_leaf_index];
  double* cuda_split_info_buffer_for_hessians = reinterpret_cast<double*>(cuda_split_info_buffer + 8);
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_index == 0) {
    cuda_leaf_output[left_leaf_index] = best_split_info->left_value;
  } else if (global_thread_index == 1) {
    cuda_leaf_output[right_leaf_index] = best_split_info->right_value;
  } else if (global_thread_index == 2) {
    cuda_split_info_buffer[0] = left_leaf_index;
  } else if (global_thread_index == 3) {
    cuda_split_info_buffer[1] = cuda_leaf_num_data[left_leaf_index];
  } else if (global_thread_index == 4) {
    cuda_split_info_buffer[2] = cuda_leaf_data_start[left_leaf_index];
  } else if (global_thread_index == 5) {
    cuda_split_info_buffer[3] = right_leaf_index;
  } else if (global_thread_index == 6) {
    cuda_split_info_buffer[4] = cuda_leaf_num_data[right_leaf_index];
  } else if (global_thread_index == 7) {
    cuda_split_info_buffer[5] = cuda_leaf_data_start[right_leaf_index];
  } else if (global_thread_index == 8) {
    cuda_split_info_buffer_for_hessians[0] = best_split_info->left_sum_hessians;
  } else if (global_thread_index == 9) {
    cuda_split_info_buffer_for_hessians[1] = best_split_info->right_sum_hessians;
  }

  if (cuda_leaf_num_data[left_leaf_index] < cuda_leaf_num_data[right_leaf_index]) {
    if (global_thread_index == 0) {
      hist_t* parent_hist_ptr = cuda_hist_pool[left_leaf_index];
      cuda_hist_pool[right_leaf_index] = parent_hist_ptr;
      cuda_hist_pool[left_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 1) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      smaller_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      smaller_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      smaller_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      larger_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      larger_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      larger_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      larger_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      larger_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      larger_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_split_info_buffer[6] = left_leaf_index;
    } else if (global_thread_index == 15) {
      cuda_split_info_buffer[7] = right_leaf_index;
    } else if (global_thread_index == 16) {
      smaller_leaf_splits->leaf_index = left_leaf_index;
    }
  } else {
    if (global_thread_index == 0) {
      larger_leaf_splits->leaf_index = left_leaf_index;
    } else if (global_thread_index == 1) {
      larger_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      larger_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      larger_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      larger_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      larger_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      smaller_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      smaller_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      smaller_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      smaller_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_hist_pool[right_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 15) {
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
    } else if (global_thread_index == 16) {
      cuda_split_info_buffer[6] = right_leaf_index;
    } else if (global_thread_index == 17) {
      cuda_split_info_buffer[7] = left_leaf_index;
    }
  }
}

__global__ void SplitInnerKernel(const int left_leaf_index, const int right_leaf_index,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices, const uint8_t* split_to_left_bit_vector,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  data_size_t* out_data_indices_in_leaf, const int split_indices_block_size_data_partition) {
  __shared__ uint16_t thread_to_left_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint16_t thread_to_right_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  uint8_t first_to_left = 0;
  uint8_t second_to_left = 0;
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[left_leaf_index];
  const data_size_t num_data_in_leaf_ref = cuda_leaf_num_data[left_leaf_index] + cuda_leaf_num_data[right_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int conflict_free_threadIdx_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + 1);
  const unsigned int global_thread_index = blockIdx.x * blockDim_x * 2 + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_thread_index];
    first_to_left = bit;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_1] = bit;
  } else {
    first_to_left = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_1] = 0;
  }
  const unsigned int conflict_free_threadIdx_x_plus_blockDim_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + blockDim_x + 1);
  const unsigned int global_thread_index_plus_blockDim_x = global_thread_index + blockDim_x;
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_thread_index_plus_blockDim_x];
    second_to_left = bit;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = bit;
  } else {
    second_to_left = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = 0;
  }
  __syncthreads();
  const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
  if (threadIdx_x == 0) {
    thread_to_left_pos[0] = 0;
    thread_to_right_pos[0] = 0;
  }
  __syncthreads();
  PrefixSum(thread_to_left_pos, split_indices_block_size_data_partition);
  __syncthreads();
  if (threadIdx_x > 0) {
    thread_to_right_pos[threadIdx_x] = (threadIdx_x - thread_to_left_pos[conflict_free_threadIdx_x_plus_1]);
  }
  thread_to_right_pos[threadIdx_x + blockDim_x] = (threadIdx_x + blockDim_x - thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]);
  __syncthreads();
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + to_right_block_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    if (first_to_left == 1) {
      left_out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      right_out_data_indices_in_leaf[thread_to_right_pos[threadIdx_x]] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    if (second_to_left == 1) {
      left_out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    } else {
      right_out_data_indices_in_leaf[thread_to_right_pos[threadIdx_x + blockDim_x]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    }
  }
}

__global__ void CopyDataIndicesKernel(
  const data_size_t num_data_in_leaf,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  if (global_thread_index < num_data_in_leaf) {
    cuda_data_indices[global_thread_index] = out_data_indices_in_leaf[global_thread_index];
  }
}

void CUDADataPartition::LaunchSplitInnerKernel(
  const data_size_t num_data_in_leaf,
  const CUDASplitInfo* best_split_info,
  const int left_leaf_index,
  const int right_leaf_index,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  data_size_t* left_leaf_num_data_ref,
  data_size_t* right_leaf_num_data_ref,
  data_size_t* left_leaf_start_ref,
  data_size_t* right_leaf_start_ref,
  double* left_leaf_sum_of_hessians_ref,
  double* right_leaf_sum_of_hessians_ref) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = std::max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  int num_blocks_final_ref = num_blocks_final - 1;
  int num_blocks_final_aligned = 1;
  while (num_blocks_final_ref > 0) {
    num_blocks_final_aligned <<= 1;
    num_blocks_final_ref >>= 1;
  }
  global_timer.Start("CUDADataPartition::AggregateBlockOffsetKernel");

  if (num_blocks_final > AGGREGATE_BLOCK_SIZE_DATA_PARTITION) {
    AggregateBlockOffsetKernel0<<<1, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_,
      cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
      cuda_leaf_num_data_, cuda_data_indices_,
      num_blocks_final);
  } else {
    AggregateBlockOffsetKernel1<<<1, num_blocks_final_aligned, 0, cuda_streams_[0]>>>(
      left_leaf_index,
      right_leaf_index,
      cuda_block_data_to_left_offset_,
      cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
      cuda_leaf_num_data_, cuda_data_indices_,
      num_blocks_final, num_blocks_final_aligned);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AggregateBlockOffsetKernel");
  global_timer.Start("CUDADataPartition::SplitInnerKernel");

  SplitInnerKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned / 2, 0, cuda_streams_[1]>>>(
    left_leaf_index, right_leaf_index, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_data_indices_, cuda_data_to_left_,
    cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
    cuda_out_data_indices_in_leaf_, split_indices_block_size_data_partition_aligned);
  global_timer.Stop("CUDADataPartition::SplitInnerKernel");

  global_timer.Start("CUDADataPartition::SplitTreeStructureKernel");
  SplitTreeStructureKernel<<<4, 5, 0, cuda_streams_[0]>>>(left_leaf_index, right_leaf_index,
    cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
    cuda_leaf_num_data_, cuda_out_data_indices_in_leaf_,
    best_split_info,
    smaller_leaf_splits,
    larger_leaf_splits,
    num_total_bin_,
    cuda_hist_,
    cuda_hist_pool_, split_indices_block_size_data_partition_aligned,

    cuda_bin_upper_bounds_, cuda_feature_num_bin_offsets_,
    cuda_leaf_output_, cuda_split_info_buffer_);
  global_timer.Stop("CUDADataPartition::SplitTreeStructureKernel");
  std::vector<int> cpu_split_info_buffer(12);
  const double* cpu_sum_hessians_info = reinterpret_cast<const double*>(cpu_split_info_buffer.data() + 8);
  global_timer.Start("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  CopyFromCUDADeviceToHostAsync<int>(cpu_split_info_buffer.data(), cuda_split_info_buffer_, 12, cuda_streams_[0]);
  global_timer.Stop("CUDADataPartition::CopyFromCUDADeviceToHostAsync");
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  const data_size_t left_leaf_num_data = cpu_split_info_buffer[1];
  const data_size_t left_leaf_data_start = cpu_split_info_buffer[2];
  const data_size_t right_leaf_num_data = cpu_split_info_buffer[4];
  global_timer.Start("CUDADataPartition::CopyDataIndicesKernel");
  CopyDataIndicesKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned, 0, cuda_streams_[2]>>>(
    left_leaf_num_data + right_leaf_num_data, cuda_out_data_indices_in_leaf_, cuda_data_indices_ + left_leaf_data_start);
  global_timer.Stop("CUDADataPartition::CopyDataIndicesKernel");
  const data_size_t right_leaf_data_start = cpu_split_info_buffer[5];
  *left_leaf_num_data_ref = left_leaf_num_data;
  *left_leaf_start_ref = left_leaf_data_start;
  *right_leaf_num_data_ref = right_leaf_num_data;
  *right_leaf_start_ref = right_leaf_data_start;
  *left_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[0];
  *right_leaf_sum_of_hessians_ref = cpu_sum_hessians_info[1];
}

__global__ void AddPredictionToScoreKernel(const double* cuda_leaf_output,
  const data_size_t* num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const data_size_t* leaf_data_start, const double learning_rate, double* cuda_scores,
  const int* cuda_data_index_to_leaf_index, const data_size_t num_data) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const int data_index = static_cast<int>(blockIdx_x * blockDim_x + threadIdx_x);
  if (data_index < num_data) {
    const int leaf_index = cuda_data_index_to_leaf_index[data_index];
    const double leaf_prediction_value = cuda_leaf_output[leaf_index] * learning_rate;
    cuda_scores[data_index] += leaf_prediction_value;
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double learning_rate, double* cuda_scores) {
  global_timer.Start("CUDADataPartition::AddPredictionToScoreKernel");
  const int num_blocks = (num_data_ + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  AddPredictionToScoreKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(cuda_leaf_output_,
    cuda_leaf_num_data_, cuda_data_indices_, cuda_leaf_data_start_, learning_rate, cuda_scores, cuda_data_index_to_leaf_index_, num_data_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AddPredictionToScoreKernel");
}

}  // namespace LightGBM

#endif  // USE_CUDA
