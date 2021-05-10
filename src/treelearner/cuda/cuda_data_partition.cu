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

__global__ void GenDataToLeftBitVectorKernel(const int* leaf_index, const data_size_t* cuda_leaf_data_start,
  const data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices, const int* best_split_feature,
  const uint32_t* best_split_threshold, const int* cuda_num_features, const uint8_t* cuda_data,
  const uint32_t* default_bin, const uint32_t* most_freq_bin, const uint8_t* default_left,
  const uint32_t* min_bin, const uint32_t* max_bin, const uint8_t* missing_is_zero, const uint8_t* missing_is_na,
  const uint8_t* mfb_is_zero, const uint8_t* mfb_is_na,
  uint8_t* cuda_data_to_left) {
  /*if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("GenDataToLeftBitVectorKernel step 0\n");
  }*/
  const int leaf_index_ref = *leaf_index;
  /*if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("GenDataToLeftBitVectorKernel leaf_index_ref = %d\n", leaf_index_ref);
  }*/
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
    /*if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("GenDataToLeftBitVectorKernel step 1\n");
    }*/
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
    /*if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("GenDataToLeftBitVectorKernel step 2\n");
    }*/
    if (local_data_index < static_cast<unsigned int>(num_data_in_leaf)) {
      /*if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GenDataToLeftBitVectorKernel step 3\n");
      }*/
      const uint32_t bin = static_cast<uint32_t>(cuda_data[global_feature_value_index]);
      if (min_bin_ref < max_bin_ref) {
        if ((missing_is_zero_ref && !mfb_is_zero_ref && bin == t_zero_bin)) {
          cuda_data_to_left[local_data_index] = split_missing_default_to_left;
        } else if (bin < min_bin_ref || bin > max_bin_ref) {
          if ((missing_is_na_ref || mfb_is_na_ref) || (missing_is_zero_ref || mfb_is_zero_ref)) {
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
        if (missing_is_zero_ref || !mfb_is_zero_ref && bin == t_zero_bin) {
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
      /*if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("GenDataToLeftBitVectorKernel step 4\n");
      }*/
    }
  }
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

__global__ void PrepareOffsetKernel(const int* leaf_index,
  const data_size_t* cuda_leaf_num_data,  const uint8_t* split_to_left_bit_vector,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const int split_indices_block_size_data_partition) {
  const unsigned int blockDim_x = blockDim.x;
  __shared__ uint32_t thread_to_left_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  //__shared__ uint32_t thread_to_right_offset_cnt[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
  //  (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1) / NUM_BANKS_DATA_PARTITION];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_threadIdx_x = CONFLICT_FREE_INDEX(threadIdx_x);
  const unsigned int global_read_index = blockIdx.x * blockDim.x * 2 + threadIdx_x;
  const data_size_t num_data_in_leaf_ref = cuda_leaf_num_data[*leaf_index];
  /*if (blockIdx.x == 0 && threadIdx_x == 0) {
    printf("PrepareOffsetKernel leaf_index = %d, num_data_in_leaf = %d\n", (*leaf_index), num_data_in_leaf_ref);
  }*/
  if (global_read_index < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_read_index];
    thread_to_left_offset_cnt[conflict_free_threadIdx_x] = bit;
    //thread_to_right_offset_cnt[conflict_free_threadIdx_x] = 1 - bit;
  } else {
    thread_to_left_offset_cnt[conflict_free_threadIdx_x] = 0;
    //thread_to_right_offset_cnt[conflict_free_threadIdx_x] = 0;
  }
  const unsigned int conflict_free_threadIdx_x_offseted = CONFLICT_FREE_INDEX(threadIdx_x + blockDim_x);
  if (global_read_index + blockDim_x < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_read_index + blockDim_x];
    thread_to_left_offset_cnt[conflict_free_threadIdx_x_offseted] = bit;
    //thread_to_right_offset_cnt[conflict_free_threadIdx_x_offseted] = 1 - bit;
  } else {
    thread_to_left_offset_cnt[conflict_free_threadIdx_x_offseted] = 0;
    //thread_to_right_offset_cnt[conflict_free_threadIdx_x_offseted] = 0;
  }
  __syncthreads();
  PrefixSum(thread_to_left_offset_cnt, split_indices_block_size_data_partition);
  //PrefixSum(thread_to_right_offset_cnt, split_indices_block_size_data_partition);
  __syncthreads();
  if (threadIdx_x == 0) {
    const unsigned int conflict_free_blockDim_x_times_2 = CONFLICT_FREE_INDEX(blockDim_x << 1);
    const data_size_t num_data_in_block = (blockIdx.x + 1) * blockDim.x * 2 <= num_data_in_leaf_ref ? static_cast<data_size_t>(blockDim_x * 2) :
      num_data_in_leaf_ref - static_cast<data_size_t>(blockIdx.x * blockDim.x * 2);
    if (num_data_in_block > 0) {
      const data_size_t data_to_left = static_cast<data_size_t>(thread_to_left_offset_cnt[conflict_free_blockDim_x_times_2]);
      block_to_left_offset_buffer[blockIdx.x + 1] = data_to_left;
      block_to_right_offset_buffer[blockIdx.x + 1] = num_data_in_block - data_to_left;
    } else {
      block_to_left_offset_buffer[blockIdx.x + 1] = 0;
      block_to_right_offset_buffer[blockIdx.x + 1] = 0;
    }
  }
}

__global__ void AggregateBlockOffsetKernel(const int* leaf_index, data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  int* cuda_cur_num_leaves,
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
  hist_t* cuda_hist, hist_t** cuda_hist_pool, const int split_indices_block_size_data_partition) {
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
    const int cur_max_leaf_index = (*cuda_cur_num_leaves);
    block_to_left_offset_buffer[0] = 0;
    const unsigned int to_left_total_cnt = block_to_left_offset_buffer[num_blocks];
    block_to_right_offset_buffer[0] = to_left_total_cnt;
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[leaf_index_ref];
    cuda_leaf_data_end[leaf_index_ref] = cuda_leaf_data_start[leaf_index_ref] + static_cast<data_size_t>(to_left_total_cnt);
    cuda_leaf_num_data[leaf_index_ref] = static_cast<data_size_t>(to_left_total_cnt);
    cuda_leaf_data_start[cur_max_leaf_index] = cuda_leaf_data_end[leaf_index_ref];
    cuda_leaf_data_end[cur_max_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[cur_max_leaf_index] = block_to_right_offset_buffer[num_blocks] - to_left_total_cnt;
    const int cuda_num_total_bin_ref = *cuda_num_total_bin;
    ++(*cuda_cur_num_leaves);
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
    }
  }
}

__global__ void SplitInnerKernel(const int* leaf_index, const int* cuda_cur_num_leaves,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices, const uint8_t* split_to_left_bit_vector,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  data_size_t* out_data_indices_in_leaf, const int split_indices_block_size_data_partition) {
  __shared__ uint8_t thread_split_to_left_bit_vector[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  __shared__ uint32_t thread_to_left_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
  __shared__ uint32_t thread_to_right_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 1 +
    (SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION + 2) / NUM_BANKS_DATA_PARTITION];
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
    thread_to_right_pos[conflict_free_threadIdx_x_plus_1] = 1 - bit;
  } else {
    thread_split_to_left_bit_vector[threadIdx_x] = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_1] = 0;
    thread_to_right_pos[conflict_free_threadIdx_x_plus_1] = 0;
  }
  const unsigned int conflict_free_threadIdx_x_plus_blockDim_x_plus_1 = CONFLICT_FREE_INDEX(threadIdx_x + blockDim_x + 1);
  const unsigned int global_thread_index_plus_blockDim_x = global_thread_index + blockDim_x;
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    const uint8_t bit = split_to_left_bit_vector[global_thread_index_plus_blockDim_x];
    thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] = bit;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = bit;
    thread_to_right_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = 1 - bit;
  } else {
    thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] = 0;
    thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = 0;
    thread_to_right_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] = 0;
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
    const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
    thread_to_left_pos[0] = to_left_block_offset;
    thread_to_right_pos[0] = to_right_block_offset;
  }
  __syncthreads();
  PrefixSum(thread_to_left_pos, split_indices_block_size_data_partition);
  PrefixSum(thread_to_right_pos, split_indices_block_size_data_partition);
  __syncthreads();
  if (global_thread_index < num_data_in_leaf_ref) {
    if (thread_split_to_left_bit_vector[threadIdx_x] == 1) {
      out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      out_data_indices_in_leaf[thread_to_right_pos[conflict_free_threadIdx_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
  if (global_thread_index_plus_blockDim_x < num_data_in_leaf_ref) {
    if (thread_split_to_left_bit_vector[threadIdx_x + blockDim_x] == 1) {
      out_data_indices_in_leaf[thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    } else {
      out_data_indices_in_leaf[thread_to_right_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1]] = cuda_data_indices_in_leaf[global_thread_index_plus_blockDim_x];
    }
  }
  /*if (thread_to_left_pos[conflict_free_threadIdx_x_plus_1] == 0) {
    printf("thread_to_left_pos[%d] = %d, global_thread_index = %d, thread_split_to_left_bit_vector[%d] = %d\n",
    conflict_free_threadIdx_x_plus_1, thread_to_left_pos[conflict_free_threadIdx_x_plus_1], global_thread_index, threadIdx_x, thread_split_to_left_bit_vector[threadIdx_x]);
  }
  if (thread_to_right_pos[conflict_free_threadIdx_x_plus_1] == 0) {
    printf("thread_to_right_pos[%d] = %d, global_thread_index = %d, thread_split_to_left_bit_vector[%d] = %d\n",
    conflict_free_threadIdx_x_plus_1, thread_to_left_pos[conflict_free_threadIdx_x_plus_1], global_thread_index, threadIdx_x, thread_split_to_left_bit_vector[threadIdx_x]);
  }
  if (thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] == 0) {
    printf("thread_to_left_pos[%d] = %d, global_thread_index = %d, thread_split_to_left_bit_vector[%d + %ds] = %d\n",
    conflict_free_threadIdx_x_plus_blockDim_x_plus_1, thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1], global_thread_index_plus_blockDim_x, threadIdx_x, blockDim_x, thread_split_to_left_bit_vector[threadIdx_x + blockDim_x]);
  }
  if (thread_to_right_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1] == 0) {
    printf("thread_to_right_pos[%d] = %d, global_thread_index = %d, thread_split_to_left_bit_vector[%d + %d] = %d\n",
    conflict_free_threadIdx_x_plus_blockDim_x_plus_1, thread_to_left_pos[conflict_free_threadIdx_x_plus_blockDim_x_plus_1], global_thread_index_plus_blockDim_x, threadIdx_x, blockDim_x, thread_split_to_left_bit_vector[threadIdx_x + blockDim_x]);
  }*/
}

/*__global__ void SplitInnerKernel(const int* leaf_index, const int* cuda_cur_num_leaves,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices, const uint8_t* split_to_left_bit_vector,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  data_size_t* out_data_indices_in_leaf) {
  __shared__ uint8_t thread_split_to_left_bit_vector[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  __shared__ uint32_t thread_to_left_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  __shared__ uint32_t thread_to_right_pos[SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION];
  const int leaf_index_ref = *leaf_index;
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[leaf_index_ref];
  const data_size_t num_data_in_leaf_ref = cuda_leaf_num_data[leaf_index_ref] + cuda_leaf_num_data[(*cuda_cur_num_leaves) - 1];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  if (global_thread_index < num_data_in_leaf_ref) {
    thread_split_to_left_bit_vector[threadIdx_x] = split_to_left_bit_vector[global_thread_index];
  } else {
    thread_split_to_left_bit_vector[threadIdx_x] = 0;
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
    const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
    thread_to_left_pos[0] = to_left_block_offset;
    thread_to_right_pos[0] = to_right_block_offset;
    for (unsigned int i = 0; i < blockDim.x - 1; ++i) {
      const unsigned int tmp_global_thread_index = blockIdx.x * blockDim.x + i;
      if (tmp_global_thread_index < num_data_in_leaf_ref) {
        if (thread_split_to_left_bit_vector[i] == 0) {
          thread_to_right_pos[i + 1] = thread_to_right_pos[i] + 1;
          thread_to_left_pos[i + 1] = thread_to_left_pos[i];
        } else {
          thread_to_left_pos[i + 1] = thread_to_left_pos[i] + 1;
          thread_to_right_pos[i + 1] = thread_to_right_pos[i];
        }
      } else {
        thread_to_left_pos[i + 1] = thread_to_left_pos[i];
        thread_to_right_pos[i + 1] = thread_to_right_pos[i];
      }
    }
  }
  __syncthreads();
  if (global_thread_index < num_data_in_leaf_ref) {
    if (thread_split_to_left_bit_vector[threadIdx_x] == 1) {
      out_data_indices_in_leaf[thread_to_left_pos[threadIdx_x]] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      out_data_indices_in_leaf[thread_to_right_pos[threadIdx_x]] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
}*/

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
  hist_t** larger_leaf_cuda_hist_pointer_pointer, const int cpu_leaf_index) {
  //Log::Warning("num_data_in_leaf = %d", num_data_in_leaf);
  const int num_blocks = std::max(80, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  //Log::Warning("num_blocks = %d, split_indices_block_size_data_partition_aligned = %d", num_blocks, split_indices_block_size_data_partition_aligned);
  auto start = std::chrono::steady_clock::now();
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  //Log::Warning("num_blocks_final = %d", num_blocks_final);
  PrepareOffsetKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned / 2>>>(
    leaf_index, cuda_leaf_num_data_, cuda_data_to_left_,
    cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_, split_indices_block_size_data_partition_aligned);
  SynchronizeCUDADevice();
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("CUDADataPartition::PrepareOffsetKernel time %f", duration);
  start = std::chrono::steady_clock::now();
  AggregateBlockOffsetKernel<<<1, split_indices_block_size_data_partition_aligned / 2>>>(leaf_index, cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
    cuda_leaf_num_data_, cuda_data_indices_,
    cuda_cur_num_leaves_,

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
    cuda_hist_pool_, split_indices_block_size_data_partition_aligned);
  const auto copy_start = std::chrono::steady_clock::now();
  CopyFromCUDADeviceToHost<data_size_t>(num_data_in_leaf_.data(), cuda_leaf_num_data_, num_leaves_);
  SynchronizeCUDADevice();
  const auto copy_end = std::chrono::steady_clock::now();
  const auto copy_duration = (static_cast<std::chrono::duration<double>>(copy_end - copy_start)).count();
  //Log::Warning("CUDADataPartition::CopyFromCUDADeviceToHost time %f", copy_duration);
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("CUDADataPartition::AggregateBlockOffsetKernel time %f", duration);
  start = std::chrono::steady_clock::now();
  SplitInnerKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned / 2>>>(
    leaf_index, cuda_cur_num_leaves_, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_data_indices_, cuda_data_to_left_,
    cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_,
    cuda_out_data_indices_in_leaf_, split_indices_block_size_data_partition_aligned);
  SynchronizeCUDADevice();
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("CUDADataPartition::SplitInnerKernel time %f", duration);
  start = std::chrono::steady_clock::now();
  CopyDataIndicesKernel<<<num_blocks_final, split_indices_block_size_data_partition_aligned>>>(
    leaf_index, cuda_cur_num_leaves_, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_out_data_indices_in_leaf_, cuda_data_indices_);
  SynchronizeCUDADevice();
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  //Log::Warning("CUDADataPartition::CopyDataIndicesKernel time %f", duration);
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
  const data_size_t* leaf_data_start, const double learning_rate, double* output_score) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const data_size_t num_data = num_data_in_leaf[blockIdx_x];
  const data_size_t* data_indices = data_indices_in_leaf + leaf_data_start[blockIdx_x];
  const double leaf_prediction_value = data_partition_leaf_output[blockIdx_x] * learning_rate;
  /*if (threadIdx_x == 0) {
    printf("leaf index = %d, leaf_prediction_value = %f\n", blockIdx_x, leaf_prediction_value);
  }*/
  for (unsigned int offset = 0; offset < static_cast<unsigned int>(num_data); offset += blockDim_x) {
    const data_size_t inner_data_index = static_cast<data_size_t>(offset + threadIdx_x);
    if (inner_data_index < num_data) {
      const data_size_t data_index = data_indices[inner_data_index];
      output_score[data_index] = leaf_prediction_value;
    }
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double learning_rate) {
  AddPredictionToScoreKernel<<<cur_num_leaves_, 1024>>>(data_partition_leaf_output_,
    cuda_leaf_num_data_, cuda_data_indices_, cuda_leaf_data_start_, learning_rate, train_data_score_tmp_);
  SynchronizeCUDADevice();
  CopyFromCUDADeviceToHost<double>(cpu_train_data_score_tmp_.data(), train_data_score_tmp_, static_cast<size_t>(num_data_));
  SynchronizeCUDADevice();
}

}  // namespace LightGBM

#endif  // USE_CUDA
