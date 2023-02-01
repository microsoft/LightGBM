/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>

#include <algorithm>

namespace LightGBM {

template <typename BIN_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramDenseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    HIST_TYPE* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.y); inner_data_index < block_num_data; inner_data_index += blockDim.y) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE, typename HIST_TYPE, size_t SHARED_HIST_SIZE>
__global__ void CUDAConstructHistogramSparseKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  __shared__ HIST_TYPE shared_hist[SHARED_HIST_SIZE];
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      HIST_TYPE* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE>
__global__ void CUDAConstructHistogramDenseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const uint32_t* column_hist_offsets,
  const uint32_t* column_hist_offsets_full,
  const int* feature_partition_column_index_offsets,
  const data_size_t num_data,
  float* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const int partition_column_start = feature_partition_column_index_offsets[blockIdx.x];
  const int partition_column_end = feature_partition_column_index_offsets[blockIdx.x + 1];
  const BIN_TYPE* data_ptr = data + static_cast<size_t>(partition_column_start) * num_data;
  const int num_columns_in_partition = partition_column_end - partition_column_start;
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  float* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (static_cast<size_t>(blockIdx_y) * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  const int column_index = static_cast<int>(threadIdx.x) + partition_column_start;
  if (threadIdx.x < static_cast<unsigned int>(num_columns_in_partition)) {
    float* shared_hist_ptr = shared_hist + (column_hist_offsets[column_index] << 1);
    for (data_size_t i = 0; i < num_iteration_this; ++i) {
      const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[static_cast<size_t>(data_index) * num_columns_in_partition + threadIdx.x]);
      const uint32_t pos = bin << 1;
      float* pos_ptr = shared_hist_ptr + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
      inner_data_index += blockDim.y;
    }
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

template <typename BIN_TYPE, typename DATA_PTR_TYPE>
__global__ void CUDAConstructHistogramSparseKernel_GlobalMemory(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const score_t* cuda_gradients,
  const score_t* cuda_hessians,
  const BIN_TYPE* data,
  const DATA_PTR_TYPE* row_ptr,
  const DATA_PTR_TYPE* partition_ptr,
  const uint32_t* column_hist_offsets_full,
  const data_size_t num_data,
  float* global_hist_buffer) {
  const int dim_y = static_cast<int>(gridDim.y * blockDim.y);
  const data_size_t num_data_in_smaller_leaf = smaller_leaf_splits->num_data_in_leaf;
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = smaller_leaf_splits->data_indices_in_leaf;
  const unsigned int num_threads_per_block = blockDim.x * blockDim.y;
  const DATA_PTR_TYPE* block_row_ptr = row_ptr + static_cast<size_t>(blockIdx.x) * (num_data + 1);
  const BIN_TYPE* data_ptr = data + partition_ptr[blockIdx.x];
  const uint32_t partition_hist_start = column_hist_offsets_full[blockIdx.x];
  const uint32_t partition_hist_end = column_hist_offsets_full[blockIdx.x + 1];
  const uint32_t num_items_in_partition = (partition_hist_end - partition_hist_start) << 1;
  const unsigned int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const int num_total_bin = column_hist_offsets_full[gridDim.x];
  float* shared_hist = global_hist_buffer + (blockIdx.y * num_total_bin + partition_hist_start) * 2;
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t block_start = (blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t* data_indices_ref_this_block = data_indices_ref + block_start;
  data_size_t block_num_data = max(0, min(num_data_in_smaller_leaf - block_start, num_data_per_thread * static_cast<data_size_t>(blockDim.y)));
  const data_size_t num_iteration_total = (block_num_data + blockDim.y - 1) / blockDim.y;
  const data_size_t remainder = block_num_data % blockDim.y;
  const data_size_t num_iteration_this = remainder == 0 ? num_iteration_total : num_iteration_total - static_cast<data_size_t>(threadIdx_y >= remainder);
  data_size_t inner_data_index = static_cast<data_size_t>(threadIdx_y);
  for (data_size_t i = 0; i < num_iteration_this; ++i) {
    const data_size_t data_index = data_indices_ref_this_block[inner_data_index];
    const DATA_PTR_TYPE row_start = block_row_ptr[data_index];
    const DATA_PTR_TYPE row_end = block_row_ptr[data_index + 1];
    const DATA_PTR_TYPE row_size = row_end - row_start;
    if (threadIdx.x < row_size) {
      const score_t grad = cuda_gradients[data_index];
      const score_t hess = cuda_hessians[data_index];
      const uint32_t bin = static_cast<uint32_t>(data_ptr[row_start + threadIdx.x]);
      const uint32_t pos = bin << 1;
      float* pos_ptr = shared_hist + pos;
      atomicAdd_block(pos_ptr, grad);
      atomicAdd_block(pos_ptr + 1, hess);
    }
    inner_data_index += blockDim.y;
  }
  __syncthreads();
  hist_t* feature_histogram_ptr = smaller_leaf_splits->hist_in_leaf + (partition_hist_start << 1);
  for (unsigned int i = thread_idx; i < num_items_in_partition; i += num_threads_per_block) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  if (cuda_row_data_->shared_hist_size() == DP_SHARED_HIST_SIZE && gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<double, DP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else if (cuda_row_data_->shared_hist_size() == SP_SHARED_HIST_SIZE && !gpu_use_dp_) {
    LaunchConstructHistogramKernelInner<float, SP_SHARED_HIST_SIZE>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else {
    Log::Fatal("Unknown shared histogram size %d", cuda_row_data_->shared_hist_size());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  if (cuda_row_data_->bit_type() == 8) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint8_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else if (cuda_row_data_->bit_type() == 16) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else if (cuda_row_data_->bit_type() == 32) {
    LaunchConstructHistogramKernelInner0<HIST_TYPE, SHARED_HIST_SIZE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else {
    Log::Fatal("Unknown bit_type = %d", cuda_row_data_->bit_type());
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner0(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  if (cuda_row_data_->row_ptr_bit_type() == 16) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else if (cuda_row_data_->row_ptr_bit_type() == 32) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint32_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else if (cuda_row_data_->row_ptr_bit_type() == 64) {
    LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint64_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else {
    if (!cuda_row_data_->is_sparse()) {
      LaunchConstructHistogramKernelInner1<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, uint16_t>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
    } else {
      Log::Fatal("Unknown row_ptr_bit_type = %d", cuda_row_data_->row_ptr_bit_type());
    }
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner1(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  if (cuda_row_data_->NumLargeBinPartition() == 0) {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, false>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  } else {
    LaunchConstructHistogramKernelInner2<HIST_TYPE, SHARED_HIST_SIZE, BIN_TYPE, PTR_TYPE, true>(cuda_smaller_leaf_splits, num_data_in_smaller_leaf);
  }
}

template <typename HIST_TYPE, size_t SHARED_HIST_SIZE, typename BIN_TYPE, typename PTR_TYPE, bool USE_GLOBAL_MEM_BUFFER>
void CUDAHistogramConstructor::LaunchConstructHistogramKernelInner2(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const data_size_t num_data_in_smaller_leaf) {
  int grid_dim_x = 0;
  int grid_dim_y = 0;
  int block_dim_x = 0;
  int block_dim_y = 0;
  CalcConstructHistogramKernelDim(&grid_dim_x, &grid_dim_y, &block_dim_x, &block_dim_y, num_data_in_smaller_leaf);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  if (!USE_GLOBAL_MEM_BUFFER) {
    if (cuda_row_data_->is_sparse()) {
      CUDAConstructHistogramSparseKernel<BIN_TYPE, PTR_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        cuda_smaller_leaf_splits,
        cuda_gradients_, cuda_hessians_,
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->GetRowPtr<PTR_TYPE>(),
        cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        num_data_);
    } else {
      CUDAConstructHistogramDenseKernel<BIN_TYPE, HIST_TYPE, SHARED_HIST_SIZE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        cuda_smaller_leaf_splits,
        cuda_gradients_, cuda_hessians_,
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        num_data_);
    }
  } else {
    if (cuda_row_data_->is_sparse()) {
      CUDAConstructHistogramSparseKernel_GlobalMemory<BIN_TYPE, PTR_TYPE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        cuda_smaller_leaf_splits,
        cuda_gradients_, cuda_hessians_,
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->GetRowPtr<PTR_TYPE>(),
        cuda_row_data_->GetPartitionPtr<PTR_TYPE>(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        num_data_,
        cuda_hist_buffer_);
    } else {
      CUDAConstructHistogramDenseKernel_GlobalMemory<BIN_TYPE><<<grid_dim, block_dim, 0, cuda_stream_>>>(
        cuda_smaller_leaf_splits,
        cuda_gradients_, cuda_hessians_,
        cuda_row_data_->GetBin<BIN_TYPE>(),
        cuda_row_data_->cuda_column_hist_offsets(),
        cuda_row_data_->cuda_partition_hist_offsets(),
        cuda_row_data_->cuda_feature_partition_column_index_offsets(),
        num_data_,
        cuda_hist_buffer_);
    }
  }
}

__global__ void SubtractHistogramKernel(
  const int num_total_bin,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  const int cuda_larger_leaf_index = cuda_larger_leaf_splits->leaf_index;
  if (cuda_larger_leaf_index >= 0) {
    const hist_t* smaller_leaf_hist = cuda_smaller_leaf_splits->hist_in_leaf;
    hist_t* larger_leaf_hist = cuda_larger_leaf_splits->hist_in_leaf;
    if (global_thread_index < 2 * num_total_bin) {
      larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
    }
  }
}

__global__ void FixHistogramKernel(
  const uint32_t* cuda_feature_num_bins,
  const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const int* cuda_need_fix_histogram_features,
  const uint32_t* cuda_need_fix_histogram_features_num_bin_aligned,
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits) {
  __shared__ hist_t shared_mem_buffer[32];
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = cuda_need_fix_histogram_features[blockIdx_x];
  const uint32_t num_bin_aligned = cuda_need_fix_histogram_features_num_bin_aligned[blockIdx_x];
  const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
  const double leaf_sum_gradients = cuda_smaller_leaf_splits->sum_of_gradients;
  const double leaf_sum_hessians = cuda_smaller_leaf_splits->sum_of_hessians;
  hist_t* feature_hist = cuda_smaller_leaf_splits->hist_in_leaf + feature_hist_offset * 2;
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t num_bin = cuda_feature_num_bins[feature_index];
  const uint32_t hist_pos = threadIdx_x << 1;
  const hist_t bin_gradient = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos] : 0.0f;
  const hist_t bin_hessian = (threadIdx_x < num_bin && threadIdx_x != most_freq_bin) ? feature_hist[hist_pos + 1] : 0.0f;
  const hist_t sum_gradient = ShuffleReduceSum<hist_t>(bin_gradient, shared_mem_buffer, num_bin_aligned);
  const hist_t sum_hessian = ShuffleReduceSum<hist_t>(bin_hessian, shared_mem_buffer, num_bin_aligned);
  if (threadIdx_x == 0) {
    feature_hist[most_freq_bin << 1] = leaf_sum_gradients - sum_gradient;
    feature_hist[(most_freq_bin << 1) + 1] = leaf_sum_hessians - sum_hessian;
  }
}

void CUDAHistogramConstructor::LaunchSubtractHistogramKernel(
  const CUDALeafSplitsStruct* cuda_smaller_leaf_splits,
  const CUDALeafSplitsStruct* cuda_larger_leaf_splits) {
  const int num_subtract_threads = 2 * num_total_bin_;
  const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
  global_timer.Start("CUDAHistogramConstructor::FixHistogramKernel");
  if (need_fix_histogram_features_.size() > 0) {
    FixHistogramKernel<<<need_fix_histogram_features_.size(), FIX_HISTOGRAM_BLOCK_SIZE, 0, cuda_stream_>>>(
      cuda_feature_num_bins_,
      cuda_feature_hist_offsets_,
      cuda_feature_most_freq_bins_,
      cuda_need_fix_histogram_features_,
      cuda_need_fix_histogram_features_num_bin_aligned_,
      cuda_smaller_leaf_splits);
  }
  global_timer.Stop("CUDAHistogramConstructor::FixHistogramKernel");
  global_timer.Start("CUDAHistogramConstructor::SubtractHistogramKernel");
  SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE, 0, cuda_stream_>>>(
    num_total_bin_,
    cuda_smaller_leaf_splits,
    cuda_larger_leaf_splits);
  global_timer.Stop("CUDAHistogramConstructor::SubtractHistogramKernel");
}

}  // namespace LightGBM

#endif  // USE_CUDA
