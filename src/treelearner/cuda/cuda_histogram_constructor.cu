/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

__global__ void CUDAConstructHistogramKernel(const int* leaf_index,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* data_indices_ptr, hist_t* feature_histogram, const int* num_feature_groups,
  const data_size_t* leaf_num_data, const uint8_t* data, const uint32_t* feature_group_offsets) {
  const unsigned int threadIdx_x = threadIdx.x;
  const int num_feature_groups_ref = *num_feature_groups;
  const int leaf_index_ref = *leaf_index;
  const data_size_t num_data_in_smaller_leaf_ref = leaf_num_data[leaf_index_ref];
  __shared__ float shared_hist[SHRAE_HIST_SIZE]; // 256 * 24 * 2, can use 24 features
  uint32_t num_bins_in_col_group = feature_group_offsets[blockDim.x];
  const uint32_t num_items_per_thread = (2 * num_bins_in_col_group + NUM_THRADS_PER_BLOCK - 1) / NUM_THRADS_PER_BLOCK;
  const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const uint32_t thread_start = thread_idx * num_items_per_thread;
  const uint32_t thread_end = thread_start + num_items_per_thread > num_bins_in_col_group * 2 ?
    num_bins_in_col_group * 2 : thread_start + num_items_per_thread;
  for (uint32_t i = thread_start; i < thread_end; ++i) {
    shared_hist[i] = 0.0f;
  }
  __syncthreads();
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_y = blockIdx.y;
  const data_size_t start = (threadIdx_y + blockIdx_y * blockDim.y) * NUM_DATA_PER_THREAD;
  const data_size_t end = start + NUM_DATA_PER_THREAD > num_data_in_smaller_leaf_ref ?
    num_data_in_smaller_leaf_ref : start + NUM_DATA_PER_THREAD;
  for (data_size_t i = start; i < end; ++i) {
    const score_t grad = cuda_gradients[i];
    const score_t hess = cuda_hessians[i];
    const data_size_t data_index = data_indices_ptr[i];
    const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + threadIdx_x]) +
      feature_group_offsets[threadIdx_x];
    const uint32_t pos = bin << 1;
    float* pos_ptr = shared_hist + pos;
    atomicAdd_system(pos_ptr, grad);
    atomicAdd_system(pos_ptr + 1, hess);
  }
  __syncthreads();
  for (uint32_t i = thread_start; i < thread_end; ++i) {
    atomicAdd_system(feature_histogram + i, shared_hist[i]);
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const int* cuda_smaller_leaf_index,
  const data_size_t* cuda_data_indices_in_smaller_leaf,
  const data_size_t* cuda_leaf_num_data) {
  const int block_dim_x = num_features_; // TODO(shiyu1994): only supports the case when the whole histogram can be loaded into shared memory
  const int block_dim_y = NUM_THRADS_PER_BLOCK / block_dim_x;
  const int grid_dim_y = ((num_data_ + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + block_dim_y - 1) / block_dim_y;
  const int grid_dim_x = (static_cast<int>(num_feature_groups_ + NUM_FEATURE_PER_THREAD_GROUP - 1) / NUM_FEATURE_PER_THREAD_GROUP);
  Log::Warning("block_dim_x = %d, block_dim_y = %d", block_dim_x, block_dim_y);
  Log::Warning("gid_dim_x = %d, grid_dim_y = %d", grid_dim_x, grid_dim_y);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  CUDAConstructHistogramKernel<<<grid_dim, block_dim>>>(cuda_smaller_leaf_index, cuda_gradients_, cuda_hessians_,
    cuda_data_indices_in_smaller_leaf, cuda_hist_, cuda_num_feature_groups_, cuda_leaf_num_data, cuda_data_,
    cuda_feature_group_bin_offsets_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
