/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

#define SHRAE_HIST_SIZE (6144)
#define NUM_DATA_PER_THREAD (1600)
#define NUM_FEATURE_PER_THREAD_GROUP (12)

__global__ void CUDAConstructHistogramKernel(const int* leaf_index,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t* data_indices_ptr, hist_t* feature_histogram, const int* num_feature_groups,
  const int* leaf_num_data_offset, const uint8_t* data, const data_size_t* num_data_in_leaf) {
  const unsigned int threadIdx_x = threadIdx.x;
  if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 0\n");
  }
  const int num_feature_groups_ref = *num_feature_groups;
  if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 1\n");
  }
  const int leaf_index_ref = *leaf_index;
  if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 2\n");
  }
  const int num_data_in_smaller_leaf_ref = *(num_data_in_leaf + leaf_index_ref);
  if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 3\n");
  }
  const int leaf_num_data_in_smaller_leaf_ref = *(leaf_num_data_offset + leaf_index_ref);
  printf("num_feature_groups_ref = %d", num_feature_groups_ref);
  printf("leaf_index_ref = %d", leaf_index_ref);
  printf("num_data_in_smaller_leaf_ref = %d", num_data_in_smaller_leaf_ref);
  printf("leaf_num_data_in_smaller_leaf_ref = %d", leaf_num_data_in_smaller_leaf_ref);
  const data_size_t* data_indices_in_smaller_leaf = data_indices_ptr + leaf_num_data_in_smaller_leaf_ref;
  //__shared__ double shared_hist[SHRAE_HIST_SIZE]; // 256 * 24, can use 12 features
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int offset = threadIdx_x % 2;
  if (threadIdx_x < 24) {
    const int feature_group_index = threadIdx_x / 2 + blockIdx_x * blockDim_x / 8 * 3;
    const data_size_t start = threadIdx_y * NUM_DATA_PER_THREAD;
    const data_size_t end = start + NUM_DATA_PER_THREAD > num_data_in_smaller_leaf_ref ?
      num_data_in_smaller_leaf_ref : start + NUM_DATA_PER_THREAD;
    if (offset == 0) {
      // handle gradient
      for (data_size_t i = start; i < end; ++i) {
        const score_t gradient = cuda_gradients[i];
        const data_size_t data_index = data_indices_in_smaller_leaf[i];
        const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + feature_group_index]);
        feature_histogram[bin << 1] += gradient;
      }
    } else {
      // handle hessian
      for (data_size_t i = start; i < end; ++i) {
        const score_t hessian = cuda_hessians[i];
        const data_size_t data_index = data_indices_in_smaller_leaf[i];
        const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + feature_group_index]);
        feature_histogram[(bin << 1) + 1] += hessian;
      }
    }
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const int* smaller_leaf_index, const data_size_t* num_data_in_leaf, const data_size_t* leaf_num_data_offset,
  const data_size_t* data_indices_ptr, const score_t* cuda_gradients, const score_t* cuda_hessians) {
  const int block_dim_x = 32;
  const int block_dim_y = 1024 / block_dim_x;
  const int grid_dim_y = ((num_data_ + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + block_dim_y - 1) / block_dim_y;
  const int grid_dim_x = (static_cast<int>(num_feature_groups_ + NUM_FEATURE_PER_THREAD_GROUP - 1) / NUM_FEATURE_PER_THREAD_GROUP);
  Log::Warning("block_dim_x = %d, block_dim_y = %d", block_dim_x, block_dim_y);
  Log::Warning("gid_dim_x = %d, grid_dim_y = %d", grid_dim_x, grid_dim_y);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  CUDAConstructHistogramKernel<<<grid_dim, block_dim>>>(smaller_leaf_index, cuda_gradients, cuda_hessians,
    data_indices_ptr, cuda_hist_, cuda_num_feature_groups_, leaf_num_data_offset, cuda_data_, num_data_in_leaf);
}

}  // namespace LightGBM

#endif  // USE_CUDA
