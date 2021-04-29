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
  const int* leaf_num_data_offset, const uint8_t* data, const data_size_t* num_data_in_leaf,
  const uint32_t* feature_group_offsets_by_col_group,
  const uint32_t* feature_group_offsets,
  const score_t* cuda_gradients_and_hessians,
  const int8_t* cuda_int_gradients,
  const int8_t* cuda_int_hessians,
  const int32_t* cuda_int_gradients_and_hessians) {
  const unsigned int threadIdx_x = threadIdx.x;
  /*if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 0\n");
  }*/
  const int num_feature_groups_ref = *num_feature_groups;
  /*if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 1\n");
  }*/
  const int leaf_index_ref = *leaf_index;
  /*if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 2\n");
  }*/
  const int num_data_in_smaller_leaf_ref = *(num_data_in_leaf + leaf_index_ref);
  /*if (threadIdx_x == 0) {
    printf("CUDAConstructHistogramKernel step 3\n");
  }*/
  const int leaf_num_data_in_smaller_leaf_ref = *(leaf_num_data_offset + leaf_index_ref);
  /*printf("num_feature_groups_ref = %d\n", num_feature_groups_ref);
  printf("leaf_index_ref = %d\n", leaf_index_ref);
  printf("num_data_in_smaller_leaf_ref = %d\n", num_data_in_smaller_leaf_ref);
  printf("leaf_num_data_in_smaller_leaf_ref = %d\n", leaf_num_data_in_smaller_leaf_ref);*/
  const data_size_t* data_indices_in_smaller_leaf = data_indices_ptr + leaf_num_data_in_smaller_leaf_ref;
  __shared__ float shared_hist[SHRAE_HIST_SIZE]; // 256 * 24 * 2, can use 24 features
  //__shared__ int32_t shared_int_hist[SHRAE_HIST_SIZE];
  //uint32_t bin_offset = feature_group_offsets[blockIdx.x * 12];
  //const uint32_t next_feature_group_start = (blockIdx.x + 1) * 12;
  //const uint32_t next_col_group_first_feature = next_feature_group_start > 28 ? 28 : next_feature_group_start;
  uint32_t num_bins_in_col_group = feature_group_offsets[blockDim.x];
  const uint32_t num_items_per_thread = (2 * num_bins_in_col_group + NUM_THRADS_PER_BLOCK - 1) / NUM_THRADS_PER_BLOCK;
  const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const uint32_t thread_start = thread_idx * num_items_per_thread;
  const uint32_t thread_end = thread_start + num_items_per_thread > num_bins_in_col_group * 2 ?
    num_bins_in_col_group * 2 : thread_start + num_items_per_thread;
  for (uint32_t i = thread_start; i < thread_end; ++i) {
    shared_hist[i] = 0.0f;
    //shared_int_hist[i] = 0;
  }
  __syncthreads();

  /*if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0) {
    printf("num_data_in_leaf = %d\n", num_data_in_smaller_leaf_ref);
    printf("num_feature_groups_ref = %d", num_feature_groups_ref);
  }*/
  const unsigned int threadIdx_y = threadIdx.y;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockIdx_y = blockIdx.y;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int offset = threadIdx_x % 2;
  //if ((threadIdx_x < 24 && blockIdx_x < 2) || (threadIdx_x < 8 && blockIdx_x == 2)) {
    //const int feature_group_index = threadIdx_x / 2 + blockIdx_x * blockDim_x / 8 * 3;
    /*if (feature_group_index >= 28) {
      printf("error feature_group_index = %d\n", feature_group_index);
    }*/
    const data_size_t start = threadIdx_y * NUM_DATA_PER_THREAD + blockIdx_y * blockDim.y * NUM_DATA_PER_THREAD;
    const data_size_t end = start + NUM_DATA_PER_THREAD > num_data_in_smaller_leaf_ref ?
      num_data_in_smaller_leaf_ref : start + NUM_DATA_PER_THREAD;
    /*if (offset == 0) {
      // handle gradient
      for (data_size_t i = start; i < end; ++i) {
        const score_t gradient = cuda_gradients[i];
        const data_size_t data_index = data_indices_in_smaller_leaf[i];
        if (data_index != i) {
          printf("error data_index = %d vs i = %d", data_index, i);
        }
        const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + feature_group_index]) +
          feature_group_offsets_by_col_group[feature_group_index];
        //shared_hist[bin << 1] += gradient;
        atomicAdd_system(shared_hist + (bin << 1), gradient);
      }
    } else {
      // handle hessian
      for (data_size_t i = start; i < end; ++i) {
        const score_t hessian = cuda_hessians[i];
        const data_size_t data_index = data_indices_in_smaller_leaf[i];
        const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + feature_group_index]) +
          feature_group_offsets_by_col_group[feature_group_index];
        //shared_hist[(bin << 1) + 1] += hessian;
        atomicAdd_system(shared_hist + ((bin << 1) + 1), hessian);
      }
    }*/
    for (data_size_t i = start; i < end; ++i) {
      const score_t grad = cuda_gradients[i];
      const score_t hess = cuda_hessians[i];
      const data_size_t data_index = data_indices_in_smaller_leaf[i];
      const uint32_t bin = static_cast<uint32_t>(data[i * num_feature_groups_ref + threadIdx_x]) +
        feature_group_offsets[threadIdx_x];
      const uint32_t pos = bin << 1;
      float* pos_ptr = shared_hist + pos;
      atomicAdd_system(pos_ptr, grad);
      atomicAdd_system(pos_ptr + 1, hess);
      //const int32_t grad = static_cast<int32_t>(cuda_int_gradients[i]);
      //const int32_t hess = static_cast<int32_t>(cuda_int_hessians[i]);
      /*const int32_t grad_and_hess = cuda_int_gradients_and_hessians[i];
      const data_size_t data_index = data_indices_in_smaller_leaf[i];
      const uint32_t bin = static_cast<uint32_t>(data[i * num_feature_groups_ref + threadIdx_x]) +
        feature_group_offsets[threadIdx_x];
      //const uint32_t pos = bin << 1;
      int32_t* pos_ptr = shared_int_hist + bin;
      atomicAdd_system(pos_ptr, grad_and_hess);*/
      //atomicAdd_system(pos_ptr + 1, hess);
    }
  //}
  __syncthreads();
  /*uint32_t bin_offset = feature_group_offsets[blockIdx.x * 12];
  const uint32_t next_feature_group_start = (blockIdx.x + 1) * 12;
  const uint32_t next_col_group_first_feature = next_feature_group_start > 28 ? 28 : next_feature_group_start;
  uint32_t num_bins_in_col_group = feature_group_offsets[next_col_group_first_feature] - bin_offset;
  const uint32_t num_items_per_thread = (2 * num_bins_in_col_group + 1023) / 1024;
  const int thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
  const uint32_t thread_start = thread_idx * num_items_per_thread;
  const uint32_t thread_end = thread_start + num_items_per_thread > num_bins_in_col_group * 2 ?
    num_bins_in_col_group * 2 : thread_start + num_items_per_thread;*/
  for (uint32_t i = thread_start; i < thread_end; ++i) {
    //feature_histogram[i + bin_offset * 2] += shared_hist[thread_idx];
    atomicAdd_system(feature_histogram + i, shared_hist[i]);
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const int* smaller_leaf_index, const data_size_t* num_data_in_leaf, const data_size_t* leaf_num_data_offset,
  const data_size_t* data_indices_ptr, const score_t* cuda_gradients, const score_t* cuda_hessians,
  const score_t* cuda_gradients_and_hessians) {
  const int block_dim_x = 28;
  const int block_dim_y = NUM_THRADS_PER_BLOCK / block_dim_x;
  const int grid_dim_y = ((num_data_ + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + block_dim_y - 1) / block_dim_y;
  const int grid_dim_x = (static_cast<int>(num_feature_groups_ + NUM_FEATURE_PER_THREAD_GROUP - 1) / NUM_FEATURE_PER_THREAD_GROUP);
  Log::Warning("block_dim_x = %d, block_dim_y = %d", block_dim_x, block_dim_y);
  Log::Warning("gid_dim_x = %d, grid_dim_y = %d", grid_dim_x, grid_dim_y);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  CUDAConstructHistogramKernel<<<grid_dim, block_dim>>>(smaller_leaf_index, cuda_gradients, cuda_hessians,
    data_indices_ptr, cuda_hist_, cuda_num_feature_groups_, leaf_num_data_offset, cuda_data_, num_data_in_leaf,
    cuda_feature_group_bin_offsets_by_col_groups_,
    cuda_feature_group_bin_offsets_, cuda_gradients_and_hessians, cuda_int_gradients_, cuda_int_hessians_,
    cuda_int_gradients_and_hessians_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
