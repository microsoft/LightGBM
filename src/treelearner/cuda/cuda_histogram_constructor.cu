/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_histogram_constructor.hpp"

namespace LightGBM {

__device__ void PrefixSum(hist_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  const unsigned int conflict_free_n_minus_1 = (n - 1);
  const hist_t last_element = elements[conflict_free_n_minus_1];
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      elements[(dst_pos)] += elements[(src_pos)];
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
      const unsigned int conflict_free_dst_pos = (dst_pos);
      const unsigned int conflict_free_src_pos = (src_pos);
      const hist_t src_val = elements[conflict_free_src_pos];
      elements[conflict_free_src_pos] = elements[conflict_free_dst_pos];
      elements[conflict_free_dst_pos] += src_val;
    }
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[(n)] = elements[conflict_free_n_minus_1] + last_element;
  }
}

__global__ void CUDAConstructHistogramKernel(const int* leaf_index,
  const score_t* cuda_gradients, const score_t* cuda_hessians,
  const data_size_t** data_indices_ptr, hist_t** feature_histogram, const int* num_feature_groups,
  const data_size_t* leaf_num_data, const uint8_t* data, const uint32_t* feature_group_offsets,
  const int* /*cuda_num_total_bin*/) {
  const unsigned int threadIdx_x = threadIdx.x;
  const int num_feature_groups_ref = *num_feature_groups;
  const int leaf_index_ref = *leaf_index;
  const int dim_y = gridDim.y * blockDim.y;
  //const int cuda_num_total_bin_ref = *cuda_num_total_bin;
  hist_t* feature_histogram_ptr = *feature_histogram;
  /*if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
    printf("construct histogram for leaf %d\n", leaf_index_ref);
  }*/
  const data_size_t num_data_in_smaller_leaf_ref = leaf_num_data[leaf_index_ref];
  const data_size_t num_data_per_thread = (num_data_in_smaller_leaf_ref + dim_y - 1) / dim_y;
  const data_size_t* data_indices_ref = *data_indices_ptr;
  __shared__ float shared_hist[SHRAE_HIST_SIZE]; // 256 * 24 * 2, can use 24 features
  uint32_t num_bins_in_col_group = feature_group_offsets[blockDim.x];
  /*if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
    printf("cuda_num_total_bin_ref = %d\n", cuda_num_total_bin_ref);
    printf("num_bins_in_col_group %d\n", num_bins_in_col_group);
  }*/
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
  const data_size_t start = (threadIdx_y + blockIdx_y * blockDim.y) * num_data_per_thread;
  const data_size_t end = start + num_data_per_thread > num_data_in_smaller_leaf_ref ?
    num_data_in_smaller_leaf_ref : start + num_data_per_thread;
  /*if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
    if (leaf_index_ref == 2) {
      for (data_size_t i = 0; i < 10; ++i) {
        printf("leaf 2 data index %d = %d\n", i, data_indices_ref[i]);
      }
      printf("===========================================\n");
      for (data_size_t i = 1030726 - 10; i < 1030726; ++i) {
        printf("leaf 2 data index %d = %d\n", i, data_indices_ref[i]);
      }
    }
  }*/
  for (data_size_t i = start; i < end; ++i) {
    const data_size_t data_index = data_indices_ref[i];
    const score_t grad = cuda_gradients[data_index];
    const score_t hess = cuda_hessians[data_index];
    const uint32_t bin = static_cast<uint32_t>(data[data_index * num_feature_groups_ref + threadIdx_x]) +
      feature_group_offsets[threadIdx_x];
    const uint32_t pos = bin << 1;
    float* pos_ptr = shared_hist + pos;
    atomicAdd_system(pos_ptr, grad);
    atomicAdd_system(pos_ptr + 1, hess);
  }
  __syncthreads();
  for (uint32_t i = thread_start; i < thread_end; ++i) {
    atomicAdd_system(feature_histogram_ptr + i, shared_hist[i]);
  }
}

void CUDAHistogramConstructor::LaunchConstructHistogramKernel(
  const int* cuda_smaller_leaf_index,
  const data_size_t** cuda_data_indices_in_smaller_leaf,
  const data_size_t* cuda_leaf_num_data,
  hist_t** cuda_leaf_hist) {
  int smaller_leaf_index = 0;
  CopyFromCUDADeviceToHost<int>(&smaller_leaf_index, cuda_smaller_leaf_index, 1);
  SynchronizeCUDADevice();
  data_size_t smaller_leaf_num_data = 0;
  CopyFromCUDADeviceToHost<data_size_t>(&smaller_leaf_num_data, cuda_leaf_num_data + smaller_leaf_index, 1);
  SynchronizeCUDADevice();
  const int block_dim_x = num_features_; // TODO(shiyu1994): only supports the case when the whole histogram can be loaded into shared memory
  const int block_dim_y = NUM_THRADS_PER_BLOCK / block_dim_x;
  const int min_grid_dim_y = 80;
  const int grid_dim_y = std::max(min_grid_dim_y, ((smaller_leaf_num_data + NUM_DATA_PER_THREAD - 1) / NUM_DATA_PER_THREAD + block_dim_y - 1) / block_dim_y);
  const int grid_dim_x = (static_cast<int>(num_feature_groups_ + NUM_FEATURE_PER_THREAD_GROUP - 1) / NUM_FEATURE_PER_THREAD_GROUP);
  //Log::Warning("smaller_leaf_num_data = %d", smaller_leaf_num_data);
  //Log::Warning("block_dim_x = %d, block_dim_y = %d", block_dim_x, block_dim_y);
  //Log::Warning("gid_dim_x = %d, grid_dim_y = %d", grid_dim_x, grid_dim_y);
  dim3 grid_dim(grid_dim_x, grid_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  CUDAConstructHistogramKernel<<<grid_dim, block_dim>>>(cuda_smaller_leaf_index, cuda_gradients_, cuda_hessians_,
    cuda_data_indices_in_smaller_leaf, cuda_leaf_hist, cuda_num_feature_groups_, cuda_leaf_num_data, cuda_data_,
    cuda_feature_group_bin_offsets_, cuda_num_total_bin_);
}

__global__ void SubtractHistogramKernel(const int* /*cuda_smaller_leaf_index*/,
  const int* cuda_larger_leaf_index, const uint8_t* cuda_feature_mfb_offsets,
  const uint32_t* cuda_feature_num_bins, const int* cuda_num_total_bin,
  hist_t** cuda_smaller_leaf_hist, hist_t** cuda_larger_leaf_hist) {
  const int cuda_num_total_bin_ref = *cuda_num_total_bin;
  const unsigned int global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
  //const int cuda_smaller_leaf_index_ref = *cuda_smaller_leaf_index;
  const int cuda_larger_leaf_index_ref = *cuda_larger_leaf_index;
  if (cuda_larger_leaf_index_ref >= 0) { 
    const hist_t* smaller_leaf_hist = *cuda_smaller_leaf_hist; //cuda_hist + (cuda_smaller_leaf_index_ref * cuda_num_total_bin_ref * 2);
    hist_t* larger_leaf_hist = *cuda_larger_leaf_hist; //cuda_hist + (cuda_larger_leaf_index_ref * cuda_num_total_bin_ref * 2);
    if (global_thread_index < 2 * cuda_num_total_bin_ref) {
      larger_leaf_hist[global_thread_index] -= smaller_leaf_hist[global_thread_index];
    }
  }
}

__global__ void FixHistogramKernel(const int* cuda_smaller_leaf_index,
  const int* cuda_larger_leaf_index,
  const uint32_t* cuda_feature_num_bins, const int* cuda_num_features,
  const int* /*cuda_num_total_bin*/, const uint32_t* cuda_feature_hist_offsets,
  const uint32_t* cuda_feature_most_freq_bins,
  const double* smaller_leaf_sum_gradients, const double* smaller_leaf_sum_hessians,
  const double* larger_leaf_sum_gradients, const double* larger_leaf_sum_hessians,
  hist_t** cuda_smaller_leaf_hist, hist_t** cuda_larger_leaf_hist) {
  const int cuda_num_features_ref = *cuda_num_features;
  const unsigned int blockIdx_x = blockIdx.x;
  const int feature_index = blockIdx_x % cuda_num_features_ref;
  const bool larger_or_smaller = static_cast<bool>(blockIdx_x / cuda_num_features_ref);
  const int leaf_index_ref = larger_or_smaller ? *cuda_larger_leaf_index : *cuda_smaller_leaf_index;
  __shared__ double hist_gradients[FIX_HISTOGRAM_SHARED_MEM_SIZE + 1];
  __shared__ double hist_hessians[FIX_HISTOGRAM_SHARED_MEM_SIZE + 1];
  if (leaf_index_ref >= 0) {
    //const int cuda_num_total_bin_ref = *cuda_num_total_bin;
    const uint32_t feature_hist_offset = cuda_feature_hist_offsets[feature_index];
    const uint32_t most_freq_bin = cuda_feature_most_freq_bins[feature_index];
    if (most_freq_bin > 0) {
      const double leaf_sum_gradients = larger_or_smaller ? *larger_leaf_sum_gradients : *smaller_leaf_sum_gradients;
      const double leaf_sum_hessians = larger_or_smaller ? *larger_leaf_sum_hessians : *smaller_leaf_sum_hessians;
      hist_t* feature_hist = larger_or_smaller ? (*cuda_larger_leaf_hist) + feature_hist_offset * 2 :
        (*cuda_smaller_leaf_hist) + feature_hist_offset * 2;
      //cuda_hist + cuda_num_total_bin_ref * 2 * leaf_index_ref + feature_hist_offset * 2;
      const unsigned int threadIdx_x = threadIdx.x;
      const uint32_t num_bin = cuda_feature_num_bins[feature_index];
      if (threadIdx_x < num_bin) {
        if (threadIdx_x == most_freq_bin) {
          hist_gradients[threadIdx_x] = 0.0f;
          hist_hessians[threadIdx_x] = 0.0f;
        } else {
          hist_gradients[threadIdx_x] = feature_hist[threadIdx_x << 1];
          hist_hessians[threadIdx_x] = feature_hist[(threadIdx_x << 1) + 1];
        }
      } else {
        hist_gradients[threadIdx_x] = 0.0f;
        hist_hessians[threadIdx_x] = 0.0f;
      }
      uint32_t num_bin_aligned = 1;
      uint32_t num_bin_to_shift = num_bin - 1;
      while (num_bin_to_shift > 0) {
        num_bin_to_shift >>= 1;
        num_bin_aligned <<= 1;
      }
      /*if (threadIdx.x == 0) {
        printf("num_bin_aligned = %d\n", num_bin_aligned);
      }*/
      __syncthreads();
      PrefixSum(hist_gradients, num_bin_aligned);
      PrefixSum(hist_hessians, num_bin_aligned);
      __syncthreads();
      if (threadIdx_x == most_freq_bin) {
        feature_hist[most_freq_bin << 1] = leaf_sum_gradients - hist_gradients[num_bin_aligned];
        feature_hist[(most_freq_bin << 1) + 1] = leaf_sum_hessians - hist_hessians[num_bin_aligned];
      }
      if (threadIdx.x == 0) {
        //printf("fix most freq bin: feature_hist_offset %d + most_freq_bin %d = %d, num_bin_aligned = %d, leaf_sum_gradients = %f, leaf_sum_hessians = %f, hist_gradients[num_bin_aligned] = %f, hist_hessians[num_bin_aligned] = %f, feature_hist[most_freq_bin << 1] = %f, feature_hist[(most_freq_bin << 1) + 1] = %f\n",
        //  feature_hist_offset, most_freq_bin, feature_hist_offset + most_freq_bin, num_bin_aligned, leaf_sum_gradients, leaf_sum_hessians, hist_gradients[num_bin_aligned], hist_hessians[num_bin_aligned], feature_hist[most_freq_bin << 1], feature_hist[(most_freq_bin << 1) + 1]);
      }
    }
  }
}

void CUDAHistogramConstructor::LaunchSubtractHistogramKernel(const int* cuda_smaller_leaf_index,
  const int* cuda_larger_leaf_index, const double* smaller_leaf_sum_gradients, const double* smaller_leaf_sum_hessians,
  const double* larger_leaf_sum_gradients, const double* larger_leaf_sum_hessians,
  hist_t** cuda_smaller_leaf_hist, hist_t** cuda_larger_leaf_hist) {
  const int num_subtract_threads = 2 * num_total_bin_;
  const int num_subtract_blocks = (num_subtract_threads + SUBTRACT_BLOCK_SIZE - 1) / SUBTRACT_BLOCK_SIZE;
  //Log::Warning("Before SubtractHistogramKernel");
  SubtractHistogramKernel<<<num_subtract_blocks, SUBTRACT_BLOCK_SIZE>>>(
    cuda_smaller_leaf_index, cuda_larger_leaf_index, cuda_feature_mfb_offsets_,
    cuda_feature_num_bins_, cuda_num_total_bin_, cuda_smaller_leaf_hist, cuda_larger_leaf_hist);
  SynchronizeCUDADevice();
  //Log::Warning("After SubtractHistogramKernel");
  FixHistogramKernel<<<2 * num_features_, FIX_HISTOGRAM_BLOCK_SIZE>>>(
    cuda_smaller_leaf_index, cuda_larger_leaf_index,
    cuda_feature_num_bins_, cuda_num_features_,
    cuda_num_total_bin_, cuda_feature_hist_offsets_,
    cuda_feature_most_freq_bins_, smaller_leaf_sum_gradients, smaller_leaf_sum_hessians,
    larger_leaf_sum_gradients, larger_leaf_sum_hessians,
    cuda_smaller_leaf_hist, cuda_larger_leaf_hist);
  SynchronizeCUDADevice();
  //Log::Warning("After FixHistogramKernel");
}

}  // namespace LightGBM

#endif  // USE_CUDA
