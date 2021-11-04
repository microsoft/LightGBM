/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include "cuda_best_split_finder.hpp"

namespace LightGBM {

__device__ void ReduceBestGainWarp(double gain, bool found, uint32_t thread_index, double* out_gain, bool* out_found, uint32_t* out_thread_index) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const bool other_found = __shfl_down_sync(mask, found, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    const uint32_t other_thread_index = __shfl_down_sync(mask, thread_index, offset);
    if ((other_found && found && other_gain > gain) || (!found && other_found)) {
      found = other_found;
      gain = other_gain;
      thread_index = other_thread_index;
    }
  }
  if (warpLane == 0) {
    *out_gain = gain;
    *out_found = found;
    *out_thread_index = thread_index;
  }
}

__device__ uint32_t ReduceBestGainBlock(double gain, bool found, uint32_t thread_index) {
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const bool other_found = __shfl_down_sync(mask, found, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    const uint32_t other_thread_index = __shfl_down_sync(mask, thread_index, offset);
    if ((other_found && found && other_gain > gain) || (!found && other_found)) {
      found = other_found;
      gain = other_gain;
      thread_index = other_thread_index;
    }
  }
  return thread_index;
}

__device__ uint32_t ReduceBestGain(double gain, bool found, uint32_t thread_index,
    double* shared_gain_buffer, bool* shared_found_buffer, uint32_t* shared_thread_index_buffer) {
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  ReduceBestGainWarp(gain, found, thread_index, shared_gain_buffer + warpID, shared_found_buffer + warpID, shared_thread_index_buffer + warpID);
  __syncthreads();
  if (warpID == 0) {
    gain = warpLane < num_warp ? shared_gain_buffer[warpLane] : kMinScore;
    found = warpLane < num_warp ? shared_found_buffer[warpLane] : false;
    thread_index = warpLane < num_warp ? shared_thread_index_buffer[warpLane] : 0;
    thread_index = ReduceBestGainBlock(gain, found, thread_index);
  }
  return thread_index;
}

__device__ void ReduceBestGainForLeaves(double* gain, int* leaves, int cuda_cur_num_leaves) {
  const unsigned int tid = threadIdx.x;
  for (unsigned int s = 1; s < cuda_cur_num_leaves; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < cuda_cur_num_leaves) {
      const uint32_t tid_s = tid + s;
      if ((leaves[tid] == -1 && leaves[tid_s] != -1) || (leaves[tid] != -1 && leaves[tid_s] != -1 && gain[tid_s] > gain[tid])) {
        gain[tid] = gain[tid_s];
        leaves[tid] = leaves[tid_s];
      }
    }
    __syncthreads();
  }
}

__device__ void ReduceBestGainForLeavesWarp(double gain, int leaf_index, double* out_gain, int* out_leaf_index) {
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_leaf_index = __shfl_down_sync(mask, leaf_index, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    if ((leaf_index != -1 && other_leaf_index != -1 && other_gain > gain) || (leaf_index == -1 && other_leaf_index != -1)) {
      gain = other_gain;
      leaf_index = other_leaf_index;
    }
  }
  if (warpLane == 0) {
    *out_gain = gain;
    *out_leaf_index = leaf_index;
  }
}

__device__ int ReduceBestGainForLeavesBlock(double gain, int leaf_index) {
  const uint32_t mask = 0xffffffff;
  for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_leaf_index = __shfl_down_sync(mask, leaf_index, offset);
    const double other_gain = __shfl_down_sync(mask, gain, offset);
    if ((leaf_index != -1 && other_leaf_index != -1 && other_gain > gain) || (leaf_index == -1 && other_leaf_index != -1)) {
      gain = other_gain;
      leaf_index = other_leaf_index;
    }
  }
  return leaf_index;
}

__device__ int ReduceBestGainForLeaves(double gain, int leaf_index, double* shared_gain_buffer, int* shared_leaf_index_buffer) {
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  ReduceBestGainForLeavesWarp(gain, leaf_index, shared_gain_buffer + warpID, shared_leaf_index_buffer + warpID);
  __syncthreads();
  if (warpID == 0) {
    gain = warpLane < num_warp ? shared_gain_buffer[warpLane] : kMinScore;
    leaf_index = warpLane < num_warp ? shared_leaf_index_buffer[warpLane] : -1;
    leaf_index = ReduceBestGainForLeavesBlock(gain, leaf_index);
  }
  return leaf_index;
}

__device__ double ThresholdL1(double s, double l1) {
  const double reg_s = fmax(0.0, fabs(s) - l1);
  if (s >= 0.0f) {
    return reg_s;
  } else {
    return -reg_s;
  }
}

__device__ double CUDABestSplitFinder::CalculateSplittedLeafOutput(double sum_gradients,
                                          double sum_hessians, double l1, const bool use_l1,
                                          double l2) {
  double ret;
  if (use_l1) {
    ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
  } else {
    ret = -sum_gradients / (sum_hessians + l2);
  }
  return ret;
}

__device__ double GetLeafGainGivenOutput(double sum_gradients,
                                      double sum_hessians, double l1, const bool use_l1,
                                      double l2, double output) {
  if (use_l1) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
  } else {
    return -(2.0 * sum_gradients * output +
              (sum_hessians + l2) * output * output);
  }
}

__device__ double GetLeafGain(double sum_gradients, double sum_hessians,
                          double l1, const bool use_l1, double l2) {
  if (use_l1) {
    const double sg_l1 = ThresholdL1(sum_gradients, l1);
    return (sg_l1 * sg_l1) / (sum_hessians + l2);
  } else {
    return (sum_gradients * sum_gradients) / (sum_hessians + l2);
  }
}

__device__ double GetSplitGains(double sum_left_gradients,
                            double sum_left_hessians,
                            double sum_right_gradients,
                            double sum_right_hessians,
                            double l1, const bool use_l1, double l2) {
  return GetLeafGain(sum_left_gradients,
                     sum_left_hessians,
                     l1, use_l1, l2) +
         GetLeafGain(sum_right_gradients,
                     sum_right_hessians,
                     l1, use_l1, l2);
}

__device__ void FindBestSplitsForLeafKernelInner(
  // input feature information
  const hist_t* feature_hist_ptr,
  const uint32_t feature_num_bin,
  const uint8_t feature_mfb_offset,
  const uint32_t feature_default_bin,
  const int inner_feature_index,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  // input task information
  const bool reverse,
  const bool skip_default_bin,
  const bool na_as_missing,
  const uint8_t assume_out_default_left,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  cuda_best_split_info->is_valid = false;

  __shared__ hist_t shared_mem_buffer[32];
  hist_t local_grad_hist = 0.0f;
  hist_t local_hess_hist = 0.0f;
  double local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  __shared__ uint32_t best_thread_index;
  __shared__ double shared_gain_buffer[32];
  __shared__ bool shared_found_buffer[32];
  __shared__ uint32_t shared_thread_index_buffer[32];
  const unsigned int threadIdx_x = threadIdx.x;
  const bool skip_sum = reverse ?
    (skip_default_bin && (feature_num_bin - 1 - threadIdx_x) == static_cast<int>(feature_default_bin)) :
    (skip_default_bin && (threadIdx_x + feature_mfb_offset) == static_cast<int>(feature_default_bin));
  const uint32_t feature_num_bin_minus_offset = feature_num_bin - feature_mfb_offset;
  if (!reverse) {
    if (threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int bin_offset = threadIdx_x << 1;
      local_grad_hist = feature_hist_ptr[bin_offset];
      local_hess_hist = feature_hist_ptr[bin_offset + 1];
    }
  } else {
    if (threadIdx_x >= static_cast<unsigned int>(na_as_missing) &&
      threadIdx_x < feature_num_bin_minus_offset && !skip_sum) {
      const unsigned int read_index = feature_num_bin_minus_offset - 1 - threadIdx_x;
      const unsigned int bin_offset = read_index << 1;
      local_grad_hist = feature_hist_ptr[bin_offset];
      local_hess_hist = feature_hist_ptr[bin_offset + 1];
    }
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    local_hess_hist += kEpsilon;
  }
  local_gain = kMinScore;
  local_grad_hist = ShufflePrefixSum(local_grad_hist, shared_mem_buffer);
  __syncthreads();
  local_hess_hist = ShufflePrefixSum(local_hess_hist, shared_mem_buffer);
  if (reverse) {
    if (threadIdx_x >= static_cast<unsigned int>(na_as_missing) && threadIdx_x <= feature_num_bin - 2 && !skip_sum) {
      const double sum_right_gradient = local_grad_hist;
      const double sum_right_hessian = local_hess_hist;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double sum_left_gradient = sum_gradients - sum_right_gradient;
      const double sum_left_hessian = sum_hessians - sum_right_hessian;
      const data_size_t left_count = num_data - right_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          lambda_l2);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(feature_num_bin - 2 - threadIdx_x);
          threshold_found = true;
        }
      }
    }
  } else {
    if (threadIdx_x <= feature_num_bin_minus_offset - 2 && !skip_sum) {
      const double sum_left_gradient = local_grad_hist;
      const double sum_left_hessian = local_hess_hist;
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          lambda_l2);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_value = static_cast<uint32_t>(threadIdx_x + feature_mfb_offset);
          threshold_found = true;
        }
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain;
    cuda_best_split_info->default_left = assume_out_default_left;
    if (reverse) {
      const double sum_right_gradient = local_grad_hist;
      const double sum_right_hessian = local_hess_hist - kEpsilon;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double sum_left_gradient = sum_gradients - sum_right_gradient;
      const double sum_left_hessian = sum_hessians - sum_right_hessian - kEpsilon;
      const data_size_t left_count = num_data - right_count;
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2, right_output);
    } else {
      const double sum_left_gradient = local_grad_hist;
      const double sum_left_hessian = local_hess_hist - kEpsilon;
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian - kEpsilon;
      const data_size_t right_count = num_data - left_count;
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2, right_output);
    }
  }
}

__device__ void FindBestSplitsForLeafKernelCategoricalInner(
  // input feature information
  const hist_t* feature_hist_ptr,
  const uint32_t feature_num_bin,
  const uint8_t feature_mfb_offset,
  const uint32_t feature_default_bin,
  const int inner_feature_index,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  // task information
  const bool is_one_hot,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  __shared__ double shared_gain_buffer[32];
  __shared__ bool shared_found_buffer[32];
  __shared__ uint32_t shared_thread_index_buffer[32];
  __shared__ uint32_t best_thread_index;
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;
  const double l2 = lambda_l2 + cat_l2;

  double local_gain = min_gain_shift;
  bool threshold_found = false;

  cuda_best_split_info->is_valid = false;

  const int bin_start = 1 - feature_mfb_offset;
  const int bin_end = feature_num_bin - feature_mfb_offset;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  if (is_one_hot) {
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const int bin_offset = (threadIdx_x << 1);
      const hist_t grad = feature_hist_ptr[bin_offset];
      const hist_t hess = feature_hist_ptr[bin_offset + 1];
      data_size_t cnt =
            static_cast<data_size_t>(__double2int_rn(hess * cnt_factor));
      if (cnt >= min_data_in_leaf && hess >= min_sum_hessian_in_leaf) {
        const data_size_t other_count = num_data - cnt;
        if (other_count >= min_data_in_leaf) {
          const double sum_other_hessian = sum_hessians - hess - kEpsilon;
          if (sum_other_hessian >= min_sum_hessian_in_leaf) {
            const double sum_other_gradient = sum_gradients - grad;
            double current_gain = GetSplitGains(
              sum_other_gradient, sum_other_hessian, grad,
              hess + kEpsilon, lambda_l1, use_l1,
              l2);
            if (current_gain > min_gain_shift) {
              local_gain = current_gain;
              threshold_found = true;
            }
          }
        }
      }
    }
    __syncthreads();
    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = 1;
      cuda_best_split_info->gain = local_gain - min_gain_shift;
      *(cuda_best_split_info->cat_threshold) = static_cast<uint32_t>(threadIdx_x + feature_mfb_offset);
      cuda_best_split_info->default_left = false;
      const int bin_offset = (threadIdx_x << 1);
      const hist_t sum_left_gradient = feature_hist_ptr[bin_offset];
      const hist_t sum_left_hessian = feature_hist_ptr[bin_offset + 1];
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2, right_output);
    }
  } else {
    __shared__ double shared_value_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    __shared__ int16_t shared_index_buffer[NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER];
    __shared__ uint16_t shared_mem_buffer_uint16[32];
    __shared__ double shared_mem_buffer_double[32];
    __shared__ int used_bin;
    uint16_t is_valid_bin = 0;
    int best_dir = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    if (threadIdx_x >= bin_start && threadIdx_x < bin_end) {
      const int bin_offset = (threadIdx_x << 1);
      const double hess = feature_hist_ptr[bin_offset + 1];
      if (__double2int_rn(hess * cnt_factor) >= cat_smooth) {
        const double grad = feature_hist_ptr[bin_offset];
        shared_value_buffer[threadIdx_x] = grad / (hess + cat_smooth);
        is_valid_bin = 1;
      } else {
        shared_value_buffer[threadIdx_x] = kMaxScore;
      }
    } else {
      shared_value_buffer[threadIdx_x] = kMaxScore;
    }
    shared_index_buffer[threadIdx_x] = threadIdx_x;
    __syncthreads();
    const int local_used_bin = ShuffleReduceSum<uint16_t>(is_valid_bin, shared_mem_buffer_uint16, blockDim.x);
    if (threadIdx_x == 0) {
      used_bin = local_used_bin;
    }
    __syncthreads();
    // TODO(shiyu1994): with more threads, this kernel may use out registers
    BitonicArgSort_1024<double, int16_t, true>(shared_value_buffer, shared_index_buffer, bin_end);
    __syncthreads();
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);

    // left to right
    double grad = 0.0f;
    double hess = 0.0f;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (shared_index_buffer[threadIdx_x] << 1);
      grad = feature_hist_ptr[bin_offset];
      hess = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    double sum_left_gradient = ShufflePrefixSum<double>(grad, shared_mem_buffer_double);
    __syncthreads();
    double sum_left_hessian = ShufflePrefixSum<double>(hess, shared_mem_buffer_double);
    // TODO(shiyu1994): constrain the split with min_data_in_group
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          l2);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    // right to left
    grad = 0.0f;
    hess = 0.0f;
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (shared_index_buffer[used_bin - 1 - threadIdx_x] << 1);
      grad = feature_hist_ptr[bin_offset];
      hess = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hess += kEpsilon;
    }
    __syncthreads();
    sum_left_gradient = ShufflePrefixSum<double>(grad, shared_mem_buffer_double);
    __syncthreads();
    sum_left_hessian = ShufflePrefixSum<double>(hess, shared_mem_buffer_double);
    // TODO(shiyu1994): constrain the split with min_data_in_group
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          l2);
        // gain with split is worse than without split
        if (current_gain > local_gain) {
          local_gain = current_gain;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = threadIdx_x + 1;
      cuda_best_split_info->gain = local_gain - min_gain_shift;
      if (best_dir == 1) {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[i] + feature_mfb_offset;
        }
      } else {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = shared_index_buffer[used_bin - 1 - i] + feature_mfb_offset;
        }
      }
      cuda_best_split_info->default_left = false;
      const hist_t sum_left_gradient = best_sum_left_gradient;
      const hist_t sum_left_hessian = best_sum_left_hessian;
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2, right_output);
    }
  }
}

__global__ void FindBestSplitsForLeafKernel(
  // input feature information
  const uint32_t* feature_hist_offsets,
  const uint8_t* feature_mfb_offsets,
  const uint32_t* feature_default_bins,
  const uint32_t* feature_num_bins,
  const int8_t* is_feature_used_bytree,
  const int8_t* is_categorical,
  // input task information
  const bool larger_only,
  const int num_tasks,
  const int* task_feature_index,
  const uint8_t* task_reverse,
  const uint8_t* task_skip_default_bin,
  const uint8_t* task_na_as_missing,
  const uint8_t* task_out_default_left,
  // input leaf information
  const int smaller_leaf_index,
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int larger_leaf_index,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  // input config parameter values
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int max_cat_to_onehot,
  // output
  CUDASplitInfo* cuda_best_split_info) {
  const unsigned int task_index = blockIdx.x % num_tasks;
  const bool is_larger = static_cast<bool>(blockIdx.x >= num_tasks || larger_only);
  const int inner_feature_index = task_feature_index[task_index];
  const double parent_gain = is_larger ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const double sum_gradients = is_larger ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (is_larger ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const double num_data = is_larger ? larger_leaf_splits->num_data_in_leaf : smaller_leaf_splits->num_data_in_leaf;
  const unsigned int output_offset = is_larger ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  if (is_feature_used_bytree[inner_feature_index]) {
    const hist_t* hist_ptr = (is_larger ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + feature_hist_offsets[inner_feature_index] * 2;
    if (is_categorical[inner_feature_index]) {
      const bool is_one_hot = feature_num_bins[inner_feature_index] <= max_cat_to_onehot;
      FindBestSplitsForLeafKernelCategoricalInner(
        // input feature information
        hist_ptr,
        feature_num_bins[inner_feature_index],
        feature_mfb_offsets[inner_feature_index],
        feature_default_bins[inner_feature_index],
        inner_feature_index,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        // input task information
        is_one_hot,
        // output parameters
        out);
    } else {
      const bool reverse = static_cast<bool>(task_reverse[task_index]);
      const bool skip_default_bin = static_cast<bool>(task_skip_default_bin[task_index]);
      const bool na_as_missing = static_cast<bool>(task_na_as_missing[task_index]);
      const bool assume_out_default_left = task_out_default_left[task_index];
      FindBestSplitsForLeafKernelInner(
        // input feature information
        hist_ptr,
        feature_num_bins[inner_feature_index],
        feature_mfb_offsets[inner_feature_index],
        feature_default_bins[inner_feature_index],
        inner_feature_index,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        // input task information
        reverse,
        skip_default_bin,
        na_as_missing,
        assume_out_default_left,
        // output parameters
        out);
    }
  } else {
    out->is_valid = false;
  }
}

__device__ void FindBestSplitsForLeafKernelInner_GlobalMemory(
  // input feature information
  const hist_t* feature_hist_ptr,
  const uint32_t feature_num_bin,
  const uint8_t feature_mfb_offset,
  const uint32_t feature_default_bin,
  const int inner_feature_index,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  // input task information
  const bool reverse,
  const bool skip_default_bin,
  const bool na_as_missing,
  const uint8_t assume_out_default_left,
  // buffer
  hist_t* hist_grad_buffer_ptr,
  hist_t* hist_hess_buffer_ptr,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  cuda_best_split_info->is_valid = false;
  double local_gain = 0.0f;
  bool threshold_found = false;
  uint32_t threshold_value = 0;
  __shared__ uint32_t best_thread_index;
  __shared__ double shared_gain_buffer[32];
  __shared__ bool shared_found_buffer[32];
  __shared__ uint32_t shared_thread_index_buffer[32];
  const unsigned int threadIdx_x = threadIdx.x;
  const uint32_t feature_num_bin_minus_offset = feature_num_bin - feature_mfb_offset;
  if (!reverse) {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; ++bin) {
      const bool skip_sum =
        (skip_default_bin && (bin + feature_mfb_offset) == static_cast<int>(feature_default_bin));
      if (!skip_sum) {
        const unsigned int bin_offset = bin << 1;
        hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
        hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
      } else {
        hist_grad_buffer_ptr[bin] = 0.0f;
        hist_hess_buffer_ptr[bin] = 0.0f;
      }
    }
  } else {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; ++bin) {
      const bool skip_sum = bin >= static_cast<unsigned int>(na_as_missing) && 
        (skip_default_bin && (feature_num_bin - 1 - bin) == static_cast<int>(feature_default_bin));
      if (!skip_sum) {
        const unsigned int read_index = feature_num_bin_minus_offset - 1 - bin;
        const unsigned int bin_offset = read_index << 1;
        hist_grad_buffer_ptr[bin] = feature_hist_ptr[bin_offset];
        hist_hess_buffer_ptr[bin] = feature_hist_ptr[bin_offset + 1];
      } else {
        hist_grad_buffer_ptr[bin] = 0.0f;
        hist_hess_buffer_ptr[bin] = 0.0f;
      }
    }
  }
  __syncthreads();
  if (threadIdx_x == 0) {
    hist_hess_buffer_ptr[0] += kEpsilon;
  }
  local_gain = kMinScore;
  GlobalMemoryPrefixSum(hist_grad_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  __syncthreads();
  GlobalMemoryPrefixSum(hist_hess_buffer_ptr, static_cast<size_t>(feature_num_bin_minus_offset));
  if (reverse) {
    for (unsigned int bin = threadIdx_x; bin < feature_num_bin_minus_offset; ++bin) {
      const bool skip_sum = (bin >= static_cast<unsigned int>(na_as_missing) &&
        (skip_default_bin && (feature_num_bin - 1 - bin) == static_cast<int>(feature_default_bin)));
      if (!skip_sum) {
        const double sum_right_gradient = hist_grad_buffer_ptr[bin];
        const double sum_right_hessian = hist_hess_buffer_ptr[bin];
        const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
        const double sum_left_gradient = sum_gradients - sum_right_gradient;
        const double sum_left_hessian = sum_hessians - sum_right_hessian;
        const data_size_t left_count = num_data - right_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
          double current_gain = GetSplitGains(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1, use_l1,
            lambda_l2);
          // gain with split is worse than without split
          if (current_gain > min_gain_shift) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = static_cast<uint32_t>(feature_num_bin - 2 - bin);
            threshold_found = true;
          }
        }
      }
    }
  } else {
    for (unsigned int bin = threadIdx_x; bin <= feature_num_bin_minus_offset - 2; ++bin) {
      const bool skip_sum =
        (skip_default_bin && (bin + feature_mfb_offset) == static_cast<int>(feature_default_bin));
      if (!skip_sum) {
        const double sum_left_gradient = hist_grad_buffer_ptr[bin];
        const double sum_left_hessian = hist_hess_buffer_ptr[bin];
        const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
        const double sum_right_gradient = sum_gradients - sum_left_gradient;
        const double sum_right_hessian = sum_hessians - sum_left_hessian;
        const data_size_t right_count = num_data - left_count;
        if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
          sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
          double current_gain = GetSplitGains(
            sum_left_gradient, sum_left_hessian, sum_right_gradient,
            sum_right_hessian, lambda_l1, use_l1,
            lambda_l2);
          // gain with split is worse than without split
          if (current_gain > min_gain_shift) {
            local_gain = current_gain - min_gain_shift;
            threshold_value = static_cast<uint32_t>(bin + feature_mfb_offset);
            threshold_found = true;
          }
        }
      }
    }
  }
  __syncthreads();
  const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx_x == 0) {
    best_thread_index = result;
  }
  __syncthreads();
  if (threshold_found && threadIdx_x == best_thread_index) {
    cuda_best_split_info->is_valid = true;
    cuda_best_split_info->threshold = threshold_value;
    cuda_best_split_info->gain = local_gain;
    cuda_best_split_info->default_left = assume_out_default_left;
    if (reverse) {
      const unsigned int best_bin = static_cast<uint32_t>(feature_num_bin - 2 - threshold_value);
      const double sum_right_gradient = hist_grad_buffer_ptr[best_bin];
      const double sum_right_hessian = hist_hess_buffer_ptr[best_bin] - kEpsilon;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double sum_left_gradient = sum_gradients - sum_right_gradient;
      const double sum_left_hessian = sum_hessians - sum_right_hessian - kEpsilon;
      const data_size_t left_count = num_data - right_count;
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2, right_output);
    } else {
      const unsigned int best_bin = static_cast<uint32_t>(threshold_value - feature_mfb_offset);
      const double sum_left_gradient = hist_grad_buffer_ptr[best_bin];
      const double sum_left_hessian = hist_hess_buffer_ptr[best_bin] - kEpsilon;
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian - kEpsilon;
      const data_size_t right_count = num_data - left_count;
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, lambda_l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, lambda_l2, right_output);
    }
  }
}

__device__ void FindBestSplitsForLeafKernelCategoricalInner_GlobalMemory(
  // input feature information
  const hist_t* feature_hist_ptr,
  const uint32_t feature_num_bin,
  const uint8_t feature_mfb_offset,
  const uint32_t feature_default_bin,
  const int inner_feature_index,
  // input config parameter values
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  // input parent node information
  const double parent_gain,
  const double sum_gradients,
  const double sum_hessians,
  const data_size_t num_data,
  // task information
  const bool is_one_hot,
  // buffer
  hist_t* hist_grad_buffer_ptr,
  hist_t* hist_hess_buffer_ptr,
  hist_t* hist_stat_buffer_ptr,
  data_size_t* hist_index_buffer_ptr,
  // output parameters
  CUDASplitInfo* cuda_best_split_info) {
  __shared__ double shared_gain_buffer[32];
  __shared__ bool shared_found_buffer[32];
  __shared__ uint32_t shared_thread_index_buffer[32];
  __shared__ uint32_t best_thread_index;
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;
  const double l2 = lambda_l2 + cat_l2;

  double local_gain = kMinScore;
  bool threshold_found = false;

  cuda_best_split_info->is_valid = false;

  const int bin_start = 1 - feature_mfb_offset;
  const int bin_end = feature_num_bin - feature_mfb_offset;
  int best_threshold = -1;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  if (is_one_hot) {
    for (int bin = bin_start + threadIdx_x; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
      const int bin_offset = (bin << 1);
      const hist_t grad = feature_hist_ptr[bin_offset];
      const hist_t hess = feature_hist_ptr[bin_offset + 1];
      data_size_t cnt =
            static_cast<data_size_t>(__double2int_rn(hess * cnt_factor));
      if (cnt >= min_data_in_leaf && hess >= min_sum_hessian_in_leaf) {
        const data_size_t other_count = num_data - cnt;
        if (other_count >= min_data_in_leaf) {
          const double sum_other_hessian = sum_hessians - hess - kEpsilon;
          if (sum_other_hessian >= min_sum_hessian_in_leaf) {
            const double sum_other_gradient = sum_gradients - grad;
            double current_gain = GetSplitGains(
              sum_other_gradient, sum_other_hessian, grad,
              hess + kEpsilon, lambda_l1, use_l1,
              l2);
            if (current_gain > min_gain_shift) {
              best_threshold = bin;
              local_gain = current_gain - min_gain_shift;
              threshold_found = true;
            }
          }
        }
      }
    }
    __syncthreads();
    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = 1;
      cuda_best_split_info->cat_threshold = new uint32_t[1];
      *(cuda_best_split_info->cat_threshold) = static_cast<uint32_t>(best_threshold);
      cuda_best_split_info->default_left = false;
      const int bin_offset = (best_threshold << 1);
      const hist_t sum_left_gradient = feature_hist_ptr[bin_offset];
      const hist_t sum_left_hessian = feature_hist_ptr[bin_offset + 1];
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2, right_output);
    }
  } else {
    __shared__ uint16_t shared_mem_buffer_uint16[32];
    __shared__ int used_bin;
    uint16_t is_valid_bin = 0;
    int best_dir = 0;
    double best_sum_left_gradient = 0.0f;
    double best_sum_left_hessian = 0.0f;
    for (int bin = 0; bin < bin_end; bin += static_cast<int>(blockDim.x)) {
      if (bin >= bin_start) {
        const int bin_offset = (bin << 1);
        const double hess = feature_hist_ptr[bin_offset + 1];
        if (__double2int_rn(hess * cnt_factor) >= cat_smooth) {
          const double grad = feature_hist_ptr[bin_offset];
          hist_stat_buffer_ptr[bin] = grad / (hess + cat_smooth);
          hist_index_buffer_ptr[bin] = threadIdx_x;
          is_valid_bin = 1;
        } else {
          hist_stat_buffer_ptr[bin] = kMaxScore;
          hist_index_buffer_ptr[bin] = -1;
        }
      }
    }
    __syncthreads();
    const int local_used_bin = ShuffleReduceSum<uint16_t>(is_valid_bin, shared_mem_buffer_uint16, blockDim.x);
    if (threadIdx_x == 0) {
      used_bin = local_used_bin;
    }
    __syncthreads();
    BitonicArgSortDevice<double, data_size_t, true, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 11>(
      hist_stat_buffer_ptr, hist_index_buffer_ptr, feature_num_bin - feature_mfb_offset);
    const int max_num_cat = min(max_cat_threshold, (used_bin + 1) / 2);
    __syncthreads();

    // left to right
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (hist_index_buffer_ptr[threadIdx_x] << 1);
      hist_grad_buffer_ptr[threadIdx_x] = feature_hist_ptr[bin_offset];
      hist_hess_buffer_ptr[threadIdx_x] = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    // TODO(shiyu1994): constrain the split with min_data_in_group
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const double sum_left_gradient = hist_grad_buffer_ptr[threadIdx_x];
      const double sum_left_hessian = hist_hess_buffer_ptr[threadIdx_x];
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          l2);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = 1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    // right to left
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const int bin_offset = (hist_index_buffer_ptr[used_bin - 1 - threadIdx_x] << 1);
      hist_grad_buffer_ptr[threadIdx_x] = feature_hist_ptr[bin_offset];
      hist_hess_buffer_ptr[threadIdx_x] = feature_hist_ptr[bin_offset + 1];
    }
    if (threadIdx_x == 0) {
      hist_hess_buffer_ptr[0] += kEpsilon;
    }
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_grad_buffer_ptr, static_cast<size_t>(bin_end));
    __syncthreads();
    GlobalMemoryPrefixSum<double>(hist_hess_buffer_ptr, static_cast<size_t>(bin_end));
    // TODO(shiyu1994): constrain the split with min_data_in_group
    if (threadIdx_x < used_bin && threadIdx_x < max_num_cat) {
      const double sum_left_gradient = hist_grad_buffer_ptr[threadIdx_x];
      const double sum_left_hessian = hist_hess_buffer_ptr[threadIdx_x];
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = num_data - left_count;
      if (sum_left_hessian >= min_sum_hessian_in_leaf && left_count >= min_data_in_leaf &&
        sum_right_hessian >= min_sum_hessian_in_leaf && right_count >= min_data_in_leaf) {
        double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          l2);
        // gain with split is worse than without split
        if (current_gain > min_gain_shift) {
          local_gain = current_gain - min_gain_shift;
          threshold_found = true;
          best_dir = -1;
          best_sum_left_gradient = sum_left_gradient;
          best_sum_left_hessian = sum_left_hessian;
        }
      }
    }
    __syncthreads();

    const uint32_t result = ReduceBestGain(local_gain, threshold_found, threadIdx_x, shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
    if (threadIdx_x == 0) {
      best_thread_index = result;
    }
    __syncthreads();
    if (threshold_found && threadIdx_x == best_thread_index) {
      cuda_best_split_info->is_valid = true;
      cuda_best_split_info->num_cat_threshold = threadIdx_x + 1;
      cuda_best_split_info->cat_threshold = new uint32_t[threadIdx_x + 1];
      cuda_best_split_info->gain = local_gain;
      if (best_dir == 1) {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = hist_index_buffer_ptr[i] + feature_mfb_offset;
        }
      } else {
        for (int i = 0; i < threadIdx_x + 1; ++i) {
          (cuda_best_split_info->cat_threshold)[i] = hist_index_buffer_ptr[used_bin - 1 - i] + feature_mfb_offset;
        }
      }
      cuda_best_split_info->default_left = false;
      const hist_t sum_left_gradient = best_sum_left_gradient;
      const hist_t sum_left_hessian = best_sum_left_hessian;
      const data_size_t left_count = static_cast<data_size_t>(__double2int_rn(sum_left_hessian * cnt_factor));
      const double sum_right_gradient = sum_gradients - sum_left_gradient;
      const double sum_right_hessian = sum_hessians - sum_left_hessian;
      const data_size_t right_count = static_cast<data_size_t>(__double2int_rn(sum_right_hessian * cnt_factor));
      const double left_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2);
      const double right_output = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2);
      cuda_best_split_info->left_sum_gradients = sum_left_gradient;
      cuda_best_split_info->left_sum_hessians = sum_left_hessian;
      cuda_best_split_info->left_count = left_count;
      cuda_best_split_info->right_sum_gradients = sum_right_gradient;
      cuda_best_split_info->right_sum_hessians = sum_right_hessian;
      cuda_best_split_info->right_count = right_count;
      cuda_best_split_info->left_value = left_output;
      cuda_best_split_info->left_gain = GetLeafGainGivenOutput(sum_left_gradient,
        sum_left_hessian, lambda_l1, use_l1, l2, left_output);
      cuda_best_split_info->right_value = right_output;
      cuda_best_split_info->right_gain = GetLeafGainGivenOutput(sum_right_gradient,
        sum_right_hessian, lambda_l1, use_l1, l2, right_output);
    }
  }
}

__global__ void FindBestSplitsForLeafKernel_GlobalMemory(
  // input feature information
  const uint32_t* feature_hist_offsets,
  const uint8_t* feature_mfb_offsets,
  const uint32_t* feature_default_bins,
  const uint32_t* feature_num_bins,
  const int8_t* is_feature_used_bytree,
  const int8_t* is_categorical,
  // input task information
  const bool larger_only,
  const int num_tasks,
  const int* task_feature_index,
  const uint8_t* task_reverse,
  const uint8_t* task_skip_default_bin,
  const uint8_t* task_na_as_missing,
  const uint8_t* task_out_default_left,
  // input leaf information
  const int smaller_leaf_index,
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const int larger_leaf_index,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  // input config parameter values
  const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf,
  const double min_gain_to_split,
  const double lambda_l1,
  const double lambda_l2,
  const double cat_smooth,
  const double cat_l2,
  const int max_cat_threshold,
  const int min_data_per_group,
  const int max_cat_to_onehot,
  // buffer
  hist_t* feature_hist_grad_buffer,
  hist_t* feature_hist_hess_buffer,
  hist_t* feature_hist_stat_buffer,
  data_size_t* feature_hist_index_buffer,
  // output
  CUDASplitInfo* cuda_best_split_info) {
  const unsigned int task_index = blockIdx.x % num_tasks;
  const bool is_larger = static_cast<bool>(blockIdx.x >= num_tasks || larger_only);
  const int inner_feature_index = task_feature_index[task_index];
  const double parent_gain = is_larger ? larger_leaf_splits->gain : smaller_leaf_splits->gain;
  const double sum_gradients = is_larger ? larger_leaf_splits->sum_of_gradients : smaller_leaf_splits->sum_of_gradients;
  const double sum_hessians = (is_larger ? larger_leaf_splits->sum_of_hessians : smaller_leaf_splits->sum_of_hessians) + 2 * kEpsilon;
  const double num_data = is_larger ? larger_leaf_splits->num_data_in_leaf : smaller_leaf_splits->num_data_in_leaf;
  const unsigned int output_offset = is_larger ? (task_index + num_tasks) : task_index;
  CUDASplitInfo* out = cuda_best_split_info + output_offset;
  if (is_feature_used_bytree[inner_feature_index]) {
    const hist_t* hist_ptr = (is_larger ? larger_leaf_splits->hist_in_leaf : smaller_leaf_splits->hist_in_leaf) + feature_hist_offsets[inner_feature_index] * 2;
    hist_t* hist_grad_buffer_ptr = feature_hist_grad_buffer + feature_hist_offsets[inner_feature_index] * 2;
    hist_t* hist_hess_buffer_ptr = feature_hist_hess_buffer + feature_hist_offsets[inner_feature_index] * 2;
    hist_t* hist_stat_buffer_ptr = feature_hist_stat_buffer + feature_hist_offsets[inner_feature_index] * 2;
    data_size_t* hist_index_buffer_ptr = feature_hist_index_buffer + feature_hist_offsets[inner_feature_index] * 2;
    if (is_categorical[inner_feature_index]) {
      const bool is_one_hot = feature_num_bins[inner_feature_index] <= max_cat_to_onehot;
      FindBestSplitsForLeafKernelCategoricalInner_GlobalMemory(
        // input feature information
        hist_ptr,
        feature_num_bins[inner_feature_index],
        feature_mfb_offsets[inner_feature_index],
        feature_default_bins[inner_feature_index],
        inner_feature_index,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        cat_smooth,
        cat_l2,
        max_cat_threshold,
        min_data_per_group,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        // input task information
        is_one_hot,
        // buffer
        hist_grad_buffer_ptr,
        hist_hess_buffer_ptr,
        hist_stat_buffer_ptr,
        hist_index_buffer_ptr,
        // output parameters
        out);
    } else {
      const bool reverse = static_cast<bool>(task_reverse[task_index]);
      const bool skip_default_bin = static_cast<bool>(task_skip_default_bin[task_index]);
      const bool na_as_missing = static_cast<bool>(task_na_as_missing[task_index]);
      const bool assume_out_default_left = task_out_default_left[task_index];
      FindBestSplitsForLeafKernelInner_GlobalMemory(
        // input feature information
        hist_ptr,
        feature_num_bins[inner_feature_index],
        feature_mfb_offsets[inner_feature_index],
        feature_default_bins[inner_feature_index],
        inner_feature_index,
        // input config parameter values
        lambda_l1,
        lambda_l2,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        min_gain_to_split,
        // input parent node information
        parent_gain,
        sum_gradients,
        sum_hessians,
        num_data,
        // input task information
        reverse,
        skip_default_bin,
        na_as_missing,
        assume_out_default_left,
        // buffer
        hist_grad_buffer_ptr,
        hist_hess_buffer_ptr,
        // output parameters
        out);
      }
  } else {
    out->is_valid = false;
  }
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits,
  const CUDALeafSplitsStruct* larger_leaf_splits,
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid) {
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  bool larger_only = false;
  if (!is_smaller_leaf_valid) {
    larger_only = true;
  }
  if (!use_global_memory_) {
    if (!larger_only) {
      FindBestSplitsForLeafKernel<<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        // input feature information
        cuda_feature_hist_offsets_,
        cuda_feature_mfb_offsets_,
        cuda_feature_default_bins_,
        cuda_feature_num_bins_,
        cuda_is_feature_used_bytree_,
        cuda_is_categorical_,
        // input task information
        larger_only,
        num_tasks_,
        cuda_task_feature_index_,
        cuda_task_reverse_,
        cuda_task_skip_default_bin_,
        cuda_task_na_as_missing_,
        cuda_task_out_default_left_,
        // input leaf information
        smaller_leaf_index,
        smaller_leaf_splits,
        larger_leaf_index,
        larger_leaf_splits,
        // configuration parameter values
        min_data_in_leaf_,
        min_sum_hessian_in_leaf_,
        min_gain_to_split_,
        lambda_l1_,
        lambda_l2_,
        cat_smooth_,
        cat_l2_,
        max_cat_threshold_,
        min_data_per_group_,
        max_cat_to_onehot_,
        // output parameters
        cuda_best_split_info_);
    }
    SynchronizeCUDADevice(__FILE__, __LINE__);
    if (larger_leaf_index >= 0) {
      FindBestSplitsForLeafKernel<<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>(
        // input feature information
        cuda_feature_hist_offsets_,
        cuda_feature_mfb_offsets_,
        cuda_feature_default_bins_,
        cuda_feature_num_bins_,
        cuda_is_feature_used_bytree_,
        cuda_is_categorical_,
        // input task information
        true,
        num_tasks_,
        cuda_task_feature_index_,
        cuda_task_reverse_,
        cuda_task_skip_default_bin_,
        cuda_task_na_as_missing_,
        cuda_task_out_default_left_,
        // input leaf information
        smaller_leaf_index,
        smaller_leaf_splits,
        larger_leaf_index,
        larger_leaf_splits,
        // configuration parameter values
        min_data_in_leaf_,
        min_sum_hessian_in_leaf_,
        min_gain_to_split_,
        lambda_l1_,
        lambda_l2_,
        cat_smooth_,
        cat_l2_,
        max_cat_threshold_,
        min_data_per_group_,
        max_cat_to_onehot_,
        // output parameters
        cuda_best_split_info_);
    }
  } else {
    if (!larger_only) {
      FindBestSplitsForLeafKernel_GlobalMemory<<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[0]>>>(
        // input feature information
        cuda_feature_hist_offsets_,
        cuda_feature_mfb_offsets_,
        cuda_feature_default_bins_,
        cuda_feature_num_bins_,
        cuda_is_feature_used_bytree_,
        cuda_is_categorical_,
        // input task information
        larger_only,
        num_tasks_,
        cuda_task_feature_index_,
        cuda_task_reverse_,
        cuda_task_skip_default_bin_,
        cuda_task_na_as_missing_,
        cuda_task_out_default_left_,
        // input leaf information
        smaller_leaf_index,
        smaller_leaf_splits,
        larger_leaf_index,
        larger_leaf_splits,
        // configuration parameter values
        min_data_in_leaf_,
        min_sum_hessian_in_leaf_,
        min_gain_to_split_,
        lambda_l1_,
        lambda_l2_,
        cat_smooth_,
        cat_l2_,
        max_cat_threshold_,
        min_data_per_group_,
        max_cat_to_onehot_,
        // buffer
        cuda_feature_hist_grad_buffer_,
        cuda_feature_hist_hess_buffer_,
        cuda_feature_hist_stat_buffer_,
        cuda_feature_hist_index_buffer_,
        // output parameters
        cuda_best_split_info_);
    }
    SynchronizeCUDADevice(__FILE__, __LINE__);
    if (larger_leaf_index >= 0) {
      FindBestSplitsForLeafKernel_GlobalMemory<<<num_tasks_, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER, 0, cuda_streams_[1]>>>(
        // input feature information
        cuda_feature_hist_offsets_,
        cuda_feature_mfb_offsets_,
        cuda_feature_default_bins_,
        cuda_feature_num_bins_,
        cuda_is_feature_used_bytree_,
        cuda_is_categorical_,
        // input task information
        true,
        num_tasks_,
        cuda_task_feature_index_,
        cuda_task_reverse_,
        cuda_task_skip_default_bin_,
        cuda_task_na_as_missing_,
        cuda_task_out_default_left_,
        // input leaf information
        smaller_leaf_index,
        smaller_leaf_splits,
        larger_leaf_index,
        larger_leaf_splits,
        // configuration parameter values
        min_data_in_leaf_,
        min_sum_hessian_in_leaf_,
        min_gain_to_split_,
        lambda_l1_,
        lambda_l2_,
        cat_smooth_,
        cat_l2_,
        max_cat_threshold_,
        min_data_per_group_,
        max_cat_to_onehot_,
        // buffer
        cuda_feature_hist_grad_buffer_,
        cuda_feature_hist_hess_buffer_,
        cuda_feature_hist_stat_buffer_,
        cuda_feature_hist_index_buffer_,
        // output parameters
        cuda_best_split_info_);
    }
  }
}

__device__ void ReduceBestSplit(bool* found, double* gain, uint32_t* shared_read_index,
  uint32_t num_features_aligned) {
  const uint32_t threadIdx_x = threadIdx.x;
  for (unsigned int s = 1; s < num_features_aligned; s <<= 1) {
    if (threadIdx_x % (2 * s) == 0 && (threadIdx_x + s) < num_features_aligned) {
      const uint32_t pos_to_compare = threadIdx_x + s;
      if ((!found[threadIdx_x] && found[pos_to_compare]) ||
        (found[threadIdx_x] && found[pos_to_compare] && gain[threadIdx_x] < gain[pos_to_compare])) {
        found[threadIdx_x] = found[pos_to_compare];
        gain[threadIdx_x] = gain[pos_to_compare];
        shared_read_index[threadIdx_x] = shared_read_index[pos_to_compare];
      }
    }
    __syncthreads();
  }
}

__global__ void SyncBestSplitForLeafKernel(const int smaller_leaf_index, const int larger_leaf_index,
  CUDASplitInfo* cuda_leaf_best_split_info,
  // input parameters
  const int* cuda_task_feature_index,
  const CUDASplitInfo* cuda_best_split_info,
  const uint32_t* cuda_feature_default_bins,
  const int num_tasks,
  const int num_tasks_aligned,
  const int num_blocks_per_leaf,
  const bool larger_only,
  const int num_leaves) {
  __shared__ double shared_gain_buffer[32];
  __shared__ bool shared_found_buffer[32];
  __shared__ uint32_t shared_thread_index_buffer[32];
  const uint32_t threadIdx_x = threadIdx.x;
  const uint32_t blockIdx_x = blockIdx.x;

  bool best_found = false;
  double best_gain = kMinScore;
  uint32_t shared_read_index = 0;

  const bool is_smaller = (blockIdx_x < static_cast<unsigned int>(num_blocks_per_leaf) && !larger_only);
  const uint32_t leaf_block_index = (is_smaller || larger_only) ? blockIdx_x : (blockIdx_x - static_cast<unsigned int>(num_blocks_per_leaf));
  const int task_index = static_cast<int>(leaf_block_index * blockDim.x + threadIdx_x);
  const uint32_t read_index = is_smaller ? static_cast<uint32_t>(task_index) : static_cast<uint32_t>(task_index + num_tasks);
  if (task_index < num_tasks) {
    best_found = cuda_best_split_info[read_index].is_valid;
    best_gain = cuda_best_split_info[read_index].gain;
    shared_read_index = read_index;
  } else {
    best_found = false;
  }

  __syncthreads();
  const uint32_t best_read_index = ReduceBestGain(best_gain, best_found, shared_read_index,
      shared_gain_buffer, shared_found_buffer, shared_thread_index_buffer);
  if (threadIdx.x == 0) {
    const int leaf_index_ref = is_smaller ? smaller_leaf_index : larger_leaf_index;
    const unsigned buffer_write_pos = static_cast<unsigned int>(leaf_index_ref) + leaf_block_index * num_leaves;
    CUDASplitInfo* cuda_split_info = cuda_leaf_best_split_info + buffer_write_pos;
    const CUDASplitInfo* best_split_info = cuda_best_split_info + best_read_index;
    if (best_split_info->is_valid) {
      *cuda_split_info = *best_split_info;
      cuda_split_info->inner_feature_index = is_smaller ? cuda_task_feature_index[best_read_index] :
        cuda_task_feature_index[static_cast<int>(best_read_index) - num_tasks];
      cuda_split_info->is_valid = true;
    } else {
      cuda_split_info->gain = kMinScore;
      cuda_split_info->is_valid = false;
    }
  }
}

__global__ void SyncBestSplitForLeafKernelAllBlocks(
  const int smaller_leaf_index,
  const int larger_leaf_index,
  const unsigned int num_blocks_per_leaf,
  const int num_leaves,
  CUDASplitInfo* cuda_leaf_best_split_info,
  const bool larger_only) {
  if (!larger_only) {
    if (blockIdx.x == 0) {
      CUDASplitInfo* smaller_leaf_split_info = cuda_leaf_best_split_info + smaller_leaf_index;
      for (unsigned int block_index = 1; block_index < num_blocks_per_leaf; ++block_index) {
        const unsigned int leaf_read_pos = static_cast<unsigned int>(smaller_leaf_index) + block_index * static_cast<unsigned int>(num_leaves);
        const CUDASplitInfo* other_split_info = cuda_leaf_best_split_info + leaf_read_pos;
        if ((other_split_info->is_valid && smaller_leaf_split_info->is_valid &&
          other_split_info->gain > smaller_leaf_split_info->gain) ||
            (!smaller_leaf_split_info->is_valid && other_split_info->is_valid)) {
            smaller_leaf_split_info->is_valid = other_split_info->is_valid;
            smaller_leaf_split_info->inner_feature_index = other_split_info->inner_feature_index;
            smaller_leaf_split_info->default_left = other_split_info->default_left;
            smaller_leaf_split_info->threshold = other_split_info->threshold;
            smaller_leaf_split_info->gain = other_split_info->gain;
            smaller_leaf_split_info->left_sum_gradients = other_split_info->left_sum_gradients;
            smaller_leaf_split_info->left_sum_hessians = other_split_info->left_sum_hessians;
            smaller_leaf_split_info->left_count = other_split_info->left_count;
            smaller_leaf_split_info->left_gain = other_split_info->left_gain;
            smaller_leaf_split_info->left_value = other_split_info->left_value;
            smaller_leaf_split_info->right_sum_gradients = other_split_info->right_sum_gradients;
            smaller_leaf_split_info->right_sum_hessians = other_split_info->right_sum_hessians;
            smaller_leaf_split_info->right_count = other_split_info->right_count;
            smaller_leaf_split_info->right_gain = other_split_info->right_gain;
            smaller_leaf_split_info->right_value = other_split_info->right_value;
        }
      }
    }
  }
  if (larger_leaf_index >= 0) {
    if (blockIdx.x == 1 || larger_only) {
      CUDASplitInfo* larger_leaf_split_info = cuda_leaf_best_split_info + larger_leaf_index;
      for (unsigned int block_index = 1; block_index < num_blocks_per_leaf; ++block_index) {
        const unsigned int leaf_read_pos = static_cast<unsigned int>(larger_leaf_index) + block_index * static_cast<unsigned int>(num_leaves);
        const CUDASplitInfo* other_split_info = cuda_leaf_best_split_info + leaf_read_pos;
        if ((other_split_info->is_valid && larger_leaf_split_info->is_valid &&
          other_split_info->gain > larger_leaf_split_info->gain) ||
            (!larger_leaf_split_info->is_valid && other_split_info->is_valid)) {
            larger_leaf_split_info->is_valid = other_split_info->is_valid;
            larger_leaf_split_info->inner_feature_index = other_split_info->inner_feature_index;
            larger_leaf_split_info->default_left = other_split_info->default_left;
            larger_leaf_split_info->threshold = other_split_info->threshold;
            larger_leaf_split_info->gain = other_split_info->gain;
            larger_leaf_split_info->left_sum_gradients = other_split_info->left_sum_gradients;
            larger_leaf_split_info->left_sum_hessians = other_split_info->left_sum_hessians;
            larger_leaf_split_info->left_count = other_split_info->left_count;
            larger_leaf_split_info->left_gain = other_split_info->left_gain;
            larger_leaf_split_info->left_value = other_split_info->left_value;
            larger_leaf_split_info->right_sum_gradients = other_split_info->right_sum_gradients;
            larger_leaf_split_info->right_sum_hessians = other_split_info->right_sum_hessians;
            larger_leaf_split_info->right_count = other_split_info->right_count;
            larger_leaf_split_info->right_gain = other_split_info->right_gain;
            larger_leaf_split_info->right_value = other_split_info->right_value;
        }
      }
    }
  }
}

__global__ void SetInvalidLeafSplitInfoKernel(
  CUDASplitInfo* cuda_leaf_best_split_info,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid,
  const int smaller_leaf_index,
  const int larger_leaf_index) {
  if (!is_smaller_leaf_valid) {
    cuda_leaf_best_split_info[smaller_leaf_index].is_valid = false;
  }
  if (!is_larger_leaf_valid && larger_leaf_index >= 0) {
    cuda_leaf_best_split_info[larger_leaf_index].is_valid = false;
  }
}

void CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel(
  const int host_smaller_leaf_index,
  const int host_larger_leaf_index,
  const bool is_smaller_leaf_valid,
  const bool is_larger_leaf_valid) {
  if (!is_smaller_leaf_valid || !is_larger_leaf_valid) {
    SetInvalidLeafSplitInfoKernel<<<1, 1>>>(
      cuda_leaf_best_split_info_,
      is_smaller_leaf_valid, is_larger_leaf_valid,
      host_smaller_leaf_index, host_larger_leaf_index);
  }
  if (!is_smaller_leaf_valid && !is_larger_leaf_valid) {
    return;
  }
  int num_tasks = num_tasks_;
  int num_tasks_aligned = 1;
  num_tasks -= 1;
  while (num_tasks > 0) {
    num_tasks_aligned <<= 1;
    num_tasks >>= 1;
  }
  const int num_blocks_per_leaf = (num_tasks_ + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK;
  if (host_larger_leaf_index >= 0 && is_smaller_leaf_valid && is_larger_leaf_valid) {
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[0]>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_,
      cuda_task_feature_index_,
      cuda_best_split_info_,
      cuda_feature_default_bins_,
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      false,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1, 0, cuda_streams_[0]>>>(
        host_smaller_leaf_index,
        host_larger_leaf_index,
        num_blocks_per_leaf,
        num_leaves_,
        cuda_leaf_best_split_info_,
        false);
    }
    SynchronizeCUDADevice(__FILE__, __LINE__);
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK, 0, cuda_streams_[1]>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_,
      cuda_task_feature_index_,
      cuda_best_split_info_,
      cuda_feature_default_bins_,
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      true,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1, 0, cuda_streams_[1]>>>(
        host_smaller_leaf_index,
        host_larger_leaf_index,
        num_blocks_per_leaf,
        num_leaves_,
        cuda_leaf_best_split_info_,
        true);
    }
  } else {
    const bool larger_only = (!is_smaller_leaf_valid && is_larger_leaf_valid);
    SyncBestSplitForLeafKernel<<<num_blocks_per_leaf, NUM_TASKS_PER_SYNC_BLOCK>>>(
      host_smaller_leaf_index,
      host_larger_leaf_index,
      cuda_leaf_best_split_info_,
      cuda_task_feature_index_,
      cuda_best_split_info_,
      cuda_feature_default_bins_,
      num_tasks_,
      num_tasks_aligned,
      num_blocks_per_leaf,
      larger_only,
      num_leaves_);
    if (num_blocks_per_leaf > 1) {
      SynchronizeCUDADevice(__FILE__, __LINE__);
      SyncBestSplitForLeafKernelAllBlocks<<<1, 1>>>(
        host_smaller_leaf_index,
        host_larger_leaf_index,
        num_blocks_per_leaf,
        num_leaves_,
        cuda_leaf_best_split_info_,
        larger_only);
    }
  }
}

__global__ void FindBestFromAllSplitsKernel(const int cur_num_leaves,
  CUDASplitInfo* cuda_leaf_best_split_info,
  int* cuda_best_split_info_buffer) {
  __shared__ double gain_shared_buffer[32];
  __shared__ int leaf_index_shared_buffer[32];
  double thread_best_gain = kMinScore;
  int thread_best_leaf_index = -1;
  const int threadIdx_x = static_cast<int>(threadIdx.x);
  for (int leaf_index = threadIdx_x; leaf_index < cur_num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    const double leaf_best_gain = cuda_leaf_best_split_info[leaf_index].gain;
    if (cuda_leaf_best_split_info[leaf_index].is_valid && leaf_best_gain > thread_best_gain) {
      thread_best_gain = leaf_best_gain;
      thread_best_leaf_index = leaf_index;
    }
  }
  const int best_leaf_index = ReduceBestGainForLeaves(thread_best_gain, thread_best_leaf_index, gain_shared_buffer, leaf_index_shared_buffer);
  if (threadIdx_x == 0) {
    cuda_best_split_info_buffer[6] = best_leaf_index;
    if (best_leaf_index != -1) {
      cuda_leaf_best_split_info[best_leaf_index].is_valid = false;
      cuda_leaf_best_split_info[cur_num_leaves].is_valid = false;
      cuda_best_split_info_buffer[7] = cuda_leaf_best_split_info[best_leaf_index].num_cat_threshold;
    }
  }
}

__global__ void PrepareLeafBestSplitInfo(const int smaller_leaf_index, const int larger_leaf_index,
  int* cuda_best_split_info_buffer,
  const CUDASplitInfo* cuda_leaf_best_split_info) {
  const unsigned int threadIdx_x = blockIdx.x;
  if (threadIdx_x == 0) {
    cuda_best_split_info_buffer[0] = cuda_leaf_best_split_info[smaller_leaf_index].inner_feature_index;
  } else if (threadIdx_x == 1) {
    cuda_best_split_info_buffer[1] = cuda_leaf_best_split_info[smaller_leaf_index].threshold;
  } else if (threadIdx_x == 2) {
    cuda_best_split_info_buffer[2] = cuda_leaf_best_split_info[smaller_leaf_index].default_left;
  }
  if (larger_leaf_index >= 0) {
    if (threadIdx_x == 3) {
      cuda_best_split_info_buffer[3] = cuda_leaf_best_split_info[larger_leaf_index].inner_feature_index;
    } else if (threadIdx_x == 4) {
      cuda_best_split_info_buffer[4] = cuda_leaf_best_split_info[larger_leaf_index].threshold;
    } else if (threadIdx_x == 5) {
      cuda_best_split_info_buffer[5] = cuda_leaf_best_split_info[larger_leaf_index].default_left;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestFromAllSplitsKernel(
  const int cur_num_leaves,
  const int smaller_leaf_index, const int larger_leaf_index,
  int* smaller_leaf_best_split_feature,
  uint32_t* smaller_leaf_best_split_threshold,
  uint8_t* smaller_leaf_best_split_default_left,
  int* larger_leaf_best_split_feature,
  uint32_t* larger_leaf_best_split_threshold,
  uint8_t* larger_leaf_best_split_default_left,
  int* best_leaf_index,
  int* num_cat_threshold) {
  FindBestFromAllSplitsKernel<<<1, NUM_THREADS_FIND_BEST_LEAF, 0, cuda_streams_[1]>>>(cur_num_leaves,
    cuda_leaf_best_split_info_,
    cuda_best_split_info_buffer_);
  PrepareLeafBestSplitInfo<<<6, 1, 0, cuda_streams_[0]>>>(smaller_leaf_index, larger_leaf_index,
    cuda_best_split_info_buffer_,
    cuda_leaf_best_split_info_);
  std::vector<int> host_leaf_best_split_info_buffer(8, 0);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(host_leaf_best_split_info_buffer.data(), cuda_best_split_info_buffer_, 8, __FILE__, __LINE__);
  *smaller_leaf_best_split_feature = host_leaf_best_split_info_buffer[0];
  *smaller_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[1]);
  *smaller_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[2]);
  if (larger_leaf_index >= 0) {
    *larger_leaf_best_split_feature = host_leaf_best_split_info_buffer[3];
    *larger_leaf_best_split_threshold = static_cast<uint32_t>(host_leaf_best_split_info_buffer[4]);
    *larger_leaf_best_split_default_left = static_cast<uint8_t>(host_leaf_best_split_info_buffer[5]);
  }
  *best_leaf_index = host_leaf_best_split_info_buffer[6];
  *num_cat_threshold = host_leaf_best_split_info_buffer[7];
}

__global__ void AllocateCatVectorsKernel(
  CUDASplitInfo* cuda_split_infos, size_t len,
  const int max_num_categories_in_split,
  const bool has_categorical_feature) {
  const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < len) {
    if (has_categorical_feature) {
      cuda_split_infos[i].cat_threshold = new uint32_t[max_num_categories_in_split];
      cuda_split_infos[i].cat_threshold_real = new int[max_num_categories_in_split];
      cuda_split_infos[i].num_cat_threshold = 0;
    } else {
      cuda_split_infos[i].cat_threshold = nullptr;
      cuda_split_infos[i].cat_threshold_real = nullptr;
      cuda_split_infos[i].num_cat_threshold = 0;
    }
  }
}

void CUDABestSplitFinder::LaunchAllocateCatVectorsKernel(CUDASplitInfo* cuda_split_infos, size_t len) const {
  const int max_num_categories_in_split = min(max_cat_threshold_ / 2, max_num_categorical_bin_);
  const int num_blocks = (static_cast<int>(len) + NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER - 1) / NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER;
  AllocateCatVectorsKernel<<<num_blocks, NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER>>>(
    cuda_split_infos, len, max_num_categories_in_split, has_categorical_feature_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
