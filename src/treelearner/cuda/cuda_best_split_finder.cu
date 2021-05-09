/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"

namespace LightGBM {

__device__ double ThresholdL1(double s, double l1) {
  const double reg_s = fmax(0.0, fabs(s) - l1);
  if (s >= 0.0f) {
    return reg_s;
  } else {
    return -reg_s;
  }
}

__device__ double CalculateSplittedLeafOutput(double sum_gradients,
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

__device__ void FindBestSplitsForLeafKernelInner(const hist_t* feature_hist_ptr,
  const uint32_t feature_num_bin, const uint8_t feature_mfb_offset,
  const uint32_t feature_default_bin, const uint8_t feature_missing_type,
  const double lambda_l1, const double lambda_l2, const double parent_gain, const data_size_t min_data_in_leaf,
  const double min_sum_hessian_in_leaf, const double min_gain_to_split,
  const double sum_gradients, const double sum_hessians, const data_size_t num_data,
  const bool reverse, const bool skip_default_bin, const bool na_as_missing,
  // output parameters
  uint32_t* output_threshold,
  double* output_gain,
  uint8_t* output_default_left,
  double* output_left_sum_gradients,
  double* output_left_sum_hessians,
  data_size_t* output_left_num_data,
  double* output_left_gain,
  double* output_left_output,
  double* output_right_sum_gradients,
  double* output_right_sum_hessians,
  data_size_t* output_right_num_data,
  double* output_right_gain,
  double* output_right_output,
  uint8_t* output_found) {
  double best_sum_left_gradient = NAN;
  double best_sum_left_hessian = NAN;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  uint32_t best_threshold = feature_num_bin;
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;

  *output_found = 0;

  if (reverse) {
    double sum_right_gradient = 0.0f;
    double sum_right_hessian = kEpsilon;
    data_size_t right_count = 0;

    int t = feature_num_bin - 1 - feature_mfb_offset - na_as_missing;
    const int t_end = 1 - feature_mfb_offset;

    // from right to left, and we don't need data in bin0
    for (; t >= t_end; --t) {
      // need to skip default bin
      if (skip_default_bin) {
        if ((t + feature_mfb_offset) == static_cast<int>(feature_default_bin)) {
          continue;
        }
      }
      const auto grad = GET_GRAD(feature_hist_ptr, t);
      const auto hess = GET_HESS(feature_hist_ptr, t);
      data_size_t cnt =
          static_cast<data_size_t>(__double2int_rn(hess * cnt_factor));
      sum_right_gradient += grad;
      sum_right_hessian += hess;
      right_count += cnt;
      // if data not enough, or sum hessian too small
      if (right_count < min_data_in_leaf ||
          sum_right_hessian < min_sum_hessian_in_leaf) {
        continue;
      }
      data_size_t left_count = num_data - right_count;
      // if data not enough
      if (left_count < min_data_in_leaf) {
        break;
      }

      double sum_left_hessian = sum_hessians - sum_right_hessian;
      // if sum hessian too small
      if (sum_left_hessian < min_sum_hessian_in_leaf) {
        break;
      }

      double sum_left_gradient = sum_gradients - sum_right_gradient;

      // current split gain
      double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          lambda_l2);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }
      *output_found = 1;
      // better split point
      if (current_gain > best_gain) {
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        // left is <= threshold, right is > threshold.  so this is t-1
        best_threshold = static_cast<uint32_t>(t - 1 + feature_mfb_offset);
        best_gain = current_gain;
      }
    }
  } else {
    double sum_left_gradient = 0.0f;
    double sum_left_hessian = kEpsilon;
    data_size_t left_count = 0;

    int t = 0;
    const int t_end = feature_num_bin - 2 - feature_mfb_offset;
    if (na_as_missing) {
      if (feature_mfb_offset == 1) {
        sum_left_gradient = sum_gradients;
        sum_left_hessian = sum_hessians - kEpsilon;
        left_count = num_data;
        for (int i = 0; i < feature_num_bin - feature_mfb_offset; ++i) {
          const auto grad = GET_GRAD(feature_hist_ptr, i);
          const auto hess = GET_HESS(feature_hist_ptr, i);
          data_size_t cnt =
              static_cast<data_size_t>(__double2int_rn(hess * cnt_factor));
          sum_left_gradient -= grad;
          sum_left_hessian -= hess;
          left_count -= cnt;
        }
        t = -1;
      }
    }

    for (; t <= t_end; ++t) {
      if (skip_default_bin) {
        if ((t + feature_mfb_offset) == static_cast<int>(feature_default_bin)) {
          continue;
        }
      }
      if (t >= 0) {
        sum_left_gradient += GET_GRAD(feature_hist_ptr, t);
        const hist_t hess = GET_HESS(feature_hist_ptr, t);
        sum_left_hessian += hess;
        left_count += static_cast<data_size_t>(
          __double2int_rn(hess * cnt_factor));
      }
      // if data not enough, or sum hessian too small
      if (left_count < min_data_in_leaf ||
          sum_left_hessian < min_sum_hessian_in_leaf) {
        continue;
      }
      data_size_t right_count = num_data - left_count;
      // if data not enough
      if (right_count < min_data_in_leaf) {
        break;
      }

      double sum_right_hessian = sum_hessians - sum_left_hessian;
      // if sum hessian too small
      if (sum_right_hessian < min_sum_hessian_in_leaf) {
        break;
      }

      double sum_right_gradient = sum_gradients - sum_left_gradient;

      // current split gain
      double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          lambda_l2);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }
      *output_found = 1;
      // better split point
      if (current_gain > best_gain) {
        best_left_count = left_count;
        best_sum_left_gradient = sum_left_gradient;
        best_sum_left_hessian = sum_left_hessian;
        best_threshold = static_cast<uint32_t>(t + feature_mfb_offset);
        best_gain = current_gain;
      }
    }
  }

  if (*output_found) {
    *output_threshold = best_threshold;
    *output_gain = best_gain - min_gain_shift;
    *output_default_left = reverse;
    *output_left_sum_gradients = best_sum_left_gradient;
    *output_left_sum_hessians = best_sum_left_hessian;
    *output_left_num_data = best_left_count;

    const double best_sum_right_gradient = sum_gradients - best_sum_left_gradient;
    const double best_sum_right_hessian = sum_hessians - best_sum_left_hessian;
    *output_right_sum_gradients = best_sum_right_gradient;
    *output_right_sum_hessians = best_sum_right_hessian;
    *output_right_num_data = num_data - best_left_count;

    *output_left_output = CalculateSplittedLeafOutput(best_sum_left_gradient,
      best_sum_left_hessian, lambda_l1, use_l1, lambda_l2);
    *output_left_gain = GetLeafGainGivenOutput(best_sum_left_gradient,
      best_sum_left_hessian, lambda_l1, use_l1, lambda_l2, *output_left_output);
    *output_right_output = CalculateSplittedLeafOutput(best_sum_right_gradient,
      best_sum_right_hessian, lambda_l1, use_l1, lambda_l2);
    *output_right_gain = GetLeafGainGivenOutput(best_sum_right_gradient,
      best_sum_right_hessian, lambda_l1, use_l1, lambda_l2, *output_right_output);
  }
}

__global__ void FindBestSplitsForLeafKernel(const hist_t* cuda_hist, const int* cuda_num_total_bin,
  const uint32_t* feature_hist_offsets, const uint8_t* feature_mfb_offsets, const uint32_t* feature_default_bins, 
  const uint8_t* feature_missing_types, const double* lambda_l1, const double* lambda_l2, const int* smaller_leaf_id,
  const int* larger_leaf_id, const double* smaller_leaf_gain, const double* larger_leaf_gain, const double* sum_gradients_in_smaller_leaf,
  const double* sum_hessians_in_smaller_leaf, const data_size_t* num_data_in_smaller_leaf, hist_t** smaller_leaf_hist,
  const double* sum_gradients_in_larger_leaf, const double* sum_hessians_in_larger_leaf,
  const data_size_t* num_data_in_larger_leaf, hist_t** larger_leaf_hist, const data_size_t* min_data_in_leaf,
  const double* min_sum_hessian_in_leaf, const double*  min_gain_to_split,
  // output
  uint32_t* cuda_best_split_threshold, uint8_t* cuda_best_split_default_left,
  double* cuda_best_split_gain, double* cuda_best_split_left_sum_gradient,
  double* cuda_best_split_left_sum_hessian, data_size_t* cuda_best_split_left_count, 
  double* cuda_best_split_left_gain, double* cuda_best_split_left_output,
  double* cuda_best_split_right_sum_gradient, double* cuda_best_split_right_sum_hessian,
  data_size_t* cuda_best_split_right_count, double* cuda_best_split_right_gain,
  double* cuda_best_split_right_output, uint8_t* cuda_best_split_found) {
  const unsigned int num_features = gridDim.x / 4;
  const unsigned int inner_feature_index = (blockIdx.x / 2) % num_features;
  const unsigned int global_block_idx = blockIdx.x;
  const bool reverse = blockIdx.x % 2 == 0 ? true : false;
  const bool smaller_or_larger = static_cast<bool>(blockIdx.x / (2 * num_features) == 0);
  const int num_bin = feature_hist_offsets[inner_feature_index + 1] - feature_hist_offsets[inner_feature_index];
  const uint8_t missing_type = feature_missing_types[inner_feature_index];
  const int leaf_index = smaller_or_larger ? *smaller_leaf_id : *larger_leaf_id;
  const double parent_gain = smaller_or_larger ? *smaller_leaf_gain : *larger_leaf_gain;
  /*if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("parent_gain = %f\n", parent_gain);
  }*/
  const double sum_gradients = smaller_or_larger ? *sum_gradients_in_smaller_leaf : *sum_gradients_in_larger_leaf;
  const double sum_hessians = smaller_or_larger ? *sum_hessians_in_smaller_leaf : *sum_hessians_in_larger_leaf;
  const double num_data_in_leaf = smaller_or_larger ? *num_data_in_smaller_leaf : *num_data_in_larger_leaf;
  uint32_t* out_threshold = cuda_best_split_threshold + global_block_idx;
  double* out_left_sum_gradients = cuda_best_split_left_sum_gradient + global_block_idx;
  double* out_left_sum_hessians = cuda_best_split_left_sum_hessian + global_block_idx;
  double* out_right_sum_gradients = cuda_best_split_right_sum_gradient + global_block_idx;
  double* out_right_sum_hessians = cuda_best_split_right_sum_hessian + global_block_idx;
  data_size_t* out_left_num_data = cuda_best_split_left_count + global_block_idx;
  data_size_t* out_right_num_data = cuda_best_split_right_count + global_block_idx;
  double* out_left_output = cuda_best_split_left_output + global_block_idx;
  double* out_right_output = cuda_best_split_right_output + global_block_idx;
  double* out_left_gain = cuda_best_split_left_gain + global_block_idx;
  double* out_right_gain = cuda_best_split_right_gain + global_block_idx;
  uint8_t* out_found = cuda_best_split_found + global_block_idx;
  uint8_t* out_default_left = cuda_best_split_default_left + global_block_idx;
  double* out_gain = cuda_best_split_gain + global_block_idx;
  if (leaf_index < 0) {
    *out_found = 0;
    return;
  }
  const int cuda_num_total_bin_ref = *cuda_num_total_bin;
  const hist_t* hist_ptr = smaller_or_larger ? *smaller_leaf_hist + feature_hist_offsets[inner_feature_index] * 2 :
    *larger_leaf_hist + feature_hist_offsets[inner_feature_index] * 2;// cuda_hist + (cuda_num_total_bin_ref * leaf_index + feature_hist_offsets[inner_feature_index]) * 2;
  if (num_bin > 2 && missing_type != 0) {
    if (missing_type == 1) {
      FindBestSplitsForLeafKernelInner(hist_ptr,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, true, false, out_threshold, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data, out_left_gain, out_left_output,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data, out_right_gain, out_right_output, out_found);
    } else {
      FindBestSplitsForLeafKernelInner(hist_ptr,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, false, true, out_threshold, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data, out_left_gain, out_left_output,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data, out_right_gain, out_right_output, out_found);
    }
  } else {
    if (reverse) {
      FindBestSplitsForLeafKernelInner(hist_ptr,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, true, false, out_threshold, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data, out_left_gain, out_left_output,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data, out_right_gain, out_right_output, out_found);
    }
    if (missing_type == 2) {
      *out_default_left = 0;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernel(const int* smaller_leaf_id, const int* larger_leaf_id,
  const double* smaller_leaf_gain, const double* larger_leaf_gain, const double* sum_gradients_in_smaller_leaf,
  const double* sum_hessians_in_smaller_leaf, const data_size_t* num_data_in_smaller_leaf, hist_t** smaller_leaf_hist,
  const double* sum_gradients_in_larger_leaf, const double* sum_hessians_in_larger_leaf,
  const data_size_t* num_data_in_larger_leaf, hist_t** larger_leaf_hist) {
  // * 2 for smaller and larger leaves, * 2 for split direction
  const int num_blocks = num_features_ * 4;
  FindBestSplitsForLeafKernel<<<num_blocks, 1>>>(cuda_hist_,
    cuda_num_total_bin_, cuda_feature_hist_offsets_,
    cuda_feature_mfb_offsets_, cuda_feature_default_bins_,
    cuda_feature_missing_type_, cuda_lambda_l1_, cuda_lambda_l2_,
    smaller_leaf_id, larger_leaf_id, smaller_leaf_gain, larger_leaf_gain,
    sum_gradients_in_smaller_leaf, sum_hessians_in_smaller_leaf, num_data_in_smaller_leaf, smaller_leaf_hist,
    sum_gradients_in_larger_leaf, sum_hessians_in_larger_leaf, num_data_in_larger_leaf, larger_leaf_hist,
    cuda_min_data_in_leaf_, cuda_min_sum_hessian_in_leaf_, cuda_min_gain_to_split_,

    cuda_best_split_threshold_, cuda_best_split_default_left_, cuda_best_split_gain_,
    cuda_best_split_left_sum_gradient_, cuda_best_split_left_sum_hessian_,
    cuda_best_split_left_count_, cuda_best_split_left_gain_, cuda_best_split_left_output_,
    cuda_best_split_right_sum_gradient_, cuda_best_split_right_sum_hessian_,
    cuda_best_split_right_count_, cuda_best_split_right_gain_, cuda_best_split_right_output_,
    cuda_best_split_found_);
}

__global__ void SyncBestSplitForLeafKernel(const int* smaller_leaf_index, const int* larger_leaf_index,
  const int* cuda_num_features, int* cuda_leaf_best_split_feature, uint8_t* cuda_leaf_best_split_default_left,
  uint32_t* cuda_leaf_best_split_threshold, double* cuda_leaf_best_split_gain,
  double* cuda_leaf_best_split_left_sum_gradient, double* cuda_leaf_best_split_left_sum_hessian,
  data_size_t* cuda_leaf_best_split_left_count, double* cuda_leaf_best_split_left_gain,
  double* cuda_leaf_best_split_left_output,
  double* cuda_leaf_best_split_right_sum_gradient, double* cuda_leaf_best_split_right_sum_hessian,
  data_size_t* cuda_leaf_best_split_right_count, double* cuda_leaf_best_split_right_gain,
  double* cuda_leaf_best_split_right_output,
  // input parameters
  const int* cuda_best_split_feature,
  const uint8_t* cuda_best_split_default_left,
  const uint32_t* cuda_best_split_threshold,
  const double* cuda_best_split_gain,
  const double* cuda_best_split_left_sum_gradient,
  const double* cuda_best_split_left_sum_hessian,
  const data_size_t* cuda_best_split_left_count,
  const double* cuda_best_split_left_gain,
  const double* cuda_best_split_left_output,
  const double* cuda_best_split_right_sum_gradient,
  const double* cuda_best_split_right_sum_hessian,
  const data_size_t* cuda_best_split_right_count,
  const double* cuda_best_split_right_gain,
  const double* cuda_best_split_right_output,
  const uint8_t* cuda_best_split_found) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const int num_features_ref = *cuda_num_features;
    const int smaller_leaf_index_ref = *smaller_leaf_index;
    const int larger_leaf_index_ref = *larger_leaf_index;

    double& smaller_leaf_best_gain = cuda_leaf_best_split_gain[smaller_leaf_index_ref];
    int& smaller_leaf_best_split_feature = cuda_leaf_best_split_feature[smaller_leaf_index_ref];
    uint8_t& smaller_leaf_best_split_default_left = cuda_leaf_best_split_default_left[smaller_leaf_index_ref];
    uint32_t& smaller_leaf_best_split_threshold = cuda_leaf_best_split_threshold[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_left_sum_gradient = cuda_leaf_best_split_left_sum_gradient[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_left_sum_hessian = cuda_leaf_best_split_left_sum_hessian[smaller_leaf_index_ref];
    data_size_t& smaller_leaf_best_split_left_count = cuda_leaf_best_split_left_count[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_left_gain = cuda_leaf_best_split_left_gain[smaller_leaf_index_ref]; 
    double& smaller_leaf_best_split_left_output = cuda_leaf_best_split_left_output[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_right_sum_gradient = cuda_leaf_best_split_right_sum_gradient[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_right_sum_hessian = cuda_leaf_best_split_right_sum_hessian[smaller_leaf_index_ref];
    data_size_t& smaller_leaf_best_split_right_count = cuda_leaf_best_split_right_count[smaller_leaf_index_ref];
    double& smaller_leaf_best_split_right_gain = cuda_leaf_best_split_right_gain[smaller_leaf_index_ref]; 
    double& smaller_leaf_best_split_right_output = cuda_leaf_best_split_right_output[smaller_leaf_index_ref];

    double& larger_leaf_best_gain = cuda_leaf_best_split_gain[larger_leaf_index_ref];
    int& larger_leaf_best_split_feature = cuda_leaf_best_split_feature[larger_leaf_index_ref];
    uint8_t& larger_leaf_best_split_default_left = cuda_leaf_best_split_default_left[larger_leaf_index_ref];
    uint32_t& larger_leaf_best_split_threshold = cuda_leaf_best_split_threshold[larger_leaf_index_ref];
    double& larger_leaf_best_split_left_sum_gradient = cuda_leaf_best_split_left_sum_gradient[larger_leaf_index_ref];
    double& larger_leaf_best_split_left_sum_hessian = cuda_leaf_best_split_left_sum_hessian[larger_leaf_index_ref];
    data_size_t& larger_leaf_best_split_left_count = cuda_leaf_best_split_left_count[larger_leaf_index_ref];
    double& larger_leaf_best_split_left_gain = cuda_leaf_best_split_left_gain[larger_leaf_index_ref];
    double& larger_leaf_best_split_left_output = cuda_leaf_best_split_left_output[larger_leaf_index_ref];
    double& larger_leaf_best_split_right_sum_gradient = cuda_leaf_best_split_right_sum_gradient[larger_leaf_index_ref];
    double& larger_leaf_best_split_right_sum_hessian = cuda_leaf_best_split_right_sum_hessian[larger_leaf_index_ref];
    data_size_t& larger_leaf_best_split_right_count = cuda_leaf_best_split_right_count[larger_leaf_index_ref];
    double& larger_leaf_best_split_right_gain = cuda_leaf_best_split_right_gain[larger_leaf_index_ref];
    double& larger_leaf_best_split_right_output = cuda_leaf_best_split_right_output[larger_leaf_index_ref];

    smaller_leaf_best_gain = kMinScore;
    larger_leaf_best_gain = kMinScore;
    int larger_leaf_offset = 2 * num_features_ref;
    for (int feature_index = 0; feature_index < num_features_ref; ++feature_index) {
      const int smaller_reverse_index = 2 * feature_index;
      const uint8_t smaller_reverse_found = cuda_best_split_found[smaller_reverse_index];
      if (smaller_reverse_found) {
        const double smaller_reverse_gain = cuda_best_split_gain[smaller_reverse_index];
        if (smaller_reverse_gain > smaller_leaf_best_gain) {
          //printf("reverse smaller leaf new best, feature_index = %d, split_gain = %f, default_left = %d, threshold = %d\n",
          //  feature_index, smaller_reverse_gain, cuda_best_split_default_left[smaller_reverse_index],
          //  cuda_best_split_threshold[smaller_reverse_index]);
          //printf("leaf index %d gain update to %f\n", smaller_leaf_index_ref, smaller_reverse_gain);
          smaller_leaf_best_gain = smaller_reverse_gain;
          smaller_leaf_best_split_feature = feature_index;
          smaller_leaf_best_split_default_left = cuda_best_split_default_left[smaller_reverse_index];
          smaller_leaf_best_split_threshold = cuda_best_split_threshold[smaller_reverse_index];
          smaller_leaf_best_split_left_sum_gradient = cuda_best_split_left_sum_gradient[smaller_reverse_index];
          smaller_leaf_best_split_left_sum_hessian = cuda_best_split_left_sum_hessian[smaller_reverse_index];
          smaller_leaf_best_split_left_count = cuda_best_split_left_count[smaller_reverse_index];
          smaller_leaf_best_split_left_gain = cuda_best_split_left_gain[smaller_reverse_index];
          //printf("leaf index %d split left gain update to %f\n", smaller_leaf_index_ref, smaller_leaf_best_split_left_gain);
          smaller_leaf_best_split_left_output = cuda_best_split_left_output[smaller_reverse_index];
          smaller_leaf_best_split_right_sum_gradient = cuda_best_split_right_sum_gradient[smaller_reverse_index];
          smaller_leaf_best_split_right_sum_hessian = cuda_best_split_right_sum_hessian[smaller_reverse_index];
          smaller_leaf_best_split_right_count = cuda_best_split_right_count[smaller_reverse_index];
          smaller_leaf_best_split_right_gain = cuda_best_split_right_gain[smaller_reverse_index];
          //printf("leaf index %d split right gain update to %f\n", smaller_leaf_index_ref, smaller_leaf_best_split_right_gain);
          smaller_leaf_best_split_right_output = cuda_best_split_right_output[smaller_reverse_index];
          /*printf("smaller_leaf_index = %d, smaller_leaf_best_gain = %f, smaller_leaf_best_split_left_sum_gradient = %f, smaller_leaf_best_split_left_sum_hessian = %f\n",
            smaller_leaf_index_ref, smaller_leaf_best_gain, smaller_leaf_best_split_left_sum_gradient, smaller_leaf_best_split_left_sum_hessian);
          printf("smaller_leaf_index = %d, smaller_leaf_best_gain = %f, smaller_leaf_best_split_right_sum_gradient = %f, smaller_leaf_best_split_right_sum_hessian = %f\n",
            smaller_leaf_index_ref, smaller_leaf_best_gain, smaller_leaf_best_split_right_sum_gradient, smaller_leaf_best_split_right_sum_hessian);*/
        }
      }
      const int smaller_non_reverse_index = 2 * feature_index + 1;
      const uint8_t smaller_non_reverse_found = cuda_best_split_found[smaller_non_reverse_index];
      if (smaller_non_reverse_found) {
        const double smaller_non_reverse_gain = cuda_best_split_gain[smaller_non_reverse_index];
        if (smaller_non_reverse_gain > smaller_leaf_best_gain) {
          //printf("non reverse smaller leaf new best, feature_index = %d, split_gain = %f, default_left = %d, threshold = %d\n",
          //  feature_index, smaller_non_reverse_gain, cuda_best_split_default_left[smaller_non_reverse_index],
          //  cuda_best_split_threshold[smaller_non_reverse_index]);
          //printf("leaf index %d gain update to %f\n", smaller_leaf_index_ref, smaller_non_reverse_gain);
          smaller_leaf_best_gain = smaller_non_reverse_gain;
          smaller_leaf_best_split_feature = feature_index;
          smaller_leaf_best_split_default_left = cuda_best_split_default_left[smaller_non_reverse_index];
          smaller_leaf_best_split_threshold = cuda_best_split_threshold[smaller_non_reverse_index];
          smaller_leaf_best_split_left_sum_gradient = cuda_best_split_left_sum_gradient[smaller_non_reverse_index];
          smaller_leaf_best_split_left_sum_hessian = cuda_best_split_left_sum_hessian[smaller_non_reverse_index];
          smaller_leaf_best_split_left_count = cuda_best_split_left_count[smaller_non_reverse_index];
          smaller_leaf_best_split_left_gain = cuda_best_split_left_gain[smaller_non_reverse_index];
          //printf("leaf index %d split left gain update to %f\n", smaller_leaf_index_ref, smaller_leaf_best_split_left_gain);
          smaller_leaf_best_split_left_output = cuda_best_split_left_output[smaller_non_reverse_index];
          smaller_leaf_best_split_right_sum_gradient = cuda_best_split_right_sum_gradient[smaller_non_reverse_index];
          smaller_leaf_best_split_right_sum_hessian = cuda_best_split_right_sum_hessian[smaller_non_reverse_index];
          smaller_leaf_best_split_right_count = cuda_best_split_right_count[smaller_non_reverse_index];
          smaller_leaf_best_split_right_gain = cuda_best_split_right_gain[smaller_non_reverse_index];
          //printf("leaf index %d split right gain update to %f\n", smaller_leaf_index_ref, smaller_leaf_best_split_right_gain);
          smaller_leaf_best_split_right_output = cuda_best_split_right_output[smaller_non_reverse_index];
          /*printf("smaller_leaf_index = %d, smaller_leaf_best_gain = %f, smaller_leaf_best_split_left_sum_gradient = %f, smaller_leaf_best_split_left_sum_hessian = %f\n",
            smaller_leaf_index_ref, smaller_leaf_best_gain, smaller_leaf_best_split_left_sum_gradient, smaller_leaf_best_split_left_sum_hessian);
          printf("smaller_leaf_index = %d, smaller_leaf_best_gain = %f, smaller_leaf_best_split_right_sum_gradient = %f, smaller_leaf_best_split_right_sum_hessian = %f\n",
            smaller_leaf_index_ref, smaller_leaf_best_gain, smaller_leaf_best_split_right_sum_gradient, smaller_leaf_best_split_right_sum_hessian);*/
        }
      }

      if (larger_leaf_index_ref >= 0) {
        const int larger_reverse_index = 2 * feature_index + larger_leaf_offset;
        const uint8_t larger_reverse_found = cuda_best_split_found[larger_reverse_index];
        if (larger_reverse_found) {
          const double larger_reverse_gain = cuda_best_split_gain[larger_reverse_index];
          if (larger_reverse_gain > larger_leaf_best_gain) {
            //printf("leaf index %d gain update to %f\n", larger_leaf_index_ref, larger_reverse_gain);
            larger_leaf_best_gain = larger_reverse_gain;
            larger_leaf_best_split_feature = feature_index;
            larger_leaf_best_split_default_left = cuda_best_split_default_left[larger_reverse_index];
            larger_leaf_best_split_threshold = cuda_best_split_threshold[larger_reverse_index];
            larger_leaf_best_split_left_sum_gradient = cuda_best_split_left_sum_gradient[larger_reverse_index];
            larger_leaf_best_split_left_sum_hessian = cuda_best_split_left_sum_hessian[larger_reverse_index];
            larger_leaf_best_split_left_count = cuda_best_split_left_count[larger_reverse_index];
            larger_leaf_best_split_left_gain = cuda_best_split_left_gain[larger_reverse_index];
            //printf("leaf index %d split left gain update to %f\n", larger_leaf_index_ref, larger_leaf_best_split_left_gain);
            larger_leaf_best_split_left_output = cuda_best_split_left_output[larger_reverse_index];
            larger_leaf_best_split_right_sum_gradient = cuda_best_split_right_sum_gradient[larger_reverse_index];
            larger_leaf_best_split_right_sum_hessian = cuda_best_split_right_sum_hessian[larger_reverse_index];
            larger_leaf_best_split_right_count = cuda_best_split_right_count[larger_reverse_index];
            larger_leaf_best_split_right_gain = cuda_best_split_right_gain[larger_reverse_index];
            //printf("leaf index %d split right gain update to %f\n", larger_leaf_index_ref, larger_leaf_best_split_right_gain);
            larger_leaf_best_split_right_output = cuda_best_split_right_output[larger_reverse_index];
            /*printf("larger_leaf_index = %d, larger_leaf_best_gain = %f, larger_leaf_best_split_left_sum_gradient = %f, larger_leaf_best_split_left_sum_hessian = %f\n",
              larger_leaf_index_ref, larger_leaf_best_gain, larger_leaf_best_split_left_sum_gradient, larger_leaf_best_split_left_sum_hessian);
            printf("larger_leaf_index = %d, larger_leaf_best_gain = %f, larger_leaf_best_split_right_sum_gradient = %f, larger_leaf_best_split_right_sum_hessian = %f\n",
              larger_leaf_index_ref, larger_leaf_best_gain, larger_leaf_best_split_right_sum_gradient, larger_leaf_best_split_right_sum_hessian);*/
          }
        }
        const int larger_non_reverse_index = 2 * feature_index + 1 + larger_leaf_offset;
        const uint8_t larger_non_reverse_found = cuda_best_split_found[larger_non_reverse_index];
        if (larger_non_reverse_found) {
          const double larger_non_reverse_gain = cuda_best_split_gain[larger_non_reverse_index];
          if (larger_non_reverse_gain > larger_leaf_best_gain) {
            //printf("leaf index %d gain update to %f\n", larger_leaf_index_ref, larger_non_reverse_gain);
            larger_leaf_best_gain = larger_non_reverse_gain;
            larger_leaf_best_split_feature = feature_index;
            larger_leaf_best_split_default_left = cuda_best_split_default_left[larger_non_reverse_index];
            larger_leaf_best_split_threshold = cuda_best_split_threshold[larger_non_reverse_index];
            larger_leaf_best_split_left_sum_gradient = cuda_best_split_left_sum_gradient[larger_non_reverse_index];
            larger_leaf_best_split_left_sum_hessian = cuda_best_split_left_sum_hessian[larger_non_reverse_index];
            larger_leaf_best_split_left_count = cuda_best_split_left_count[larger_non_reverse_index];
            larger_leaf_best_split_left_gain = cuda_best_split_left_gain[larger_non_reverse_index];
            //printf("leaf index %d split left gain update to %f\n", larger_leaf_index_ref, larger_leaf_best_split_left_gain);
            larger_leaf_best_split_left_output = cuda_best_split_left_output[larger_non_reverse_index];
            larger_leaf_best_split_right_sum_gradient = cuda_best_split_right_sum_gradient[larger_non_reverse_index];
            larger_leaf_best_split_right_sum_hessian = cuda_best_split_right_sum_hessian[larger_non_reverse_index];
            larger_leaf_best_split_right_count = cuda_best_split_right_count[larger_non_reverse_index];
            larger_leaf_best_split_right_gain = cuda_best_split_right_gain[larger_non_reverse_index];
            //printf("leaf index %d split right gain update to %f\n", larger_leaf_index_ref, larger_leaf_best_split_right_gain);
            larger_leaf_best_split_right_output = cuda_best_split_right_output[larger_non_reverse_index];
            /*printf("larger_leaf_index = %d, larger_leaf_best_gain = %f, larger_leaf_best_split_left_sum_gradient = %f, larger_leaf_best_split_left_sum_hessian = %f\n",
              larger_leaf_index_ref, larger_leaf_best_gain, larger_leaf_best_split_left_sum_gradient, larger_leaf_best_split_left_sum_hessian);
            printf("larger_leaf_index = %d, larger_leaf_best_gain = %f, larger_leaf_best_split_right_sum_gradient = %f, larger_leaf_best_split_right_sum_hessian = %f\n",
              larger_leaf_index_ref, larger_leaf_best_gain, larger_leaf_best_split_right_sum_gradient, larger_leaf_best_split_right_sum_hessian);*/
          }
        }
      }
    }
  }
}

void CUDABestSplitFinder::LaunchSyncBestSplitForLeafKernel(const int* smaller_leaf_index, const int* larger_leaf_index) {
  SyncBestSplitForLeafKernel<<<1, 1>>>(smaller_leaf_index, larger_leaf_index,
    cuda_num_features_, cuda_leaf_best_split_feature_, cuda_leaf_best_split_default_left_,
    cuda_leaf_best_split_threshold_, cuda_leaf_best_split_gain_,
    cuda_leaf_best_split_left_sum_gradient_, cuda_leaf_best_split_left_sum_hessian_,
    cuda_leaf_best_split_left_count_, cuda_leaf_best_split_left_gain_,
    cuda_leaf_best_split_left_output_,
    cuda_leaf_best_split_right_sum_gradient_, cuda_leaf_best_split_right_sum_hessian_,
    cuda_leaf_best_split_right_count_, cuda_leaf_best_split_right_gain_,
    cuda_leaf_best_split_right_output_,
    cuda_best_split_feature_,
    cuda_best_split_default_left_,
    cuda_best_split_threshold_,
    cuda_best_split_gain_,
    cuda_best_split_left_sum_gradient_,
    cuda_best_split_left_sum_hessian_,
    cuda_best_split_left_count_,
    cuda_best_split_left_gain_,
    cuda_best_split_left_output_,
    cuda_best_split_right_sum_gradient_,
    cuda_best_split_right_sum_hessian_,
    cuda_best_split_right_count_,
    cuda_best_split_right_gain_,
    cuda_best_split_right_output_,
    cuda_best_split_found_);
}

__global__ void FindBestFromAllSplitsKernel(const int* cuda_cur_num_leaves,
  const double* cuda_leaf_best_split_gain, int* out_best_leaf) {
  const int cuda_cur_num_leaves_ref = *cuda_cur_num_leaves;
  double best_gain = kMinScore;
  for (int leaf_index = 0; leaf_index < cuda_cur_num_leaves_ref; ++leaf_index) {
    const double leaf_best_gain = cuda_leaf_best_split_gain[leaf_index];
    //printf("cuda_leaf_best_split_gain[%d] = %f\n", leaf_index, leaf_best_gain);
    if (leaf_best_gain > best_gain) {
      best_gain = leaf_best_gain;
      *out_best_leaf = leaf_index;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestFromAllSplitsKernel(const int* cuda_cur_num_leaves) {
  FindBestFromAllSplitsKernel<<<1, 1>>>(cuda_cur_num_leaves, cuda_leaf_best_split_gain_, cuda_best_leaf_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
