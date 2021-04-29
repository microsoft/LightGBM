/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_best_split_finder.hpp"

namespace LightGBM {
/*
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
  double* output_gain,
  uint8_t* output_default_left,
  double* output_left_sum_gradients,
  double* output_left_sum_hessians,
  data_size_t* output_left_num_data,
  double* output_right_sum_gradients,
  double* output_right_sum_hessians,
  data_size_t* output_right_num_data) {

  double best_sum_left_gradient = NAN;
  double best_sum_left_hessian = NAN;
  double best_gain = kMinScore;
  data_size_t best_left_count = 0;
  uint32_t best_threshold = feature_num_bin;
  const double cnt_factor = num_data / sum_hessians;
  const bool use_l1 = lambda_l1 > 0.0f;
  const double min_gain_shift = parent_gain + min_gain_to_split;

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
          static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
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

      double sum_left_hessian = sum_hessian - sum_right_hessian;
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
        sum_left_gradient = sum_gradient;
        sum_left_hessian = sum_hessian - kEpsilon;
        left_count = num_data;
        for (int i = 0; i < feature_num_bin - feature_mfb_offset; ++i) {
          const auto grad = GET_GRAD(feature_hist_ptr, i);
          const auto hess = GET_HESS(feature_hist_ptr, i);
          data_size_t cnt =
              static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
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
        const hist_t* hess = GET_HESS(feature_hist_ptr, t);
        sum_left_hessian += hess;
        left_count += static_cast<data_size_t>(
            Common::RoundInt(hess * cnt_factor));
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

      double sum_right_hessian = sum_hessian - sum_left_hessian;
      // if sum hessian too small
      if (sum_right_hessian < min_sum_hessian_in_leaf) {
        break;
      }

      double sum_right_gradient = sum_gradient - sum_left_gradient;

      // current split gain
      double current_gain = GetSplitGains(
          sum_left_gradient, sum_left_hessian, sum_right_gradient,
          sum_right_hessian, lambda_l1, use_l1,
          lambda_l2);
      // gain with split is worse than without split
      if (current_gain <= min_gain_shift) {
        continue;
      }

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
}

__global__ void FindBestSplitsForLeafKernel(const hist_t* leaf_hist_ptr,
  const uint32_t* feature_hist_offsets, const uint8_t* feature_mfb_offsets, const uint32_t* feature_default_bins, 
  const uint8_t* feature_missing_types, const double* lambda_l1, const double* lambda_l2, const int* smaller_leaf_id,
  const int* larger_leaf_id, const double* smaller_leaf_gain, const double* larger_leaf_gain, const double* sum_gradients_in_smaller_leaf,
  const double* sum_hessians_in_smaller_leaf, const data_size_t* num_data_in_smaller_leaf,
  const double* sum_gradients_in_larger_leaf, const double* sum_hessians_in_larger_leaf,
  const data_size_t* num_data_in_larger_leaf, const data_size_t* min_data_in_leaf,
  const double* min_sum_hessian_in_leaf, const double*  min_gain_to_split,
  // output
  uint8_t* cuda_best_split_default_left, double* cuda_best_split_gain, double* cuda_best_split_left_sum_gradient,
  double* cuda_best_split_left_sum_hessian, data_size_t* cuda_best_split_left_count,
  double* cuda_best_split_right_sum_gradient, double* cuda_best_split_right_sum_hessian,
  data_size_t* cuda_best_split_right_count) {
  const unsigned int num_features = blockDim.x / 2;
  const unsigned int inner_feature_index = blockIdx.x % num_features;
  const unsigned int threadIdx = threadIdx.x;
  const unsigned int global_threadIdx = threadIdx + blockIdx.x * blockDim.x;
  const bool reverse = threadIdx == 0 ? true : false;
  const bool smaller_or_larger_leaf = static_cast<bool>(blockIdx.x / num_features);
  const int num_bin = feature_hist_offsets[inner_feature_index + 1] - feature_hist_offsets[inner_feature_index];
  const uint8_t missing_type = feature_missing_type[inner_feature_index];
  const int leaf_index = smaller_or_larger ? *smaller_leaf_id : *larger_leaf_id;
  const double parent_gain = smaller_or_larger ? *smaller_leaf_gain : *larger_leaf_gain;
  const double sum_gradients = smaller_or_larger ? *sum_gradients_in_smaller_leaf : *sum_gradients_in_larger_leaf;
  const double sum_hessians = smaller_or_larger ? *sum_hessians_in_smaller_leaf : *sum_hessians_in_larger_leaf;
  const double num_data_in_leaf = smaller_or_larger ? *num_data_in_smaller_leaf : *num_data_in_larger_leaf;
  double* out_left_sum_gradients = cuda_best_split_left_sum_gradient + global_threadIdx;
  double* out_left_sum_hessians = cuda_best_split_left_sum_hessian + global_threadIdx;
  double* out_right_sum_gradients = cuda_best_split_right_sum_gradient + global_threadIdx;
  double* out_right_sum_hessians = cuda_best_split_right_sum_hessian + global_threadIdx;
  data_size_t* out_left_num_data = cuda_best_split_left_count + global_threadIdx;
  data_size_t* out_right_num_data = cuda_best_split_right_count + global_threadIdx;
  uint8_t* out_default_left = cuda_best_split_default_left + global_threadIdx;
  double* out_gain = cuda_best_split_gain + global_threadIdx;
  if (num_bin > 2 && missing_type != 0) {
    if (missing_type == 1) {
      FindBestSplitsForLeafKernelInner(leaf_hist_ptr + leaf_index,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, *parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, true, false, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data);
    } else {
      FindBestSplitsForLeafKernelInner(leaf_hist_ptr + leaf_index,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, *parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, false, true, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data);
    }
  } else {
    if (reverse) {
      FindBestSplitsForLeafKernelInner(leaf_hist_ptr + leaf_index,
        num_bin, feature_mfb_offsets[inner_feature_index], feature_default_bins[inner_feature_index],
        feature_missing_types[inner_feature_index], *lambda_l1, *lambda_l2, *parent_gain,
        *min_data_in_leaf, *min_sum_hessian_in_leaf, *min_gain_to_split, sum_gradients, sum_hessians,
        num_data_in_leaf, reverse, true, false, out_gain, out_default_left,
        out_left_sum_gradients, out_left_sum_hessians, out_left_num_data,
        out_right_sum_gradients, out_right_sum_hessians, out_right_num_data);
    }
    if (missing_type == 2) {
      *out_default_left = 0;
    }
  }
}

void CUDABestSplitFinder::LaunchFindBestSplitsForLeafKernel(const int* smaller_leaf_id, const int* larger_leaf_id,
  const double* smaller_leaf_gain, const double* larger_leaf_gain, const double* sum_gradients_in_smaller_leaf,
  const double* sum_hessians_in_smaller_leaf, const data_size_t* num_data_in_smaller_leaf,
  const double* sum_gradients_in_larger_leaf, const double* sum_hessians_in_larger_leaf,
  const data_size_t* num_data_in_larger_leaf, const data_size_t* min_data_in_leaf,
  const double* min_sum_hessian_in_leaf) {
  const int leaf_id_ref = *leaf_id;
  const int num_total_bin_ref = *num_total_bin_;
  // * 2 for smaller and larger leaves, * 2 for split direction
  const int num_blocks = num_features_ * 2;
  FindBestSplitsForLeafKernel<<<num_blocks, 2>>>(cuda_hist_, cuda_feature_hist_offsets_,
    cuda_feature_mfb_offsets_, cuda_feature_default_bins_,
    cuda_feature_missing_type_, cuda_lambda_l1_,
    smaller_leaf_id, larger_leaf_id, smaller_leaf_gain, larger_leaf_gain,
    sum_gradients_in_smaller_leaf, sum_hessians_in_smaller_leaf, num_data_in_smaller_leaf,
    sum_gradients_in_larger_leaf, sum_hessians_in_larger_leaf, num_data_in_larger_leaf,
    cuda_min_data_in_leaf_, cuda_min_sum_hessian_in_leaf_, cuda_min_gain_to_split,

    cuda_best_split_default_left_, cuda_best_split_gain_,
    cuda_best_split_left_sum_gradient_, cuda_best_split_left_sum_hessian_,
    cuda_best_split_left_count_, cuda_best_split_right_sum_gradient_,
    cuda_best_split_right_sum_hessian_, cuda_best_split_right_count_);
}
*/
}  // namespace LightGBM

#endif  // USE_CUDA
