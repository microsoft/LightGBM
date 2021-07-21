/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

template <bool USE_INDICES>
__global__ void AddPredictionToScoreKernel(
  // dataset information
  const data_size_t num_data,
  void* const* cuda_data_by_column,
  const uint8_t* cuda_column_bit_type,
  const uint32_t* cuda_feature_min_bin,
  const uint32_t* cuda_feature_max_bin,
  const uint32_t* cuda_feature_offset,
  const uint32_t* cuda_feature_default_bin,
  const uint32_t* cuda_feature_most_freq_bin,
  const int* cuda_feature_to_column,
  const data_size_t* cuda_used_indices,
  // tree information
  const uint32_t* cuda_threshold_in_bin,
  const int8_t* cuda_decision_type,
  const int* cuda_split_feature_inner,
  const int* cuda_left_child,
  const int* cuda_right_child,
  const double* cuda_leaf_value,
  // output
  double* score) {
  const data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const data_size_t data_index = USE_INDICES ? cuda_used_indices[inner_data_index] : inner_data_index;
  if (data_index < num_data) {
    int node = 0;
    while (node >= 0) {
      const int split_feature_inner = cuda_split_feature_inner[node];
      const int column = cuda_feature_to_column[split_feature_inner];
      const uint32_t default_bin = cuda_feature_default_bin[split_feature_inner];
      const uint32_t most_freq_bin = cuda_feature_most_freq_bin[split_feature_inner];
      const uint32_t max_bin = cuda_feature_max_bin[split_feature_inner];
      const uint32_t min_bin = cuda_feature_min_bin[split_feature_inner];
      const uint32_t offset = cuda_feature_offset[split_feature_inner];
      const uint8_t column_bit_type = cuda_column_bit_type[column];
      uint32_t bin = 0;
      if (column_bit_type == 8) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint8_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == 16) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint16_t*>(cuda_data_by_column[column]))[data_index]);
      } else if (column_bit_type == 32) {
        bin = static_cast<uint32_t>((reinterpret_cast<const uint32_t*>(cuda_data_by_column[column]))[data_index]);
      }
      if (bin >= min_bin && bin <= max_bin) {
        bin = bin - min_bin + offset;
      } else {
        bin = most_freq_bin;
      }
      const int8_t decision_type = cuda_decision_type[node];
      const uint32_t threshold_in_bin = cuda_threshold_in_bin[node];
      const int8_t missing_type = ((decision_type >> 2) & 3);
      const bool default_left = ((decision_type & kDefaultLeftMask) > 0);
      if ((missing_type == 1 && bin == default_bin) || (missing_type == 2 && bin == max_bin)) {
        if (default_left) {
          node = cuda_left_child[node];
        } else {
          node = cuda_right_child[node];
        }
      } else {
        if (bin <= threshold_in_bin) {
          node = cuda_left_child[node];
        } else {
          node = cuda_right_child[node];
        }
      }
    }
    score[data_index] += cuda_leaf_value[~node];
  }
}

void CUDATree::LaunchAddPredictionToScoreKernel(
  const Dataset* data,
  const data_size_t* used_data_indices,
  data_size_t num_data,
  double* score) const {
  const CUDAColumnData* cuda_column_data = data->cuda_column_data();
  if (cuda_column_data == nullptr) {
    Log::Warning("error cuda_column_data is nullptr");
  }
  const int num_blocks = (num_data + num_threads_per_block_add_prediction_to_score_ - 1) / num_threads_per_block_add_prediction_to_score_;
  if (used_data_indices == nullptr) {
    AddPredictionToScoreKernel<false><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      nullptr,
      // tree information
      cuda_threshold_in_bin_,
      cuda_decision_type_,
      cuda_split_feature_inner_,
      cuda_left_child_,
      cuda_right_child_,
      cuda_leaf_value_,
      // output
      score);
  } else {
    AddPredictionToScoreKernel<true><<<num_blocks, num_threads_per_block_add_prediction_to_score_>>>(
      // dataset information
      num_data,
      cuda_column_data->cuda_data_by_column(),
      cuda_column_data->cuda_column_bit_type(),
      cuda_column_data->cuda_feature_min_bin(),
      cuda_column_data->cuda_feature_max_bin(),
      cuda_column_data->cuda_feature_offset(),
      cuda_column_data->cuda_feature_default_bin(),
      cuda_column_data->cuda_feature_most_freq_bin(),
      cuda_column_data->cuda_feature_to_column(),
      used_data_indices,
      // tree information
      cuda_threshold_in_bin_,
      cuda_decision_type_,
      cuda_split_feature_inner_,
      cuda_left_child_,
      cuda_right_child_,
      cuda_leaf_value_,
      // output
      score);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

__global__ void ShrinkageKernel(const double rate, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] *= rate;
  }
}

void CUDATree::LaunchShrinkageKernel(const double rate) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  ShrinkageKernel<<<num_blocks, num_threads_per_block>>>(rate, cuda_leaf_value_, num_leaves_);
}

}  // namespace LightGBM
