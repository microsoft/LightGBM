/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear):
Tree(max_leaves, track_branch_features, is_linear),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDAMemory();
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {}

void CUDATree::InitCUDAMemory() {
  AllocateCUDAMemoryOuter<int>(&cuda_left_child_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_right_child_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_split_feature_inner_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_split_feature_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_leaf_depth_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<int>(&cuda_leaf_parent_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemoryOuter<uint32_t>(&cuda_threshold_in_bin_,
                                    static_cast<size_t>(max_leaves_),
                                    __FILE__,
                                    __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_threshold_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<int8_t>(&cuda_decision_type_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_leaf_value_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_internal_weight_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_internal_value_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_leaf_weight_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_leaf_count_,
                                       static_cast<size_t>(max_leaves_),
                                       __FILE__,
                                       __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_internal_count_,
                                       static_cast<size_t>(max_leaves_),
                                       __FILE__,
                                       __LINE__);
  AllocateCUDAMemoryOuter<float>(&cuda_split_gain_,
                                       static_cast<size_t>(max_leaves_),
                                       __FILE__,
                                       __LINE__);
  SetCUDAMemoryOuter<int>(cuda_leaf_parent_, 0, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_leaf_value_, 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<double>(cuda_leaf_weight_, 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemoryOuter<int>(cuda_leaf_parent_, -1, 1, __FILE__, __LINE__);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
}

void CUDATree::InitCUDA() {
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_left_child_,
                                    left_child_.data(),
                                    left_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_right_child_,
                                    right_child_.data(),
                                    right_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_split_feature_inner_,
                                    split_feature_inner_.data(),
                                    split_feature_inner_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_split_feature_,
                                    split_feature_.data(),
                                    split_feature_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_threshold_in_bin_,
                                    threshold_in_bin_.data(),
                                    threshold_in_bin_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_threshold_,
                                    threshold_.data(),
                                    threshold_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int8_t>(&cuda_decision_type_,
                                    decision_type_.data(),
                                    decision_type_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_leaf_value_,
                                    leaf_value_.data(),
                                    leaf_value_.size(),
                                    __FILE__,
                                    __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

int CUDATree::Split(const int leaf_index,
           const int real_feature_index,
           const double real_threshold,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info) {
  LaunchSplitKernel(leaf_index, real_feature_index, real_threshold, missing_type, cuda_split_info);
  ++num_leaves_;
  return num_leaves_ - 1;
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const {
  LaunchAddPredictionToScoreKernel(data, nullptr, num_data, score);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const {
  // TODO(shiyu1994): used_data_indices should reside on GPU
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, score);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

void CUDATree::ToHost() {
  left_child_.resize(max_leaves_ - 1);
  right_child_.resize(max_leaves_ - 1);
  split_feature_inner_.resize(max_leaves_ - 1);
  split_feature_.resize(max_leaves_ - 1);
  threshold_in_bin_.resize(max_leaves_ - 1);
  threshold_.resize(max_leaves_ - 1);
  decision_type_.resize(max_leaves_ - 1, 0);
  split_gain_.resize(max_leaves_ - 1);
  leaf_parent_.resize(max_leaves_);
  leaf_value_.resize(max_leaves_);
  leaf_weight_.resize(max_leaves_);
  leaf_count_.resize(max_leaves_);
  internal_value_.resize(max_leaves_ - 1);
  internal_weight_.resize(max_leaves_ - 1);
  internal_count_.resize(max_leaves_ - 1);
  leaf_depth_.resize(max_leaves_);

  const size_t num_leaves_size = static_cast<size_t>(num_leaves_);
  CopyFromCUDADeviceToHostOuter<int>(left_child_.data(), cuda_left_child_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int>(right_child_.data(), cuda_right_child_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int>(split_feature_inner_.data(), cuda_split_feature_inner_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int>(split_feature_.data(), cuda_split_feature_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<uint32_t>(threshold_in_bin_.data(), cuda_threshold_in_bin_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(threshold_.data(), cuda_threshold_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int8_t>(decision_type_.data(), cuda_decision_type_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<float>(split_gain_.data(), cuda_split_gain_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int>(leaf_parent_.data(), cuda_leaf_parent_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(leaf_value_.data(), cuda_leaf_value_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(leaf_weight_.data(), cuda_leaf_weight_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<data_size_t>(leaf_count_.data(), cuda_leaf_count_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(internal_value_.data(), cuda_internal_value_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(internal_weight_.data(), cuda_internal_weight_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<data_size_t>(internal_count_.data(), cuda_internal_count_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<int>(leaf_depth_.data(), cuda_leaf_depth_, num_leaves_size, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM
