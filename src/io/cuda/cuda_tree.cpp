/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear,
  const int gpu_device_id, const bool has_categorical_feature):
Tree(max_leaves, track_branch_features, is_linear),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  if (gpu_device_id >= 0) {
    SetCUDADevice(gpu_device_id, __FILE__, __LINE__);
  } else {
    SetCUDADevice(0, __FILE__, __LINE__);
  }
  if (has_categorical_feature) {
    cuda_cat_boundaries_.Resize(max_leaves);
    cuda_cat_boundaries_inner_.Resize(max_leaves);
  }
  InitCUDAMemory();
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {
  DeallocateCUDAMemory<int>(&cuda_left_child_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_right_child_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_split_feature_inner_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_split_feature_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_leaf_depth_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int>(&cuda_leaf_parent_, __FILE__, __LINE__);
  DeallocateCUDAMemory<uint32_t>(&cuda_threshold_in_bin_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_threshold_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_internal_weight_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_internal_value_, __FILE__, __LINE__);
  DeallocateCUDAMemory<int8_t>(&cuda_decision_type_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_leaf_value_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_leaf_count_, __FILE__, __LINE__);
  DeallocateCUDAMemory<double>(&cuda_leaf_weight_, __FILE__, __LINE__);
  DeallocateCUDAMemory<data_size_t>(&cuda_internal_count_, __FILE__, __LINE__);
  DeallocateCUDAMemory<float>(&cuda_split_gain_, __FILE__, __LINE__);
  gpuAssert(cudaStreamDestroy(cuda_stream_), __FILE__, __LINE__);
}

void CUDATree::InitCUDAMemory() {
  AllocateCUDAMemory<int>(&cuda_left_child_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<int>(&cuda_right_child_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<int>(&cuda_split_feature_inner_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<int>(&cuda_split_feature_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<int>(&cuda_leaf_depth_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<int>(&cuda_leaf_parent_,
                               static_cast<size_t>(max_leaves_),
                               __FILE__,
                               __LINE__);
  AllocateCUDAMemory<uint32_t>(&cuda_threshold_in_bin_,
                                    static_cast<size_t>(max_leaves_),
                                    __FILE__,
                                    __LINE__);
  AllocateCUDAMemory<double>(&cuda_threshold_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<int8_t>(&cuda_decision_type_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<double>(&cuda_leaf_value_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<double>(&cuda_internal_weight_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<double>(&cuda_internal_value_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<double>(&cuda_leaf_weight_,
                             static_cast<size_t>(max_leaves_),
                             __FILE__,
                             __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_leaf_count_,
                                  static_cast<size_t>(max_leaves_),
                                  __FILE__,
                                  __LINE__);
  AllocateCUDAMemory<data_size_t>(&cuda_internal_count_,
                                       static_cast<size_t>(max_leaves_),
                                       __FILE__,
                                       __LINE__);
  AllocateCUDAMemory<float>(&cuda_split_gain_,
                                 static_cast<size_t>(max_leaves_),
                                 __FILE__,
                                 __LINE__);
  SetCUDAMemory<double>(cuda_leaf_value_, 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemory<double>(cuda_leaf_weight_, 0.0f, 1, __FILE__, __LINE__);
  SetCUDAMemory<int>(cuda_leaf_parent_, -1, 1, __FILE__, __LINE__);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::InitCUDA() {
  InitCUDAMemoryFromHostMemory<int>(&cuda_left_child_,
                                    left_child_.data(),
                                    left_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_right_child_,
                                    right_child_.data(),
                                    right_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_split_feature_inner_,
                                    split_feature_inner_.data(),
                                    split_feature_inner_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_split_feature_,
                                    split_feature_.data(),
                                    split_feature_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_threshold_in_bin_,
                                    threshold_in_bin_.data(),
                                    threshold_in_bin_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_threshold_,
                                    threshold_.data(),
                                    threshold_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_leaf_depth_,
                                    leaf_depth_.data(),
                                    leaf_depth_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int8_t>(&cuda_decision_type_,
                                       decision_type_.data(),
                                       decision_type_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_internal_weight_,
                                       internal_weight_.data(),
                                       internal_weight_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_internal_value_,
                                       internal_value_.data(),
                                       internal_value_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_internal_count_,
                                       internal_count_.data(),
                                       internal_count_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<data_size_t>(&cuda_leaf_count_,
                                       leaf_count_.data(),
                                       leaf_count_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<float>(&cuda_split_gain_,
                                       split_gain_.data(),
                                       split_gain_.size(),
                                       __FILE__,
                                       __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_leaf_value_,
                                    leaf_value_.data(),
                                    leaf_value_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<double>(&cuda_leaf_weight_,
                                    leaf_weight_.data(),
                                    leaf_weight_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemory<int>(&cuda_leaf_parent_,
                                    leaf_parent_.data(),
                                    leaf_parent_.size(),
                                    __FILE__,
                                    __LINE__);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

int CUDATree::Split(const int leaf_index,
           const int real_feature_index,
           const double real_threshold,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info) {
  LaunchSplitKernel(leaf_index, real_feature_index, real_threshold, missing_type, cuda_split_info);
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  ++num_leaves_;
  return num_leaves_ - 1;
}

int CUDATree::SplitCategorical(const int leaf_index,
           const int real_feature_index,
           const MissingType missing_type,
           const CUDASplitInfo* cuda_split_info,
           uint32_t* cuda_bitset,
           size_t cuda_bitset_len,
           uint32_t* cuda_bitset_inner,
           size_t cuda_bitset_inner_len) {
  LaunchSplitCategoricalKernel(leaf_index, real_feature_index,
    missing_type, cuda_split_info,
    cuda_bitset_len, cuda_bitset_inner_len);
  cuda_bitset_.PushBack(cuda_bitset, cuda_bitset_len);
  cuda_bitset_inner_.PushBack(cuda_bitset_inner, cuda_bitset_inner_len);
  ++num_leaves_;
  ++num_cat_;
  RecordBranchFeatures(leaf_index, num_leaves_, real_feature_index);
  return num_leaves_ - 1;
}

void CUDATree::RecordBranchFeatures(const int left_leaf_index,
                                    const int right_leaf_index,
                                    const int real_feature_index) {
  if (track_branch_features_) {
    branch_features_[right_leaf_index] = branch_features_[left_leaf_index];
    branch_features_[right_leaf_index].push_back(real_feature_index);
    branch_features_[left_leaf_index].push_back(real_feature_index);
  }
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const {
  LaunchAddPredictionToScoreKernel(data, nullptr, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    const data_size_t* used_data_indices,
                                    data_size_t num_data, double* score) const {
  LaunchAddPredictionToScoreKernel(data, used_data_indices, num_data, score);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

inline void CUDATree::AddBias(double val) {
  Tree::AddBias(val);
  LaunchAddBiasKernel(val);
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
  CopyFromCUDADeviceToHost<int>(left_child_.data(), cuda_left_child_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(right_child_.data(), cuda_right_child_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_inner_.data(), cuda_split_feature_inner_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(split_feature_.data(), cuda_split_feature_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<uint32_t>(threshold_in_bin_.data(), cuda_threshold_in_bin_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(threshold_.data(), cuda_threshold_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int8_t>(decision_type_.data(), cuda_decision_type_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<float>(split_gain_.data(), cuda_split_gain_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_parent_.data(), cuda_leaf_parent_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(leaf_weight_.data(), cuda_leaf_weight_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(leaf_count_.data(), cuda_leaf_count_, num_leaves_size, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_value_.data(), cuda_internal_value_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<double>(internal_weight_.data(), cuda_internal_weight_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<data_size_t>(internal_count_.data(), cuda_internal_count_, num_leaves_size - 1, __FILE__, __LINE__);
  CopyFromCUDADeviceToHost<int>(leaf_depth_.data(), cuda_leaf_depth_, num_leaves_size, __FILE__, __LINE__);

  if (num_cat_ > 0) {
    cuda_cat_boundaries_inner_.Resize(num_cat_ + 1);
    cuda_cat_boundaries_.Resize(num_cat_ + 1);
    cat_boundaries_ = cuda_cat_boundaries_.ToHost();
    cat_boundaries_inner_ = cuda_cat_boundaries_inner_.ToHost();
    cat_threshold_ = cuda_bitset_.ToHost();
    cat_threshold_inner_ = cuda_bitset_inner_.ToHost();
  }

  SynchronizeCUDADevice(__FILE__, __LINE__);
}

void CUDATree::SyncLeafOutputFromHostToCUDA() {
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_, leaf_value_.data(), leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::SyncLeafOutputFromCUDAToHost() {
  CopyFromCUDADeviceToHost<double>(leaf_value_.data(), cuda_leaf_value_, leaf_value_.size(), __FILE__, __LINE__);
}

void CUDATree::AsConstantTree(double val) {
  Tree::AsConstantTree(val);
  CopyFromHostToCUDADevice<double>(cuda_leaf_value_, &val, 1, __FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
