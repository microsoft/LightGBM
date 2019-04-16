/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <vector>

#include "parallel_tree_learner.h"

namespace LightGBM {


template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::FeatureParallelTreeLearner(const Config* config)
  :TREELEARNER_T(config) {
}

template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::~FeatureParallelTreeLearner() {
}

template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  input_buffer_.resize((sizeof(SplitInfo) + sizeof(uint32_t) * this->config_->max_cat_threshold) * 2);
  output_buffer_.resize((sizeof(SplitInfo) + sizeof(uint32_t) * this->config_->max_cat_threshold) * 2);
}


template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // get feature partition
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) { continue; }
    if (this->is_feature_used_[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      num_bins_distributed[cur_min_machine] += this->train_data_->FeatureNumBin(inner_feature_index);
      this->is_feature_used_[inner_feature_index] = false;
    }
  }
  // get local used features
  for (auto fid : feature_distribution[rank_]) {
    this->is_feature_used_[fid] = true;
  }
}

template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  TREELEARNER_T::FindBestSplitsFromHistograms(is_feature_used, use_subtract);
  SplitInfo smaller_best_split, larger_best_split;
  // get best split at smaller leaf
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }
  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);
  // update best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()] = smaller_best_split;
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()] = larger_best_split;
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class FeatureParallelTreeLearner<GPUTreeLearner>;
template class FeatureParallelTreeLearner<SerialTreeLearner>;
}  // namespace LightGBM
