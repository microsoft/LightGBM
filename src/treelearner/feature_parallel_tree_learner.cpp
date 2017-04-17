#include "parallel_tree_learner.h"

#include <cstring>

#include <vector>

namespace LightGBM {


template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::FeatureParallelTreeLearner(const TreeConfig* tree_config)
  :TREELEARNER_T(tree_config) {
}

template <typename TREELEARNER_T>
FeatureParallelTreeLearner<TREELEARNER_T>::~FeatureParallelTreeLearner() {

}

template <typename TREELEARNER_T>
void FeatureParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  input_buffer_.resize(sizeof(SplitInfo) * 2);
  output_buffer_.resize(sizeof(SplitInfo) * 2);
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
void FeatureParallelTreeLearner<TREELEARNER_T>::FindBestSplitsForLeaves() {
  SplitInfo smaller_best, larger_best;
  // get best split at smaller leaf
  smaller_best = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }
  // sync global best info
  std::memcpy(input_buffer_.data(), &smaller_best, sizeof(SplitInfo));
  std::memcpy(input_buffer_.data() + sizeof(SplitInfo), &larger_best, sizeof(SplitInfo));

  Network::Allreduce(input_buffer_.data(), sizeof(SplitInfo) * 2, sizeof(SplitInfo),
                     output_buffer_.data(), &SplitInfo::MaxReducer);
  // copy back
  std::memcpy(&smaller_best, output_buffer_.data(), sizeof(SplitInfo));
  std::memcpy(&larger_best, output_buffer_.data() + sizeof(SplitInfo), sizeof(SplitInfo));
  // update best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()] = smaller_best;
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()] = larger_best;
  }
}

// instantiate template classes, otherwise linker cannot find the code
template class FeatureParallelTreeLearner<GPUTreeLearner>;
template class FeatureParallelTreeLearner<SerialTreeLearner>;
}  // namespace LightGBM
