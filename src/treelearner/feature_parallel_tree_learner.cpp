#include "parallel_tree_learner.h"

#include <cstring>

#include <vector>

namespace LightGBM {

FeatureParallelTreeLearner::FeatureParallelTreeLearner(const TreeConfig* tree_config)
  :SerialTreeLearner(tree_config) {
}

FeatureParallelTreeLearner::~FeatureParallelTreeLearner() {

}
void FeatureParallelTreeLearner::Init(const Dataset* train_data) {
  SerialTreeLearner::Init(train_data);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  input_buffer_.resize(sizeof(SplitInfo) * 2);
  output_buffer_.resize(sizeof(SplitInfo) * 2);
}



void FeatureParallelTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  // get feature partition
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  for (int i = 0; i < train_data_->num_features(); ++i) {
    if (is_feature_used_[i]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(i);
      num_bins_distributed[cur_min_machine] += train_data_->FeatureAt(i)->num_bin();
      is_feature_used_[i] = false;
    }
  }
  // get local used features
  for (auto fid : feature_distribution[rank_]) {
    is_feature_used_[fid] = true;
  }
}

void FeatureParallelTreeLearner::FindBestSplitsForLeaves() {
  int smaller_best_feature = -1, larger_best_feature = -1;
  SplitInfo smaller_best, larger_best;
  // get best split at smaller leaf
  std::vector<double> gains;
  for (size_t i = 0; i < smaller_leaf_splits_->BestSplitPerFeature().size(); ++i) {
    gains.push_back(smaller_leaf_splits_->BestSplitPerFeature()[i].gain);
  }
  smaller_best_feature = static_cast<int>(ArrayArgs<double>::ArgMax(gains));
  smaller_best = smaller_leaf_splits_->BestSplitPerFeature()[smaller_best_feature];
  // get best split at larger leaf
  if (larger_leaf_splits_->LeafIndex() >= 0) {
    gains.clear();
    for (size_t i = 0; i < larger_leaf_splits_->BestSplitPerFeature().size(); ++i) {
      gains.push_back(larger_leaf_splits_->BestSplitPerFeature()[i].gain);
    }
    larger_best_feature = static_cast<int>(ArrayArgs<double>::ArgMax(gains));
    larger_best = larger_leaf_splits_->BestSplitPerFeature()[larger_best_feature];
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
  best_split_per_leaf_[smaller_leaf_splits_->LeafIndex()] = smaller_best;
  if (larger_leaf_splits_->LeafIndex() >= 0) {
    best_split_per_leaf_[larger_leaf_splits_->LeafIndex()] = larger_best;
  }
}

}  // namespace LightGBM
