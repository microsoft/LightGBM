#include "cegb_tree_learner.h"

namespace LightGBM {

void CEGBTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian){
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  coupled_features_used.clear();
  coupled_features_used.resize(train_data->num_features());
}


void CEGBTreeLearner::Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf){
  const SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  const int inner_feature_index = train_data_->InnerFeatureIndex(best_split_info.feature);
  coupled_features_used[inner_feature_index] = true;
  SerialTreeLearner::Split(tree, best_leaf, left_leaf, right_leaf);
}

void CEGBTreeLearner::ResetConfig(const Config* config){
  tradeoff = config->cegb_tradeoff;
  coupled_feature_penalty = config->cegb_penalty_feature_coupled;
  SerialTreeLearner::ResetConfig(config);
}

void CEGBTreeLearner::ResetTrainingData(const Dataset* data){
  throw std::runtime_error("Cannot reset training data for CEGB");
}

void CEGBTreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract){
  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  OMP_INIT_EX();
  // find splits
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) { continue; }
    const int tid = omp_get_thread_num();
    SplitInfo smaller_split;
    train_data_->FixHistogram(feature_index,
                              smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                              smaller_leaf_splits_->num_data_in_leaf(),
                              smaller_leaf_histogram_array_[feature_index].RawData());
    int real_fidx = train_data_->RealFeatureIndex(feature_index);
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      smaller_leaf_splits_->min_constraint(),
      smaller_leaf_splits_->max_constraint(),
      &smaller_split);
    smaller_split.feature = real_fidx;
    if(!coupled_features_used[feature_index]){
      smaller_split.gain -= tradeoff*coupled_feature_penalty[real_fidx];
    }
    if (smaller_split > smaller_best[tid]) {
      smaller_best[tid] = smaller_split;
    }
    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                larger_leaf_splits_->num_data_in_leaf(),
                                larger_leaf_histogram_array_[feature_index].RawData());
    }
    SplitInfo larger_split;
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->min_constraint(),
      larger_leaf_splits_->max_constraint(),
      &larger_split);
    larger_split.feature = real_fidx;
    if(!coupled_features_used[feature_index]){
      larger_split.gain -= tradeoff*coupled_feature_penalty[real_fidx];
    }
    if (larger_split > larger_best[tid]) {
      larger_best[tid] = larger_split;
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->LeafIndex();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }
}
} // namespace LightGBM
