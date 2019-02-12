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
} // namespace LightGBM
