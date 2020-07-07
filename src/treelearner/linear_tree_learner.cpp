/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "linear_tree_learner.h"

#include <Eigen/Dense>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

namespace LightGBM {


Tree* LinearTreeLearner::Train(const score_t* gradients, const score_t *hessians) {
  Common::FunctionTimer fun_timer("SerialTreeLearner::Train", global_timer);
  gradients_ = gradients;
  hessians_ = hessians;
  int num_threads = OMP_NUM_THREADS();
  if (share_state_->num_threads != num_threads && share_state_->num_threads > 0) {
    Log::Warning(
        "Detected that num_threads changed during training (from %d to %d), "
        "it may cause unexpected errors.",
        share_state_->num_threads, num_threads);
  }
  share_state_->num_threads = num_threads;

  // some initial works before training
  BeforeTrain();

  bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves, track_branch_features, true));
  auto tree_prt = tree.get();
  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  int init_splits = ForceSplits(tree_prt, &left_leaf, &right_leaf, &cur_depth);

  for (int split = init_splits; split < config_->num_leaves - 1; ++split) {
    // some initial works before finding best split
    if (BeforeFindBestSplit(tree_prt, left_leaf, right_leaf)) {
      // find best threshold for every feature
      FindBestSplits(tree_prt);
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
      break;
    }
    // split tree with best leaf
    Split(tree_prt, best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  CalculateLinear(tree_prt);
  tree_prt->SetLinear(true);

  Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

void LinearTreeLearner::CalculateLinear(Tree* tree) {

  /*
  auto split_features = std::vector<int>();
  for (int j = 0; j < train_data_->num_features(); ++j) {
    auto bin_mapper = train_data_->FeatureBinMapper(j);
    if (bin_mapper->bin_type() == BinType::NumericalBin) {
      split_features.push_back(j);
    }
  }
  */

  OMP_INIT_EX();
#pragma omp parallel for schedule(static)
  for (int leaf_num = 0; leaf_num < tree->num_leaves(); ++leaf_num) {
    OMP_LOOP_EX_BEGIN();

  
    // get split features
    int node = tree->leaf_parent(leaf_num);
    auto split_features = std::vector<int>();
    while (node >= 0) {
      int feat = tree->split_feature_inner(node);
      int feat_raw = tree->split_feature(node);
      auto bin_mapper = train_data_->FeatureBinMapper(feat);
      if (bin_mapper->bin_type() == BinType::NumericalBin) {
        split_features.push_back(feat);
      }
      node = tree->internal_parent(node);
    }
    std::sort(split_features.begin(), split_features.end());
    auto new_end = std::unique(split_features.begin(), split_features.end());
    split_features.erase(new_end, split_features.end());
    
    // get matrix of feature values
    const data_size_t* ind = data_partition_->indices();
    int idx = data_partition_->leaf_begin(leaf_num);
    int num_data = data_partition_->leaf_count(leaf_num);
    // refer to Eq 3 of https://arxiv.org/pdf/1802.05640.pdf
    auto nan_row = std::vector<bool>(num_data, false);
    auto num_nan_rows = 0;
    Eigen::MatrixXd X_full(num_data, split_features.size() + 1);  // matrix of feature values
    for (int feat_num = 0; feat_num < split_features.size(); ++feat_num) {
      int feat = split_features[feat_num];
      for (int i = 0; i < num_data; ++i) {
        int row_idx = ind[idx + i];
        double val = train_data_->get_data(row_idx, feat);
        if (nan_row[i]) {
          continue;
        }
        if (isnan(val) || isinf(val)) {
          nan_row[i] = true;
          ++num_nan_rows;
        } else {
          X_full(i, feat_num) = val;
        }
      }
    }
    int num_valid_data = num_data - num_nan_rows;
    if (num_valid_data < split_features.size() + 1) {
      tree->SetLeafConst(leaf_num, tree->LeafOutput(leaf_num));
    } else {

      Eigen::MatrixXd X(num_valid_data, split_features.size() + 1);
      int curr_row = 0;
      for (int i = 0; i < num_data; ++i) {
        if (!nan_row[i]) {
          X.row(curr_row) = X_full.row(i);
          ++curr_row;
        }
      }
      // final column is all ones
      for (int i = 0; i < num_valid_data; ++i) {
        X(i, split_features.size()) = 1;
      }
      Eigen::VectorXd h(num_valid_data, 1);
      curr_row = 0;
      for (int i = 0; i < num_data; ++i) {
        if (!nan_row[i]) {
          h(curr_row, 0) = hessians_[ind[idx + i]];
          ++curr_row;
        }
      }
      Eigen::MatrixXd XTHX(split_features.size() + 1, split_features.size() + 1);
      XTHX = X.transpose() * h.asDiagonal() * X;
      for (int i = 0; i < split_features.size(); ++i) {
        XTHX(i, i) += config_->linear_lambda;
      }
      Eigen::MatrixXd grad(num_valid_data, 1);
      curr_row = 0;
      for (int i = 0; i < num_data; ++i) {
        if (!nan_row[i]) {
          grad(curr_row) = gradients_[ind[idx + i]];
          ++curr_row;
        }
      }
      Eigen::MatrixXd coeffs = - XTHX.fullPivLu().inverse() * X.transpose() * grad;
      // remove features with very small coefficients
      std::vector<double> coeffs_vec;
      std::vector<int> split_features_new;
      for (int i = 0; i < split_features.size(); ++i) {
        if (coeffs(i) < -kZeroThreshold || coeffs(i) > kZeroThreshold) {
          coeffs_vec.push_back(coeffs(i));
          split_features_new.push_back(split_features[i]);
        }
      }
      // update the tree properties
      tree->SetLeafFeaturesInner(leaf_num, split_features_new);
      std::vector<int> split_features_raw(split_features_new.size());
      for (int i = 0; i < split_features_new.size(); ++i) {
        split_features_raw[i] = train_data_->RealFeatureIndex(split_features_new[i]);
      }
      tree->SetLeafFeatures(leaf_num, split_features_raw);
      tree->SetLeafCoeffs(leaf_num, coeffs_vec);
      double const_term = coeffs(split_features.size());
      tree->SetLeafConst(leaf_num, const_term);
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

}  // namespace LightGBM
