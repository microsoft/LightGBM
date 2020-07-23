/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "linear_tree_learner.h"

#include <Eigen/Dense>
#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

namespace LightGBM {

  void LinearTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
    SerialTreeLearner::Init(train_data, is_constant_hessian);
    curr_pred_ = std::vector<double>(num_data_, 0);
    is_nan_ = std::vector<int8_t>(num_data_, 0);
  }


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

  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves, true, true));
  tree->SetLinear(true);
  auto tree_prt = tree.get();
  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  int init_splits = ForceSplits(tree_prt, &left_leaf, &right_leaf, &cur_depth);

  std::fill(curr_pred_.begin(), curr_pred_.end(), 0);
  std::fill(is_nan_.begin(), is_nan_.end(), 0);

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
    std::vector<int> parent_features = tree_prt->LeafFeaturesInner(best_leaf);
    std::vector<double> parent_coeffs = tree_prt->LeafCoeffs(best_leaf);
    double parent_const = tree_prt->LeafConst(best_leaf);
    int raw_feat = best_leaf_SplitInfo.feature;
    Split(tree_prt, best_leaf, &left_leaf, &right_leaf);
    CalculateLinear(tree_prt, left_leaf, raw_feat, parent_features, parent_coeffs, parent_const,
                    best_leaf_SplitInfo.left_sum_gradient, best_leaf_SplitInfo.left_sum_hessian);
    CalculateLinear(tree_prt, right_leaf, raw_feat, parent_features, parent_coeffs, parent_const,
                    best_leaf_SplitInfo.right_sum_gradient, best_leaf_SplitInfo.right_sum_hessian);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

void LinearTreeLearner::CalculateLinear(Tree* tree, int leaf_num, int raw_feat,
                                        const std::vector<int>& parent_features,
                                        const std::vector<double>& parent_coeffs,
                                        const double& parent_const,
                                        const double& sum_grad, const double& sum_hess) {
  int inner_feat = train_data_->InnerFeatureIndex(raw_feat);
  auto bin_mapper = train_data_->FeatureBinMapper(inner_feat);

  // calculate coefficients using the additive method described in https://arxiv.org/pdf/1802.05640.pdf
  // the coefficients vector is given by
  // - (X_T * H * X + lambda) ^ (-1) * (X_T * H * y + g_T X)
  // where X is the matrix where the first column is the feature values and the second is all ones,
  // y is the vector of current predictions
  // H is the diagonal matrix of the hessian,
  // lambda is the diagonal matrix with diagonal entries equal to the regularisation term linear_lambda
  // g is the vector of gradients
  // the subscript _T denotes the transpose
  int idx = data_partition_->leaf_begin(leaf_num);
  const data_size_t* ind = data_partition_->indices();
  const double* feat_ptr = train_data_->raw_index(raw_feat);
  int prev_feat = tree->LeafNewFeature(leaf_num);
  const double* prev_feat_ptr = nullptr;
  double prev_coeff = 0;
  double prev_const = tree->LeafNewConst(leaf_num);
  if (prev_feat > -1) {
    prev_feat_ptr = train_data_->raw_index(prev_feat);
    prev_coeff = tree->LeafNewCoeff(leaf_num);
  }
  data_size_t num_data = data_partition_->leaf_count(leaf_num);
  double leaf_output = tree->LeafOutput(leaf_num);
  auto features = parent_features;
  auto coeffs = parent_coeffs;
  double constant_term;
  bool can_solve = true;
  std::function<double(const double*, int, double, double)> update_prev;
  if (prev_feat > -1) {
    update_prev = [](const double* prev_feat_ptr, int row_idx, double prev_coeff, double prev_const) {
      return prev_feat_ptr[row_idx] * prev_coeff + prev_const;
    };
  } else {
    update_prev = [](const double* prev_feat_ptr, int row_idx, double prev_coeff, double prev_const) {
      return prev_const;
    };
  }

  if (bin_mapper->bin_type() != BinType::NumericalBin) {
    can_solve = false;
  } else {
    double XTHX_00 = 0, XTHX_01 = 0, XTHX_11 = 0;
    double XTHy_0 = 0, XTHy_1 = 0;
    double gTX_0 = 0, gTX_1 = 0;
#pragma omp parallel for schedule(static, 512) reduction(+:XTHX_00,XTHX_01,XTHX_11,XTHy_0,XTHy_1,gTX_0,gTX_1) if (num_data > 1024)
    for (int i = 0; i < num_data; ++i) {
      int row_idx = ind[idx + i];
      double x = feat_ptr[row_idx];
      if (std::isnan(x) || std::isinf(x)) {
        is_nan_[row_idx] = 1;
        continue;
      }
      double h = hessians_[row_idx];
      double g = gradients_[row_idx];
      double y;
      if (is_nan_[row_idx]) {
        y = leaf_output;
      } else {
        y = curr_pred_[row_idx];
        y += update_prev(prev_feat_ptr, row_idx, prev_coeff, prev_const);
      }
      curr_pred_[row_idx] = y;
      XTHX_00 += x * x * h;
      XTHX_01 += h * x;
      XTHX_11 += h;
      XTHy_0 += x * h * y;
      XTHy_1 += h * y;
      gTX_0 += g * x;
      gTX_1 += g;
    }
    XTHX_01 += config_->linear_lambda;
    double det = XTHX_00 * XTHX_11 - XTHX_01 * XTHX_01;
    if (det > -kEpsilon && det < kEpsilon) {
      can_solve = false;
    } else {
      double feat_coeff = - (XTHX_11 * (XTHy_0 + gTX_0) - XTHX_01 * (XTHy_1 + gTX_1)) / det;
      constant_term = - (- XTHX_01 * (XTHy_0 + gTX_0) + XTHX_00 * (XTHy_1 + gTX_1)) / det;
      // add the new coeff to the parent coeffs
      if (feat_coeff < -kZeroThreshold || feat_coeff > kZeroThreshold) {
        tree->SetLeafNewFeature(leaf_num, raw_feat);
        tree->SetLeafNewCoeff(leaf_num, feat_coeff);
        bool feature_exists = false;
        for (int feat_num = 0; feat_num < parent_features.size(); ++feat_num) {
          if (parent_features[feat_num] == raw_feat) {
            coeffs[feat_num] += feat_coeff;
            feature_exists = true;
            break;
          }
        }
        if (!feature_exists && (feat_coeff < -kZeroThreshold || feat_coeff > kZeroThreshold)) {
          features.push_back(raw_feat);
          coeffs.push_back(feat_coeff);
        }
      }
    }
  }
  if (!can_solve) {
    // just calculate an adjustment to the constant term
    double hy = 0;
    double linear_lambda = config_->linear_lambda;
#pragma omp parallel for schedule(static, 512) reduction(+:hy) if (num_data > 1024)
    for (int i = 0; i < num_data; ++i) {
      int row_idx = ind[idx + i];
      double y;
      if (is_nan_[row_idx]) {
        y = leaf_output;
      } else {
        y += update_prev(prev_feat_ptr, row_idx, prev_coeff, prev_const);
      }
      curr_pred_[row_idx] = y;
      hy += hessians_[row_idx] * y;
    }
    if (sum_hess > kEpsilon) {
      constant_term = -(hy + sum_grad) / (sum_hess + linear_lambda);
    } else {
      constant_term = 0;
    }
    
    // update curr_pred
#pragma omp parallel for schedule(static, 512) if (num_data > 1024)
    for (int i = 0; i < num_data; ++i) {
      int row_idx = ind[idx + i];
      if (is_nan_[row_idx]) {
        curr_pred_[row_idx] = leaf_output;
      } else {
        curr_pred_[row_idx] += constant_term;
      }
    }
  }
  // update the tree properties
  tree->SetLeafNewConst(leaf_num, constant_term);
  tree->SetLeafFeaturesInner(leaf_num, features);
  std::vector<int> features_raw(features.size());
  for (int i = 0; i < features.size(); ++i) {
    features_raw[i] = train_data_->RealFeatureIndex(features[i]);
  }
  tree->SetLeafFeatures(leaf_num, features_raw);
  tree->SetLeafCoeffs(leaf_num, coeffs);
  tree->SetLeafConst(leaf_num, constant_term + parent_const);
  }
}
