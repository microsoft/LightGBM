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
    leaf_map_ = std::vector<int>(num_data_, -1);
    contains_nan_ = std::vector<int8_t>(num_features_, 0);
    any_nan_ = false;
#pragma omp parallel for schedule(dynamic)
    for (int feat = 0; feat < num_features_; ++feat) {
      auto bin_mapper = train_data_->FeatureBinMapper(feat);
      if (bin_mapper->bin_type() == BinType::NumericalBin) {
        const double* feat_ptr = train_data_->raw_index(feat);
        for (int i = 0; i < num_data_; ++i) {
          if (std::isnan(feat_ptr[i])) {
            contains_nan_[feat] = 1;
            any_nan_ = true;
            break;
          }
        }
      }
    }
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
    Split(tree_prt, best_leaf, &left_leaf, &right_leaf);
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }

  bool has_nan = false;
  if (any_nan_) {
    for (int i = 0; i < tree->num_leaves() - 1 ; ++i) {
      if (contains_nan_[tree_prt->split_feature_inner(i)]) {
        has_nan = true;
        break;
      }
    }
  }

  if (has_nan) {
    LinearTreeLearner::CalculateLinear<true>(tree_prt);
  } else {
    LinearTreeLearner::CalculateLinear<false>(tree_prt);
  }

  Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

template<bool HAS_NAN>
void LinearTreeLearner::CalculateLinear(Tree* tree) {

  bool data_has_nan = false;
  std::fill(leaf_map_.begin(), leaf_map_.end(), -1);

  // map data to leaf number
  const data_size_t* ind = data_partition_->indices();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    data_size_t idx = data_partition_->leaf_begin(i);
    for (int j = 0; j < data_partition_->leaf_count(i); ++j) {
      leaf_map_[ind[idx + j]] = i;
    }
  }

  // calculate coefficients using the additive method described in https://arxiv.org/pdf/1802.05640.pdf
  // the coefficients vector is given by
  // - (X_T * H * X + lambda) ^ (-1) * (X_T * H * y + g_T X)
  // where:
  // X is the matrix where the first column is the feature values and the second is all ones,
  // y is the vector of current predictions
  // H is the diagonal matrix of the hessian,
  // lambda is the diagonal matrix with diagonal entries equal to the regularisation term linear_lambda
  // g is the vector of gradients
  // the subscript _T denotes the transpose

  // create array of numerical features pointers to raw data, and coefficient matrices, for each leaf
  std::vector<std::vector<int>> leaf_features;
  std::vector<std::vector<const double*>> raw_data_ptr;
  std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> XTHX;
  std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> XTg;

  int max_num_features = 0;
  for (int i = 0; i < tree->num_leaves(); ++i) {
    std::vector<int> raw_features = tree->branch_features(i);
    std::sort(raw_features.begin(), raw_features.end());
    auto new_end = std::unique(raw_features.begin(), raw_features.end());
    raw_features.erase(new_end, raw_features.end());
    std::vector<int> numerical_features;
    std::vector<const double*> data_ptr;
    for (int j = 0; j < raw_features.size(); ++j) {
      int feat = train_data_->InnerFeatureIndex(raw_features[j]);
      auto bin_mapper = train_data_->FeatureBinMapper(feat);
      if (bin_mapper->bin_type() == BinType::NumericalBin) {
        numerical_features.push_back(feat);
        data_ptr.push_back(train_data_->raw_index(feat));
      }
    }
    leaf_features.push_back(numerical_features);
    raw_data_ptr.push_back(data_ptr);
    // store only upper triangular half of matrix as an array, in row-major order
    XTHX.push_back(std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>
      ((numerical_features.size() + 1) * (numerical_features.size() + 2) / 2 + 8, 0));
    XTg.push_back(std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>
      (numerical_features.size() + 1 + 8, 0.0));
    if (numerical_features.size() > max_num_features) {
      max_num_features = numerical_features.size();
    }
  }

  std::vector<std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>>> XTHX_by_thread;
  std::vector<std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>>> XTg_by_thread;
  std::vector<std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>> curr_row; 
  for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
    XTHX_by_thread.push_back(XTHX);
    XTg_by_thread.push_back(XTg);
    curr_row.push_back(std::vector<double, Common::AlignmentAllocator<double, kAlignedSize>>(max_num_features + 1 + 8));  
  }
#pragma omp parallel for schedule(static) if (num_data_ > 1024)
  for (int i = 0; i < num_data_; ++i) {
    int tid = omp_get_thread_num();
    int leaf_num = leaf_map_[i];
    if (leaf_num < 0) {
      continue;
    }
    bool nan_found = false;
    data_size_t num_feat = leaf_features[leaf_num].size();
    for (int feat = 0; feat < num_feat;  ++feat) {
      double val = raw_data_ptr[leaf_num][feat][i];
      if (HAS_NAN) {
        if (std::isnan(val)) {
          nan_found = true;
          break;
        }
      }
      curr_row[tid][feat] = val;
    }
    if (HAS_NAN) {
      if (nan_found) {
        continue;
      }
    }
    curr_row[tid][num_feat] = 1;
    double h = hessians_[i];
    double g = gradients_[i];
    int j = 0;
    for (int feat1 = 0; feat1 < num_feat + 1; ++feat1) {
      double f1_val = curr_row[tid][feat1];
      for (int feat2 = feat1; feat2 < num_feat + 1; ++feat2) {
        XTHX_by_thread[tid][leaf_num][j] += f1_val * curr_row[tid][feat2] * h;
        ++j;
      }
      XTg_by_thread[tid][leaf_num][feat1] += f1_val * g;
    }
  }
  // aggregate results from different threads
  for (int tid = 0; tid < OMP_NUM_THREADS(); ++tid) {
    for (int leaf_num = 0; leaf_num < tree->num_leaves(); ++leaf_num) {
      int num_feat = leaf_features[leaf_num].size();
      for (int j = 0; j < (num_feat + 1) * (num_feat + 2) / 2; ++j) {
        XTHX[leaf_num][j] += XTHX_by_thread[tid][leaf_num][j];
      }
      for (int feat1 = 0; feat1 < num_feat + 1; ++feat1) {
        XTg[leaf_num][feat1] += XTg_by_thread[tid][leaf_num][feat1];
      }
    }
  }
  // copy into eigen matrices and solve
 #pragma omp parallel for schedule(static)
  for (int leaf_num = 0; leaf_num < tree->num_leaves(); ++leaf_num) {
    int num_feat = leaf_features[leaf_num].size();
    Eigen::MatrixXd XTHX_mat(num_feat + 1, num_feat + 1);
    Eigen::MatrixXd XTg_mat(num_feat + 1, 1);
    int j = 0;
    for (int feat1 = 0; feat1 < num_feat + 1; ++feat1) {
      for (int feat2 = feat1; feat2 < num_feat + 1; ++feat2) {
        XTHX_mat(feat1, feat2) = XTHX[leaf_num][j];
        XTHX_mat(feat2, feat1) = XTHX_mat(feat1, feat2);
        if ((feat1 == feat2) && (feat1 < num_feat)){
          XTHX_mat(feat1, feat2) += config_->linear_lambda;
        }
        ++j;
      }
      XTg_mat(feat1) = XTg[leaf_num][feat1];
    }
    Eigen::MatrixXd coeffs = - XTHX_mat.fullPivLu().inverse() * XTg_mat;
    // remove features with very small coefficients
    std::vector<double> coeffs_vec;
    std::vector<int> features_new;
    for (int i = 0; i < leaf_features[leaf_num].size(); ++i) {
      if (coeffs(i) < -kZeroThreshold || coeffs(i) > kZeroThreshold) {
        coeffs_vec.push_back(coeffs(i));
        int feat = leaf_features[leaf_num][i];
        features_new.push_back(feat);
        if (contains_nan_[feat]) {
          data_has_nan = true;
        }
      }
    }
    // update the tree properties
    tree->SetLeafFeaturesInner(leaf_num, features_new);
    std::vector<int> features_raw(features_new.size());
    for (int i = 0; i < features_new.size(); ++i) {
      features_raw[i] = train_data_->RealFeatureIndex(features_new[i]);
    }
    tree->SetLeafFeatures(leaf_num, features_raw);
    tree->SetLeafCoeffs(leaf_num, coeffs_vec);
    tree->SetLeafConst(leaf_num, coeffs(num_feat));
    tree->SetNan(data_has_nan);
  }
  }

}
