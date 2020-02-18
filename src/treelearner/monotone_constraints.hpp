/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_

#include <limits>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "split_info.hpp"

namespace LightGBM {

struct CostEfficientGradientBoosting;

struct LearnerState {
  const Config *config_;
  const Dataset *train_data_;
  const Tree *tree;
  std::unique_ptr<CostEfficientGradientBoosting> &cegb_;

  LearnerState(const Config *config_, const Dataset *train_data_,
               const Tree *tree,
               std::unique_ptr<CostEfficientGradientBoosting> &cegb_)
      : config_(config_), train_data_(train_data_), tree(tree), cegb_(cegb_) {};
};

struct ConstraintEntry {
  double min = -std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::max();

  ConstraintEntry() {}

  void Reset() {
    min = -std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::max();
  }

  void UpdateMin(double new_min) { min = std::max(new_min, min); }

  void UpdateMax(double new_max) { max = std::min(new_max, max); }

  bool UpdateMinAndReturnBoolIfChanged(double new_min) {
    if (new_min > min) {
      min = new_min;
      return true;
    }
    return false;
  }

  bool UpdateMaxAndReturnBoolIfChanged(double new_max) {
    if (new_max < max) {
      max = new_max;
      return true;
    }
    return false;
  }
};

template <typename ConstraintEntry>
class LeafConstraints {
 public:
  explicit LeafConstraints(int num_leaves) : num_leaves_(num_leaves) {
    entries_.resize(num_leaves_);
    leaves_to_update.reserve(num_leaves_);
  }

  void Reset() {
    for (auto& entry : entries_) {
      entry.Reset();
    }
  }

  void UpdateConstraintsWithMid(bool is_numerical_split, int leaf, int new_leaf,
                         int8_t monotone_type, double right_output,
                         double left_output) {
    entries_[new_leaf] = entries_[leaf];
    if (is_numerical_split) {
      double mid = (left_output + right_output) / 2.0f;
      if (monotone_type < 0) {
        entries_[leaf].UpdateMin(mid);
        entries_[new_leaf].UpdateMax(mid);
      } else if (monotone_type > 0) {
        entries_[leaf].UpdateMax(mid);
        entries_[new_leaf].UpdateMin(mid);
      }
    }
  }

  void UpdateConstraintsWithOutputs(bool is_numerical_split, int leaf,
                                    int new_leaf, int8_t monotone_type,
                                    double right_output, double left_output) {
    entries_[new_leaf] = entries_[leaf];
    if (is_numerical_split) {
      if (monotone_type < 0) {
        entries_[leaf].UpdateMin(right_output);
        entries_[new_leaf].UpdateMax(left_output);
      } else if (monotone_type > 0) {
        entries_[leaf].UpdateMax(right_output);
        entries_[new_leaf].UpdateMin(left_output);
      }
    }
  }

  void GoUpToFindLeavesToUpdate(int node_idx, std::vector<int> &features,
                                std::vector<uint32_t> &thresholds,
                                std::vector<bool> &is_in_right_split,
                                int split_feature, const SplitInfo &split_info,
                                uint32_t split_threshold,
                                std::vector<SplitInfo> &best_split_per_leaf_,
                                LearnerState &learner_state) {

    int parent_idx = learner_state.tree->node_parent(node_idx);
    if (parent_idx != -1) {
      int inner_feature = learner_state.tree->split_feature_inner(parent_idx);
      int8_t monotone_type =
          learner_state.train_data_->FeatureMonotone(inner_feature);
      bool is_right_split =
          learner_state.tree->right_child(parent_idx) == node_idx;
      bool split_contains_new_information = true;
      bool is_split_numerical =
          learner_state.train_data_->FeatureBinMapper(inner_feature)
              ->bin_type() == BinType::NumericalBin;

      // only branches containing leaves that are contiguous to the original
      // leaf need to be updated
      if (is_split_numerical) {
        for (unsigned int i = 0; i < features.size(); ++i) {
          if (features[i] == inner_feature &&
              (is_in_right_split[i] == is_right_split)) {
            split_contains_new_information = false;
            break;
          }
        }
      }

      if (split_contains_new_information) {
        if (monotone_type != 0) {
          int left_child_idx = learner_state.tree->left_child(parent_idx);
          int right_child_idx = learner_state.tree->right_child(parent_idx);
          bool left_child_is_curr_idx = (left_child_idx == node_idx);
          int node_idx_to_pass =
              (left_child_is_curr_idx) ? right_child_idx : left_child_idx;
          bool take_min = (monotone_type < 0) ? left_child_is_curr_idx
                                              : !left_child_is_curr_idx;

          GoDownToFindLeavesToUpdate(
              node_idx_to_pass, features, thresholds, is_in_right_split,
              take_min, split_feature, split_info, true, true, split_threshold,
              best_split_per_leaf_,
              learner_state);
        }

        is_in_right_split.push_back(
            learner_state.tree->right_child(parent_idx) == node_idx);
        thresholds.push_back(learner_state.tree->threshold_in_bin(parent_idx));
        features.push_back(learner_state.tree->split_feature_inner(parent_idx));
      }

      if (parent_idx != 0) {
        LeafConstraints::GoUpToFindLeavesToUpdate(
            parent_idx, features, thresholds, is_in_right_split, split_feature,
            split_info, split_threshold, best_split_per_leaf_,
            learner_state);
      }
    }
  }

  void GoDownToFindLeavesToUpdate(
      int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, int maximum,
      int split_feature, const SplitInfo &split_info, bool use_left_leaf,
      bool use_right_leaf, uint32_t split_threshold,
      std::vector<SplitInfo> &best_split_per_leaf_,
      LearnerState &learner_state) {

    if (node_idx < 0) {
      int leaf_idx = ~node_idx;

      // if a leaf is at max depth then there is no need to update it
      int max_depth = learner_state.config_->max_depth;
      if (learner_state.tree->leaf_depth(leaf_idx) >= max_depth &&
          max_depth > 0) {
        return;
      }

      // splits that are not to be used shall not be updated
      if (best_split_per_leaf_[leaf_idx].gain == kMinScore) {
        return;
      }

      std::pair<double, double> min_max_constraints;
      bool something_changed = false;
      if (use_right_leaf && use_left_leaf) {
        min_max_constraints =
            std::minmax(split_info.right_output, split_info.left_output);
      } else if (use_right_leaf && !use_left_leaf) {
        min_max_constraints = std::pair<double, double>(
            split_info.right_output, split_info.right_output);
      } else {
        min_max_constraints = std::pair<double, double>(split_info.left_output,
                                                        split_info.left_output);
      }

#ifdef DEBUG
      if (maximum) {
        CHECK(min_max_constraints.first >=
              learner_state.tree->LeafOutput(leaf_idx));
      } else {
        CHECK(min_max_constraints.second <=
              learner_state.tree->LeafOutput(leaf_idx));
      }
#endif

      if (!maximum) {
        something_changed = entries_[leaf_idx].UpdateMinAndReturnBoolIfChanged(
            min_max_constraints.second);
      } else {
        something_changed = entries_[leaf_idx].UpdateMaxAndReturnBoolIfChanged(
            min_max_constraints.first);
      }
      if (!something_changed) {
        return;
      }
      leaves_to_update.push_back(leaf_idx);

    } else {
      // check if the children are contiguous with the original leaf
      std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
          learner_state.tree, node_idx, features, thresholds, is_in_right_split,
          learner_state.train_data_);
      int inner_feature = learner_state.tree->split_feature_inner(node_idx);
      uint32_t threshold = learner_state.tree->threshold_in_bin(node_idx);
      bool is_split_numerical =
          learner_state.train_data_->FeatureBinMapper(inner_feature)
              ->bin_type() == BinType::NumericalBin;
      bool use_left_leaf_for_update = true;
      bool use_right_leaf_for_update = true;
      if (is_split_numerical && inner_feature == split_feature) {
        if (threshold >= split_threshold) {
          use_left_leaf_for_update = false;
        }
        if (threshold <= split_threshold) {
          use_right_leaf_for_update = false;
        }
      }

      if (keep_going_left_right.first) {
        GoDownToFindLeavesToUpdate(
            learner_state.tree->left_child(node_idx), features, thresholds,
            is_in_right_split, maximum, split_feature, split_info,
            use_left_leaf, use_right_leaf_for_update && use_right_leaf,
            split_threshold, best_split_per_leaf_, learner_state);
      }
      if (keep_going_left_right.second) {
        GoDownToFindLeavesToUpdate(
            learner_state.tree->right_child(node_idx), features, thresholds,
            is_in_right_split, maximum, split_feature, split_info,
            use_left_leaf_for_update && use_left_leaf, use_right_leaf,
            split_threshold, best_split_per_leaf_, learner_state);
      }
    }
  }

  void GoUpToFindLeavesToUpdate(int node_idx, int split_feature,
                                const SplitInfo &split_info,
                                uint32_t split_threshold,
                                std::vector<SplitInfo> &best_split_per_leaf_,
                                LearnerState &learner_state) {
    int depth = learner_state.tree->leaf_depth(
                    ~learner_state.tree->left_child(node_idx)) - 1;

    std::vector<int> features;
    std::vector<uint32_t> thresholds;
    std::vector<bool> is_in_right_split;

    features.reserve(depth);
    thresholds.reserve(depth);
    is_in_right_split.reserve(depth);

    GoUpToFindLeavesToUpdate(
        node_idx, features, thresholds, is_in_right_split, split_feature,
        split_info, split_threshold, best_split_per_leaf_,
        learner_state);
  }

  std::pair<bool, bool> ShouldKeepGoingLeftRight(
      const Tree *tree, int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, const Dataset *train_data_) {
    int inner_feature = tree->split_feature_inner(node_idx);
    uint32_t threshold = tree->threshold_in_bin(node_idx);
    bool is_split_numerical =
      train_data_->FeatureBinMapper(inner_feature)->bin_type() == BinType::NumericalBin;

    bool keep_going_right = true;
    bool keep_going_left = true;
    // left and right nodes are checked to find out if they are contiguous with the original leaf
    // if so the algorithm should keep going down these nodes to update constraints
    if (is_split_numerical) {
      for (unsigned int i = 0; i < features.size(); ++i) {
        if (features[i] == inner_feature) {
          if (threshold >= thresholds[i] && !is_in_right_split[i]) {
            keep_going_right = false;
          }
          if (threshold <= thresholds[i] && is_in_right_split[i]) {
            keep_going_left = false;
          }
        }
      }
    }
    return std::pair<bool, bool>(keep_going_left, keep_going_right);
  }

  const ConstraintEntry& Get(int leaf_idx) const { return entries_[leaf_idx]; }

  std::vector<int> leaves_to_update;

 private:
  int num_leaves_;
  std::vector<ConstraintEntry> entries_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
