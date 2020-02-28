/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "split_info.hpp"

namespace LightGBM {

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

class LeafConstraintsBase {
 public:
  virtual ~LeafConstraintsBase(){};
  virtual const ConstraintEntry &Get(int leaf_idx) const = 0;
  virtual void Reset() = 0;
  virtual void BeforeSplit(const Tree *tree, int leaf, int new_leaf,
                           int8_t monotone_type) = 0;
  virtual std::vector<int> Update(
      const Tree *tree, bool is_numerical_split,
      int leaf, int new_leaf, int8_t monotone_type, double right_output,
      double left_output, int split_feature, const SplitInfo &split_info,
      const std::vector<SplitInfo> &best_split_per_leaf) = 0;

  inline static LeafConstraintsBase *Create(const Config *config, int num_leaves);
};

class BasicLeafConstraints : public LeafConstraintsBase {
 public:
  explicit BasicLeafConstraints(int num_leaves) : num_leaves_(num_leaves) {
    entries_.resize(num_leaves_);
  }

  void Reset() override {
    for (auto &entry : entries_) {
      entry.Reset();
    }
  }

  void BeforeSplit(const Tree *, int, int, int8_t) override {}

  std::vector<int> Update(const Tree *,
                          bool is_numerical_split, int leaf, int new_leaf,
                          int8_t monotone_type, double right_output,
                          double left_output, int, const SplitInfo &,
                          const std::vector<SplitInfo> &) override {
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
    return std::vector<int>();
  }

  const ConstraintEntry &Get(int leaf_idx) const { return entries_[leaf_idx]; }

 private:
  int num_leaves_;
  std::vector<ConstraintEntry> entries_;
};

class FastLeafConstraints : public BasicLeafConstraints {
 public:
  explicit FastLeafConstraints(const Config *config, int num_leaves)
      : BasicLeafConstraints(num_leaves), config_(config) {
    leaf_is_in_monotone_subtree_.resize(num_leaves_, false);
    node_parent_.resize(num_leaves_, 0);
    leaves_to_update_.reserve(num_leaves_);
  }

  void Reset() override {
    BasicLeafConstraints::Reset();
    std::fill_n(leaf_is_in_monotone_subtree_.begin(), num_leaves_, false);
    std::fill_n(node_parent_.begin(), num_leaves_, 0);
    leaves_to_update_.clear();
  }

  void BeforeSplit(const Tree *tree, int leaf, int new_leaf,
                   int8_t monotone_type) override {
    if (monotone_type != 0 || leaf_is_in_monotone_subtree_[leaf]) {
      leaf_is_in_monotone_subtree_[leaf] = true;
      leaf_is_in_monotone_subtree_[new_leaf] = true;
    }
    node_parent_[new_leaf - 1] = tree->leaf_parent(leaf);
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

  std::vector<int> Update(const Tree *tree, bool is_numerical_split, int leaf,
                          int new_leaf, int8_t monotone_type,
                          double right_output, double left_output,
                          int split_feature, const SplitInfo &split_info,
                          const std::vector<SplitInfo> &best_split_per_leaf) {
    leaves_to_update_.clear();
    UpdateConstraintsWithOutputs(is_numerical_split, leaf, new_leaf,
                                 monotone_type, right_output, left_output);

    GoUpToFindLeavesToUpdate(tree, tree->leaf_parent(new_leaf), split_feature,
                             split_info, split_info.threshold,
                             best_split_per_leaf);
    return leaves_to_update_;
  }

  void GoUpToFindLeavesToUpdate(
      const Tree *tree, int node_idx, std::vector<int> &features,
      std::vector<uint32_t> &thresholds, std::vector<bool> &is_in_right_split,
      int split_feature, const SplitInfo &split_info, uint32_t split_threshold,
      const std::vector<SplitInfo> &best_split_per_leaf) {
    int parent_idx = node_parent_[node_idx];
    if (parent_idx != -1) {
      int inner_feature = tree->split_feature_inner(parent_idx);
      int feature = tree->split_feature(parent_idx);
      int8_t monotone_type = config_->monotone_constraints[feature];
      bool is_right_split = tree->right_child(parent_idx) == node_idx;
      bool split_contains_new_information = true;
      bool is_split_numerical = tree->IsNumericalSplit(node_idx);

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
          int left_child_idx = tree->left_child(parent_idx);
          int right_child_idx = tree->right_child(parent_idx);
          bool left_child_is_curr_idx = (left_child_idx == node_idx);
          int node_idx_to_pass =
              (left_child_is_curr_idx) ? right_child_idx : left_child_idx;
          bool take_min = (monotone_type < 0) ? left_child_is_curr_idx
                                              : !left_child_is_curr_idx;

          GoDownToFindLeavesToUpdate(tree, node_idx_to_pass, features,
                                     thresholds, is_in_right_split, take_min,
                                     split_feature, split_info, true, true,
                                     split_threshold, best_split_per_leaf);
        }

        is_in_right_split.push_back(tree->right_child(parent_idx) == node_idx);
        thresholds.push_back(tree->threshold_in_bin(parent_idx));
        features.push_back(tree->split_feature_inner(parent_idx));
      }

      if (parent_idx != 0) {
        GoUpToFindLeavesToUpdate(tree, parent_idx, features, thresholds,
                                 is_in_right_split, split_feature, split_info,
                                 split_threshold, best_split_per_leaf);
      }
    }
  }

  void GoDownToFindLeavesToUpdate(
      const Tree *tree, int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, int maximum,
      int split_feature, const SplitInfo &split_info, bool use_left_leaf,
      bool use_right_leaf, uint32_t split_threshold,
      const std::vector<SplitInfo> &best_split_per_leaf) {
    if (node_idx < 0) {
      int leaf_idx = ~node_idx;

      // splits that are not to be used shall not be updated, included leaf at
      // max depth
      if (best_split_per_leaf[leaf_idx].gain == kMinScore) {
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
      leaves_to_update_.push_back(leaf_idx);

    } else {
      // check if the children are contiguous with the original leaf
      std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
          tree, node_idx, features, thresholds, is_in_right_split);
      int inner_feature = tree->split_feature_inner(node_idx);
      uint32_t threshold = tree->threshold_in_bin(node_idx);
      bool is_split_numerical = tree->IsNumericalSplit(node_idx);
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
        GoDownToFindLeavesToUpdate(tree, tree->left_child(node_idx), features,
                                   thresholds, is_in_right_split, maximum,
                                   split_feature, split_info, use_left_leaf,
                                   use_right_leaf_for_update && use_right_leaf,
                                   split_threshold, best_split_per_leaf);
      }
      if (keep_going_left_right.second) {
        GoDownToFindLeavesToUpdate(
            tree, tree->right_child(node_idx), features, thresholds,
            is_in_right_split, maximum, split_feature, split_info,
            use_left_leaf_for_update && use_left_leaf, use_right_leaf,
            split_threshold, best_split_per_leaf);
      }
    }
  }

  void GoUpToFindLeavesToUpdate(
      const Tree *tree, int node_idx, int split_feature,
      const SplitInfo &split_info, uint32_t split_threshold,
      const std::vector<SplitInfo> &best_split_per_leaf) {
    int depth = tree->leaf_depth(~tree->left_child(node_idx)) - 1;

    std::vector<int> features;
    std::vector<uint32_t> thresholds;
    std::vector<bool> is_in_right_split;

    features.reserve(depth);
    thresholds.reserve(depth);
    is_in_right_split.reserve(depth);

    GoUpToFindLeavesToUpdate(tree, node_idx, features, thresholds,
                             is_in_right_split, split_feature, split_info,
                             split_threshold, best_split_per_leaf);
  }

  std::pair<bool, bool> ShouldKeepGoingLeftRight(
      const Tree *tree, int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split) {
    int inner_feature = tree->split_feature_inner(node_idx);
    uint32_t threshold = tree->threshold_in_bin(node_idx);
    bool is_split_numerical = tree->IsNumericalSplit(node_idx);

    bool keep_going_right = true;
    bool keep_going_left = true;
    // left and right nodes are checked to find out if they are contiguous with
    // the original leaf if so the algorithm should keep going down these nodes
    // to update constraints
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

  const ConstraintEntry &Get(int leaf_idx) const { return entries_[leaf_idx]; }

 private:
  const Config *config_;
  int num_leaves_;
  std::vector<ConstraintEntry> entries_;
  std::vector<int> leaves_to_update_;
  // add parent node information
  std::vector<int> node_parent_;
  // Keeps track of the monotone splits above the leaf
  std::vector<bool> leaf_is_in_monotone_subtree_;
};

LeafConstraintsBase *LeafConstraintsBase::Create(const Config *config,
                                                 int num_leaves) {
  if (config->monotone_constraints_method == "intermediate") {
    return new FastLeafConstraints(config, num_leaves);
  }
  return new BasicLeafConstraints(num_leaves);
}

}  // namespace LightGBM
#endif  // LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
