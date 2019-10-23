#include "monotone_constraints.hpp"
#include "serial_tree_learner.h"
#include "feature_histogram.hpp"
#include "cost_effective_gradient_boosting.hpp"

namespace LightGBM {

void LeafConstraints::SetChildrenConstraintsFastMethod(
    std::vector<LeafConstraints> &constraints_per_leaf, int *right_leaf,
    int *left_leaf, int8_t monotone_type, double right_output,
    double left_output, bool is_numerical_split) {
  constraints_per_leaf[*right_leaf] = constraints_per_leaf[*left_leaf];
  if (is_numerical_split) {
    // depending on the monotone type we set constraints on the future splits
    // these constraints may be updated later in the algorithm
    if (monotone_type < 0) {
      constraints_per_leaf[*left_leaf].SetMinConstraint(right_output);
      constraints_per_leaf[*right_leaf].SetMaxConstraint(left_output);
    } else if (monotone_type > 0) {
      constraints_per_leaf[*left_leaf].SetMaxConstraint(right_output);
      constraints_per_leaf[*right_leaf].SetMinConstraint(left_output);
    }
  }
}

// this function goes through the tree to find how the split that
// has just been performed is going to affect the constraints of other leaves
void LeafConstraints::GoUpToFindLeavesToUpdate(
    int node_idx, std::vector<int> &features, std::vector<uint32_t> &thresholds,
    std::vector<bool> &is_in_right_split, int split_feature,
    const SplitInfo &split_info, double previous_leaf_output,
    uint32_t split_threshold, std::vector<SplitInfo> &best_split_per_leaf_,
    const std::vector<int8_t> &is_feature_used_, int num_threads_,
    int num_features_, HistogramPool &histogram_pool_,
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

    // only branches containing leaves that are contiguous to the original leaf
    // need to be updated
    for (unsigned int i = 0; i < features.size(); ++i) {
      if ((features[i] == inner_feature && is_split_numerical) &&
          (is_in_right_split[i] == is_right_split)) {
        split_contains_new_information = false;
        break;
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
            node_idx_to_pass, features, thresholds, is_in_right_split, take_min,
            split_feature, split_info, previous_leaf_output, true, true,
            split_threshold, best_split_per_leaf_, is_feature_used_,
            num_threads_, num_features_, histogram_pool_, learner_state);
      }

      is_in_right_split.push_back(learner_state.tree->right_child(parent_idx) ==
                                  node_idx);
      thresholds.push_back(learner_state.tree->threshold_in_bin(parent_idx));
      features.push_back(learner_state.tree->split_feature_inner(parent_idx));
    }

    if (parent_idx != 0) {
      LeafConstraints::GoUpToFindLeavesToUpdate(
          parent_idx, features, thresholds, is_in_right_split, split_feature,
          split_info, previous_leaf_output, split_threshold,
          best_split_per_leaf_, is_feature_used_, num_threads_, num_features_,
          histogram_pool_, learner_state);
    }
  }
}

// this function goes through the tree to find how the split that was just made
// is
// going to affect other leaves
void LeafConstraints::GoDownToFindLeavesToUpdate(
    int node_idx, const std::vector<int> &features,
    const std::vector<uint32_t> &thresholds,
    const std::vector<bool> &is_in_right_split, int maximum, int split_feature,
    const SplitInfo &split_info, double previous_leaf_output,
    bool use_left_leaf, bool use_right_leaf, uint32_t split_threshold,
    std::vector<SplitInfo> &best_split_per_leaf_,
    const std::vector<int8_t> &is_feature_used_, int num_threads_,
    int num_features_, HistogramPool &histogram_pool_,
    LearnerState &learner_state) {
  if (node_idx < 0) {
    int leaf_idx = ~node_idx;

    // if leaf is at max depth then there is no need to update it
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
    bool something_changed;
    if (use_right_leaf && use_left_leaf) {
      min_max_constraints =
          std::minmax(split_info.right_output, split_info.left_output);
    } else if (use_right_leaf && !use_left_leaf) {
      min_max_constraints = std::pair<double, double>(split_info.right_output,
                                                      split_info.right_output);
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

    if (!learner_state.config_->monotone_precise_mode) {
      if (!maximum) {
        something_changed =
            learner_state.constraints_per_leaf_[leaf_idx]
                .SetMinConstraintAndReturnChange(min_max_constraints.second);
      } else {
        something_changed =
            learner_state.constraints_per_leaf_[leaf_idx]
                .SetMaxConstraintAndReturnChange(min_max_constraints.first);
      }
      if (!something_changed) {
        return;
      }
    } else {
      if (!maximum) {
        // both functions need to be called in this order
        // because they modify the struct
        something_changed =
            learner_state.constraints_per_leaf_[leaf_idx]
                .CrossesMinConstraint(min_max_constraints.second);
        something_changed = learner_state.constraints_per_leaf_[leaf_idx]
                                .IsInMinConstraints(previous_leaf_output) ||
                            something_changed;
      } else {
        // both functions need to be called in this order
        // because they modify the struct
        something_changed =
            learner_state.constraints_per_leaf_[leaf_idx]
                .CrossesMaxConstraint(min_max_constraints.first);
        something_changed = learner_state.constraints_per_leaf_[leaf_idx]
                                .IsInMaxConstraints(previous_leaf_output) ||
                            something_changed;
      }
      // if constraints have changed, then best splits need to be updated
      // otherwise, we can just continue and go to the next split
      if (!something_changed) {
        return;
      }
    }
    UpdateBestSplitsFromHistograms(
        best_split_per_leaf_[leaf_idx], leaf_idx,
        learner_state.tree->leaf_depth(leaf_idx), is_feature_used_,
        num_threads_, num_features_, histogram_pool_, learner_state);
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
          previous_leaf_output, use_left_leaf,
          use_right_leaf_for_update && use_right_leaf, split_threshold,
          best_split_per_leaf_, is_feature_used_, num_threads_, num_features_,
          histogram_pool_, learner_state);
    }
    if (keep_going_left_right.second) {
      GoDownToFindLeavesToUpdate(
          learner_state.tree->right_child(node_idx), features, thresholds,
          is_in_right_split, maximum, split_feature, split_info,
          previous_leaf_output, use_left_leaf_for_update && use_left_leaf,
          use_right_leaf, split_threshold, best_split_per_leaf_,
          is_feature_used_, num_threads_, num_features_, histogram_pool_,
          learner_state);
    }
  }
}

// this function checks if the original leaf and the children of the node that
// is
// currently being visited are contiguous, and if so, the children should be
// visited too
std::pair<bool, bool> LeafConstraints::ShouldKeepGoingLeftRight(
    const Tree *tree, int node_idx, const std::vector<int> &features,
    const std::vector<uint32_t> &thresholds,
    const std::vector<bool> &is_in_right_split, const Dataset *train_data_) {
  int inner_feature = tree->split_feature_inner(node_idx);
  uint32_t threshold = tree->threshold_in_bin(node_idx);
  bool is_split_numerical = train_data_->FeatureBinMapper(inner_feature)
                                ->bin_type() == BinType::NumericalBin;

  bool keep_going_right = true;
  bool keep_going_left = true;
  // we check if the left and right node are contiguous with the original leaf
  // if so we should keep going down these nodes to update constraints
  for (unsigned int i = 0; i < features.size(); ++i) {
    if (features[i] == inner_feature) {
      if (is_split_numerical) {
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

// this function updates the best split for each leaf
// it is called only when monotone constraints exist
void LeafConstraints::UpdateBestSplitsFromHistograms(
    SplitInfo &split, int leaf, int depth,
    const std::vector<int8_t> &is_feature_used_, int num_threads_,
    int num_features_, HistogramPool &histogram_pool_,
    LearnerState &learner_state) {
  std::vector<SplitInfo> bests(num_threads_);
  std::vector<bool> should_split_be_worse(num_threads_, false);

  // the feature histogram is retrieved
  FeatureHistogram *histogram_array_;
  histogram_pool_.Get(leaf, &histogram_array_);

  OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    // the feature that are supposed to be used are computed
    if (!is_feature_used_[feature_index])
      continue;
    if (!histogram_array_[feature_index].is_splittable()) {
      continue;
    }

    // loop through the features to find the best one just like in the
    // FindBestSplitsFromHistograms function
    const int tid = omp_get_thread_num();
    int real_fidx = learner_state.train_data_->RealFeatureIndex(feature_index);

    // if the monotone precise mode is disabled or if the constraints have to be
    // updated,
    // but are not exclusively worse, then we update the constraints and the
    // best split
    if (!learner_state.config_->monotone_precise_mode ||
        (learner_state.constraints_per_leaf_[leaf].ToBeUpdated(feature_index) &&
         !learner_state.constraints_per_leaf_[leaf]
              .AreActualConstraintsWorse(feature_index))) {

      SerialTreeLearner::ComputeBestSplitForFeature(
          split.left_sum_gradient + split.right_sum_gradient,
          split.left_sum_hessian + split.right_sum_hessian,
          split.left_count + split.right_count, feature_index, histogram_array_,
          bests, leaf, depth, tid, real_fidx, learner_state, true);
    } else {
      if (learner_state.cegb_->GetSplitInfo(
              leaf * learner_state.train_data_->num_features() +
              feature_index) > bests[tid]) {
        bests[tid] = learner_state.cegb_->GetSplitInfo(
            leaf * learner_state.train_data_->num_features() + feature_index);
        should_split_be_worse[tid] =
            learner_state.constraints_per_leaf_[leaf]
                .AreActualConstraintsWorse(feature_index);
      }
    }

    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  // if the best split that has been found previously actually doesn't have the
  // true constraints
  // but worse ones that were not computed before to optimize the computation
  // time,
  // then we update every split and every constraints that should be updated
  if (should_split_be_worse[best_idx]) {
    std::fill(bests.begin(), bests.end(), SplitInfo());
#pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
    for (int feature_index = 0; feature_index < num_features_;
         ++feature_index) {
      OMP_LOOP_EX_BEGIN();
      if (!is_feature_used_[feature_index])
        continue;
      if (!histogram_array_[feature_index].is_splittable()) {
        continue;
      }

      const int tid = omp_get_thread_num();
      int real_fidx =
          learner_state.train_data_->RealFeatureIndex(feature_index);

      if (learner_state.constraints_per_leaf_[leaf]
              .AreActualConstraintsWorse(feature_index)) {
        ;
      } else {
#ifdef DEBUG
        CHECK(!learner_state.constraints_per_leaf_[leaf]
                   .ToBeUpdated(feature_index));
#endif
        if (learner_state.cegb_->GetSplitInfo(
                leaf * learner_state.train_data_->num_features() +
                feature_index) > bests[tid]) {
          bests[tid] = learner_state.cegb_->GetSplitInfo(
              leaf * learner_state.train_data_->num_features() + feature_index);
        }
      }

      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  }

  // note: the gains may differ for the same set of constraints due to the
  // non-deterministic OMP reduction.
  split = bests[best_idx];
}

} // namespace LightGBM
