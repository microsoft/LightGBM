#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_

#include <vector>
#include <LightGBM/network.h>
#include "split_info.hpp"
#include <LightGBM/tree.h>
#include "data_partition.hpp"

namespace LightGBM {

struct CostEfficientGradientBoosting;
struct CurrentConstraints;
class HistogramPool;
struct LeafConstraints;

struct LearnerState {
  const Config *config_;
  const DataPartition *data_partition_;
  const Dataset *train_data_;
  std::vector<LeafConstraints> &constraints_per_leaf_;
  const Tree *tree;
  CurrentConstraints &current_constraints;
  std::unique_ptr<CostEfficientGradientBoosting> &cegb_;

  LearnerState(const Config *config_,
               const DataPartition *data_partition_,
               const Dataset *train_data_,
               std::vector<LeafConstraints> &constraints_per_leaf_,
               const Tree *tree, CurrentConstraints &current_constraints,
               std::unique_ptr<CostEfficientGradientBoosting> &cegb_)
      : config_(config_), data_partition_(data_partition_),
        train_data_(train_data_), constraints_per_leaf_(constraints_per_leaf_),
        tree(tree), current_constraints(current_constraints), cegb_(cegb_) {};
};

// the purpose of this structure is to store the constraints for one leaf
// when the monotone precise mode is disabled, then it will just store
// one min and one max constraint
// but if the monotone precise mode is enabled, then it may store a
// large number of constraints for different thresholds and features
struct LeafConstraints {
  std::vector<std::vector<double> > min_constraints;
  std::vector<std::vector<double> > max_constraints;
  // the constraint number i is valid on the slice
  // [thresholds[i]:threshold[i+1])
  // if threshold[i+1] does not exist, then it is valid for thresholds following
  // threshold[i]
  std::vector<std::vector<uint32_t> > min_thresholds;
  std::vector<std::vector<uint32_t> > max_thresholds;
  // These 2 vectors keep track of which constraints over which features
  // have to be upated
  std::vector<bool> min_to_be_updated;
  std::vector<bool> max_to_be_updated;
  // This vector keeps track of the constraints that we didn't update for some
  // features, because they could only be worse, and another better split was
  // available, so we didn't need to compute them yet, but we may need to in the
  // future
  std::vector<bool> are_actual_constraints_worse;

  static void SetChildrenConstraintsFastMethod(
      std::vector<LeafConstraints> &constraints_per_leaf, int *right_leaf,
      int *left_leaf, int8_t monotone_type, double right_output,
      double left_output, bool is_numerical_split);

  static void GoUpToFindLeavesToUpdate(
      int node_idx, std::vector<int> &features,
      std::vector<uint32_t> &thresholds, std::vector<bool> &is_in_right_split,
      int split_feature, const SplitInfo &split_info,
      double previous_leaf_output, uint32_t split_threshold,
      std::vector<SplitInfo> &best_split_per_leaf_,
      const std::vector<int8_t> &is_feature_used_, int num_threads_,
      int num_features_, HistogramPool &histogram_pool_,
      LearnerState &learner_state);

  static void GoUpToFindLeavesToUpdate(
      int node_idx, int split_feature, const SplitInfo &split_info,
      double previous_leaf_output, uint32_t split_threshold,
      std::vector<SplitInfo> &best_split_per_leaf_,
      const std::vector<int8_t> &is_feature_used_, int num_threads_,
      int num_features_, HistogramPool &histogram_pool_,
      LearnerState &learner_state) {
    int depth = learner_state.tree->leaf_depth(
                    ~learner_state.tree->left_child(node_idx)) -
                1;

    std::vector<int> features;
    std::vector<uint32_t> thresholds;
    std::vector<bool> is_in_right_split;

    features.reserve(depth);
    thresholds.reserve(depth);
    is_in_right_split.reserve(depth);

    GoUpToFindLeavesToUpdate(node_idx, features, thresholds, is_in_right_split,
                             split_feature, split_info, previous_leaf_output,
                             split_threshold, best_split_per_leaf_,
                             is_feature_used_, num_threads_, num_features_,
                             histogram_pool_, learner_state);
  }

  static void GoDownToFindLeavesToUpdate(
      int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, int maximum,
      int split_feature, const SplitInfo &split_info,
      double previous_leaf_output, bool use_left_leaf, bool use_right_leaf,
      uint32_t split_threshold, std::vector<SplitInfo> &best_split_per_leaf_,
      const std::vector<int8_t> &is_feature_used_, int num_threads_,
      int num_features_, HistogramPool &histogram_pool_,
      LearnerState &learner_state);

  static std::pair<bool, bool> ShouldKeepGoingLeftRight(
      const Tree *tree, int node_idx, const std::vector<int> &features,
      const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, const Dataset *train_data_);

  static void
  UpdateBestSplitsFromHistograms(SplitInfo &split, int leaf, int depth,
                                 const std::vector<int8_t> &is_feature_used_,
                                 int num_threads_, int num_features_,
                                 HistogramPool &histogram_pool_,
                                 LearnerState &learner_state);

  bool IsInConstraints(double element,
                       const std::vector<std::vector<double> > &constraints,
                       std::vector<bool> &to_be_updated) {
    bool ret = false;
    for (unsigned int i = 0; i < constraints.size(); i++) {
      for (unsigned int j = 0; j < constraints[i].size(); j++) {
        if (element == constraints[i][j]) {
          ret = true;
          to_be_updated[i] = true;
          are_actual_constraints_worse[i] = false;
        }
      }
    }
    return ret;
  }

  bool IsInMinConstraints(double min) {
    return IsInConstraints(min, min_constraints, min_to_be_updated);
  }

  bool IsInMaxConstraints(double max) {
    return IsInConstraints(max, max_constraints, max_to_be_updated);
  }

  void SetConstraint(double element,
                     std::vector<std::vector<double> > &constraints,
                     bool is_operator_greater) const {
    for (unsigned int i = 0; i < constraints.size(); i++) {
      for (unsigned int j = 0; j < constraints[i].size(); j++) {
        if ((is_operator_greater && element > constraints[i][j]) ||
            (!is_operator_greater && element < constraints[i][j])) {
          constraints[i][j] = element;
        }
      }
    }
  }

  // this function is the same as the previous one, but it also returns
  // if it actually modified something or not
  bool
  SetConstraintAndReturnChange(double element,
                               std::vector<std::vector<double> > &constraints,
                               bool is_operator_greater) const {
    bool something_changed = false;
    for (unsigned int i = 0; i < constraints.size(); i++) {
      for (unsigned int j = 0; j < constraints[i].size(); j++) {
        if ((is_operator_greater && element > constraints[i][j]) ||
            (!is_operator_greater && element < constraints[i][j])) {
          constraints[i][j] = element;
          something_changed = true;
        }
      }
    }
    return something_changed;
  }

  // this function checks if the element passed as a parameter would actually update
  // the constraints if they were to be set it as an additional constraint
  bool CrossesConstraint(double element,
                         std::vector<std::vector<double> > &constraints,
                         bool is_operator_greater,
                         std::vector<bool> &to_be_updated) {
    bool ret = false;
    for (unsigned int i = 0; i < constraints.size(); i++) {
      for (unsigned int j = 0; j < constraints[i].size(); j++) {
        if ((is_operator_greater && element > constraints[i][j]) ||
            (!is_operator_greater && element < constraints[i][j])) {
          ret = true;
          to_be_updated[i] = true;
          are_actual_constraints_worse[i] = true;
        }
      }
    }
    return ret;
  }

  bool SetMinConstraintAndReturnChange(double min) {
    return SetConstraintAndReturnChange(min, min_constraints, true);
  }

  bool SetMaxConstraintAndReturnChange(double max) {
    return SetConstraintAndReturnChange(max, max_constraints, false);
  }

  void SetMinConstraint(double min) {
    SetConstraint(min, min_constraints, true);
  }

  void SetMaxConstraint(double max) {
    SetConstraint(max, max_constraints, false);
  }

  bool CrossesMinConstraint(double min) {
    return CrossesConstraint(min, min_constraints, true, min_to_be_updated);
  }

  bool CrossesMaxConstraint(double max) {
    return CrossesConstraint(max, max_constraints, false, max_to_be_updated);
  }

  void ResetUpdates(unsigned int i) {
#ifdef DEBUG
    CHECK(i < are_actual_constraints_worse.size());
#endif
    are_actual_constraints_worse[i] = false;
    min_to_be_updated[i] = false;
    max_to_be_updated[i] = false;
  }

  // when the monotone precise mode is disabled, then we can just store
  // 1 min and 1 max constraints per leaf, so we call this constructor
  LeafConstraints() {
    min_constraints.push_back(
        std::vector<double>(1, -std::numeric_limits<double>::max()));
    max_constraints.push_back(
        std::vector<double>(1, std::numeric_limits<double>::max()));
    min_thresholds.push_back(std::vector<uint32_t>(1, 0));
    max_thresholds.push_back(std::vector<uint32_t>(1, 0));
  }

  // when the monotone precise mode is enabled, then for each feature,
  // we need to sort an array of constraints
  LeafConstraints(unsigned int num_features) {
    min_constraints.resize(num_features);
    max_constraints.resize(num_features);

    min_thresholds.resize(num_features);
    max_thresholds.resize(num_features);

    min_to_be_updated.resize(num_features, false);
    max_to_be_updated.resize(num_features, false);
    are_actual_constraints_worse.resize(num_features, false);

    for (unsigned int i = 0; i < num_features; i++) {
      // The number 32 has no real meaning here, but during our experiments,
      // we found that the number of constraints per feature was well below 32, so by
      // allocating this space, we may save some time because we won't have to allocate it later
      min_constraints[i].reserve(32);
      max_constraints[i].reserve(32);

      min_thresholds[i].reserve(32);
      max_thresholds[i].reserve(32);

      min_constraints[i].push_back(-std::numeric_limits<double>::max());
      max_constraints[i].push_back(std::numeric_limits<double>::max());

      min_thresholds[i].push_back(0);
      max_thresholds[i].push_back(0);
    }
  }

  bool AreActualConstraintsWorse(unsigned int feature_idx) const {
    return are_actual_constraints_worse[feature_idx];
  }

  bool ToBeUpdated(unsigned int feature_idx) const {
    return min_to_be_updated[feature_idx] || max_to_be_updated[feature_idx];
  }

  bool MinToBeUpdated(unsigned int feature_idx) const {
    return min_to_be_updated[feature_idx];
  }

  bool MaxToBeUpdated(unsigned int feature_idx) const {
    return max_to_be_updated[feature_idx];
  }

  LeafConstraints(const LeafConstraints &constraints)
      : min_constraints(constraints.min_constraints),
        max_constraints(constraints.max_constraints),
        min_thresholds(constraints.min_thresholds),
        max_thresholds(constraints.max_thresholds),
        min_to_be_updated(constraints.min_to_be_updated),
        max_to_be_updated(constraints.max_to_be_updated),
        are_actual_constraints_worse(constraints.are_actual_constraints_worse) {
  }

  // When we reset the constraints, then we just need to write that the constraints
  // are +/- inf, starting from the threshold 0
  void Reset() {
    for (unsigned int i = 0; i < min_constraints.size(); i++) {
      min_constraints[i].resize(1);
      max_constraints[i].resize(1);
      min_thresholds[i].resize(1);
      max_thresholds[i].resize(1);

      min_constraints[i][0] = -std::numeric_limits<double>::max();
      max_constraints[i][0] = std::numeric_limits<double>::max();
      min_thresholds[i][0] = 0;
      max_thresholds[i][0] = 0;
    }
  }

  static double ComputeMonotoneSplitGainPenalty(int depth, double penalization,
                                                double epsilon = 1e-10) {
    if (penalization >= depth + 1.) {
      return epsilon;
    }
    if (penalization <= 1.) {
      return 1. - penalization / pow(2., depth) + epsilon;
    }
    return 1. - pow(2, penalization - 1. - depth) + epsilon;
  }
};

struct SplittingConstraints {
  std::vector<double> cumulative_min_constraint_right_to_left;
  std::vector<double> cumulative_max_constraint_right_to_left;
  std::vector<double> cumulative_min_constraint_left_to_right;
  std::vector<double> cumulative_max_constraint_left_to_right;

  std::vector<uint32_t> thresholds_min_constraints;
  std::vector<uint32_t> thresholds_max_constraints;

  unsigned int index_min_constraint_left_to_right;
  unsigned int index_min_constraint_right_to_left;
  unsigned int index_max_constraint_left_to_right;
  unsigned int index_max_constraint_right_to_left;
  bool update_is_necessary;

  SplittingConstraints() {};

  SplittingConstraints(
      std::vector<double> &cumulative_min_constraint_right_to_left,
      std::vector<double> &cumulative_min_constraint_left_to_right,
      std::vector<double> &cumulative_max_constraint_right_to_left,
      std::vector<double> &cumulative_max_constraint_left_to_right,
      std::vector<uint32_t> &thresholds_min_constraints,
      std::vector<uint32_t> &thresholds_max_constraints) {
    this->cumulative_min_constraint_right_to_left =
        cumulative_min_constraint_right_to_left;
    this->cumulative_min_constraint_left_to_right =
        cumulative_min_constraint_left_to_right;
    this->cumulative_max_constraint_right_to_left =
        cumulative_max_constraint_right_to_left;
    this->cumulative_max_constraint_left_to_right =
        cumulative_max_constraint_left_to_right;

    this->thresholds_min_constraints = thresholds_min_constraints;
    this->thresholds_max_constraints = thresholds_max_constraints;
  }

  static void CumulativeExtremum(
      const double &(*extremum_function)(const double &, const double &),
      bool is_direction_from_left_to_right,
      std::vector<double> &cumulative_extremum) {
    if (cumulative_extremum.size() == 1) {
      return;
    }
#ifdef DEBUG
    CHECK(cumulative_extremum.size() != 0);
#endif

    std::size_t n_exts = cumulative_extremum.size();
    int step = is_direction_from_left_to_right ? 1 : -1;
    std::size_t start = is_direction_from_left_to_right ? 0 : n_exts - 1;
    std::size_t end = is_direction_from_left_to_right ? n_exts - 1 : 0;

    for (auto i = start; i != end; i = i + step) {
      cumulative_extremum[i + step] = extremum_function(
          cumulative_extremum[i + step], cumulative_extremum[i]);
    }
  }

  void ComputeCumulativeExtremums() {
    const double &(*min)(const double &, const double &) = std::min<double>;
    const double &(*max)(const double &, const double &) = std::max<double>;

    CumulativeExtremum(max, true, cumulative_min_constraint_left_to_right);
    CumulativeExtremum(max, false, cumulative_min_constraint_right_to_left);
    CumulativeExtremum(min, true, cumulative_max_constraint_left_to_right);
    CumulativeExtremum(min, false, cumulative_max_constraint_right_to_left);
  }

  void InitializeIndices(int dir) {
    if (dir == -1) {
      index_min_constraint_left_to_right = thresholds_min_constraints.size() - 1;
      index_min_constraint_right_to_left = thresholds_min_constraints.size() - 1;
      index_max_constraint_left_to_right = thresholds_max_constraints.size() - 1;
      index_max_constraint_right_to_left = thresholds_max_constraints.size() - 1;
      update_is_necessary = !(thresholds_max_constraints.size() == 1 &&
                              thresholds_min_constraints.size() == 1);
    } else {
      index_min_constraint_left_to_right = 0;
      index_min_constraint_right_to_left = 0;
      index_max_constraint_left_to_right = 0;
      index_max_constraint_right_to_left = 0;
    }
  }

  void UpdateIndices(int dir, const int8_t bias, int t) {
    if (dir == -1) {
      if (update_is_necessary) {
        while (
            static_cast<int>(
                thresholds_min_constraints[index_min_constraint_left_to_right]) >
            t + bias - 1) {
          index_min_constraint_left_to_right -= 1;
        }
        while (
            static_cast<int>(
                thresholds_min_constraints[index_min_constraint_right_to_left]) >
            t + bias) {
          index_min_constraint_right_to_left -= 1;
        }
        while (
            static_cast<int>(
                thresholds_max_constraints[index_max_constraint_left_to_right]) >
            t + bias - 1) {
          index_max_constraint_left_to_right -= 1;
        }
        while (
            static_cast<int>(
                thresholds_max_constraints[index_max_constraint_right_to_left]) >
            t + bias) {
          index_max_constraint_right_to_left -= 1;
        }
      }
#ifdef DEBUG
      CHECK(index_min_constraint_left_to_right <
            thresholds_min_constraint.size());
      CHECK(index_min_constraint_right_to_left <
            thresholds_min_constraint.size());
      CHECK(index_max_constraint_left_to_right <
            thresholds_max_constraint.size());
      CHECK(index_max_constraint_right_to_left <
            thresholds_max_constraint.size());
#endif
    } else {
// current split gain
#ifdef DEBUG
      CHECK(index_min_constraint_left_to_right <
            thresholds_min_constraint.size());
      CHECK(index_min_constraint_right_to_left <
            thresholds_min_constraint.size());
      CHECK(index_max_constraint_left_to_right <
            thresholds_max_constraint.size());
      CHECK(index_max_constraint_right_to_left <
            thresholds_max_constraint.size());
#endif
    }
  }

  double CurrentMinConstraintRight() const {
    return cumulative_min_constraint_right_to_left
        [index_min_constraint_right_to_left];
  }

  double CurrentMaxConstraintRight() const {
    return cumulative_max_constraint_right_to_left
        [index_max_constraint_right_to_left];
  }

  double CurrentMinConstraintLeft() const {
    return cumulative_min_constraint_left_to_right
        [index_min_constraint_left_to_right];
  }

  double CurrentMaxConstraintLeft() const {
    return cumulative_max_constraint_left_to_right
        [index_max_constraint_left_to_right];
  }

  void Reserve(int space_to_reserve) {
    cumulative_max_constraint_right_to_left.reserve(space_to_reserve);
    cumulative_max_constraint_left_to_right.reserve(space_to_reserve);
    cumulative_min_constraint_right_to_left.reserve(space_to_reserve);
    cumulative_min_constraint_left_to_right.reserve(space_to_reserve);
    thresholds_max_constraints.reserve(space_to_reserve);
    thresholds_min_constraints.reserve(space_to_reserve);
  }

  void InitializeConstraints() {
    thresholds_min_constraints.resize(1);
    thresholds_max_constraints.resize(1);

    cumulative_min_constraint_right_to_left.resize(1);
    cumulative_min_constraint_left_to_right.resize(1);
    cumulative_max_constraint_right_to_left.resize(1);
    cumulative_max_constraint_left_to_right.resize(1);

    cumulative_min_constraint_right_to_left[0] = -std::numeric_limits<double>::max();
    cumulative_min_constraint_left_to_right[0] = -std::numeric_limits<double>::max();
    cumulative_max_constraint_right_to_left[0] = std::numeric_limits<double>::max();
    cumulative_max_constraint_left_to_right[0] = std::numeric_limits<double>::max();

    thresholds_min_constraints[0] = 0;
    thresholds_max_constraints[0] = 0;
  }

  void Set(const LeafConstraints &leaf_constraints) {
    cumulative_min_constraint_right_to_left[0] = leaf_constraints.min_constraints[0][0];
    cumulative_max_constraint_right_to_left[0] = leaf_constraints.max_constraints[0][0];

    cumulative_min_constraint_left_to_right[0] = leaf_constraints.min_constraints[0][0];
    cumulative_max_constraint_left_to_right[0] = leaf_constraints.max_constraints[0][0];

    thresholds_min_constraints[0] = leaf_constraints.min_thresholds[0][0];
    thresholds_max_constraints[0] = leaf_constraints.max_thresholds[0][0];
  }

  void CheckCoherenceWithLeafOutput(double leaf_output,
                                    double EPS) {
    CHECK(cumulative_min_constraint_left_to_right == cumulative_min_constraint_right_to_left);
    CHECK(cumulative_max_constraint_left_to_right == cumulative_max_constraint_right_to_left);
    for (const auto &x : cumulative_max_constraint_left_to_right) {
      CHECK(leaf_output <= EPS + x);
      CHECK(x > -std::numeric_limits<double>::max());
    }
    for (const auto &x : cumulative_min_constraint_right_to_left) {
      CHECK(leaf_output + EPS >= x);
      CHECK(x < std::numeric_limits<double>::max());
    }
  }
};

struct CurrentConstraints {
  std::vector<SplittingConstraints> splitting_constraints_vector;

  const int space_to_reserve_non_monotone_precise_mode;
  const int space_to_reserve_monotone_precise_mode;

  // the number 32 has no real meaning here, but during our experiments,
  // we found that the number of constraints per feature was well below 32, so
  // by allocating this space, we may save some time because we won't have to
  // allocate it later
  CurrentConstraints()
      : space_to_reserve_non_monotone_precise_mode(1),
        space_to_reserve_monotone_precise_mode(32) {};

  void Init(int num_threads_, const Config *config_) {
    splitting_constraints_vector.resize(num_threads_);

    int space_to_reserve = space_to_reserve_monotone_precise_mode;
    if (!config_->monotone_precise_mode) {
      space_to_reserve = space_to_reserve_non_monotone_precise_mode;
    }

    for (int i = 0; i < num_threads_; ++i) {
      splitting_constraints_vector[i].Reserve(space_to_reserve);
      InitializeConstraints(i);
    }
  }

  SplittingConstraints& operator[](unsigned int i) {
    return splitting_constraints_vector[i];
  }

  // initializing constraints is just writing that the constraints should +/-
  // inf from threshold 0
  void InitializeConstraints(unsigned int tid) {
    splitting_constraints_vector[tid].InitializeConstraints();
  }

  void Set(const LeafConstraints &leaf_constraints, unsigned int tid) {
    splitting_constraints_vector[tid].Set(leaf_constraints);
  }

  void CheckCoherenceWithLeafOutput(double leaf_output, unsigned int tid,
                                    double EPS) {
    splitting_constraints_vector[tid]
        .CheckCoherenceWithLeafOutput(leaf_output, EPS);
  }
};

struct BestConstraints {
  double best_min_constraint_right;
  double best_max_constraint_right;
  double best_min_constraint_left;
  double best_max_constraint_left;

  void Init() {
    best_min_constraint_left = NAN;
    best_max_constraint_left = NAN;
    best_min_constraint_right = NAN;
    best_max_constraint_right = NAN;
  }

  void Update(SplittingConstraints *constraints) {
    best_min_constraint_right = constraints->CurrentMinConstraintRight();
    best_max_constraint_right = constraints->CurrentMaxConstraintRight();
    best_min_constraint_left = constraints->CurrentMinConstraintLeft();
    best_max_constraint_left = constraints->CurrentMaxConstraintLeft();
  }
};

} // namespace LightGBM
#endif // LightGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
