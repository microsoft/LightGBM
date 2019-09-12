#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_

#include <vector>

namespace LightGBM {

// the purpose of this structure is to store the constraints for one leaf
// when the monotone precise mode is disabled, then it will just store
// one min and one max constraint
// but if the monotone precise mode is enabled, then it may store a
// large number of constraints for different thresholds and features
struct LeafConstraints {
  std::vector<std::vector<double> > min_constraints;
  std::vector<std::vector<double> > max_constraints;
  // the constraint number i is valid on the slice [thresholds[i]:threshold[i+1])
  // if threshold[i+1] does not exist, then it is valid for thresholds following threshold[i]
  std::vector<std::vector<uint32_t> > min_thresholds;
  std::vector<std::vector<uint32_t> > max_thresholds;
  // These 2 vectors keep track of which constraints over which features
  // have to be upated
  std::vector<bool> min_to_be_updated;
  std::vector<bool> max_to_be_updated;
  // This vector keeps track of the constraints that we didn't update for some
  // features, because they could only be worse, and another better split was
  // available, so we didn't need to compute them yet, but we may need to in the future
  std::vector<bool> are_actual_constraints_worse;

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

struct CurrentConstraints {
  std::vector<std::vector<double> > dummy_min_constraints;
  std::vector<std::vector<double> > min_constraints;
  std::vector<std::vector<double> > dummy_max_constraints;
  std::vector<std::vector<double> > max_constraints;

  std::vector<std::vector<uint32_t> > thresholds_min_constraints;
  std::vector<std::vector<uint32_t> > thresholds_max_constraints;

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
    dummy_min_constraints.resize(num_threads_);
    min_constraints.resize(num_threads_);
    dummy_max_constraints.resize(num_threads_);
    max_constraints.resize(num_threads_);

    thresholds_min_constraints.resize(num_threads_);
    thresholds_max_constraints.resize(num_threads_);

    int space_to_reserve = space_to_reserve_monotone_precise_mode;
    if (!config_->monotone_precise_mode) {
      space_to_reserve = space_to_reserve_non_monotone_precise_mode;
    }

    for (int i = 0; i < num_threads_; ++i) {
      dummy_min_constraints[i].reserve(space_to_reserve);
      min_constraints[i].reserve(space_to_reserve);
      dummy_max_constraints[i].reserve(space_to_reserve);
      max_constraints[i].reserve(space_to_reserve);

      thresholds_min_constraints[i].reserve(space_to_reserve);
      thresholds_max_constraints[i].reserve(space_to_reserve);

      InitializeConstraints(i);
    }
  }

  // initializing constraints is just writing that the constraints should +/-
  // inf from threshold 0
  void InitializeConstraints(unsigned int tid) {
    thresholds_min_constraints[tid].resize(1);
    thresholds_max_constraints[tid].resize(1);

    dummy_min_constraints[tid].resize(1);
    min_constraints[tid].resize(1);
    dummy_max_constraints[tid].resize(1);
    max_constraints[tid].resize(1);

    dummy_min_constraints[tid][0] = -std::numeric_limits<double>::max();
    min_constraints[tid][0] = -std::numeric_limits<double>::max();
    dummy_max_constraints[tid][0] = std::numeric_limits<double>::max();
    max_constraints[tid][0] = std::numeric_limits<double>::max();

    thresholds_min_constraints[tid][0] = 0;
    thresholds_max_constraints[tid][0] = 0;
  }

  void Set(const LeafConstraints &leaf_constraints, unsigned int tid) {
    dummy_min_constraints[tid][0] = leaf_constraints.min_constraints[0][0];
    dummy_max_constraints[tid][0] = leaf_constraints.max_constraints[0][0];

    min_constraints[tid][0] = leaf_constraints.min_constraints[0][0];
    max_constraints[tid][0] = leaf_constraints.max_constraints[0][0];

    thresholds_min_constraints[tid][0] = leaf_constraints.min_thresholds[0][0];
    thresholds_max_constraints[tid][0] = leaf_constraints.max_thresholds[0][0];
  }

  void CheckCoherenceWithLeafOutput(double leaf_output, unsigned int tid,
                                    double EPS) {
    CHECK(dummy_min_constraints[tid] == min_constraints[tid]);
    CHECK(dummy_max_constraints[tid] == max_constraints[tid]);
    for (const auto &x : max_constraints[tid]) {
      CHECK(leaf_output <= EPS + x);
      CHECK(x > -std::numeric_limits<double>::max());
    }
    for (const auto &x : dummy_min_constraints[tid]) {
      CHECK(leaf_output + EPS >= x);
      CHECK(x < std::numeric_limits<double>::max());
    }
  }
};

} // namespace LightGBM
#endif // LightGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
