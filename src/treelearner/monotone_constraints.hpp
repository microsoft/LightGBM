#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_

#include <algorithm>

namespace LightGBM {

struct LeafConstraints {
  double min_constraint;
  double max_constraint;

  static void SetChildrenConstraintsFastMethod(
    std::vector<LeafConstraints> &constraints_per_leaf, int *right_leaf,
    int *left_leaf, int8_t monotone_type, double right_output,
    double left_output, bool is_numerical_split);

  LeafConstraints() {
    Reset();
  }

  void Reset() {
    min_constraint = -std::numeric_limits<double>::max();
    max_constraint = std::numeric_limits<double>::max();
  }

  void UpdateMinConstraint(double min) {
    min_constraint = std::max(min, min_constraint);
  }

  void UpdateMaxConstraint(double max) {
    max_constraint = std::min(max, max_constraint);
  }
};

} // namespace LightGBM
#endif // LightGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
