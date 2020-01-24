#include "monotone_constraints.hpp"

namespace LightGBM {

void LeafConstraints::SetChildrenConstraintsFastMethod(
    std::vector<LeafConstraints> &constraints_per_leaf, int *right_leaf,
    int *left_leaf, int8_t monotone_type, double right_output,
    double left_output, bool is_numerical_split) {
  constraints_per_leaf[*right_leaf] = constraints_per_leaf[*left_leaf];
  if (is_numerical_split) {
    double mid = (left_output + right_output) / 2.0f;
    if (monotone_type < 0) {
      constraints_per_leaf[*left_leaf].UpdateMinConstraint(mid);
      constraints_per_leaf[*right_leaf].UpdateMaxConstraint(mid);
    } else if (monotone_type > 0) {
      constraints_per_leaf[*left_leaf].UpdateMaxConstraint(mid);
      constraints_per_leaf[*right_leaf].UpdateMinConstraint(mid);
    }
  }
}

} // namespace LightGBM
