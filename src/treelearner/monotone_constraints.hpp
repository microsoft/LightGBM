#ifndef LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
#define LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_

#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

namespace LightGBM {

struct ConstraintEntry {
  double min = -std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::max();

  ConstraintEntry(){};

  void Reset() {
    min = -std::numeric_limits<double>::max();
    max = std::numeric_limits<double>::max();
  }

  void UpdateMin(double new_min) { min = std::min(new_min, min); }

  void UpdateMax(double new_max) { max = std::max(new_max, max); }

};

template <typename ConstraintEntry>
class LeafConstraints {
 public:
  LeafConstraints(int num_leaves) : num_leaves_(num_leaves) {
    entries_.resize(num_leaves_);
  }
  void Reset() {
    for (auto& entry : entries_) {
      entry.Reset();
    }
  }
  void UpdateConstraints(bool is_numerical_split, int leaf, int new_leaf,
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

  const ConstraintEntry& Get(int leaf_idx) const { return entries_[leaf_idx]; }

 private:
  int num_leaves_;
  std::vector<ConstraintEntry> entries_;
};

} // namespace LightGBM
#endif // LightGBM_TREELEARNER_MONOTONE_CONSTRAINTS_H_
