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
};

template <typename ConstraintEntry>
class LeafConstraints {
 public:
  explicit LeafConstraints(int num_leaves) : num_leaves_(num_leaves) {
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

}  // namespace LightGBM
#endif  // LIGHTGBM_TREELEARNER_MONOTONE_CONSTRAINTS_HPP_
