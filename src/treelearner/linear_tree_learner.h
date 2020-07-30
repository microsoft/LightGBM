/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_

#include "serial_tree_learner.h"

namespace LightGBM {

/*!
* \brief Tree learner with linear model at each leaf.
*/
class LinearTreeLearner: public SerialTreeLearner {
 public:

  explicit LinearTreeLearner(const Config* config) : SerialTreeLearner(config) {};

  void Init(const Dataset* train_data, bool is_constant_hessian) override;
   
  Tree* Train(const score_t* gradients, const score_t *hessians);

  template<bool HAS_NAN>
  void CalculateLinear(Tree* tree);

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
    if (tree->has_nan()) {
      AddPredictionToScoreInner<true>(tree, out_score);
    } else {
      AddPredictionToScoreInner<false>(tree, out_score);
    }
  }

  template<bool HAS_NAN>
  void AddPredictionToScoreInner(const Tree* tree, double* out_score) const {
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      double leaf_output = tree->LeafOutput(i);
      double leaf_const = tree->LeafConst(i);
      std::vector<double> leaf_coeffs = tree->LeafCoeffs(i);
      std::vector<int> feat_arr = tree->LeafFeaturesInner(i);
      std::vector<const double*> feat_ptr_arr;
      for (int feat : feat_arr) {
        feat_ptr_arr.push_back(train_data_->raw_index(feat));
      }
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        int row_idx = tmp_idx[j];
        double output = leaf_const;
        bool nan_found = false;
        for (int feat_num = 0; feat_num < feat_arr.size(); ++feat_num) {
          double val = feat_ptr_arr[feat_num][row_idx];
          if (HAS_NAN) {
            if (std::isnan(val)) {
              nan_found = true;
              break;
            } 
          }
          output += val * leaf_coeffs[feat_num];
        }
        if (HAS_NAN) {
          if (nan_found) {
            out_score[row_idx] += leaf_output;
          } else {
            out_score[row_idx] += output;
          }
        } else {
          out_score[row_idx] += output;
        }
      }
    }
  }

private:
  /*! \brief whether numerical features contain any nan values */
  std::vector<int8_t> contains_nan_;
  /*! whether any numerical feature contains a nan value */
  bool any_nan_;
  /*! \brief map dataset to leaves */
  std::vector<int> leaf_map_;
};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_
