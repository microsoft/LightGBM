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

  void CalculateLinear(Tree* tree, int leaf, int feat,
                       const std::vector<int>& parent_features,
                       const std::vector<double>& parent_coeffs,
                       const double& parent_const, const double& sum_grad, const double& sum_hess,
                       int& num_nan);

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
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
        if (is_nan_[row_idx]) {
          out_score[row_idx] += leaf_output;
          continue;
        }
        double output = leaf_const;
        for (int feat_num = 0; feat_num < feat_arr.size(); ++feat_num) {
          output += feat_ptr_arr[feat_num][row_idx] * leaf_coeffs[feat_num];
        }
        out_score[row_idx] += output;
      }
    }
  }

private:
  /*! \brief Temporary storage for calculating additive linear model */
  std::vector<double> curr_pred_;
  /*! \brief Temporary storage for calculating additive linear model */
  std::vector<int8_t> is_nan_;
  /*! \brief Temporary storage for calculating additive linear model */
  std::vector<int> nan_ind_;
};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_
