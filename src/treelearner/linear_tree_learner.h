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

  void CalculateLinear(Tree* tree, int leaf,
                       const std::vector<int>& parent_features,
                       const std::vector<double>& parent_coeffs,
                       const double& parent_const, const double& sum_grad, const double& sum_hess);

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      int num_feat = tree->LeafFeaturesInner(i).size();
      double leaf_output = tree->LeafOutput(i);
      double leaf_const = tree->LeafConst(i);
      std::vector<double> leaf_coeffs = tree->LeafCoeffs(i);
      for (int feat_num = 0; feat_num < num_feat; ++feat_num) {
        int feat = tree->LeafFeaturesInner(i)[feat_num];
        const double* feat_ptr = train_data_->raw_index(feat);
        for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
          if (is_nan_[tmp_idx[j]] == 0) {
            double feat_val = feat_ptr[tmp_idx[j]];
            out_score[tmp_idx[j]] += feat_val * leaf_coeffs[feat_num];
          }
        }
      }
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        if (is_nan_[tmp_idx[j]] == 1) {
          out_score[tmp_idx[j]] += leaf_output;
        } else {
          out_score[tmp_idx[j]] += leaf_const;
        }
      }
    }
  }
private:
  /*! \brief Temporary storage for calculating additive linear model */
  std::vector<double> curr_pred_;
  /*! \brief Temporary storage for calculating additive linear model */
  std::vector<int8_t> is_nan_;
};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_
