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
   
  Tree* Train(const score_t* gradients, const score_t *hessians);

  void CalculateLinear(Tree* tree);

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double leaf_const = tree->LeafConst(i);
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      int num_feat = tree->LeafFeaturesInner(i).size();
      std::vector<std::unique_ptr<BinIterator>> iter(num_feat);
      for (int feat_num = 0; feat_num < num_feat; ++feat_num) {
        int feat = tree->LeafFeaturesInner(i)[feat_num];
        iter[feat_num].reset(train_data_->FeatureIterator(feat));
      }
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        double add_score = leaf_const;
        bool nan_found = false;
        for (int feat_num = 0; feat_num < num_feat; ++feat_num) {
          int feat = tree->LeafFeaturesInner(i)[feat_num];
          double feat_val = train_data_->get_data(tmp_idx[j], feat);
          if (isnan(feat_val) || isinf(feat_val)) {
            nan_found = true;
            break;
          } else {
            add_score += feat_val * tree->LeafCoeffs(i)[feat_num];
          }
        }
        if (nan_found) {
          out_score[tmp_idx[j]] += tree->LeafOutput(i);
        } else {
          out_score[tmp_idx[j]] += add_score;
        }
      }
    }
  }

};
}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LINEAR_TREE_LEARNER_H_
