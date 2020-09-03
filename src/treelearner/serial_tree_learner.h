/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SERIAL_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_SERIAL_TREE_LEARNER_H_

#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/random.h>

#include <string>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "col_sampler.hpp"
#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "monotone_constraints.hpp"
#include "split_info.hpp"

#ifdef USE_GPU
// Use 4KBytes aligned allocator for ordered gradients and ordered hessians when GPU is enabled.
// This is necessary to pin the two arrays in memory and make transferring faster.
#include <boost/align/aligned_allocator.hpp>
#endif

namespace LightGBM {

using json11::Json;

/*! \brief forward declaration */
class CostEfficientGradientBoosting;

/*!
* \brief Used for learning a tree by single machine
*/
class SerialTreeLearner: public TreeLearner {
 public:
  friend CostEfficientGradientBoosting;
  explicit SerialTreeLearner(const Config* config);

  ~SerialTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void InitLinear(const Dataset* train_data, const int max_leaves) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override {
    ResetTrainingDataInner(train_data, is_constant_hessian, true);
  }

  void ResetIsConstantHessian(bool is_constant_hessian) override {
    share_state_->is_constant_hessian = is_constant_hessian;
  }

  virtual void ResetTrainingDataInner(const Dataset* train_data,
                                      bool is_constant_hessian,
                                      bool reset_multi_val_bin);

  void ResetConfig(const Config* config) override;

  inline void SetForcedSplit(const Json* forced_split_json) override {
    if (forced_split_json != nullptr && !forced_split_json->is_null()) {
      forced_split_json_ = forced_split_json;
    } else {
      forced_split_json_ = nullptr;
    }
  }

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  /*! \brief Create array mapping dataset to leaf index, used for linear trees */
  void GetLeafMap(Tree* tree);

  template<bool HAS_NAN>
  void CalculateLinear(Tree* tree, bool is_refit, const score_t* gradients, const score_t* hessians, bool is_first_tree);

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override {
    if (subset == nullptr) {
      data_partition_->SetUsedDataIndices(used_indices, num_data);
      share_state_->is_use_subrow = false;
    } else {
      ResetTrainingDataInner(subset, share_state_->is_constant_hessian, false);
      share_state_->is_use_subrow = true;
      share_state_->is_subrow_copied = false;
      share_state_->bagging_use_indices = used_indices;
      share_state_->bagging_indices_cnt = num_data;
    }
  }

  void AddPredictionToScore(const Tree* tree,
                            double* out_score) const override {
    CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
    if (tree->is_linear()) {
      CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
      if (tree->has_nan()) {
        AddPredictionToScoreInner<true>(tree, out_score);
      } else {
        AddPredictionToScoreInner<false>(tree, out_score);
      }
    } else {
      if (tree->num_leaves() <= 1) {
        return;
      }
#pragma omp parallel for schedule(static, 1)
      for (int i = 0; i < tree->num_leaves(); ++i) {
        double output = static_cast<double>(tree->LeafOutput(i));
        data_size_t cnt_leaf_data = 0;
        auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
        for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
          out_score[tmp_idx[j]] += output;
        }
      }
    }
  }

  template<bool HAS_NAN>
  void AddPredictionToScoreInner(const Tree* tree, double* out_score) const {
    int num_leaves = tree->num_leaves();
    std::vector<double> leaf_const(num_leaves);
    std::vector<std::vector<double>> leaf_coeff(num_leaves);
    std::vector<std::vector<const float*>> feat_ptr(num_leaves);
    std::vector<double> leaf_output(num_leaves);
    std::vector<int> leaf_num_features(num_leaves);
    for (int leaf_num = 0; leaf_num < num_leaves; ++leaf_num) {
      leaf_const[leaf_num] = tree->LeafConst(leaf_num);
      leaf_coeff[leaf_num] = tree->LeafCoeffs(leaf_num);
      leaf_output[leaf_num] = tree->LeafOutput(leaf_num);
      for (int feat : tree->LeafFeaturesInner(leaf_num)) {
        feat_ptr[leaf_num].push_back(train_data_->raw_index(feat));
      }
      leaf_num_features[leaf_num] = feat_ptr[leaf_num].size();
    }
    OMP_INIT_EX();
#pragma omp parallel for schedule(static) if (num_data_ > 1024)
    for (int i = 0; i < num_data_; ++i) {
      OMP_LOOP_EX_BEGIN();
      int leaf_num = leaf_map_[i];
      if (leaf_num < 0) {
        continue;
      }
      double output = leaf_const[leaf_num];
      int num_feat = leaf_num_features[leaf_num];
      if (HAS_NAN) {
        bool nan_found = false;
        for (int feat_ind = 0; feat_ind < num_feat; ++feat_ind) {
          float val = feat_ptr[leaf_num][feat_ind][i];
          if (std::isnan(val)) {
            nan_found = true;
            break;
          }
          output += val * leaf_coeff[leaf_num][feat_ind];
        }
        if (nan_found) {
          out_score[i] += leaf_output[leaf_num];
        } else {
          out_score[i] += output;
        }
      } else {
        for (int feat_ind = 0; feat_ind < num_feat; ++feat_ind) {
          output += feat_ptr[leaf_num][feat_ind][i] * leaf_coeff[leaf_num][feat_ind];
        }
        out_score[i] += output;
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }


  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

 protected:
  void ComputeBestSplitForFeature(FeatureHistogram* histogram_array_,
                                  int feature_index, int real_fidx,
                                  bool is_feature_used, int num_data,
                                  const LeafSplits* leaf_splits,
                                  SplitInfo* best_split);

  void GetShareStates(const Dataset* dataset, bool is_constant_hessian, bool is_first_time);

  void RecomputeBestSplitForLeaf(int leaf, SplitInfo* split);

  /*!
  * \brief Some initial works before training
  */
  virtual void BeforeTrain();

  /*!
  * \brief Some initial works before FindBestSplit
  */
  virtual bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf);

  virtual void FindBestSplits(const Tree* tree);

  virtual void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract);

  virtual void FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract, const Tree*);

  /*!
  * \brief Partition tree and data according best split.
  * \param tree Current tree, will be splitted on this function.
  * \param best_leaf The index of leaf that will be splitted.
  * \param left_leaf The index of left leaf after splitted.
  * \param right_leaf The index of right leaf after splitted.
  */
  inline virtual void Split(Tree* tree, int best_leaf, int* left_leaf,
    int* right_leaf) {
    SplitInner(tree, best_leaf, left_leaf, right_leaf, true);
  }

  void SplitInner(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf,
                  bool update_cnt);

  /* Force splits with forced_split_json dict and then return num splits forced.*/
  int32_t ForceSplits(Tree* tree, int* left_leaf, int* right_leaf,
                      int* cur_depth);

  /*!
  * \brief Get the number of data in a leaf
  * \param leaf_idx The index of leaf
  * \return The number of data in the leaf_idx leaf
  */
  inline virtual data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const;

  /*! \brief number of data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  /*! \brief training data */
  const Dataset* train_data_;
  /*! \brief gradients of current iteration */
  const score_t* gradients_;
  /*! \brief hessians of current iteration */
  const score_t* hessians_;
  /*! \brief training data partition on leaves */
  std::unique_ptr<DataPartition> data_partition_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;
  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;
  /*! \brief store best split per feature for all leaves */
  std::vector<SplitInfo> splits_per_leaf_;
  /*! \brief stores minimum and maximum constraints for each leaf */
  std::unique_ptr<LeafConstraintsBase> constraints_;

  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplits> larger_leaf_splits_;
#ifdef USE_GPU
  /*! \brief gradients of current iteration, ordered for cache optimized, aligned to 4K page */
  std::vector<score_t, boost::alignment::aligned_allocator<score_t, 4096>> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized, aligned to 4K page */
  std::vector<score_t, boost::alignment::aligned_allocator<score_t, 4096>> ordered_hessians_;
#else
  /*! \brief gradients of current iteration, ordered for cache optimized */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized */
  std::vector<score_t, Common::AlignmentAllocator<score_t, kAlignedSize>> ordered_hessians_;
#endif
  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const Config* config_;
  ColSampler col_sampler_;
  const Json* forced_split_json_;
  std::unique_ptr<TrainingShareStates> share_state_;
  std::unique_ptr<CostEfficientGradientBoosting> cegb_;
  /*! \brief whether numerical features contain any nan values, used for linear model */
  std::vector<int8_t> contains_nan_;
  /*! whether any numerical feature contains a nan value, used for linear model */
  bool any_nan_;
  /*! \brief map dataset to leaves, used for linear model */
  std::vector<int> leaf_map_;
  /*! \brief temporary storage for calculating linear model */
  std::vector<std::vector<float>> XTHX_;
  std::vector<std::vector<float>> XTg_;
  std::vector<std::vector<std::vector<float>>> XTHX_by_thread_;
  std::vector<std::vector<std::vector<float>>> XTg_by_thread_;
};

inline data_size_t SerialTreeLearner::GetGlobalDataCountInLeaf(int leaf_idx) const {
  if (leaf_idx >= 0) {
    return data_partition_->leaf_count(leaf_idx);
  } else {
    return 0;
  }
}

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_SERIAL_TREE_LEARNER_H_
