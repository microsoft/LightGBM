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
#include <LightGBM/utils/random.h>

#include <string>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "split_info.hpp"
#include "monotone_constraints.hpp"

#ifdef USE_GPU
// Use 4KBytes aligned allocator for ordered gradients and ordered hessians when GPU is enabled.
// This is necessary to pin the two arrays in memory and make transferring faster.
#include <boost/align/aligned_allocator.hpp>
#endif

using namespace json11;

namespace LightGBM {
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

  void ResetTrainingData(const Dataset* train_data) override;

  void ResetConfig(const Config* config) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian,
              const Json& forced_split_json) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) override;

  void SetBaggingData(const data_size_t* used_indices, data_size_t num_data) override {
    data_partition_->SetUsedDataIndices(used_indices, num_data);
  }

  void AddPredictionToScore(const Tree* tree, double* out_score) const override {
    if (tree->num_leaves() <= 1) { return; }
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        out_score[tmp_idx[j]] += output;
      }
    }
  }

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

 protected:
  virtual std::vector<int8_t> GetUsedFeatures(bool is_tree_level);
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

  virtual void
  FindBestSplitsFromHistograms(const std::vector<int8_t> &is_feature_used,
                               bool use_subtract, const Tree *tree);

  virtual void UpdateBestSplitsFromHistograms(SplitInfo &split, int leaf,
                                              int depth, const Tree *tree);

  /*!
  * \brief Partition tree and data according best split.
  * \param tree Current tree, will be splitted on this function.
  * \param best_leaf The index of leaf that will be splitted.
  * \param left_leaf The index of left leaf after splitted.
  * \param right_leaf The index of right leaf after splitted.
  */
  virtual void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf);

  /* Force splits with forced_split_json dict and then return num splits forced.*/
  virtual int32_t ForceSplits(Tree* tree, const Json& forced_split_json, int* left_leaf,
                              int* right_leaf, int* cur_depth,
                              bool *aborted_last_force_split);

  /*!
  * \brief Get the number of data in a leaf
  * \param leaf_idx The index of leaf
  * \return The number of data in the leaf_idx leaf
  */
  inline virtual data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const;

  void ComputeBestSplitForFeature(double sum_gradient, double sum_hessian,
                                  data_size_t num_data, int feature_index,
                                  FeatureHistogram *histogram_array_,
                                  std::vector<SplitInfo> &bests, int leaf_index,
                                  int depth, const int tid, int real_fidx,
                                  const Tree *tree, bool update = false);

  void ComputeConstraintsPerThreshold(int feature, const Tree *tree,
                                      int node_idx, unsigned int tid,
                                      bool per_threshold, bool compute_min,
                                      bool compute_max, uint32_t it_start,
                                      uint32_t it_end);

  void ComputeConstraintsPerThreshold(int feature, const Tree *tree,
                                      int node_idx, unsigned int tid,
                                      bool per_threshold = true,
                                      bool compute_min = true,
                                      bool compute_max = true) {
    ComputeConstraintsPerThreshold(feature, tree, node_idx, tid, per_threshold,
                                   compute_min, compute_max, 0,
                                   train_data_->NumBin(feature));
  }

  void ComputeConstraintsPerThresholdInSubtree(
      int split_feature, int monotone_feature, const Tree *tree, int node_idx,
      bool maximum, uint32_t it_start, uint32_t it_end,
      const std::vector<int> &features, const std::vector<uint32_t> &thresholds,
      const std::vector<bool> &is_in_right_split, unsigned int tid,
      bool per_threshold);

  static double ComputeMonotoneSplitGainPenalty(int depth, double penalization,
                                                double epsilon = 1e-10);

  void GoDownToFindLeavesToUpdate(const Tree *tree, int node_idx,
                                  const std::vector<int> &features,
                                  const std::vector<uint32_t> &thresholds,
                                  const std::vector<bool> &is_in_right_split,
                                  int maximum, int split_feature,
                                  const SplitInfo &split_info,
                                  double previous_leaf_output,
                                  bool use_left_leaf, bool use_right_leaf,
                                  uint32_t split_threshold);

  /* Once we made a split, the constraints on other leaves may change.
     We need to update them to remain coherent. */
  void GoUpToFindLeavesToUpdate(const Tree *tree, int node_idx,
                                std::vector<int> &features,
                                std::vector<uint32_t> &thresholds,
                                std::vector<bool> &is_in_right_split,
                                int split_feature, const SplitInfo &split_info,
                                double previous_leaf_output,
                                uint32_t split_threshold);

  void GoUpToFindLeavesToUpdate(const Tree *tree, int node_idx,
                                int split_feature, const SplitInfo &split_info,
                                double previous_leaf_output,
                                uint32_t split_threshold) {
    int depth = tree->leaf_depth(~tree->left_child(node_idx)) - 1;

    std::vector<int> features;
    std::vector<uint32_t> thresholds;
    std::vector<bool> is_in_right_split;

    features.reserve(depth);
    thresholds.reserve(depth);
    is_in_right_split.reserve(depth);

    GoUpToFindLeavesToUpdate(tree, node_idx, features, thresholds,
                             is_in_right_split, split_feature, split_info,
                             previous_leaf_output, split_threshold);
  }

  std::pair<bool, bool>
  ShouldKeepGoingLeftRight(const Tree *tree, int node_idx,
                           const std::vector<int> &features,
                           const std::vector<uint32_t> &thresholds,
                           const std::vector<bool> &is_in_right_split);

  std::pair<bool, bool>
  LeftRightContainsRelevantInformation(bool maximum, int inner_feature,
                                       bool split_feature_is_inner_feature);

  void InitializeConstraints(unsigned int tid);

  void UpdateConstraints(std::vector<std::vector<double> > &constraints,
                         std::vector<std::vector<uint32_t> > &thresholds,
                         double extremum, uint32_t it_start, uint32_t it_end,
                         int split_feature, int tid, bool maximum);

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
  /*! \brief used for generate used features */
  Random random_;
  /*! \brief used for sub feature training, is_feature_used_[i] = false means don't used feature i */
  std::vector<int8_t> is_feature_used_;
  /*! \brief used feature indices in current tree */
  std::vector<int> used_feature_indices_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;

  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;

  std::vector<Constraints> constraints_per_leaf_;
  /*! \brief store best split per feature for all leaves */
  std::vector<SplitInfo> splits_per_leaf_;

  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplits> larger_leaf_splits_;
  std::vector<int> valid_feature_indices_;

#ifdef USE_GPU
  /*! \brief gradients of current iteration, ordered for cache optimized, aligned to 4K page */
  std::vector<score_t, boost::alignment::aligned_allocator<score_t, 4096>> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized, aligned to 4K page */
  std::vector<score_t, boost::alignment::aligned_allocator<score_t, 4096>> ordered_hessians_;
#else
  /*! \brief gradients of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_gradients_;
  /*! \brief hessians of current iteration, ordered for cache optimized */
  std::vector<score_t> ordered_hessians_;
#endif

  /*! \brief Store ordered bin */
  std::vector<std::unique_ptr<OrderedBin>> ordered_bins_;
  /*! \brief True if has ordered bin */
  bool has_ordered_bin_ = false;
  /*! \brief  is_data_in_leaf_[i] != 0 means i-th data is marked */
  std::vector<char> is_data_in_leaf_;
  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const Config* config_;
  int num_threads_;
  std::vector<int> ordered_bin_indices_;
  bool is_constant_hessian_;
  std::unique_ptr<CostEfficientGradientBoosting> cegb_;

  std::vector<std::vector<double> > dummy_min_constraints;
  std::vector<std::vector<double> > min_constraints;
  std::vector<std::vector<double> > dummy_max_constraints;
  std::vector<std::vector<double> > max_constraints;

  std::vector<std::vector<uint32_t> > thresholds_min_constraints;
  std::vector<std::vector<uint32_t> > thresholds_max_constraints;

  std::vector<std::vector<int> > features;
  std::vector<std::vector<uint32_t> > thresholds;
  std::vector<std::vector<bool> > is_in_right_split;
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
