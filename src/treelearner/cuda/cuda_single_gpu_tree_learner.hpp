/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
#define LIGHTGBM_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_

#include <memory>
#include <vector>

#ifdef USE_CUDA_EXP

#include "cuda_leaf_splits.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_partition.hpp"
#include "cuda_best_split_finder.hpp"

#include "../serial_tree_learner.h"

namespace LightGBM {

#define CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE (1024)

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
  explicit CUDASingleGPUTreeLearner(const Config* config, const bool boosting_on_cuda);

  ~CUDASingleGPUTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt, const double* train_score) const override;

  void ResetConfig(const Config* config) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

  void ResetBoostingOnGPU(const bool boosting_on_gpu) override;

 protected:
  void BeforeTrain() override;

  void ReduceLeafStat(CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const;

  void LaunchReduceLeafStatKernel(const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
    const int* leaf_parent, const int* left_child, const int* right_child,
    const int num_leaves, const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const;

  void ConstructBitsetForCategoricalSplit(const CUDASplitInfo* best_split_info);

  void LaunchConstructBitsetForCategoricalSplitKernel(const CUDASplitInfo* best_split_info);

  void AllocateBitset();

  void SelectFeatureByNode(const Tree* tree);

  #ifdef DEUBG
  void CheckSplitValid(
    const int left_leaf, const int right_leaf,
    const double sum_left_gradients, const double sum_right_gradients);
  #endif  // DEBUG

  // GPU device ID
  int gpu_device_id_;
  // number of threads on CPU
  int num_threads_;

  // CUDA components for tree training

  // leaf splits information for smaller and larger leaves
  std::unique_ptr<CUDALeafSplits> cuda_smaller_leaf_splits_;
  std::unique_ptr<CUDALeafSplits> cuda_larger_leaf_splits_;
  // data partition that partitions data indices into different leaves
  std::unique_ptr<CUDADataPartition> cuda_data_partition_;
  // for histogram construction
  std::unique_ptr<CUDAHistogramConstructor> cuda_histogram_constructor_;
  // for best split information finding, given the histograms
  std::unique_ptr<CUDABestSplitFinder> cuda_best_split_finder_;

  std::vector<int> leaf_best_split_feature_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_hessians_;
  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
  int num_cat_threshold_;
  bool has_categorical_feature_;
  // whether need to select features by node
  bool select_features_by_node_;

  std::vector<int> categorical_bin_to_value_;
  std::vector<int> categorical_bin_offsets_;

  mutable double* cuda_leaf_gradient_stat_buffer_;
  mutable double* cuda_leaf_hessian_stat_buffer_;
  mutable data_size_t leaf_stat_buffer_size_;
  mutable data_size_t refit_num_data_;
  uint32_t* cuda_bitset_;
  size_t cuda_bitset_len_;
  uint32_t* cuda_bitset_inner_;
  size_t cuda_bitset_inner_len_;
  size_t* cuda_block_bitset_len_buffer_;
  int* cuda_categorical_bin_to_value_;
  int* cuda_categorical_bin_offsets_;

  /*! \brief gradients on CUDA */
  score_t* cuda_gradients_;
  /*! \brief hessians on CUDA */
  score_t* cuda_hessians_;
  /*! \brief whether boosting is done on CUDA */
  bool boosting_on_cuda_;

  #ifdef DEBUG
  /*! \brief gradients on CPU */
  std::vector<score_t> host_gradients_;
  /*! \brief hessians on CPU */
  std::vector<score_t> host_hessians_;
  #endif  // DEBUG
};

}  // namespace LightGBM

#else  // USE_CUDA_EXP

// When GPU support is not compiled in, quit with an error message

namespace LightGBM {

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit CUDASingleGPUTreeLearner(const Config* tree_config, const bool /*boosting_on_cuda*/) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner experimental version was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA_EXP=1");
    }
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_TREELEARNER_CUDA_CUDA_SINGLE_GPU_TREE_LEARNER_HPP_
