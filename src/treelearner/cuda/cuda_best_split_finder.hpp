/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_
#define LIGHTGBM_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>

#include <vector>

#include <LightGBM/cuda/cuda_random.hpp>
#include <LightGBM/cuda/cuda_split_info.hpp>

#include "cuda_leaf_splits.hpp"

#define NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER (256)
#define NUM_THREADS_FIND_BEST_LEAF (256)
#define NUM_TASKS_PER_SYNC_BLOCK (1024)

namespace LightGBM {

struct SplitFindTask {
  int inner_feature_index;
  bool reverse;
  bool skip_default_bin;
  bool na_as_missing;
  bool assume_out_default_left;
  bool is_categorical;
  bool is_one_hot;
  uint32_t hist_offset;
  uint8_t mfb_offset;
  uint32_t num_bin;
  uint32_t default_bin;
  int rand_threshold;
};

class CUDABestSplitFinder {
 public:
  CUDABestSplitFinder(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets,
    const bool select_features_by_node,
    const Config* config);

  ~CUDABestSplitFinder();

  void InitFeatureMetaInfo(const Dataset* train_data);

  void Init();

  void InitCUDAFeatureMetaInfo();

  void BeforeTrain(const std::vector<int8_t>& is_feature_used_bytree);

  void FindBestSplitsForLeaf(
    const CUDALeafSplitsStruct* smaller_leaf_splits,
    const CUDALeafSplitsStruct* larger_leaf_splits,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const data_size_t num_data_in_smaller_leaf,
    const data_size_t num_data_in_larger_leaf,
    const double sum_hessians_in_smaller_leaf,
    const double sum_hessians_in_larger_leaf);

  const CUDASplitInfo* FindBestFromAllSplits(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index,
    int* num_cat_threshold);

  void ResetTrainingData(
    const hist_t* cuda_hist,
    const Dataset* train_data,
    const std::vector<uint32_t>& feature_hist_offsets);

  void ResetConfig(const Config* config, const hist_t* cuda_hist);

  void SetUsedFeatureByNode(const std::vector<int8_t>& is_feature_used_by_smaller_node,
                            const std::vector<int8_t>& is_feature_used_by_larger_node);

 private:
  #define LaunchFindBestSplitsForLeafKernel_PARAMS \
    const CUDALeafSplitsStruct* smaller_leaf_splits, \
    const CUDALeafSplitsStruct* larger_leaf_splits, \
    const int smaller_leaf_index, \
    const int larger_leaf_index, \
    const bool is_smaller_leaf_valid, \
    const bool is_larger_leaf_valid

  void LaunchFindBestSplitsForLeafKernel(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND>
  void LaunchFindBestSplitsForLeafKernelInner0(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1>
  void LaunchFindBestSplitsForLeafKernelInner1(LaunchFindBestSplitsForLeafKernel_PARAMS);

  template <bool USE_RAND, bool USE_L1, bool USE_SMOOTHING>
  void LaunchFindBestSplitsForLeafKernelInner2(LaunchFindBestSplitsForLeafKernel_PARAMS);

  #undef LaunchFindBestSplitsForLeafKernel_PARAMS

  void LaunchSyncBestSplitForLeafKernel(
    const int host_smaller_leaf_index,
    const int host_larger_leaf_index,
    const bool is_smaller_leaf_valid,
    const bool is_larger_leaf_valid);

  void LaunchFindBestFromAllSplitsKernel(
    const int cur_num_leaves,
    const int smaller_leaf_index,
    const int larger_leaf_index,
    int* smaller_leaf_best_split_feature,
    uint32_t* smaller_leaf_best_split_threshold,
    uint8_t* smaller_leaf_best_split_default_left,
    int* larger_leaf_best_split_feature,
    uint32_t* larger_leaf_best_split_threshold,
    uint8_t* larger_leaf_best_split_default_left,
    int* best_leaf_index,
    data_size_t* num_cat_threshold);

  void AllocateCatVectors(CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len);

  void LaunchAllocateCatVectorsKernel(CUDASplitInfo* cuda_split_infos, uint32_t* cat_threshold_vec, int* cat_threshold_real_vec, size_t len);

  void LaunchInitCUDARandomKernel();

  // Host memory
  int num_features_;
  int num_leaves_;
  int max_num_bin_in_feature_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_default_bins_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<MissingType> feature_missing_type_;
  double lambda_l1_;
  double lambda_l2_;
  data_size_t min_data_in_leaf_;
  double min_sum_hessian_in_leaf_;
  double min_gain_to_split_;
  double cat_smooth_;
  double cat_l2_;
  int max_cat_threshold_;
  int min_data_per_group_;
  int max_cat_to_onehot_;
  bool extra_trees_;
  int extra_seed_;
  bool use_smoothing_;
  double path_smooth_;
  std::vector<cudaStream_t> cuda_streams_;
  // for best split find tasks
  std::vector<SplitFindTask> split_find_tasks_;
  int num_tasks_;
  // use global memory
  bool use_global_memory_;
  // number of total bins in the dataset
  const int num_total_bin_;
  // has categorical feature
  bool has_categorical_feature_;
  // maximum number of bins of categorical features
  int max_num_categorical_bin_;
  // marks whether a feature is categorical
  std::vector<int8_t> is_categorical_;
  // whether need to select features by node
  bool select_features_by_node_;

  // CUDA memory, held by this object
  // for per leaf best split information
  CUDASplitInfo* cuda_leaf_best_split_info_;
  // for best split information when finding best split
  CUDASplitInfo* cuda_best_split_info_;
  // best split information buffer, to be copied to host
  int* cuda_best_split_info_buffer_;
  // find best split task information
  CUDAVector<SplitFindTask> cuda_split_find_tasks_;
  int8_t* cuda_is_feature_used_bytree_;
  // used when finding best split with global memory
  hist_t* cuda_feature_hist_grad_buffer_;
  hist_t* cuda_feature_hist_hess_buffer_;
  hist_t* cuda_feature_hist_stat_buffer_;
  data_size_t* cuda_feature_hist_index_buffer_;
  uint32_t* cuda_cat_threshold_leaf_;
  int* cuda_cat_threshold_real_leaf_;
  uint32_t* cuda_cat_threshold_feature_;
  int* cuda_cat_threshold_real_feature_;
  int max_num_categories_in_split_;
  // used for extremely randomized trees
  CUDAVector<CUDARandom> cuda_randoms_;
  // features used by node
  CUDAVector<int8_t> is_feature_used_by_smaller_node_;
  CUDAVector<int8_t> is_feature_used_by_larger_node_;

  // CUDA memory, held by other object
  const hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_TREELEARNER_CUDA_CUDA_BEST_SPLIT_FINDER_HPP_
