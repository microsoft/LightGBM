/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
#define LIGHTGBM_CUDA_DATA_SPLITTER_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include <LightGBM/bin.h>
#include "new_cuda_utils.hpp"
#include "cuda_leaf_splits.hpp"
#include "cuda_split_info.hpp"

// TODO(shiyu1994): adjust these values according to different CUDA and GPU versions
#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)
#define SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION (512)
#define NUM_BANKS_DATA_PARTITION (32)
#define LOG_NUM_BANKS_DATA_PARTITION (5)
#define AGGREGATE_BLOCK_SIZE_DATA_PARTITION (1024)

namespace LightGBM {

class CUDADataPartition {
 public:
  CUDADataPartition(
    const Dataset* train_data,
    const int num_total_bin,
    const int num_leaves,
    const int num_threads,
    const data_size_t* cuda_num_data,
    hist_t* cuda_hist);

  void Init();

  void BeforeTrain(const data_size_t* data_indices);

  void Split(
    // input best split info
    const CUDASplitInfo* best_split_info,
    // for leaf information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    std::vector<data_size_t>* leaf_num_data,
    std::vector<data_size_t>* leaf_data_start,
    std::vector<double>* leaf_sum_hessians,
    const std::vector<int>& leaf_best_split_feature,
    const std::vector<uint32_t>& leaf_best_split_threshold,
    const std::vector<uint8_t>& leaf_best_split_default_left,
    int* smaller_leaf_index,
    int* larger_leaf_index,
    const int leaf_index,
    const int cur_max_leaf_index);

  Tree* GetCPUTree();

  void UpdateTrainScore(const double learning_rate, double* cuda_scores);

  const data_size_t* cuda_leaf_data_start() const { return cuda_leaf_data_start_; }

  const data_size_t* cuda_leaf_data_end() const { return cuda_leaf_data_end_; }

  const data_size_t* cuda_leaf_num_data() const { return cuda_leaf_num_data_; }

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_; }

  const int* cuda_cur_num_leaves() const { return cuda_cur_num_leaves_; }

  const int* tree_split_leaf_index() const { return tree_split_leaf_index_; }

  const int* tree_inner_feature_index() const { return tree_inner_feature_index_; }

  const uint32_t* tree_threshold() const { return tree_threshold_; }

  const double* tree_threshold_real() const { return tree_threshold_real_; }

  const double* tree_left_output() const { return tree_left_output_; }

  const double* tree_right_output() const { return tree_right_output_; }

  const data_size_t* tree_left_count() const { return tree_left_count_; }

  const data_size_t* tree_right_count() const { return tree_right_count_; }

  const double* tree_left_sum_hessian() const { return tree_left_sum_hessian_; }

  const double* tree_right_sum_hessian() const { return tree_right_sum_hessian_; }

  const double* tree_gain() const { return tree_gain_; }

  const uint8_t* tree_default_left() const { return tree_default_left_; }

 private:
  void CalcBlockDim(const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim);

  void CalcBlockDimInCopy(const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim);

  void GenDataToLeftBitVector(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start,
    const int left_leaf_index, const int right_leaf_index);

  void SplitInner(
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    std::vector<data_size_t>* cpu_leaf_num_data,
    std::vector<data_size_t>* cpu_leaf_data_start,
    std::vector<double>* cpu_leaf_sum_hessians,
    int* smaller_leaf_index,
    int* larger_leaf_index,
    const int leaf_index);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    std::vector<data_size_t>* cpu_leaf_num_data,
    std::vector<data_size_t>* cpu_leaf_data_start,
    std::vector<double>* cpu_leaf_sum_hessians,
    int* smaller_leaf_index,
    int* larger_leaf_index,
    const int cpu_leaf_index);

  void LaunchGenDataToLeftBitVectorKernel(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start,
    const int left_leaf_index, const int right_leaf_index);

  template <typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelMaxIsMinInner(
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_bin_to_left,
    const int column_index,
    const int num_blocks_final,
    const int split_indices_block_size_data_partition_aligned,
    const int split_feature_index,
    const data_size_t leaf_data_start,
    const data_size_t num_data_in_leaf,
    const uint32_t th,
    const uint32_t t_zero_bin,
    const uint32_t most_freq_bin,
    const uint32_t max_bin,
    const uint32_t min_bin,
    const uint8_t split_default_to_left,
    const uint8_t split_missing_default_to_left,
    const int left_leaf_index,
    const int right_leaf_index,
    const int default_leaf_index,
    const int missing_default_leaf_index);

  template <typename BIN_TYPE>
  void LaunchGenDataToLeftBitVectorKernelMaxIsNotMinInner(
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const int column_index,
    const int num_blocks_final,
    const int split_indices_block_size_data_partition_aligned,
    const int split_feature_index,
    const data_size_t leaf_data_start,
    const data_size_t num_data_in_leaf,
    const uint32_t th,
    const uint32_t t_zero_bin,
    const uint32_t most_freq_bin,
    const uint32_t max_bin,
    const uint32_t min_bin,
    const uint8_t split_default_to_left,
    const uint8_t split_missing_default_to_left,
    const int left_leaf_index,
    const int right_leaf_index,
    const int default_leaf_index,
    const int missing_default_leaf_index);

  template <typename BIN_TYPE>
  void LaunchUpdateDataIndexToLeafIndexKernel(const data_size_t cuda_leaf_data_start,
    const data_size_t num_data_in_leaf, const data_size_t* cuda_data_indices,
    const uint32_t th, const BIN_TYPE* column_data,
    // values from feature
    const uint32_t t_zero_bin, const uint32_t max_bin_ref, const uint32_t min_bin_ref,
    int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
    const int default_leaf_index, const int missing_default_leaf_index,
    const bool missing_is_zero, const bool missing_is_na, const bool mfb_is_zero, const bool mfb_is_na, const bool max_to_left,
    const int num_blocks, const int block_size);

  void LaunchPrefixSumKernel(uint32_t* cuda_elements);

  void LaunchAddPredictionToScoreKernel(const double learning_rate, double* cuda_scores);


  // Host memory

  // dataset information
  /*! \brief number of training data */
  const data_size_t num_data_;
  /*! \brief number of features in training data */
  const int num_features_;
  /*! \brief number of total bins in training data */
  const int num_total_bin_;
  /*! \brief upper bounds of feature histogram bins */
  std::vector<std::vector<double>> bin_upper_bounds_;
  /*! \brief number of bins per feature */
  std::vector<int> feature_num_bins_;
  /*! \brief bin data stored by column */
  const CUDAColumnData* cuda_column_data_;

  // config information
  /*! \brief maximum number of leaves in a tree */
  const int num_leaves_;
  /*! \brief number of threads */
  const int num_threads_;

  // tree structure information
  /*! \brief current number of leaves in tree */
  int cur_num_leaves_;

  // split algorithm related
  /*! \brief maximum number of blocks to aggregate after finding bit vector by blocks */
  int max_num_split_indices_blocks_;

  // CUDA streams
  /*! \brief cuda streams used for asynchronizing kernel computing and memory copy */
  std::vector<cudaStream_t> cuda_streams_;


  // CUDA memory, held by this object

  // tree structure information
  /*! \brief data indices by leaf */
  data_size_t* cuda_data_indices_;
  /*! \brief start position of each leaf in cuda_data_indices_ */
  data_size_t* cuda_leaf_data_start_;
  /*! \brief end position of each leaf in cuda_data_indices_  */
  data_size_t* cuda_leaf_data_end_;
  /*! \brief number of data in each leaf */
  data_size_t* cuda_leaf_num_data_;
  /*! \brief currnet number of leaves in tree */
  int* cuda_cur_num_leaves_;
  /*! \brief records the histogram of each leaf */
  hist_t** cuda_hist_pool_;
  /*! \brief records the value of each leaf */
  double* cuda_leaf_output_;

  // split data algorithm related
  /*! \brief marks whether each data goes to left or right, 1 for left, and 0 for right */
  uint8_t* cuda_data_to_left_;
  /*! \brief maps data index to leaf index, for adding scores to training data set */
  int* cuda_data_index_to_leaf_index_;
  /*! \brief prefix sum of number of data going to left in all blocks */
  data_size_t* cuda_block_data_to_left_offset_;
  /*! \brief prefix sum of number of data going to right in all blocks */
  data_size_t* cuda_block_data_to_right_offset_;
  /*! \brief buffer for splitting data indices, will be copied back to cuda_data_indices_ after split */
  data_size_t* cuda_out_data_indices_in_leaf_;

  // split tree structure algorithm related
  /*! \brief buffer to store split information, prepared to be copied to cpu */
  int* cuda_split_info_buffer_;
  /*! \brief the sequence of leaf indices being split during tree growing */
  int* tree_split_leaf_index_;
  /*! \brief the sequence of inner split indices during tree growing */
  int* tree_inner_feature_index_;
  /*! \brief the sequence of inner threshold during tree growing */
  uint32_t* tree_threshold_;
  /*! \brief the sequence of real threshold during tree growing */
  double* tree_threshold_real_;
  /*! \brief the sequence of left child output value of splits during tree growing */
  double* tree_left_output_;
  /*! \brief the sequence of right child output value of splits during tree growing */
  double* tree_right_output_;
  /*! \brief the sequence of left child data number value of splits during tree growing */
  data_size_t* tree_left_count_;
  /*! \brief the sequence of right child data number value of splits during tree growing */
  data_size_t* tree_right_count_;
  /*! \brief the sequence of left child hessian sum value of splits during tree growing */
  double* tree_left_sum_hessian_;
  /*! \brief the sequence of right child hessian sum value of splits during tree growing */
  double* tree_right_sum_hessian_;
  /*! \brief the sequence of split gains during tree growing */
  double* tree_gain_;
  /*! \brief the sequence of split default left during tree growing */
  uint8_t* tree_default_left_;

  // dataset information
  /*! \brief upper bounds of bin boundaries for feature histograms */
  double* cuda_bin_upper_bounds_;
  /*! \brief the bin offsets of features, used to access cuda_bin_upper_bounds_ */
  int* cuda_feature_num_bin_offsets_;


  // CUDA memory, held by other object

  // dataset information
  /*! \brief number of data in training set, for intialization of cuda_leaf_num_data_ and cuda_leaf_data_end_ */
  const data_size_t* cuda_num_data_;
  /*! \brief beginning of histograms, for initialization of cuda_hist_pool_ */
  hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
