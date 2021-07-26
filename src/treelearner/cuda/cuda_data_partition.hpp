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

#include "cuda_leaf_splits.hpp"
#include <LightGBM/cuda/cuda_split_info.hpp>

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
    hist_t* cuda_hist);

  void Init();

  void BeforeTrain(const data_size_t* data_indices);

  void Split(
    // input best split info
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    const int leaf_best_split_feature,
    const uint32_t leaf_best_split_threshold,
    const uint8_t leaf_best_split_default_left,
    const data_size_t num_data_in_leaf,
    const data_size_t leaf_data_start,
    // for leaf information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians);

  void UpdateTrainScore(const double learning_rate, double* cuda_scores);

  const data_size_t* cuda_data_indices() const { return cuda_data_indices_; }

 private:
  void CalcBlockDim(
    const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim);

  void CalcBlockDimInCopy(
    const data_size_t num_data_in_leaf,
    int* grid_dim,
    int* block_dim);

  void GenDataToLeftBitVector(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

  void SplitInner(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians);

  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitInnerKernel(
    // input best split info
    const data_size_t num_data_in_leaf,
    const CUDASplitInfo* best_split_info,
    const int left_leaf_index,
    const int right_leaf_index,
    // for leaf splits information update
    CUDALeafSplitsStruct* smaller_leaf_splits,
    CUDALeafSplitsStruct* larger_leaf_splits,
    // gather information for CPU, used for launching kernels
    data_size_t* left_leaf_num_data,
    data_size_t* right_leaf_num_data,
    data_size_t* left_leaf_start,
    data_size_t* right_leaf_start,
    double* left_leaf_sum_of_hessians,
    double* right_leaf_sum_of_hessians);

  void LaunchGenDataToLeftBitVectorKernel(
    const data_size_t num_data_in_leaf,
    const int split_feature_index,
    const uint32_t split_threshold,
    const uint8_t split_default_left,
    const data_size_t leaf_data_start,
    const int left_leaf_index,
    const int right_leaf_index);

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
  void LaunchUpdateDataIndexToLeafIndexKernel(
    const data_size_t num_data_in_leaf,
    const data_size_t* data_indices_in_leaf,
    const uint32_t th,
    const BIN_TYPE* column_data,
    // values from feature
    const uint32_t t_zero_bin,
    const uint32_t max_bin_ref,
    const uint32_t min_bin_ref,
    int* cuda_data_index_to_leaf_index,
    const int left_leaf_index,
    const int right_leaf_index,
    const int default_leaf_index,
    const int missing_default_leaf_index,
    const bool missing_is_zero,
    const bool missing_is_na,
    const bool mfb_is_zero,
    const bool mfb_is_na,
    const bool max_to_left,
    const int num_blocks,
    const int block_size);

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

  // dataset information
  /*! \brief upper bounds of bin boundaries for feature histograms */
  double* cuda_bin_upper_bounds_;
  /*! \brief the bin offsets of features, used to access cuda_bin_upper_bounds_ */
  int* cuda_feature_num_bin_offsets_;
  /*! \brief number of data in training set, for intialization of cuda_leaf_num_data_ and cuda_leaf_data_end_ */
  data_size_t* cuda_num_data_;


  // CUDA memory, held by other object

  // dataset information
  /*! \brief beginning of histograms, for initialization of cuda_hist_pool_ */
  hist_t* cuda_hist_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
