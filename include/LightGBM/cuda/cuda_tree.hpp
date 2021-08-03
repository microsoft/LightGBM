/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_
#define LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/cuda/cuda_split_info.hpp>
#include <LightGBM/tree.h>
#include "../bin.h"

namespace LightGBM {

__device__ void SetDecisionTypeCUDA(int8_t* decision_type, bool input, int8_t mask);

__device__ void SetMissingTypeCUDA(int8_t* decision_type, int8_t input);

__device__ bool GetDecisionTypeCUDA(int8_t decision_type, int8_t mask);

__device__ int8_t GetMissingTypeCUDA(int8_t decision_type);

__device__ bool IsZeroCUDA(double fval);

class CUDATree : public Tree {
 public:
  /*!
  * \brief Constructor
  * \param max_leaves The number of max leaves
  * \param track_branch_features Whether to keep track of ancestors of leaf nodes
  * \param is_linear Whether the tree has linear models at each leaf
  */
  explicit CUDATree(int max_leaves, bool track_branch_features, bool is_linear);

  explicit CUDATree(const Tree* host_tree);

  ~CUDATree() noexcept;

  int Split(const int leaf_index,
             const int real_feature_index,
             const double real_threshold,
             const MissingType missing_type,
             const CUDASplitInfo* cuda_split_info);

  /*!
  * \brief Adding prediction value of this tree model to scores
  * \param data The dataset
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            data_size_t num_data,
                            double* score) const override;

  /*!
  * \brief Adding prediction value of this tree model to scores
  * \param data The dataset
  * \param used_data_indices Indices of used data
  * \param num_data Number of total data
  * \param score Will add prediction to score
  */
  void AddPredictionToScore(const Dataset* data,
                            const data_size_t* used_data_indices,
                            data_size_t num_data, double* score) const override;

  const int* cuda_left_child() const { return cuda_left_child_; }

  const int* cuda_right_child() const { return cuda_right_child_; }

  const int* cuda_split_feature_inner() const { return cuda_split_feature_inner_; }

  const int* cuda_split_feature() const { return cuda_split_feature_; }

  const uint32_t* cuda_threshold_in_bin() const { return cuda_threshold_in_bin_; }

  const double* cuda_threshold() const { return cuda_threshold_; }

  const int8_t* cuda_decision_type() const { return cuda_decision_type_; }

  const double* cuda_leaf_value() const { return cuda_leaf_value_; }

  inline void Shrinkage(double rate) override;

  void ToHost();

 private:
  void InitCUDAMemory();

  void InitCUDA();

  void LaunchSplitKernel(const int leaf_index,
                         const int real_feature_index,
                         const double real_threshold,
                         const MissingType missing_type,
                         const CUDASplitInfo* cuda_split_info);

  void LaunchAddPredictionToScoreKernel(const Dataset* data,
                                        const data_size_t* used_data_indices,
                                        data_size_t num_data, double* score) const;

  void LaunchShrinkageKernel(const double rate);

  int* cuda_left_child_;
  int* cuda_right_child_;
  int* cuda_split_feature_inner_;
  int* cuda_split_feature_;
  int* cuda_leaf_depth_;
  int* cuda_leaf_parent_;
  uint32_t* cuda_threshold_in_bin_;
  double* cuda_threshold_;
  double* cuda_internal_weight_;
  double* cuda_internal_value_;
  int8_t* cuda_decision_type_;
  double* cuda_leaf_value_;
  data_size_t* cuda_leaf_count_;
  double* cuda_leaf_weight_;
  data_size_t* cuda_internal_count_;
  float* cuda_split_gain_;

  cudaStream_t cuda_stream_;

  const int num_threads_per_block_add_prediction_to_score_;
};

}  //namespace LightGBM

#endif  // LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_
