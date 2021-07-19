/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_
#define LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_

#include <LightGBM/cuda/cuda_column_data.hpp>
#include <LightGBM/tree.h>

namespace LightGBM {

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

 private:
  void InitCUDA();

  void LaunchAddPredictionToScoreKernel(const Dataset* data,
                                  data_size_t num_data,
                                  double* score) const;

  void LaunchAddPredictionToScoreKernel(const Dataset* data,
                                  const data_size_t* used_data_indices,
                                  data_size_t num_data, double* score) const;

  void LaunchShrinkageKernel(const double rate);

  int* cuda_left_child_;
  int* cuda_right_child_;
  int* cuda_split_feature_inner_;
  int* cuda_split_feature_;
  uint32_t* cuda_threshold_in_bin_;
  double* cuda_threshold_;
  int8_t* cuda_decision_type_;
  double* cuda_leaf_value_;

  const int num_threads_per_block_add_prediction_to_score_;
};

}  //namespace LightGBM

#endif  // LIGHTGBM_IO_CUDA_CUDA_TREE_HPP_
