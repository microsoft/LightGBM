/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_
#define LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_

#include <LightGBM/meta.h>

#include <limits>
#include <vector>

#include "data_partition.hpp"

namespace LightGBM {

/*!
* \brief used to find split candidates for a leaf
*/
class LeafSplits {
 public:
  explicit LeafSplits(data_size_t num_data)
    :num_data_in_leaf_(num_data), num_data_(num_data),
    data_indices_(nullptr) {
  }
  void ResetNumData(data_size_t num_data) {
    num_data_ = num_data;
    num_data_in_leaf_ = num_data;
  }
  ~LeafSplits() {
  }

  /*!

  * \brief Init split on current leaf on partial data. 
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(int leaf, const DataPartition* data_partition, double sum_gradients, double sum_hessians) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    min_val_ = -std::numeric_limits<double>::max();
    max_val_ = std::numeric_limits<double>::max();
  }

  void SetValueConstraint(double min, double max) {
    min_val_ = min;
    max_val_ = max;
  }


  /*!
  * \brief Init splits on current leaf, it will traverse all data to sum up the results
  * \param gradients
  * \param hessians
  */
  void Init(const score_t* gradients, const score_t* hessians) {
    num_data_in_leaf_ = num_data_;
    leaf_index_ = 0;
    data_indices_ = nullptr;
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum_gradients, tmp_sum_hessians)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      tmp_sum_gradients += gradients[i];
      tmp_sum_hessians += hessians[i];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
    min_val_ = -std::numeric_limits<double>::max();
    max_val_ = std::numeric_limits<double>::max();
  }

  /*!
  * \brief Init splits on current leaf of partial data.
  * \param leaf Index of current leaf
  * \param data_partition current data partition
  * \param gradients
  * \param hessians
  */
  void Init(int leaf, const DataPartition* data_partition, const score_t* gradients, const score_t* hessians) {
    leaf_index_ = leaf;
    data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
    double tmp_sum_gradients = 0.0f;
    double tmp_sum_hessians = 0.0f;
#pragma omp parallel for schedule(static) reduction(+:tmp_sum_gradients, tmp_sum_hessians)
    for (data_size_t i = 0; i < num_data_in_leaf_; ++i) {
      data_size_t idx = data_indices_[i];
      tmp_sum_gradients += gradients[idx];
      tmp_sum_hessians += hessians[idx];
    }
    sum_gradients_ = tmp_sum_gradients;
    sum_hessians_ = tmp_sum_hessians;
    min_val_ = -std::numeric_limits<double>::max();
    max_val_ = std::numeric_limits<double>::max();
  }


  /*!
  * \brief Init splits on current leaf, only update sum_gradients and sum_hessians
  * \param sum_gradients
  * \param sum_hessians
  */
  void Init(double sum_gradients, double sum_hessians) {
    leaf_index_ = 0;
    sum_gradients_ = sum_gradients;
    sum_hessians_ = sum_hessians;
    min_val_ = -std::numeric_limits<double>::max();
    max_val_ = std::numeric_limits<double>::max();
  }

  /*!
  * \brief Init splits on current leaf
  */
  void Init() {
    leaf_index_ = -1;
    data_indices_ = nullptr;
    num_data_in_leaf_ = 0;
    min_val_ = -std::numeric_limits<double>::max();
    max_val_ = std::numeric_limits<double>::max();
  }


  /*! \brief Get current leaf index */
  int LeafIndex() const { return leaf_index_; }

  /*! \brief Get numer of data in current leaf */
  data_size_t num_data_in_leaf() const { return num_data_in_leaf_; }

  /*! \brief Get sum of gradients of current leaf */
  double sum_gradients() const { return sum_gradients_; }

  /*! \brief Get sum of hessians of current leaf */
  double sum_hessians() const { return sum_hessians_; }

  double max_constraint() const { return max_val_; }

  double min_constraint() const { return min_val_; }

  /*! \brief Get indices of data of current leaf */
  const data_size_t* data_indices() const { return data_indices_; }


 private:
  /*! \brief current leaf index */
  int leaf_index_;
  /*! \brief number of data on current leaf */
  data_size_t num_data_in_leaf_;
  /*! \brief number of all training data */
  data_size_t num_data_;
  /*! \brief sum of gradients of current leaf */
  double sum_gradients_;
  /*! \brief sum of hessians of current leaf */
  double sum_hessians_;
  /*! \brief indices of data of current leaf */
  const data_size_t* data_indices_;
  double min_val_;
  double max_val_;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LEAF_SPLITS_HPP_
