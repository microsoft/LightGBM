#ifndef LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_
#define LIGHTGBM_TREELEARNER_LEAF_SPLITS_HPP_

#include <LightGBM/meta.h>
#include "data_partition.hpp"
#include "split_info.hpp"

#include <vector>

namespace LightGBM {

/*!
* \brief used to find split candidates for a leaf
*/
class LeafSplits {
public:
  LeafSplits(int num_feature, data_size_t num_data)
    :num_data_in_leaf_(num_data), num_data_(num_data), num_features_(num_feature),
    data_indices_(nullptr) {
    best_split_per_feature_.resize(num_features_);
    for (int i = 0; i < num_features_; ++i) {
      best_split_per_feature_[i].feature = i;
    }
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
    for (SplitInfo& split_info : best_split_per_feature_) {
      split_info.Reset();
    }
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
    for (SplitInfo& split_info : best_split_per_feature_) {
      split_info.Reset();
    }
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
    for (SplitInfo& split_info : best_split_per_feature_) {
      split_info.Reset();
    }
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
    for (SplitInfo& split_info : best_split_per_feature_) {
      split_info.Reset();
    }
  }

  /*!
  * \brief Init splits on current leaf
  */
  void Init() {
    leaf_index_ = -1;
    for (SplitInfo& split_info : best_split_per_feature_) {
      split_info.Reset();
    }
  }

  /*! \brief Get best splits on all features */
  std::vector<SplitInfo>& BestSplitPerFeature() { return best_split_per_feature_;}

  /*! \brief Get current leaf index */
  int LeafIndex() const { return leaf_index_; }

  /*! \brief Get numer of data in current leaf */
  data_size_t num_data_in_leaf() const { return num_data_in_leaf_; }

  /*! \brief Get sum of gradients of current leaf */
  double sum_gradients() const { return sum_gradients_; }
  
  /*! \brief Get sum of hessians of current leaf */
  double sum_hessians() const { return sum_hessians_; }

  /*! \brief Get indices of data of current leaf */
  const data_size_t* data_indices() const { return data_indices_; }


private:
  /*! \brief store best splits of all feature on current leaf */
  std::vector<SplitInfo> best_split_per_feature_;
  /*! \brief current leaf index */
  int leaf_index_;
  /*! \brief number of data on current leaf */
  data_size_t num_data_in_leaf_;
  /*! \brief number of all training data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  /*! \brief sum of gradients of current leaf */
  double sum_gradients_;
  /*! \brief sum of hessians of current leaf */
  double sum_hessians_;
  /*! \brief indices of data of current leaf */
  const data_size_t* data_indices_;
};

}  // namespace LightGBM
#endif   // LightGBM_TREELEARNER_LEAF_SPLITS_HPP_
