/*!
 * Copyright (c) 2023 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_PAIRWISE_FEATURE_GROUP_H_
#define LIGHTGBM_PAIRWISE_FEATURE_GROUP_H_

#include "feature_group.h"

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

namespace LightGBM {

/*! \brief Using to store data and providing some operations on one feature
group*/
class PairwiseRankingFeatureGroup: public FeatureGroup {
 public:
  /*!
  * \brief Constructor
  * \param num_feature number of features of this group
  * \param bin_mappers Bin mapper for features
  * \param num_data Total number of data
  * \param is_enable_sparse True if enable sparse feature
  */

  PairwiseRankingFeatureGroup(const FeatureGroup& other, int num_data) {
    num_feature_ = other.num_feature_;
    is_multi_val_ = false;
    is_dense_multi_val_ = false;
    is_sparse_ = false;
    num_total_bin_ = other.num_total_bin_;
    bin_offsets_ = other.bin_offsets_;
    num_data_ = num_data;

    bin_mappers_.reserve(other.bin_mappers_.size());
    for (auto& bin_mapper : other.bin_mappers_) {
      bin_mappers_.emplace_back(new BinMapper(*bin_mapper));
    }
    CreateBinData(num_data, is_multi_val_, !is_sparse_, is_sparse_);
  }

  /*!
   * \brief Constructor from memory when data is present
   * \param memory Pointer of memory
   * \param num_all_data Number of global data
   * \param local_used_indices Local used indices, empty means using all data
   * \param group_id Id of group
   */
  PairwiseRankingFeatureGroup(const void* memory,
                              data_size_t num_all_data,
                              const std::vector<data_size_t>& local_used_indices,
                              int group_id) {
    // TODO(shiyu1994)
  }

  /*!
   * \brief Constructor from definition in memory (without data)
   * \param memory Pointer of memory
   * \param local_used_indices Local used indices, empty means using all data
   */
  FeatureGroup(const void* memory, data_size_t num_data, int group_id) {
    // TODO(shiyu1994)
  }

  /*! \brief Destructor */
  ~PairwiseRankingFeatureGroup() {}

  /*!
   * \brief Load the overall definition of the feature group from binary serialized data
   * \param memory Pointer of memory
   * \param group_id Id of group
   */
  const char* LoadDefinitionFromMemory(const void* memory, int group_id) {
    // TODO(shiyu1994)
  }

  inline BinIterator* SubFeatureIterator(int sub_feature) {
    // TODO(shiyu1994)
  }

  inline void FinishLoad() {
    // TODO(shiyu1994)
  }

  inline BinIterator* FeatureGroupIterator() {
    // TODO(shiyu1994)
  }

 private:
  void CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
    // TODO(shiyu1994)
  }

  
  /*! \brief Pairwise data index to original data indices for ranking with pairwise features  */
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  /*! \brief Number of pairwise data */
  data_size_t num_data_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_PAIRWISE_FEATURE_GROUP_H_
