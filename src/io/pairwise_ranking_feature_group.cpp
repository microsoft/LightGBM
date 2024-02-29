/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <LightGBM/pairwise_ranking_feature_group.h>
#include <LightGBM/utils/threading.h>

#include "pairwise_lambdarank_bin.hpp"

namespace LightGBM {

PairwiseRankingFeatureGroup::PairwiseRankingFeatureGroup(const FeatureGroup& other, int num_original_data, const int is_first_or_second_in_pairing, int num_pairs, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map):
  FeatureGroup(other, num_original_data), paired_ranking_item_index_map_(paired_ranking_item_index_map), num_data_(num_pairs), is_first_or_second_in_pairing_(is_first_or_second_in_pairing) {

  CreateBinData(num_original_data, is_multi_val_, !is_sparse_, is_sparse_);

  Threading::For<data_size_t>(0, num_original_data, 512, [this, &other] (int block_index, data_size_t block_start, data_size_t block_end) {
    for (int feature_index = 0; feature_index < num_feature_; ++feature_index) {
      std::unique_ptr<BinIterator> bin_iterator(other.SubFeatureIterator(feature_index));
      bin_iterator->Reset(block_start);
      for (data_size_t index = block_start; index < block_end; ++index) {
        PushBinData(block_index, feature_index, index, bin_iterator->Get(index));
      }
    }
  });

  FinishLoad();
}

void PairwiseRankingFeatureGroup::CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
  CHECK(!is_multi_val);  // do not support multi-value bin for now
  if (is_multi_val) {
    multi_bin_data_.clear();
    for (int i = 0; i < num_feature_; ++i) {
      int addi = bin_mappers_[i]->GetMostFreqBin() == 0 ? 0 : 1;
      if (bin_mappers_[i]->sparse_rate() >= kSparseThreshold) {
        if (is_first_or_second_in_pairing_ == 0) {
          multi_bin_data_.emplace_back(Bin::CreateSparsePairwiseRankingFirstBin(
              num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
        } else {
          multi_bin_data_.emplace_back(Bin::CreateSparsePairwiseRankingSecondBin(
              num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
        }
      } else {
        if (is_first_or_second_in_pairing_ == 0) {
          multi_bin_data_.emplace_back(
              Bin::CreateDensePairwiseRankingFirstBin(num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
        } else {
          multi_bin_data_.emplace_back(
              Bin::CreateDensePairwiseRankingSecondBin(num_data, bin_mappers_[i]->num_bin() + addi, num_data_, paired_ranking_item_index_map_));
        }
      }
    }
    is_multi_val_ = true;
  } else {
    if (force_sparse ||
        (!force_dense && num_feature_ == 1 &&
          bin_mappers_[0]->sparse_rate() >= kSparseThreshold)) {
      is_sparse_ = true;
      if (is_first_or_second_in_pairing_ == 0) {
        bin_data_.reset(Bin::CreateSparsePairwiseRankingFirstBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
      } else {
        bin_data_.reset(Bin::CreateSparsePairwiseRankingSecondBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
      }
    } else {
      is_sparse_ = false;
      if (is_first_or_second_in_pairing_ == 0) {
        bin_data_.reset(Bin::CreateDensePairwiseRankingFirstBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
      } else {
        bin_data_.reset(Bin::CreateDensePairwiseRankingSecondBin(num_data, num_total_bin_, num_data_, paired_ranking_item_index_map_));
      }
    }
    is_multi_val_ = false;
  }
}

}  // namespace LightGBM
