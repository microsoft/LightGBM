/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <LightGBM/pairwise_ranking_feature_group.h>
#include "pairwise_lambdarank_bin.hpp"

namespace LightGBM {

template <template<typename> typename PAIRWISE_BIN_TYPE>
void PairwiseRankingFeatureGroup::CreateBinDataInner(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
  if (is_multi_val) {
    multi_bin_data_.clear();
    for (int i = 0; i < num_feature_; ++i) {
      uint32_t most_freq_bin = bin_mappers_[i]->GetMostFreqBin();
      int addi = most_freq_bin == 0 ? 0 : 1;
      if (!is_multi_val) {
        uint32_t min_bin = bin_offsets_[i];
        uint32_t max_bin = bin_offsets_[i + 1] - 1;
      } else {
        uint32_t min_bin = 1;
        uint32_t max_bin = bin_mappers_[i]->num_bin() - 1 + addi;
      }
      if (bin_mappers_[i]->sparse_rate() >= kSparseThreshold) {
        multi_bin_data_.emplace_back(Bin::CreateSparsePairwiseRankingBin<PAIRWISE_BIN_TYPE>(
            num_data, bin_mappers_[i]->num_bin() + addi, paired_ranking_item_index_map_));
      } else {
        multi_bin_data_.emplace_back(
            Bin::CreateDensePairwiseRankingBin<PAIRWISE_BIN_TYPE>(num_data, bin_mappers_[i]->num_bin() + addi, paired_ranking_item_index_map_));
      }
    }
    is_multi_val_ = true;
  } else {
    if (force_sparse ||
        (!force_dense && num_feature_ == 1 &&
          bin_mappers_[0]->sparse_rate() >= kSparseThreshold)) {
      is_sparse_ = true;
      bin_data_.reset(Bin::CreateSparsePairwiseRankingBin<PAIRWISE_BIN_TYPE>(num_data, num_total_bin_, paired_ranking_item_index_map_));
    } else {
      is_sparse_ = false;
      bin_data_.reset(Bin::CreateDensePairwiseRankingBin<PAIRWISE_BIN_TYPE>(num_data, num_total_bin_, paired_ranking_item_index_map_));
    }
    is_multi_val_ = false;
  }
}

void PairwiseRankingFeatureGroup::CreateBinData(int num_data, bool is_multi_val, bool force_dense, bool force_sparse) {
  if (is_first_or_second_in_pairing_ == 0) {
    CreateBinDataInner<PairwiseRankingFirstBin>(num_data, is_multi_val, force_dense, force_sparse);
  } else {
    CreateBinDataInner<PairwiseRankingSecondBin>(num_data, is_multi_val, force_dense, force_sparse);
  }
}

} // namespace LightGBM
