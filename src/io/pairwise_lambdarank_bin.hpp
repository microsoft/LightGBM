/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_

#include <LightGBM/bin.h>

namespace LightGBM {

template <typename VAL_T>
class PairwiseRankingFirstIterator: public BinIterator {
 public:
  PairwiseRankingFirstIterator(const Bin* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_ = unpaired_bin_->GetIterator();
    unpaired_bin_iterator_->Reset();
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = 0;
    prev_val_ = 0;
  }

  ~PairwiseRankingFirstIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map[idx].first
    if (data_index != prev_index_) {
      CHECK_GT(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_i_->Get(data_index);
    }
    prev_index_ = data_index;
    return prev_val_;
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map[idx].first
    if (data_index != prev_index_) {
      CHECK_GT(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_i_->RawGet(data_index);
    }
    prev_index_ = data_index;
    return prev_val_;
  }

  void Reset(data_size_t idx) {
    unpaired_bin_iterator_->Reset(idx);
    prev_index_ = 0;
    prev_val_ = 0;
  }

 private:
  const Bin* unpaired_bin_;
  BinIterator* unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const data_size_t prev_index_;
  const uint32_t prev_val_;
};

template <typename VAL_T>
class PairwiseRankingSecondIterator: public BinIterator {
 public:
  PairwiseRankingSecondIterator(const Bin* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_ = unpaired_bin_->GetIterator();
    unpaired_bin_iterator_->Reset();
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = 0;
    prev_val_ = 0;
  }

  ~PairwiseRankingSecondIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map[idx].second
    if (data_index < prev_index_) {
      unpaired_bin_iterator_i_.Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_i_->Get(data_index);
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map[idx].second
    if (data_index < prev_index_) {
      unpaired_bin_iterator_i_.Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_i_->RawGet(data_index);
  }

  void Reset(data_size_t idx) {
    unpaired_bin_iterator_->Reset(idx);
    prev_index_ = 0;
  }

 private:
  const Bin* unpaired_bin_;
  BinIterator* unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const data_size_t prev_index_;
};

}  // LightGBM

#endif  // LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
