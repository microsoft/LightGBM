/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_

#include <LightGBM/bin.h>

namespace LightGBM {

template <typename BIN_TYPE>
class PairwiseRankingFirstBin;

template <typename BIN_TYPE>
class PairwiseRankingSecondBin;

template <typename BIN_TYPE>
class PairwiseRankingFirstIterator: public BinIterator {
 friend PairwiseRankingFirstBin<BIN_TYPE>;
 public:
  PairwiseRankingFirstIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_ = unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin);
    unpaired_bin_iterator_->Reset(0);
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = 0;
    prev_val_ = 0;
  }

  ~PairwiseRankingFirstIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].first;
    if (data_index != prev_index_) {
      CHECK_GT(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_->Get(data_index);
    }
    prev_index_ = data_index;
    return prev_val_;
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].first;
    if (data_index != prev_index_) {
      CHECK_GT(data_index, prev_index_);
      prev_val_ = unpaired_bin_iterator_->RawGet(data_index);
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
  const BIN_TYPE* unpaired_bin_;
  BinIterator* unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
  uint32_t prev_val_;
};

template <typename BIN_TYPE>
class PairwiseRankingSecondIterator: public BinIterator {
 friend PairwiseRankingSecondBin<BIN_TYPE>;
 public:
  PairwiseRankingSecondIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_ = unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin);
    unpaired_bin_iterator_->Reset(0);
    paired_ranking_item_index_map_ = paired_ranking_item_index_map;
    prev_index_ = 0;
  }

  ~PairwiseRankingSecondIterator() {}

  uint32_t Get(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].second;
    if (data_index < prev_index_) {
      unpaired_bin_iterator_->Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_->Get(data_index);
  }

  uint32_t RawGet(data_size_t idx) {
    const data_size_t data_index = paired_ranking_item_index_map_[idx].second;
    if (data_index < prev_index_) {
      unpaired_bin_iterator_->Reset(0);
    }
    prev_index_ = data_index;
    return unpaired_bin_iterator_->RawGet(data_index);
  }

  void Reset(data_size_t idx) {
    unpaired_bin_iterator_->Reset(idx);
    prev_index_ = 0;
  }

 private:
  const BIN_TYPE* unpaired_bin_;
  BinIterator* unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
};

template <typename BIN_TYPE, template<typename> typename ITERATOR_TYPE>
class PairwiseRankingBin: public BIN_TYPE {
 public:
  PairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const BIN_TYPE* unpaired_bin): BIN_TYPE(0), paired_ranking_item_index_map_(paired_ranking_item_index_map), unpaired_bin_(unpaired_bin) {
    num_data_ = num_data;
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new ITERATOR_TYPE<BIN_TYPE>(unpaired_bin_.get(), paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

 protected:
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const std::shared_ptr<const BIN_TYPE> unpaired_bin_;
  data_size_t num_data_;
};

template <typename BIN_TYPE>
class PairwiseRankingFirstBin: public PairwiseRankingBin<BIN_TYPE, PairwiseRankingFirstIterator> {
 public:
  PairwiseRankingFirstBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const BIN_TYPE* unpaired_bin): PairwiseRankingBin<BIN_TYPE, PairwiseRankingFirstIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
};

template <typename BIN_TYPE>
class PairwiseRankingSecondBin: public PairwiseRankingBin<BIN_TYPE, PairwiseRankingSecondIterator> {
 public:
  PairwiseRankingSecondBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const BIN_TYPE* unpaired_bin): PairwiseRankingBin<BIN_TYPE, PairwiseRankingSecondIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
};

}  // LightGBM

#endif  // LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
