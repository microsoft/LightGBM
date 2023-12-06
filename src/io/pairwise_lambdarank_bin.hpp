/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
#define LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_

#include <memory>
#include <utility>

#include <LightGBM/bin.h>

#include "dense_bin.hpp"
#include "sparse_bin.hpp"

namespace LightGBM {

template <typename BIN_TYPE>
class PairwiseRankingFirstBin;

template <typename BIN_TYPE>
class PairwiseRankingSecondBin;

template <typename BIN_TYPE>
class PairwiseRankingFirstIterator: public BinIterator {
 public:
  friend PairwiseRankingFirstBin<BIN_TYPE>;

  PairwiseRankingFirstIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
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
  std::unique_ptr<BinIterator> unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
  uint32_t prev_val_;
};

template <typename BIN_TYPE>
class PairwiseRankingSecondIterator: public BinIterator {
 public:
  friend PairwiseRankingSecondBin<BIN_TYPE>;

  PairwiseRankingSecondIterator(const BIN_TYPE* unpaired_bin, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, const uint32_t min_bin, const uint32_t max_bin, const uint32_t most_freq_bin) {
    unpaired_bin_ = unpaired_bin;
    unpaired_bin_iterator_.reset(unpaired_bin_->GetIterator(min_bin, max_bin, most_freq_bin));
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
  std::unique_ptr<BinIterator> unpaired_bin_iterator_;
  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  data_size_t prev_index_;
};

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
class PairwiseRankingBin: public BIN_TYPE {
 public:
  PairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, BIN_TYPE* unpaired_bin): BIN_TYPE(0), paired_ranking_item_index_map_(paired_ranking_item_index_map), unpaired_bin_(unpaired_bin) {
    num_data_ = num_data;
  }

  virtual ~PairwiseRankingBin() {}

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override {
    return new ITERATOR_TYPE<BIN_TYPE>(unpaired_bin_.get(), paired_ranking_item_index_map_, min_bin, max_bin, most_freq_bin);
  }

  void InitStreaming(uint32_t num_thread, int32_t omp_max_threads) override;

  void Push(int tid, data_size_t idx, uint32_t value) override;

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override;

  void SaveBinaryToFile(BinaryWriter* writer) const override;

  void LoadFromMemory(const void* memory,
    const std::vector<data_size_t>& local_used_indices) override;

  size_t SizesInByte() const override;

  data_size_t num_data() const override;

  void* get_data() override {
    return unpaired_bin_->get_data();
  }

  void ReSize(data_size_t num_data) override;

  data_size_t Split(uint32_t /*min_bin*/, uint32_t /*max_bin*/,
                    uint32_t /*default_bin*/, uint32_t /*most_freq_bin*/,
                    MissingType /*missing_type*/, bool /*default_left*/,
                    uint32_t /*threshold*/, const data_size_t* /*data_indices*/,
                    data_size_t /*cnt*/,
                    data_size_t* /*lte_indices*/,
                    data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
  }

  data_size_t SplitCategorical(
      uint32_t /*min_bin*/, uint32_t /*max_bin*/, uint32_t /*most_freq_bin*/,
      const uint32_t* /*threshold*/, int /*num_threshold*/,
      const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
      data_size_t* /*lte_indices*/, data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
  }

  data_size_t Split(uint32_t /*max_bin*/, uint32_t /*default_bin*/,
                            uint32_t /*most_freq_bin*/, MissingType /*missing_type*/,
                            bool /*default_left*/, uint32_t /*threshold*/,
                            const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
                            data_size_t* /*lte_indices*/,
                            data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
  }

  data_size_t SplitCategorical(
      uint32_t /*max_bin*/, uint32_t /*most_freq_bin*/, const uint32_t* /*threshold*/,
      int /*num_threshold*/, const data_size_t* /*data_indices*/, data_size_t /*cnt*/,
      data_size_t* /*lte_indices*/, data_size_t* /*gt_indices*/) const override {
    Log::Fatal("Not implemented.");
  }

  const void* GetColWiseData(uint8_t* /*bit_type*/, bool* /*is_sparse*/, std::vector<BinIterator*>* /*bin_iterator*/, const int /*num_threads*/) const override {
    Log::Fatal("Not implemented.");
  }

  const void* GetColWiseData(uint8_t* /*bit_type*/, bool* /*is_sparse*/, BinIterator** /*bin_iterator*/) const override {
    Log::Fatal("Not implemented.");
  }

 protected:

  virtual data_size_t get_unpaired_index(const data_size_t paired_index) const = 0;

  const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map_;
  const std::unique_ptr<BIN_TYPE> unpaired_bin_;
  data_size_t num_data_;
};

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
class DensePairwiseRankingBin: public PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE> {
 public:
  DensePairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>(num_data, paired_ranking_item_index_map, unpaired_bin) {}

  void ConstructHistogram(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt8(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt16(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt32(
    const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override;

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogram(data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt8(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt8(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt16(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt16(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt32(const data_size_t* data_indices, data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  void ConstructHistogramInt32(data_size_t start, data_size_t end,
                                       const score_t* ordered_gradients, hist_t* out) const override;

  data_size_t Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const override;

  data_size_t Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const override;

 private:
  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
  void ConstructHistogramInner(const data_size_t* data_indices,
                               data_size_t start, data_size_t end,
                               const score_t* ordered_gradients,
                               const score_t* ordered_hessians,
                               hist_t* out) const;

  template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
  void ConstructHistogramIntInner(const data_size_t* data_indices,
                                  data_size_t start, data_size_t end,
                                  const score_t* ordered_gradients,
                                  hist_t* out) const;

  template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
  data_size_t SplitInner(uint32_t min_bin, uint32_t max_bin,
                         uint32_t default_bin, uint32_t most_freq_bin,
                         bool default_left, uint32_t threshold,
                         const data_size_t* data_indices, data_size_t cnt,
                         data_size_t* lte_indices,
                         data_size_t* gt_indices) const;
};

template <typename VAL_T, template<typename> class ITERATOR_TYPE>
class SparsePairwiseRankingBin: public PairwiseRankingBin<SparseBin<VAL_T>, ITERATOR_TYPE> {
 public:
  SparsePairwiseRankingBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): PairwiseRankingBin<SparseBin<VAL_T>, ITERATOR_TYPE>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
};

template <typename VAL_T, bool IS_4BIT>
class DensePairwiseRankingFirstBin: public DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingFirstIterator> {
 public:
  DensePairwiseRankingFirstBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingFirstIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].first;
  }
};

template <typename VAL_T, bool IS_4BIT>
class DensePairwiseRankingSecondBin: public DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingSecondIterator> {
 public:
  DensePairwiseRankingSecondBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, DenseBin<VAL_T, IS_4BIT>* unpaired_bin): DensePairwiseRankingBin<VAL_T, IS_4BIT, PairwiseRankingSecondIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].second;
  }
};

template <typename VAL_T>
class SparsePairwiseRankingFirstBin: public SparsePairwiseRankingBin<VAL_T, PairwiseRankingFirstIterator> {
 public:
  SparsePairwiseRankingFirstBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): SparsePairwiseRankingBin<VAL_T, PairwiseRankingFirstIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].first;
  }
};

template <typename VAL_T>
class SparsePairwiseRankingSecondBin: public SparsePairwiseRankingBin<VAL_T, PairwiseRankingSecondIterator> {
 public:
  SparsePairwiseRankingSecondBin(data_size_t num_data, const std::pair<data_size_t, data_size_t>* paired_ranking_item_index_map, SparseBin<VAL_T>* unpaired_bin): SparsePairwiseRankingBin<VAL_T, PairwiseRankingSecondIterator>(num_data, paired_ranking_item_index_map, unpaired_bin) {}
 private:
  data_size_t get_unpaired_index(const data_size_t paired_index) const {
    return this->paired_ranking_item_index_map_[paired_index].second;
  }
};


}  // namespace LightGBM

#endif  // LIGHTGBM_IO_PAIRWISE_LAMBDARANK_BIN_HPP_
