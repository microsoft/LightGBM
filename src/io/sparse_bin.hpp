/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_SPARSE_BIN_HPP_
#define LIGHTGBM_IO_SPARSE_BIN_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <limits>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace LightGBM {

template <typename VAL_T> class SparseBin;

const size_t kNumFastIndex = 64;

template <typename VAL_T>
class SparseBinIterator: public BinIterator {
public:
  SparseBinIterator(const SparseBin<VAL_T>* bin_data,
    uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin)
    : bin_data_(bin_data), min_bin_(static_cast<VAL_T>(min_bin)),
    max_bin_(static_cast<VAL_T>(max_bin)),
    most_freq_bin_(static_cast<VAL_T>(most_freq_bin)) {
    if (most_freq_bin_ == 0) {
      offset_ = 1;
    } else {
      offset_ = 0;
    }
    Reset(0);
  }
  SparseBinIterator(const SparseBin<VAL_T>* bin_data, data_size_t start_idx)
    : bin_data_(bin_data) {
    Reset(start_idx);
  }

  inline uint32_t RawGet(data_size_t idx) override;
  inline VAL_T InnerRawGet(data_size_t idx);

  inline uint32_t Get(data_size_t idx) override {
    VAL_T ret = InnerRawGet(idx);
    if (ret >= min_bin_ && ret <= max_bin_) {
      return ret - min_bin_ + offset_;
    } else {
      return most_freq_bin_;
    }
  }

  inline void Reset(data_size_t idx) override;

private:
  const SparseBin<VAL_T>* bin_data_;
  data_size_t cur_pos_;
  data_size_t i_delta_;
  VAL_T min_bin_;
  VAL_T max_bin_;
  VAL_T most_freq_bin_;
  uint8_t offset_;
};

template <typename VAL_T>
class SparseBin: public Bin {
public:
  friend class SparseBinIterator<VAL_T>;

  explicit SparseBin(data_size_t num_data)
    : num_data_(num_data) {
    int num_threads = 1;
    #pragma omp parallel
    #pragma omp master
    {
      num_threads = omp_get_num_threads();
    }
    push_buffers_.resize(num_threads);
  }

  ~SparseBin() {
  }

  void ReSize(data_size_t num_data) override {
    num_data_ = num_data;
  }

  void Push(int tid, data_size_t idx, uint32_t value) override {
    auto cur_bin = static_cast<VAL_T>(value);
    if (cur_bin != 0) {
      push_buffers_[tid].emplace_back(idx, cur_bin);
    }
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override;

  #define ACC_GH(hist, i, g, h) \
  const auto ti = static_cast<int>(i) << 1; \
  hist[ti] += g; \
  hist[ti + 1] += h; \

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(data_indices[start], &i_delta, &cur_pos);
    data_size_t i = start;
    for (;;) {
      if (cur_pos < data_indices[i]) {
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) { break; }
      } else if (cur_pos > data_indices[i]) {
        if (++i >= end) { break; }
      } else {
        const VAL_T bin = vals_[i_delta];
        ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
        if (++i >= end) { break; }
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) { break; }
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(start, &i_delta, &cur_pos);
    while (cur_pos < start && i_delta < num_vals_) {
      cur_pos += deltas_[++i_delta];
    }
    while (cur_pos < end && i_delta < num_vals_) {
      const VAL_T bin = vals_[i_delta];
      ACC_GH(out, bin, ordered_gradients[cur_pos], ordered_hessians[cur_pos]);
      cur_pos += deltas_[++i_delta];
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(data_indices[start], &i_delta, &cur_pos);
    data_size_t i = start;
    for (;;) {
      if (cur_pos < data_indices[i]) {
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) { break; }
      } else if (cur_pos > data_indices[i]) {
        if (++i >= end) { break; }
      } else {
        const VAL_T bin = vals_[i_delta];
        ACC_GH(out, bin, ordered_gradients[i], 1.0f);
        if (++i >= end) { break; }
        cur_pos += deltas_[++i_delta];
        if (i_delta >= num_vals_) { break; }
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    data_size_t i_delta, cur_pos;
    InitIndex(start, &i_delta, &cur_pos);
    while (cur_pos < start && i_delta < num_vals_) {
      cur_pos += deltas_[++i_delta];
    }
    while (cur_pos < end && i_delta < num_vals_) {
      const VAL_T bin = vals_[i_delta];
      ACC_GH(out, bin, ordered_gradients[cur_pos], 1.0f);
      cur_pos += deltas_[++i_delta];
    }
  }
  #undef ACC_GH

  inline void NextNonzeroFast(data_size_t* i_delta,
    data_size_t* cur_pos) const {
    *cur_pos += deltas_[++(*i_delta)];
    if (*i_delta >= num_vals_) {
      *cur_pos = num_data_;
    }
  }

  inline bool NextNonzero(data_size_t* i_delta,
    data_size_t* cur_pos) const {
    *cur_pos += deltas_[++(*i_delta)];
    if (*i_delta < num_vals_) {
      return true;
    } else {
      *cur_pos = num_data_;
      return false;
    }
  }


  data_size_t Split(
    uint32_t min_bin, uint32_t max_bin, uint32_t default_bin, uint32_t most_freq_bin, MissingType missing_type, bool default_left,
    uint32_t threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    if (num_data <= 0) { return 0; }
    VAL_T th = static_cast<VAL_T>(threshold + min_bin);
    const VAL_T minb = static_cast<VAL_T>(min_bin);
    const VAL_T maxb = static_cast<VAL_T>(max_bin);
    VAL_T t_default_bin = static_cast<VAL_T>(min_bin + default_bin);
    VAL_T t_most_freq_bin = static_cast<VAL_T>(min_bin + most_freq_bin);
    if (most_freq_bin == 0) {
      th -= 1;
      t_default_bin -= 1;
      t_most_freq_bin -= 1;
    }
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    data_size_t* default_indices = gt_indices;
    data_size_t* default_count = &gt_count;
    data_size_t* missing_default_indices = gt_indices;
    data_size_t* missing_default_count = &gt_count;
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    if (most_freq_bin <= threshold) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    if (missing_type == MissingType::NaN) {
      if (default_left) {
        missing_default_indices = lte_indices;
        missing_default_count = &lte_count;
      }
      for (data_size_t i = 0; i < num_data; ++i) {
        const data_size_t idx = data_indices[i];
        const VAL_T bin = iterator.InnerRawGet(idx);
        if (bin == maxb) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else if (bin < minb || bin > maxb || t_most_freq_bin == bin) {
          default_indices[(*default_count)++] = idx;
        } else if (bin > th) {
          gt_indices[gt_count++] = idx;
        } else {
          lte_indices[lte_count++] = idx;
        }
      }
    } else {
      if ((default_left && missing_type == MissingType::Zero)
        || (default_bin <= threshold && missing_type != MissingType::Zero)) {
        missing_default_indices = lte_indices;
        missing_default_count = &lte_count;
      }
      if (default_bin == most_freq_bin) {
        for (data_size_t i = 0; i < num_data; ++i) {
          const data_size_t idx = data_indices[i];
          const VAL_T bin = iterator.InnerRawGet(idx);
          if (bin < minb || bin > maxb || t_most_freq_bin == bin) {
            missing_default_indices[(*missing_default_count)++] = idx;
          } else if (bin > th) {
            gt_indices[gt_count++] = idx;
          } else {
            lte_indices[lte_count++] = idx;
          }
        }
      } else {
        for (data_size_t i = 0; i < num_data; ++i) {
          const data_size_t idx = data_indices[i];
          const VAL_T bin = iterator.InnerRawGet(idx);
          if (bin == t_default_bin) {
            missing_default_indices[(*missing_default_count)++] = idx;
          } else if (bin < minb || bin > maxb || t_most_freq_bin == bin) {
            default_indices[(*default_count)++] = idx;
          } else if (bin > th) {
            gt_indices[gt_count++] = idx;
          } else {
            lte_indices[lte_count++] = idx;
          }
        }
      }
    }
    return lte_count;
  }

  data_size_t SplitCategorical(
    uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin,
    const uint32_t* threshold, int num_threahold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    if (num_data <= 0) { return 0; }
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    data_size_t* default_indices = gt_indices;
    data_size_t* default_count = &gt_count;
    if (Common::FindInBitset(threshold, num_threahold, most_freq_bin)) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      uint32_t bin = iterator.InnerRawGet(idx);
      if (bin < min_bin || bin > max_bin) {
        default_indices[(*default_count)++] = idx;
      } else if (Common::FindInBitset(threshold, num_threahold, bin - min_bin)) {
        lte_indices[lte_count++] = idx;
      } else {
        gt_indices[gt_count++] = idx;
      }
    }
    return lte_count;
  }

  data_size_t num_data() const override { return num_data_; }

  void FinishLoad() override {
    // get total non zero size
    size_t pair_cnt = 0;
    for (size_t i = 0; i < push_buffers_.size(); ++i) {
      pair_cnt += push_buffers_[i].size();
    }
    std::vector<std::pair<data_size_t, VAL_T>>& idx_val_pairs = push_buffers_[0];
    idx_val_pairs.reserve(pair_cnt);

    for (size_t i = 1; i < push_buffers_.size(); ++i) {
      idx_val_pairs.insert(idx_val_pairs.end(), push_buffers_[i].begin(), push_buffers_[i].end());
      push_buffers_[i].clear();
      push_buffers_[i].shrink_to_fit();
    }
    // sort by data index
    std::sort(idx_val_pairs.begin(), idx_val_pairs.end(),
      [](const std::pair<data_size_t, VAL_T>& a, const std::pair<data_size_t, VAL_T>& b) {
        return a.first < b.first;
      });
    // load delta array
    LoadFromPair(idx_val_pairs);
  }

  void LoadFromPair(const std::vector<std::pair<data_size_t, VAL_T>>& idx_val_pairs) {
    deltas_.clear();
    vals_.clear();
    // transform to delta array
    data_size_t last_idx = 0;
    for (size_t i = 0; i < idx_val_pairs.size(); ++i) {
      const data_size_t cur_idx = idx_val_pairs[i].first;
      const VAL_T bin = idx_val_pairs[i].second;
      data_size_t cur_delta = cur_idx - last_idx;
      // disallow the multi-val in one row
      if (i > 0 && cur_delta == 0) { continue; }
      while (cur_delta >= 256) {
        deltas_.push_back(255);
        vals_.push_back(0);
        cur_delta -= 255;
      }
      deltas_.push_back(static_cast<uint8_t>(cur_delta));
      vals_.push_back(bin);
      last_idx = cur_idx;
    }
    // avoid out of range
    deltas_.push_back(0);
    num_vals_ = static_cast<data_size_t>(vals_.size());

    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    // generate fast index
    GetFastIndex();
  }

  void GetFastIndex() {
    fast_index_.clear();
    // get shift cnt
    data_size_t mod_size = (num_data_ + kNumFastIndex - 1) / kNumFastIndex;
    data_size_t pow2_mod_size = 1;
    fast_index_shift_ = 0;
    while (pow2_mod_size < mod_size) {
      pow2_mod_size <<= 1;
      ++fast_index_shift_;
    }
    // build fast index
    data_size_t i_delta = -1;
    data_size_t cur_pos = 0;
    data_size_t next_threshold = 0;
    while (NextNonzero(&i_delta, &cur_pos)) {
      while (next_threshold <= cur_pos) {
        fast_index_.emplace_back(i_delta, cur_pos);
        next_threshold += pow2_mod_size;
      }
    }
    // avoid out of range
    while (next_threshold < num_data_) {
      fast_index_.emplace_back(num_vals_ - 1, cur_pos);
      next_threshold += pow2_mod_size;
    }
    fast_index_.shrink_to_fit();
  }

  void SaveBinaryToFile(const VirtualFileWriter* writer) const override {
    writer->Write(&num_vals_, sizeof(num_vals_));
    writer->Write(deltas_.data(), sizeof(uint8_t) * (num_vals_ + 1));
    writer->Write(vals_.data(), sizeof(VAL_T) * num_vals_);
  }

  size_t SizesInByte() const override {
    return sizeof(num_vals_) + sizeof(uint8_t) * (num_vals_ + 1)
      + sizeof(VAL_T) * num_vals_;
  }

  void LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) override {
    const char* mem_ptr = reinterpret_cast<const char*>(memory);
    data_size_t tmp_num_vals = *(reinterpret_cast<const data_size_t*>(mem_ptr));
    mem_ptr += sizeof(tmp_num_vals);
    const uint8_t* tmp_delta = reinterpret_cast<const uint8_t*>(mem_ptr);
    mem_ptr += sizeof(uint8_t) * (tmp_num_vals + 1);
    const VAL_T* tmp_vals = reinterpret_cast<const VAL_T*>(mem_ptr);

    deltas_.clear();
    vals_.clear();
    num_vals_ = tmp_num_vals;
    for (data_size_t i = 0; i < num_vals_; ++i) {
      deltas_.push_back(tmp_delta[i]);
      vals_.push_back(tmp_vals[i]);
    }
    deltas_.push_back(0);
    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    if (local_used_indices.empty()) {
      // generate fast index
      GetFastIndex();
    } else {
      std::vector<std::pair<data_size_t, VAL_T>> tmp_pair;
      data_size_t cur_pos = 0;
      data_size_t j = -1;
      for (data_size_t i = 0; i < static_cast<data_size_t>(local_used_indices.size()); ++i) {
        const data_size_t idx = local_used_indices[i];
        while (cur_pos < idx && j < num_vals_) {
          NextNonzero(&j, &cur_pos);
        }
        if (cur_pos == idx && j < num_vals_ && vals_[j] > 0) {
          // new row index is i
          tmp_pair.emplace_back(i, vals_[j]);
        }
      }
      LoadFromPair(tmp_pair);
    }
  }

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const SparseBin<VAL_T>*>(full_bin);
    deltas_.clear();
    vals_.clear();
    data_size_t start = 0;
    if (num_used_indices > 0) {
      start = used_indices[0];
    }
    SparseBinIterator<VAL_T> iterator(other_bin, start);
    // transform to delta array
    data_size_t last_idx = 0;
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      auto bin = iterator.InnerRawGet(used_indices[i]);
      if (bin > 0) {
        data_size_t cur_delta = i - last_idx;
        while (cur_delta >= 256) {
          deltas_.push_back(255);
          vals_.push_back(0);
          cur_delta -= 255;
        }
        deltas_.push_back(static_cast<uint8_t>(cur_delta));
        vals_.push_back(bin);
        last_idx = i;
      }
    }
    // avoid out of range
    deltas_.push_back(0);
    num_vals_ = static_cast<data_size_t>(vals_.size());

    // reduce memory cost
    deltas_.shrink_to_fit();
    vals_.shrink_to_fit();

    // generate fast index
    GetFastIndex();
  }

  SparseBin<VAL_T>* Clone() override;

  SparseBin<VAL_T>(const SparseBin<VAL_T>& other)
    : num_data_(other.num_data_), deltas_(other.deltas_), vals_(other.vals_),
    num_vals_(other.num_vals_), push_buffers_(other.push_buffers_),
    fast_index_(other.fast_index_), fast_index_shift_(other.fast_index_shift_) {
  }

  void InitIndex(data_size_t start_idx, data_size_t * i_delta, data_size_t * cur_pos) const {
    auto idx = start_idx >> fast_index_shift_;
    if (static_cast<size_t>(idx) < fast_index_.size()) {
      const auto fast_pair = fast_index_[start_idx >> fast_index_shift_];
      *i_delta = fast_pair.first;
      *cur_pos = fast_pair.second;
    } else {
      *i_delta = -1;
      *cur_pos = 0;
    }
  }

private:

  data_size_t num_data_;
  std::vector<uint8_t, Common::AlignmentAllocator<uint8_t, kAlignedSize>> deltas_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, kAlignedSize>> vals_;
  data_size_t num_vals_;
  std::vector<std::vector<std::pair<data_size_t, VAL_T>>> push_buffers_;
  std::vector<std::pair<data_size_t, data_size_t>> fast_index_;
  data_size_t fast_index_shift_;
};

template<typename VAL_T>
SparseBin<VAL_T>* SparseBin<VAL_T>::Clone() {
  return new SparseBin(*this);
}

template <typename VAL_T>
inline uint32_t SparseBinIterator<VAL_T>::RawGet(data_size_t idx) {
  return InnerRawGet(idx);
}

template <typename VAL_T>
inline VAL_T SparseBinIterator<VAL_T>::InnerRawGet(data_size_t idx) {
  while (cur_pos_ < idx) {
    bin_data_->NextNonzeroFast(&i_delta_, &cur_pos_);
  }
  if (cur_pos_ == idx) {
    return bin_data_->vals_[i_delta_];
  } else {
    return 0;
  }
}

template <typename VAL_T>
inline void SparseBinIterator<VAL_T>::Reset(data_size_t start_idx) {
  bin_data_->InitIndex(start_idx, &i_delta_, &cur_pos_);
}

template <typename VAL_T>
BinIterator* SparseBin<VAL_T>::GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new SparseBinIterator<VAL_T>(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
#endif   // LightGBM_IO_SPARSE_BIN_HPP_
