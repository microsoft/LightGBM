#ifndef LIGHTGBM_IO_SPARSE_BIN_HPP_
#define LIGHTGBM_IO_SPARSE_BIN_HPP_

#include <LightGBM/utils/log.h>

#include <LightGBM/bin.h>

#include <omp.h>

#include <cstring>
#include <cstdint>

#include <vector>

namespace LightGBM {

template <typename VAL_T>
class SparseBin;

const size_t kNumFastIndex = 64;
const uint8_t kMaxDelta = 255;

template <typename VAL_T>
class SparseBinIterator: public BinIterator {
public:
  SparseBinIterator(const SparseBin<VAL_T>* bin_data, data_size_t start_idx)
    : bin_data_(bin_data) {
    Reset(start_idx);
  }

  inline VAL_T InnerGet(data_size_t idx);

  inline uint32_t Get(data_size_t idx) override {
    return InnerGet(idx);
  }

  inline void Reset(data_size_t idx);
private:
  const SparseBin<VAL_T>* bin_data_;
  data_size_t cur_pos_;
  data_size_t i_delta_;
};

template <typename VAL_T>
class OrderedSparseBin;

template <typename VAL_T>
class SparseBin: public Bin {
public:
  friend class SparseBinIterator<VAL_T>;
  friend class OrderedSparseBin<VAL_T>;

  SparseBin(data_size_t num_data, int default_bin)
    : num_data_(num_data) {
    default_bin_ = static_cast<VAL_T>(default_bin);
    if (default_bin_ != 0) {
      Log::Info("Warning: sparse feature with negative values, treating negative values as zero");
    }
#pragma omp parallel
#pragma omp master
    {
      num_threads_ = omp_get_num_threads();
    }
    for (int i = 0; i < num_threads_; ++i) {
      push_buffers_.emplace_back();
    }
  }

  ~SparseBin() {
  }

  void Push(int tid, data_size_t idx, uint32_t value) override {
    // not store zero data
    if (value <= default_bin_) { return; }
    push_buffers_[tid].emplace_back(idx, static_cast<VAL_T>(value));
  }

  BinIterator* GetIterator(data_size_t start_idx) const override;

  void ConstructHistogram(const data_size_t*, data_size_t, const score_t*,
    const score_t*, HistogramBinEntry*) const override {
    // Will use OrderedSparseBin->ConstructHistogram() instead
    Log::Fatal("Using OrderedSparseBin->ConstructHistogram() instead");
  }

  inline bool NextNonzero(data_size_t* i_delta,
    data_size_t* cur_pos) const {
    ++(*i_delta);
    *cur_pos += deltas_[*i_delta];
    data_size_t factor = 1;
    while (*i_delta < num_vals_ && vals_[*i_delta] == 0) {
      ++(*i_delta);
      factor *= kMaxDelta;
      *cur_pos += deltas_[*i_delta] * factor;
    }
    if (*i_delta >= 0 && *i_delta < num_vals_) {
      return true;
    } else {
      return false;
    }
  }

  virtual data_size_t Split(unsigned int threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    // not need to split
    if (num_data <= 0) { return 0; }
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      VAL_T bin = iterator.InnerGet(idx);
      if (bin > threshold) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  }

  data_size_t num_data() const override { return num_data_; }

  OrderedBin* CreateOrderedBin() const override;

  void FinishLoad() override {
    // get total non zero size
    size_t non_zero_size = 0;
    for (size_t i = 0; i < push_buffers_.size(); ++i) {
      non_zero_size += push_buffers_[i].size();
    }
    // merge
    non_zero_pair_.reserve(non_zero_size);
    for (size_t i = 0; i < push_buffers_.size(); ++i) {
      non_zero_pair_.insert(non_zero_pair_.end(), push_buffers_[i].begin(), push_buffers_[i].end());
      push_buffers_[i].clear();
      push_buffers_[i].shrink_to_fit();
    }
    push_buffers_.clear();
    push_buffers_.shrink_to_fit();
    // sort by data index
    std::sort(non_zero_pair_.begin(), non_zero_pair_.end(),
      [](const std::pair<data_size_t, VAL_T>& a, const std::pair<data_size_t, VAL_T>& b) {
      return a.first < b.first;
    });
    // load detla array
    LoadFromPair(non_zero_pair_);
    // free memory
    non_zero_pair_.clear();
    non_zero_pair_.shrink_to_fit();
  }

  void LoadFromPair(const std::vector<std::pair<data_size_t, VAL_T>>& non_zero_pair) {
    deltas_.clear();
    vals_.clear();
    // transform to delta array
    data_size_t last_idx = 0;
    for (size_t i = 0; i < non_zero_pair.size(); ++i) {
      const data_size_t cur_idx = non_zero_pair[i].first;
      const VAL_T bin = non_zero_pair[i].second;
      data_size_t cur_delta = cur_idx - last_idx;
      while (cur_delta > kMaxDelta) {
        deltas_.push_back(cur_delta %  kMaxDelta);
        vals_.push_back(0);
        cur_delta /= kMaxDelta;
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
      while (next_threshold < cur_pos) {
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

  void SaveBinaryToFile(FILE* file) const override {
    fwrite(&num_vals_, sizeof(num_vals_), 1, file);
    fwrite(deltas_.data(), sizeof(uint8_t), num_vals_ + 1, file);
    fwrite(vals_.data(), sizeof(VAL_T), num_vals_, file);
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
        if (cur_pos == idx && j < num_vals_) {
          // new row index is i
          tmp_pair.emplace_back(i, vals_[j]);
        }
      }
      LoadFromPair(tmp_pair);
    }

  }

protected:
  data_size_t num_data_;
  std::vector<std::pair<data_size_t, VAL_T>> non_zero_pair_;
  std::vector<uint8_t> deltas_;
  std::vector<VAL_T> vals_;
  data_size_t num_vals_;
  int num_threads_;
  std::vector<std::vector<std::pair<data_size_t, VAL_T>>> push_buffers_;
  std::vector<std::pair<data_size_t, data_size_t>> fast_index_;
  data_size_t fast_index_shift_;
  VAL_T default_bin_;
};

template <typename VAL_T>
inline VAL_T SparseBinIterator<VAL_T>::InnerGet(data_size_t idx) {
  while (cur_pos_ < idx && i_delta_ < bin_data_->num_vals_) {
    bin_data_->NextNonzero(&i_delta_, &cur_pos_);
  }
  if (cur_pos_ == idx && i_delta_ < bin_data_->num_vals_ && i_delta_ >= 0) {
    return bin_data_->vals_[i_delta_];
  } else {
    return 0;
  }
}

template <typename VAL_T>
inline void SparseBinIterator<VAL_T>::Reset(data_size_t start_idx) {
  const auto fast_pair = bin_data_->fast_index_[start_idx >> bin_data_->fast_index_shift_];
  i_delta_ = fast_pair.first;
  cur_pos_ = fast_pair.second;
}

template <typename VAL_T>
BinIterator* SparseBin<VAL_T>::GetIterator(data_size_t start_idx) const {
  return new SparseBinIterator<VAL_T>(this, start_idx);
}


template <typename VAL_T>
class SparseCategoricalBin: public SparseBin<VAL_T> {
public:
  SparseCategoricalBin(data_size_t num_data, int default_bin)
    : SparseBin<VAL_T>(num_data, default_bin) {
  }

  virtual data_size_t Split(unsigned int threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    // not need to split
    if (num_data <= 0) { return 0; }
    SparseBinIterator<VAL_T> iterator(this, data_indices[0]);
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      VAL_T bin = iterator.InnerGet(idx);
      if (bin != threshold) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  }
};


}  // namespace LightGBM
#endif   // LightGBM_IO_SPARSE_BIN_HPP_
