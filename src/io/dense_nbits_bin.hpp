/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_
#define LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

class Dense4bitsBin;

class Dense4bitsBinIterator : public BinIterator {
public:
  explicit Dense4bitsBinIterator(const Dense4bitsBin* bin_data, uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin)
    : bin_data_(bin_data), min_bin_(static_cast<uint8_t>(min_bin)),
    max_bin_(static_cast<uint8_t>(max_bin)),
    most_freq_bin_(static_cast<uint8_t>(most_freq_bin)) {
    if (most_freq_bin_ == 0) {
      offset_ = 1;
    } else {
      offset_ = 0;
    }
  }
  inline uint32_t RawGet(data_size_t idx) override;
  inline uint32_t Get(data_size_t idx) override;
  inline void Reset(data_size_t) override {}

private:
  const Dense4bitsBin* bin_data_;
  uint8_t min_bin_;
  uint8_t max_bin_;
  uint8_t most_freq_bin_;
  uint8_t offset_;
};

class Dense4bitsBin : public Bin {
public:
  friend Dense4bitsBinIterator;
  explicit Dense4bitsBin(data_size_t num_data)
    : num_data_(num_data) {
    int len = (num_data_ + 1) / 2;
    data_.resize(len, static_cast<uint8_t>(0));
    buf_ = std::vector<uint8_t>(len, static_cast<uint8_t>(0));
  }

  ~Dense4bitsBin() {
  }

  void Push(int, data_size_t idx, uint32_t value) override {
    const int i1 = idx >> 1;
    const int i2 = (idx & 1) << 2;
    const uint8_t val = static_cast<uint8_t>(value) << i2;
    if (i2 == 0) {
      data_[i1] = val;
    } else {
      buf_[i1] = val;
    }
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
      const int len = (num_data_ + 1) / 2;
      data_.resize(len);
    }
  }

  inline BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override;

  #define ACC_GH(hist, i, g, h) \
  const auto ti = (i) << 1; \
  hist[ti] += g; \
  hist[ti + 1] += h; \

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    const data_size_t pf_offset = 64;
    const data_size_t pf_end = end - pf_offset - kCacheLineSize;
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + (data_indices[i + pf_offset] >> 1));
      const data_size_t idx = data_indices[i];
      const auto bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
    for (; i < end; i++) {
      const data_size_t idx = data_indices[i];
      const auto bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    const data_size_t pf_offset = 64;
    const data_size_t pf_end = end - pf_offset - kCacheLineSize;
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + ((i + pf_offset) >> 1));
      const auto bin = (data_[i >> 1] >> ((i & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
    for (; i < end; i++) {
      const auto bin = (data_[i >> 1] >> ((i & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    const data_size_t pf_offset = 64;
    const data_size_t pf_end = end - pf_offset - kCacheLineSize;
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + (data_indices[i + pf_offset] >> 1));
      const data_size_t idx = data_indices[i];
      const auto bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
    for (; i < end; i++) {
      const data_size_t idx = data_indices[i];
      const auto bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    const data_size_t pf_offset = 64;
    const data_size_t pf_end = end - pf_offset - kCacheLineSize;
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + ((i + pf_offset) >> 1));
      const auto bin = (data_[i >> 1] >> ((i & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
    for (; i < end; i++) {
      const auto bin = (data_[i >> 1] >> ((i & 1) << 2)) & 0xf;
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
  }

  #undef ACC_GH

  data_size_t Split(
    uint32_t min_bin, uint32_t max_bin, uint32_t default_bin, uint32_t most_freq_bin, MissingType missing_type, bool default_left,
    uint32_t threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    if (num_data <= 0) { return 0; }
    uint8_t th = static_cast<uint8_t>(threshold + min_bin);
    const uint8_t minb = static_cast<uint8_t>(min_bin);
    const uint8_t maxb = static_cast<uint8_t>(max_bin);
    uint8_t t_default_bin = static_cast<uint8_t>(min_bin + default_bin);
    uint8_t t_most_freq_bin = static_cast<uint8_t>(min_bin + most_freq_bin);
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
        const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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
          const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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
          const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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
    data_size_t* default_indices = gt_indices;
    data_size_t* default_count = &gt_count;
    if (Common::FindInBitset(threshold, num_threahold, most_freq_bin)) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      const uint32_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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
    if (buf_.empty()) { return; }
    int len = (num_data_ + 1) / 2;
    for (int i = 0; i < len; ++i) {
      data_[i] |= buf_[i];
    }
    buf_.clear();
  }

  void LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) override {
    const uint8_t* mem_data = reinterpret_cast<const uint8_t*>(memory);
    if (!local_used_indices.empty()) {
      const data_size_t rest = num_data_ & 1;
      for (int i = 0; i < num_data_ - rest; i += 2) {
        // get old bins
        data_size_t idx = local_used_indices[i];
        const auto bin1 = static_cast<uint8_t>((mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        idx = local_used_indices[i + 1];
        const auto bin2 = static_cast<uint8_t>((mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        // add
        const int i1 = i >> 1;
        data_[i1] = (bin1 | (bin2 << 4));
      }
      if (rest) {
        data_size_t idx = local_used_indices[num_data_ - 1];
        data_[num_data_ >> 1] = (mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      }
    } else {
      for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = mem_data[i];
      }
    }
  }

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const Dense4bitsBin*>(full_bin);
    const data_size_t rest = num_used_indices & 1;
    for (int i = 0; i < num_used_indices - rest; i += 2) {
      data_size_t idx = used_indices[i];
      const auto bin1 = static_cast<uint8_t>((other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
      idx = used_indices[i + 1];
      const auto bin2 = static_cast<uint8_t>((other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
      const int i1 = i >> 1;
      data_[i1] = (bin1 | (bin2 << 4));
    }
    if (rest) {
      data_size_t idx = used_indices[num_used_indices - 1];
      data_[num_used_indices >> 1] = (other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
    }
  }

  void SaveBinaryToFile(const VirtualFileWriter* writer) const override {
    writer->Write(data_.data(), sizeof(uint8_t) * data_.size());
  }

  size_t SizesInByte() const override {
    return sizeof(uint8_t)* data_.size();
  }

  Dense4bitsBin* Clone() override {
    return new Dense4bitsBin(*this);
  }

protected:
  Dense4bitsBin(const Dense4bitsBin& other)
    : num_data_(other.num_data_), data_(other.data_), buf_(other.buf_) {
  }

  data_size_t num_data_;
  std::vector<uint8_t, Common::AlignmentAllocator<uint8_t, kAlignedSize>> data_;
  std::vector<uint8_t> buf_;
};

uint32_t Dense4bitsBinIterator::Get(data_size_t idx) {
  const auto bin = (bin_data_->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
  if (bin >= min_bin_ && bin <= max_bin_) {
    return bin - min_bin_ + offset_;
  } else {
    return most_freq_bin_;
  }
}

uint32_t Dense4bitsBinIterator::RawGet(data_size_t idx) {
  return (bin_data_->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
}

inline BinIterator* Dense4bitsBin::GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new Dense4bitsBinIterator(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_
