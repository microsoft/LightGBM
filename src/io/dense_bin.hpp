/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_DENSE_BIN_HPP_
#define LIGHTGBM_IO_DENSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

template <typename VAL_T>
class DenseBin;

template <typename VAL_T>
class DenseBinIterator: public BinIterator {
 public:
  explicit DenseBinIterator(const DenseBin<VAL_T>* bin_data, uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin)
    : bin_data_(bin_data), min_bin_(static_cast<VAL_T>(min_bin)),
    max_bin_(static_cast<VAL_T>(max_bin)),
    most_freq_bin_(static_cast<VAL_T>(most_freq_bin)) {
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
  const DenseBin<VAL_T>* bin_data_;
  VAL_T min_bin_;
  VAL_T max_bin_;
  VAL_T most_freq_bin_;
  uint8_t offset_;
};
/*!
* \brief Used to store bins for dense feature
* Use template to reduce memory cost
*/
template <typename VAL_T>
class DenseBin: public Bin {
public:
  friend DenseBinIterator<VAL_T>;
  explicit DenseBin(data_size_t num_data)
    : num_data_(num_data), data_(num_data_, static_cast<VAL_T>(0)) {
  }

  ~DenseBin() {
  }

  void Push(int, data_size_t idx, uint32_t value) override {
    data_[idx] = static_cast<VAL_T>(value);
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
      data_.resize(num_data_);
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
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset - kCacheLineSize / sizeof(VAL_T);
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + data_indices[i + pf_offset]);
      const VAL_T bin = data_[data_indices[i]];
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
    for (; i < end; i++) {
      const VAL_T bin = data_[data_indices[i]];
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset - kCacheLineSize / sizeof(VAL_T);
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + i + pf_offset);
      const VAL_T bin = data_[i];
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
    for (; i < end; i++) {
      const VAL_T bin = data_[i];
      ACC_GH(out, bin, ordered_gradients[i], ordered_hessians[i]);
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset - kCacheLineSize / sizeof(VAL_T);
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + data_indices[i + pf_offset]);
      const VAL_T bin = data_[data_indices[i]];
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
    for (; i < end; i++) {
      const VAL_T bin = data_[data_indices[i]];
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset - kCacheLineSize / sizeof(VAL_T);
    data_size_t i = start;
    for (; i < pf_end; i++) {
      PREFETCH_T0(data_.data() + i + pf_offset);
      const VAL_T bin = data_[i];
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
    for (; i < end; i++) {
      const VAL_T bin = data_[i];
      ACC_GH(out, bin, ordered_gradients[i], 1.0f);
    }
  }
  #undef ACC_GH

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
        const VAL_T bin = data_[idx];
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
          const VAL_T bin = data_[idx];
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
          const VAL_T bin = data_[idx];
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
      const uint32_t bin = data_[idx];
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

  void FinishLoad() override {}

  void LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) override {
    const VAL_T* mem_data = reinterpret_cast<const VAL_T*>(memory);
    if (!local_used_indices.empty()) {
      for (int i = 0; i < num_data_; ++i) {
        data_[i] = mem_data[local_used_indices[i]];
      }
    } else {
      for (int i = 0; i < num_data_; ++i) {
        data_[i] = mem_data[i];
      }
    }
  }

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const DenseBin<VAL_T>*>(full_bin);
    for (int i = 0; i < num_used_indices; ++i) {
      data_[i] = other_bin->data_[used_indices[i]];
    }
  }

  void SaveBinaryToFile(const VirtualFileWriter* writer) const override {
    writer->Write(data_.data(), sizeof(VAL_T) * num_data_);
  }

  size_t SizesInByte() const override {
    return sizeof(VAL_T)* num_data_;
  }

  DenseBin<VAL_T>* Clone() override;

private:
  data_size_t num_data_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, kAlignedSize>> data_;

  DenseBin<VAL_T>(const DenseBin<VAL_T>& other)
    : num_data_(other.num_data_), data_(other.data_) {
  }
};

template<typename VAL_T>
DenseBin<VAL_T>* DenseBin<VAL_T>::Clone() {
  return new DenseBin<VAL_T>(*this);
}

template <typename VAL_T>
uint32_t DenseBinIterator<VAL_T>::Get(data_size_t idx) {
  auto ret = bin_data_->data_[idx];
  if (ret >= min_bin_ && ret <= max_bin_) {
    return ret - min_bin_ + offset_;
  } else {
    return most_freq_bin_;
  }
}

template <typename VAL_T>
inline uint32_t DenseBinIterator<VAL_T>::RawGet(data_size_t idx) {
  return bin_data_->data_[idx];
}

template <typename VAL_T>
BinIterator* DenseBin<VAL_T>::GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new DenseBinIterator<VAL_T>(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
#endif   // LightGBM_IO_DENSE_BIN_HPP_
