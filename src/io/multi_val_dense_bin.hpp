/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_


#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {


template <typename VAL_T>
class MultiValDenseBin : public MultiValBin {
public:

  explicit MultiValDenseBin(data_size_t num_data, int num_bin, int num_feature)
    : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature) {
    data_.resize(static_cast<size_t>(num_data_) * num_feature_, static_cast<VAL_T>(0));
  }

  ~MultiValDenseBin() {
  }

  data_size_t num_data() const override {
    return num_data_;
  }

  int num_bin() const override {
    return num_bin_;
  }


  void PushOneRow(int , data_size_t idx, const std::vector<uint32_t>& values) override {
    auto start = RowPtr(idx);
#ifdef DEBUG
    CHECK(num_feature_ == static_cast<int>(values.size()));
#endif  // DEBUG
    for (auto i = 0; i < num_feature_; ++i) {
      data_[start + i] = static_cast<VAL_T>(values[i]);
    }
  }

  void FinishLoad() override {

  }

  bool IsSparse() override{
    return false;
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
    }
  }

  #define ACC_GH(hist, i, g, h) \
  const auto ti = static_cast<int>(i) << 1; \
  hist[ti] += g; \
  hist[ti + 1] += h; \

  template<bool use_indices, bool use_prefetch, bool use_hessians>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians, hist_t* out) const {
    data_size_t i = start;
    if (use_prefetch) {
      const data_size_t pf_offset = 32 / sizeof(VAL_T);
      const data_size_t pf_end = end - pf_offset;

      for (; i < pf_end; ++i) {
        const auto idx = use_indices ? data_indices[i] : i;
        const auto pf_idx = use_indices ? data_indices[i + pf_offset] : i + pf_offset;
        PREFETCH_T0(gradients + pf_idx);
        if (use_hessians) {
          PREFETCH_T0(hessians + pf_idx);
        }
        PREFETCH_T0(data_.data() + RowPtr(pf_idx));
        const auto j_start = RowPtr(idx);
        for (auto j = j_start; j < j_start + num_feature_; ++j) {
          const VAL_T bin = data_[j];
          if (use_hessians) {
            ACC_GH(out, bin, gradients[idx], hessians[idx]);
          } else {
            ACC_GH(out, bin, gradients[idx], 1.0f);
          }
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = use_indices ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);
      for (auto j = j_start; j < j_start + num_feature_; ++j) {
        const VAL_T bin = data_[j];
        if (use_hessians) {
          ACC_GH(out, bin, gradients[idx], hessians[idx]);
        } else {
          ACC_GH(out, bin, gradients[idx], 1.0f);
        }
      }
    }
  }
  #undef ACC_GH

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end, gradients, hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    ConstructHistogramInner<false, false, true>(nullptr, start, end, gradients, hessians, out);
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end, gradients, nullptr, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(nullptr, start, end, gradients, nullptr, out);
  }

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    data_.clear();
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      for (int64_t j = other_bin->RowPtr(used_indices[i]); j < other_bin->RowPtr(used_indices[i] + 1); ++j) {
        data_.push_back(other_bin->data_[j]);
      }
    }
  }

  inline int64_t RowPtr(data_size_t idx) const {
    return static_cast<int64_t>(idx) * num_feature_;
  }

  MultiValDenseBin<VAL_T>* Clone() override;

private:
  data_size_t num_data_;
  int num_bin_;
  int num_feature_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>> data_;

  MultiValDenseBin<VAL_T>(const MultiValDenseBin<VAL_T>& other)
    : num_data_(other.num_data_), num_bin_(other.num_bin_), num_feature_(other.num_feature_), data_(other.data_) {
  }
};

template<typename VAL_T>
MultiValDenseBin<VAL_T>* MultiValDenseBin<VAL_T>::Clone() {
  return new MultiValDenseBin<VAL_T>(*this);
}



}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
