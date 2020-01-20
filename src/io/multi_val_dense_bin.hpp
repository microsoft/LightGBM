/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_


#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

namespace LightGBM {


template <typename VAL_T>
class MultiValDenseBin : public MultiValBin {
public:

  explicit MultiValDenseBin(data_size_t num_data, int num_bin)
    : num_data_(num_data), num_bin_(num_bin) {
    row_ptr_.resize(num_data_ + 1, 0);
    data_.reserve(num_data_);
    int num_threads = 1;
    #pragma omp parallel
    #pragma omp master
    {
      num_threads = omp_get_num_threads();
    }
    if (num_threads > 1) {
      t_data_.resize(num_threads - 1);
    }
  }

  ~MultiValDenseBin() {
  }

  data_size_t num_data() const override {
    return num_data_;
  }

  int num_bin() const override {
    return num_bin_;
  }


  void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>& values) override {
    row_ptr_[idx + 1] = static_cast<data_size_t>(values.size());
    if (tid == 0) {
      for (auto val : values) {
        data_.push_back(val);
      }
    } else {
      for (auto val : values) {
        t_data_[tid - 1].push_back(val);
      }
    }
  }

  void FinishLoad() override {
    for (data_size_t i = 0; i < num_data_; ++i) {
      row_ptr_[i + 1] += row_ptr_[i];
    }
    if (t_data_.size() > 0) {
      size_t offset = data_.size();
      data_.resize(row_ptr_[num_data_]);
      for (size_t tid = 0; tid < t_data_.size(); ++tid) {
        std::memcpy(data_.data() + offset, t_data_[tid].data(), t_data_[tid].size() * sizeof(VAL_T));
        offset += t_data_[tid].size();
        t_data_[tid].clear();
      }
    }
    row_ptr_.shrink_to_fit();
    data_.shrink_to_fit();
    t_data_.clear();
    t_data_.shrink_to_fit();
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

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(row_ptr_.data() + data_indices[i + prefetch_size]);
        PREFETCH_T0(gradients + data_indices[i + prefetch_size]);
        PREFETCH_T0(hessians + data_indices[i + prefetch_size]);
        PREFETCH_T0(data_.data() + row_ptr_[data_indices[i + prefetch_size]]);
      }
      for (data_size_t idx = RowPtr(data_indices[i]); idx < RowPtr(data_indices[i] + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[data_indices[i]], hessians[data_indices[i]]);
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients, const score_t* hessians,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(row_ptr_.data() + i + prefetch_size);
        PREFETCH_T0(gradients + i + prefetch_size);
        PREFETCH_T0(hessians + i + prefetch_size);
        PREFETCH_T0(data_.data() + row_ptr_[i + prefetch_size]);
      }
      for (data_size_t idx = RowPtr(i); idx < RowPtr(i + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[i], hessians[i]);
      }
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(row_ptr_.data() + data_indices[i + prefetch_size]);
        PREFETCH_T0(gradients + data_indices[i + prefetch_size]);
        PREFETCH_T0(data_.data() +  row_ptr_[data_indices[i + prefetch_size]]);
      }
      for (data_size_t idx = RowPtr(data_indices[i]); idx < RowPtr(data_indices[i] + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[data_indices[i]], 1.0f);
      }
    }
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* gradients,
    hist_t* out) const override {
    const data_size_t prefetch_size = 16;
    for (data_size_t i = start; i < end; ++i) {
      if (prefetch_size + i < end) {
        PREFETCH_T0(row_ptr_.data() + i + prefetch_size);
        PREFETCH_T0(gradients + i + prefetch_size);
        PREFETCH_T0(data_.data() + row_ptr_[i + prefetch_size]);
      }
      for (data_size_t idx = RowPtr(i); idx < RowPtr(i + 1); ++idx) {
        const VAL_T bin = data_[idx];
        ACC_GH(out, bin, gradients[i], 1.0f);
      }
    }
  }
  #undef ACC_GH

  void CopySubset(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const MultiValDenseBin<VAL_T>*>(full_bin);
    row_ptr_.resize(num_data_ + 1, 0);
    data_.clear();
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      for (data_size_t j = other_bin->row_ptr_[used_indices[i]]; j < other_bin->row_ptr_[used_indices[i] + 1]; ++j) {
        data_.push_back(other_bin->data_[j]);
      }
      row_ptr_[i + 1] = row_ptr_[i] + other_bin->row_ptr_[used_indices[i] + 1] - other_bin->row_ptr_[used_indices[i]];
    }
  }

  inline data_size_t RowPtr(data_size_t idx) const {
    return row_ptr_[idx];
  }

  MultiValDenseBin<VAL_T>* Clone() override;

private:
  data_size_t num_data_;
  int num_bin_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>> data_;
  std::vector<data_size_t, Common::AlignmentAllocator<data_size_t, 32>> row_ptr_;
  std::vector<std::vector<VAL_T>> t_data_;

  MultiValDenseBin<VAL_T>(const MultiValDenseBin<VAL_T>& other)
    : num_data_(other.num_data_), data_(other.data_), row_ptr_(other.row_ptr_){
  }
};

template<typename VAL_T>
MultiValDenseBin<VAL_T>* MultiValDenseBin<VAL_T>::Clone() {
  return new MultiValDenseBin<VAL_T>(*this);
}



}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_DENSE_BIN_HPP_
