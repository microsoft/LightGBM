/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "pairwise_lambdarank_bin.hpp"

namespace LightGBM {

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads) {
  unpaired_bin_->InitStreaming(num_thread, omp_max_threads);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::Push(int tid, data_size_t idx, uint32_t value) {
  unpaired_bin_->Push(tid, idx, value);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) {
  unpaired_bin_->CopySubrow(full_bin, used_indices, num_used_indices);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SaveBinaryToFile(BinaryWriter* writer) const {
  unpaired_bin_->SaveBinaryToFile(writer);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) {
  unpaired_bin_->LoadFromMemory(memory, local_used_indices);
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SizesInByte() const {
  return unpaired_bin_->SizesInByte();
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::num_data() const {
  return unpaired_bin_->num_data();
}

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::ReSize(data_size_t num_data) {
  return unpaired_bin_->ReSize(num_data);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInner(const data_size_t* data_indices,
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const {
  data_size_t i = start;
  hist_t* grad = out;
  hist_t* hess = out + 1;
  hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess);
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto idx = get_unpaired_index(paired_idx);
      const auto paired_pf_idx =
          USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      const auto pf_idx = get_unpaired_index(paired_pf_idx);
      if (IS_4BIT) {
        PREFETCH_T0(unpaired_bin_->data_.data() + (pf_idx >> 1));
      } else {
        PREFETCH_T0(unpaired_bin_->data_.data() + pf_idx);
      }
      const auto ti = static_cast<uint32_t>(unpaired_bin_->data(idx)) << 1;
      if (USE_HESSIAN) {
        grad[ti] += ordered_gradients[i];
        hess[ti] += ordered_hessians[i];
      } else {
        grad[ti] += ordered_gradients[i];
        ++cnt[ti];
      }
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto ti = static_cast<uint32_t>(unpaired_bin_->data(idx)) << 1;
    if (USE_HESSIAN) {
      grad[ti] += ordered_gradients[i];
      hess[ti] += ordered_hessians[i];
    } else {
      grad[ti] += ordered_gradients[i];
      ++cnt[ti];
    }
  }
}

template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
void ConstructHistogramIntInner(const data_size_t* data_indices,
                              data_size_t start, data_size_t end,
                              const score_t* ordered_gradients,
                              hist_t* out) const {
  data_size_t i = start;
  PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
  const int16_t* gradients_ptr = reinterpret_cast<const int16_t*>(ordered_gradients);
  const VAL_T* data_ptr_base = unpaired_bin_->data_.data();
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto paired_pf_idx =
          USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      const auto idx = get_unpaired_index(paired_idx);
      const auto pf_idx = get_unpaired_index(paired_pf_idx);
      if (IS_4BIT) {
        PREFETCH_T0(data_ptr_base + (pf_idx >> 1));
      } else {
        PREFETCH_T0(data_ptr_base + pf_idx);
      }
      const auto ti = static_cast<uint32_t>(unpaired_bin_->data(idx));
      const int16_t gradient_16 = gradients_ptr[i];
      if (USE_HESSIAN) {
        const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
        out_ptr[ti] += gradient_packed;
      } else {
        const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (1);
        out_ptr[ti] += gradient_packed;
      }
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto ti = static_cast<uint32_t>(unpaired_bin_->data(idx));
    const int16_t gradient_16 = gradients_ptr[i];
    if (USE_HESSIAN) {
      const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (gradient_16 & 0xff);
      out_ptr[ti] += gradient_packed;
    } else {
      const PACKED_HIST_T gradient_packed = HIST_BITS == 8 ? gradient_16 :
          (static_cast<PACKED_HIST_T>(static_cast<int8_t>(gradient_16 >> 8)) << HIST_BITS) | (1);
      out_ptr[ti] += gradient_packed;
    }
  }
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const {
  ConstructHistogramInner<true, true, true>(
        data_indices, start, end, ordered_gradients, ordered_hessians, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::(data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const override {
  ConstructHistogramInner<false, false, true>(
      nullptr, start, end, ordered_gradients, ordered_hessians, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                              ordered_gradients, nullptr, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramInner<false, false, false>(
      nullptr, start, end, ordered_gradients, nullptr, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, true, int16_t, 8>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, true, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, false, int16_t, 8>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, false, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, true, int32_t, 16>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, true, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, false, int32_t, 16>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, false, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, true, int64_t, 32>(
      data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, true, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<true, true, false, int64_t, 32>(
    data_indices, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const override {
  ConstructHistogramIntInner<false, false, false, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE, bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
data_size_t PairwiseRankingBin<DenseBin<VAL_T, IS_4BIT>, ITERATOR_TYPE>::SplitInner(uint32_t min_bin, uint32_t max_bin,
                        uint32_t default_bin, uint32_t most_freq_bin,
                        bool default_left, uint32_t threshold,
                        const data_size_t* data_indices, data_size_t cnt,
                        data_size_t* lte_indices,
                        data_size_t* gt_indices) const {
  auto th = static_cast<VAL_T>(threshold + min_bin);
  auto t_zero_bin = static_cast<VAL_T>(min_bin + default_bin);
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;
  }
  const auto minb = static_cast<VAL_T>(min_bin);
  const auto maxb = static_cast<VAL_T>(max_bin);
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
  if (MISS_IS_ZERO || MISS_IS_NA) {
    if (default_left) {
      missing_default_indices = lte_indices;
      missing_default_count = &lte_count;
    }
  }
  if (min_bin < max_bin) {
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t idx = data_indices[i];
      const auto bin = data(idx);
      if ((MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
          (MISS_IS_NA && !MFB_IS_NA && bin == maxb)) {
        missing_default_indices[(*missing_default_count)++] = idx;
      } else if ((USE_MIN_BIN && (bin < minb || bin > maxb)) ||
                  (!USE_MIN_BIN && bin == 0)) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else {
          default_indices[(*default_count)++] = idx;
        }
      } else if (bin > th) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
  } else {
    data_size_t* max_bin_indices = gt_indices;
    data_size_t* max_bin_count = &gt_count;
    if (maxb <= th) {
      max_bin_indices = lte_indices;
      max_bin_count = &lte_count;
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t idx = data_indices[i];
      const auto bin = data(idx);
      if (MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        missing_default_indices[(*missing_default_count)++] = idx;
      } else if (bin != maxb) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else {
          default_indices[(*default_count)++] = idx;
        }
      } else {
        if (MISS_IS_NA && !MFB_IS_NA) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else {
          max_bin_indices[(*max_bin_count)++] = idx;
        }
      }
    }
  }
  return lte_count;
}

}  // namespace LightGBM
