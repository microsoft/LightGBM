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

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::InitStreaming(uint32_t num_thread, int32_t omp_max_threads);

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::Push(int tid, data_size_t idx, uint32_t value) {
  unpaired_bin_->Push(tid, idx, value);
}

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::Push(int tid, data_size_t idx, uint32_t value);

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) {
  unpaired_bin_->CopySubrow(full_bin, used_indices, num_used_indices);
}

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices);

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SaveBinaryToFile(BinaryWriter* writer) const {
  unpaired_bin_->SaveBinaryToFile(writer);
}

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::SaveBinaryToFile(BinaryWriter* writer) const;

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) {
  unpaired_bin_->LoadFromMemory(memory, local_used_indices);
}

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices);

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::SizesInByte() const {
  return unpaired_bin_->SizesInByte();
}

template size_t PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::SizesInByte() const;
template size_t PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::SizesInByte() const;

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
data_size_t PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::num_data() const {
  return unpaired_bin_->num_data();
}

template data_size_t PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::num_data() const;
template data_size_t PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::num_data() const;

template <typename BIN_TYPE, template<typename> class ITERATOR_TYPE>
void PairwiseRankingBin<BIN_TYPE, ITERATOR_TYPE>::ReSize(data_size_t num_data) {
  return unpaired_bin_->ReSize(num_data);
}

template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingFirstIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint8_t, true>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint8_t, false>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint16_t, false>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<DenseBin<uint32_t, false>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint8_t>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint16_t>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);
template void PairwiseRankingBin<SparseBin<uint32_t>, PairwiseRankingSecondIterator>::ReSize(data_size_t num_data);

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInner(
  const data_size_t* data_indices,
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const {
  data_size_t i = start;
  hist_t* grad = out;
  hist_t* hess = out + 1;
  hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess);
  const VAL_T* base_data_ptr = reinterpret_cast<const VAL_T*>(this->unpaired_bin_->get_data());
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto idx = this->get_unpaired_index(paired_idx);
      const auto paired_pf_idx =
          USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      const auto pf_idx = this->get_unpaired_index(paired_pf_idx);
      if (IS_4BIT) {
        PREFETCH_T0(base_data_ptr + (pf_idx >> 1));
      } else {
        PREFETCH_T0(base_data_ptr + pf_idx);
      }
      const auto ti = static_cast<uint32_t>(this->unpaired_bin_->data(idx)) << 1;
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
    const auto paired_idx = USE_INDICES ? data_indices[i] : i;
    const auto idx = this->get_unpaired_index(paired_idx);
    const auto ti = static_cast<uint32_t>(this->unpaired_bin_->data(idx)) << 1;
    if (USE_HESSIAN) {
      grad[ti] += ordered_gradients[i];
      hess[ti] += ordered_hessians[i];
    } else {
      grad[ti] += ordered_gradients[i];
      ++cnt[ti];
    }
  }
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN, typename PACKED_HIST_T, int HIST_BITS>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramIntInner(
  const data_size_t* data_indices,
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  data_size_t i = start;
  PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
  const int16_t* gradients_ptr = reinterpret_cast<const int16_t*>(ordered_gradients);
  const VAL_T* data_ptr_base = reinterpret_cast<const VAL_T*>(this->unpaired_bin_->get_data());
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto paired_idx = USE_INDICES ? data_indices[i] : i;
      const auto paired_pf_idx =
          USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      const auto idx = this->get_unpaired_index(paired_idx);
      const auto pf_idx = this->get_unpaired_index(paired_pf_idx);
      if (IS_4BIT) {
        PREFETCH_T0(data_ptr_base + (pf_idx >> 1));
      } else {
        PREFETCH_T0(data_ptr_base + pf_idx);
      }
      const auto ti = static_cast<uint32_t>(this->unpaired_bin_->data(idx));
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
    const auto paired_idx = USE_INDICES ? data_indices[i] : i;
    const auto idx = this->get_unpaired_index(paired_idx);
    const auto ti = static_cast<uint32_t>(this->unpaired_bin_->data(idx));
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
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const {
  ConstructHistogramInner<true, true, true>(
        data_indices, start, end, ordered_gradients, ordered_hessians, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start, data_size_t end,
  const score_t* ordered_gradients, const score_t* ordered_hessians,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const {
  ConstructHistogramInner<false, false, true>(
      nullptr, start, end, ordered_gradients, ordered_hessians, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* ordered_hessians,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramInner<true, true, false>(data_indices, start, end,
                                              ordered_gradients, nullptr, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramInner<false, false, false>(
      nullptr, start, end, ordered_gradients, nullptr, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogram(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int16_t, 8>(
      data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int16_t, 8>(
    data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int16_t, 8>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt8(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int32_t, 16>(
      data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int32_t, 16>(
    data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int32_t, 16>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt16(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, true, int64_t, 32>(
      data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, true, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  const score_t* /*ordered_hessians*/,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<true, true, false, int64_t, 32>(
    data_indices, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  const data_size_t* data_indices, data_size_t start,
  data_size_t end, const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
void DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const {
  ConstructHistogramIntInner<false, false, false, int64_t, 32>(
      nullptr, start, end, ordered_gradients, out);
}

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template void DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::ConstructHistogramInt32(
  data_size_t start, data_size_t end,
  const score_t* ordered_gradients,
  hist_t* out) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool USE_MIN_BIN>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::SplitInner(uint32_t min_bin, uint32_t max_bin,
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
      const data_size_t paired_idx = data_indices[i];
      const data_size_t idx = this->get_unpaired_index(paired_idx);
      const auto bin = this->unpaired_bin_->data(idx);
      if ((MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
          (MISS_IS_NA && !MFB_IS_NA && bin == maxb)) {
        missing_default_indices[(*missing_default_count)++] = paired_idx;
      } else if ((USE_MIN_BIN && (bin < minb || bin > maxb)) ||
                  (!USE_MIN_BIN && bin == 0)) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          default_indices[(*default_count)++] = paired_idx;
        }
      } else if (bin > th) {
        gt_indices[gt_count++] = paired_idx;
      } else {
        lte_indices[lte_count++] = paired_idx;
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
      const data_size_t paired_idx = data_indices[i];
      const data_size_t idx = this->get_unpaired_index(paired_idx);
      const auto bin = this->unpaired_bin_->data(idx);
      if (MISS_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        missing_default_indices[(*missing_default_count)++] = paired_idx;
      } else if (bin != maxb) {
        if ((MISS_IS_NA && MFB_IS_NA) || (MISS_IS_ZERO && MFB_IS_ZERO)) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          default_indices[(*default_count)++] = paired_idx;
        }
      } else {
        if (MISS_IS_NA && !MFB_IS_NA) {
          missing_default_indices[(*missing_default_count)++] = paired_idx;
        } else {
          max_bin_indices[(*max_bin_count)++] = paired_idx;
        }
      }
    }
  }
  return lte_count;
}

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const {
  #define ARGUMENTS                                                        \
  min_bin, max_bin, default_bin, most_freq_bin, default_left, threshold, \
      data_indices, cnt, lte_indices, gt_indices
    if (missing_type == MissingType::None) {
      return SplitInner<false, false, false, false, true>(ARGUMENTS);
    } else if (missing_type == MissingType::Zero) {
      if (default_bin == most_freq_bin) {
        return SplitInner<true, false, true, false, true>(ARGUMENTS);
      } else {
        return SplitInner<true, false, false, false, true>(ARGUMENTS);
      }
    } else {
      if (max_bin == most_freq_bin + min_bin && most_freq_bin > 0) {
        return SplitInner<false, true, false, true, true>(ARGUMENTS);
      } else {
        return SplitInner<false, true, false, false, true>(ARGUMENTS);
      }
    }
#undef ARGUMENTS
}

template data_size_t DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::Split(uint32_t min_bin, uint32_t max_bin,
                    uint32_t default_bin, uint32_t most_freq_bin,
                    MissingType missing_type, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t cnt,
                    data_size_t* lte_indices,
                    data_size_t* gt_indices) const;

template <typename VAL_T, bool IS_4BIT, template<typename> class ITERATOR_TYPE>
data_size_t DensePairwiseRankingBin<VAL_T, IS_4BIT, ITERATOR_TYPE>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const {
#define ARGUMENTS                                                  \
  1, max_bin, default_bin, most_freq_bin, default_left, threshold, \
      data_indices, cnt, lte_indices, gt_indices
    if (missing_type == MissingType::None) {
      return SplitInner<false, false, false, false, false>(ARGUMENTS);
    } else if (missing_type == MissingType::Zero) {
      if (default_bin == most_freq_bin) {
        return SplitInner<true, false, true, false, false>(ARGUMENTS);
      } else {
        return SplitInner<true, false, false, false, false>(ARGUMENTS);
      }
    } else {
      if (max_bin == most_freq_bin + 1 && most_freq_bin > 0) {
        return SplitInner<false, true, false, true, false>(ARGUMENTS);
      } else {
        return SplitInner<false, true, false, false, false>(ARGUMENTS);
      }
    }
#undef ARGUMENTS
}

template data_size_t DensePairwiseRankingBin<uint8_t, true, PairwiseRankingFirstIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, false, PairwiseRankingFirstIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint16_t, false, PairwiseRankingFirstIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint32_t, false, PairwiseRankingFirstIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, true, PairwiseRankingSecondIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint8_t, false, PairwiseRankingSecondIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint16_t, false, PairwiseRankingSecondIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

template data_size_t DensePairwiseRankingBin<uint32_t, false, PairwiseRankingSecondIterator>::Split(uint32_t max_bin, uint32_t default_bin,
                            uint32_t most_freq_bin, MissingType missing_type,
                            bool default_left, uint32_t threshold,
                            const data_size_t* data_indices, data_size_t cnt,
                            data_size_t* lte_indices,
                            data_size_t* gt_indices) const;

}  // namespace LightGBM
