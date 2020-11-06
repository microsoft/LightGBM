/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <memory>
#include <vector>
#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

namespace LightGBM {

struct TrainingShareStates {
  int num_threads = 0;
  bool is_colwise = true;
  bool is_use_subcol = false;
  bool is_use_subrow = false;
  bool is_subrow_copied = false;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;
  std::unique_ptr<MultiValBin> multi_val_bin;
  std::unique_ptr<MultiValBin> multi_val_bin_subset;
  std::vector<uint32_t> hist_move_src;
  std::vector<uint32_t> hist_move_dest;
  std::vector<uint32_t> hist_move_size;

  virtual void SetMultiValBin(MultiValBin* bin, data_size_t num_data) = 0;

  virtual void HistMove() = 0;

  virtual void HistMerge() = 0;

  virtual void ResizeHistBuf(hist_t* hist_data) = 0;

  virtual void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) = 0;

  virtual void InitTrain() {
    const auto cur_multi_val_bin = (is_use_subcol || is_use_subrow)
          ? multi_val_bin_subset.get()
          : multi_val_bin.get();
    if (cur_multi_val_bin == nullptr) {
      return;
    }
    num_bin_ = cur_multi_val_bin->num_bin();
    num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    min_block_size_ = std::min<int>(static_cast<int>(0.3f * num_bin_ /
      cur_multi_val_bin->num_element_per_row()) + 1, 1024);
  }

  template <bool USE_INDICES, bool ORDERED>
  bool ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const score_t* gradients,
                          const score_t* hessians,
                          hist_t* hist_data) {
    const auto cur_multi_val_bin = (is_use_subcol || is_use_subrow)
          ? multi_val_bin_subset.get()
          : multi_val_bin.get();
    if (cur_multi_val_bin == nullptr) {
      return false;
    }
    n_data_block_ = 1;
    data_block_size_ = num_data;
    Threading::BlockInfo<data_size_t>(num_threads, num_data, min_block_size_,
                                      max_block_size_, &n_data_block_, &data_block_size_);
    ResizeHistBuf(hist_data);
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int block_id = 0; block_id < n_data_block_; ++block_id) {
      OMP_LOOP_EX_BEGIN();
      data_size_t start = block_id * data_block_size_;
      data_size_t end = std::min(start + data_block_size_, num_data);
      if (USE_INDICES) {
        if (ORDERED) {
          ConstructHistogramsForBlock(
            cur_multi_val_bin, start, end, data_indices, gradients, hessians,
            block_id, true, true
          );
        } else {
          ConstructHistogramsForBlock(
            cur_multi_val_bin, start, end, data_indices, gradients, hessians,
            block_id, true, false
          );
        }
      } else {
        ConstructHistogramsForBlock(
            cur_multi_val_bin, start, end, data_indices, gradients, hessians,
            block_id, false, false
          );
      }
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    return true;
  }

  static TrainingShareStates* CreateTrainingShareStates(bool single_precision_hist_buffer);

  virtual ~TrainingShareStates() {}

protected:
  size_t kHistBufferEntrySize;
  int num_bin_aligned_;
  int num_bin_;
  int max_block_size_;
  int min_block_size_;
  int n_data_block_;
  int data_block_size_;
  hist_t* origin_hist_data_ = nullptr;
};

struct TrainingShareStatesFloat : public TrainingShareStates {
  std::vector<float, Common::AlignmentAllocator<float, kAlignedSize>> hist_buf;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> temp_buf;

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(float);
    max_block_size_ = 100000;
    if (bin == nullptr) {
      return;
    }
    int num_blocks = (num_data + max_block_size_ - 1) / max_block_size_;
    num_blocks = std::max(num_threads, num_blocks);
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    size_t new_size = static_cast<size_t>(num_bin_aligned_) * 2 * num_blocks;
    if (new_size > hist_buf.size()) {
      hist_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2 * num_blocks);
    }
    if (is_use_subcol) {
      temp_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2);
    }
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data;
    size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
    if (hist_buf.size() < new_buf_size) {
      hist_buf.resize(new_buf_size);
    }
    if (temp_buf.size() < static_cast<size_t>(num_bin_aligned_) * 2 && is_use_subcol) {
      temp_buf.resize(static_cast<size_t>(num_bin_aligned_) * 2);
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) override {
    float* data_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (block_id);
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
        sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                 gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = temp_buf.data();
    }
    std::memset(reinterpret_cast<void*>(dst), 0, num_bin_ * kHistEntrySize);
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 0; tid < n_data_block_; ++tid) {
        auto src_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = temp_buf.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }
};

struct TrainingShareStatesFloatWithBuffer : public TrainingShareStatesFloat {

  void SetMultiValBin(MultiValBin* bin, data_size_t /*num_data*/) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(float);
    max_block_size_ = 100000;
    if (bin == nullptr) {
      return;
    }
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    const size_t thread_buf_size = static_cast<size_t>(num_bin_aligned_) * num_threads * 2;
    temp_buf.resize(thread_buf_size, 0.0f);
    hist_buf.resize(thread_buf_size, 0.0f);
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data;
    const size_t new_thread_buf_size = static_cast<size_t>(num_bin_aligned_) * num_threads * 2;
    if (new_thread_buf_size > temp_buf.size()) {
      temp_buf.resize(new_thread_buf_size, 0.0f);
    }
    if (new_thread_buf_size > hist_buf.size()) {
      hist_buf.resize(new_thread_buf_size, 0.0f);
    }
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < static_cast<int>(temp_buf.size()); ++i) {
      temp_buf[i] = 0.0f;
    }
    if (!is_use_subcol) {
      #pragma omp parallel for schedule(static) num_threads(num_threads)
      for (int i = 0; i < num_bin_ * 2; ++i) {
        hist_data[i] = 0.0f;
      }
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int /*block_id*/,
    bool use_indices, bool ordered) override {
    int thread_id = omp_get_thread_num();
    float* data_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * thread_id;
    const int num_bin = sub_multi_val_bin->num_bin();
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
        sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                 gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
    double* thread_buf_ptr = origin_hist_data_;
    if (thread_id == 0) {
      if (is_use_subcol) {
        thread_buf_ptr = temp_buf.data();
      }
    } else {
      thread_buf_ptr = temp_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (thread_id);
    }
    for (int i = 0; i < 2 * num_bin; ++i) {
      thread_buf_ptr[i] += data_ptr[i];
    }
  }

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = temp_buf.data();
    }
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 1; tid < num_threads; ++tid) {
        auto src_ptr = temp_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = temp_buf.data();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }
};

struct TrainingShareStatesDouble : public TrainingShareStates {
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf;

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data) override {
    num_threads = OMP_NUM_THREADS();
    kHistBufferEntrySize = 2 * sizeof(hist_t);
    max_block_size_ = num_data;
    if (bin == nullptr) {
      return;
    }
    multi_val_bin.reset(bin);
    num_bin_aligned_ =
        (bin->num_bin() + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    size_t new_size = static_cast<size_t>(num_bin_aligned_) * 2 * num_threads;
    if (new_size > hist_buf.size()) {
      hist_buf.resize(new_size);
    }
  }

  void ResizeHistBuf(hist_t* hist_data) override {
    origin_hist_data_ = hist_data;
    size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
    if (hist_buf.size() < new_buf_size) {
      hist_buf.resize(new_buf_size);
    }
  }

  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    bool use_indices, bool ordered) override {
    hist_t* data_ptr = origin_hist_data_;
    if (block_id == 0) {
      if (is_use_subcol) {
        data_ptr = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
      }
    } else {
      data_ptr = hist_buf.data() +
        static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
    }
    const int num_bin = sub_multi_val_bin->num_bin();
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin * kHistBufferEntrySize);
    if (use_indices) {
      if (ordered) {
        sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                 gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  void HistMerge() override {
    int n_bin_block = 1;
    int bin_block_size = num_bin_;
    Threading::BlockInfo<data_size_t>(num_threads, num_bin_, 512, &n_bin_block,
                                    &bin_block_size);
    hist_t* dst = origin_hist_data_;
    if (is_use_subcol) {
      dst = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
    }
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int t = 0; t < n_bin_block; ++t) {
      const int start = t * bin_block_size;
      const int end = std::min(start + bin_block_size, num_bin_);
      for (int tid = 1; tid < n_data_block_; ++tid) {
        auto src_ptr = hist_buf.data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid - 1);
        for (int i = start * 2; i < end * 2; ++i) {
          dst[i] += src_ptr[i];
        }
      }
    }
  }

  void HistMove() override {
    if (!is_use_subcol) {
      return;
    }
    hist_t* src = hist_buf.data() + hist_buf.size() - 2 * static_cast<size_t>(num_bin_aligned_);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src.size()); ++i) {
      std::copy_n(src + hist_move_src[i], hist_move_size[i],
                  origin_hist_data_ + hist_move_dest[i]);
    }
  }
};

} // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
