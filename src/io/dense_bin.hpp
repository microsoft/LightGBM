#ifndef LIGHTGBM_IO_DENSE_BIN_HPP_
#define LIGHTGBM_IO_DENSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <vector>
#include <cstring>
#include <cstdint>

namespace LightGBM {

/*!
* \brief Used to store bins for dense feature
* Use template to reduce memory cost
*/
template <typename VAL_T>
class DenseBin: public Bin {
public:
  DenseBin(data_size_t num_data, int default_bin)
    : num_data_(num_data) {
    data_.resize(num_data_);
    VAL_T default_bin_T = static_cast<VAL_T>(default_bin);
    std::fill(data_.begin(), data_.end(), default_bin_T);
  }

  ~DenseBin() {
  }

  void Push(int, data_size_t idx, uint32_t value) override {
    data_[idx] = static_cast<VAL_T>(value);
  }

  inline uint32_t Get(data_size_t idx) const {
    return static_cast<uint32_t>(data_[idx]);
  }

  BinIterator* GetIterator(data_size_t start_idx) const override;

  void ConstructHistogram(const data_size_t* data_indices, data_size_t num_data,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    HistogramBinEntry* out) const override {
    // use 4-way unrolling, will be faster
    if (data_indices != nullptr) {  // if use part of data
      data_size_t rest = num_data % 4;
      data_size_t i = 0;
      for (; i < num_data - rest; i += 4) {
        VAL_T bin0 = data_[data_indices[i]];
        VAL_T bin1 = data_[data_indices[i + 1]];
        VAL_T bin2 = data_[data_indices[i + 2]];
        VAL_T bin3 = data_[data_indices[i + 3]];

        out[bin0].sum_gradients += ordered_gradients[i];
        out[bin1].sum_gradients += ordered_gradients[i + 1];
        out[bin2].sum_gradients += ordered_gradients[i + 2];
        out[bin3].sum_gradients += ordered_gradients[i + 3];

        out[bin0].sum_hessians += ordered_hessians[i];
        out[bin1].sum_hessians += ordered_hessians[i + 1];
        out[bin2].sum_hessians += ordered_hessians[i + 2];
        out[bin3].sum_hessians += ordered_hessians[i + 3];

        ++out[bin0].cnt;
        ++out[bin1].cnt;
        ++out[bin2].cnt;
        ++out[bin3].cnt;
      }
      for (; i < num_data; ++i) {
        VAL_T bin = data_[data_indices[i]];
        out[bin].sum_gradients += ordered_gradients[i];
        out[bin].sum_hessians += ordered_hessians[i];
        ++out[bin].cnt;
      }
    } else {  // use full data
      data_size_t rest = num_data % 4;
      data_size_t i = 0;
      for (; i < num_data - rest; i += 4) {
        VAL_T bin0 = data_[i];
        VAL_T bin1 = data_[i + 1];
        VAL_T bin2 = data_[i + 2];
        VAL_T bin3 = data_[i + 3];

        out[bin0].sum_gradients += ordered_gradients[i];
        out[bin1].sum_gradients += ordered_gradients[i + 1];
        out[bin2].sum_gradients += ordered_gradients[i + 2];
        out[bin3].sum_gradients += ordered_gradients[i + 3];

        out[bin0].sum_hessians += ordered_hessians[i];
        out[bin1].sum_hessians += ordered_hessians[i + 1];
        out[bin2].sum_hessians += ordered_hessians[i + 2];
        out[bin3].sum_hessians += ordered_hessians[i + 3];

        ++out[bin0].cnt;
        ++out[bin1].cnt;
        ++out[bin2].cnt;
        ++out[bin3].cnt;
      }
      for (; i < num_data; ++i) {
        VAL_T bin = data_[i];
        out[bin].sum_gradients += ordered_gradients[i];
        out[bin].sum_hessians += ordered_hessians[i];
        ++out[bin].cnt;
      }
    }
  }

  virtual data_size_t Split(unsigned int threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    for (data_size_t i = 0; i < num_data; ++i) {
      data_size_t idx = data_indices[i];
      if (data_[idx] > threshold) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  }
  data_size_t num_data() const override { return num_data_; }

  /*! \brief not ordered bin for dense feature */
  OrderedBin* CreateOrderedBin() const override { return nullptr; }

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

  void SaveBinaryToFile(FILE* file) const override {
    fwrite(data_.data(), sizeof(VAL_T), num_data_, file);
  }

  size_t SizesInByte() const override {
    return sizeof(VAL_T) * num_data_;
  }

protected:
  data_size_t num_data_;
  std::vector<VAL_T> data_;
};

template <typename VAL_T>
class DenseBinIterator: public BinIterator {
public:
  explicit DenseBinIterator(const DenseBin<VAL_T>* bin_data)
    : bin_data_(bin_data) {
  }
  uint32_t Get(data_size_t idx) override {
    return bin_data_->Get(idx);
  }
private:
  const DenseBin<VAL_T>* bin_data_;
};

template <typename VAL_T>
BinIterator* DenseBin<VAL_T>::GetIterator(data_size_t) const {
  return new DenseBinIterator<VAL_T>(this);
}

template <typename VAL_T>
class DenseCategoricalBin: public DenseBin<VAL_T> {
public:
  DenseCategoricalBin(data_size_t num_data, int default_bin)
    : DenseBin<VAL_T>(num_data, default_bin) {
  }

  virtual data_size_t Split(unsigned int threshold, data_size_t* data_indices, data_size_t num_data,
    data_size_t* lte_indices, data_size_t* gt_indices) const override {
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    for (data_size_t i = 0; i < num_data; ++i) {
      data_size_t idx = data_indices[i];
      if (DenseBin<VAL_T>::data_[idx] != threshold) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  }
};

}  // namespace LightGBM
#endif   // LightGBM_IO_DENSE_BIN_HPP_
