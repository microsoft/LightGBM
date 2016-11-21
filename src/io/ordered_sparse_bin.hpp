#ifndef LIGHTGBM_IO_ORDERED_SPARSE_BIN_HPP_
#define LIGHTGBM_IO_ORDERED_SPARSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstring>
#include <cstdint>

#include <vector>
#include <mutex>
#include <algorithm>

#include "sparse_bin.hpp"

namespace LightGBM {

/*!
* \brief Interface for ordered bin data. efficient for construct histogram, especially for sparse bin
*        There are 2 advantages by using ordered bin.
*        1. group the data by leafs to improve the cache hit.
*        2. only store the non-zero bin, which can speed up the histogram consturction for sparse features.
*        However it brings additional cost: it need re-order the bins after every split, which will cost much for dense feature.
*        So we only using ordered bin for sparse situations.
*/
template <typename VAL_T>
class OrderedSparseBin: public OrderedBin {
public:
  /*! \brief Pair to store one bin entry */
  struct SparsePair {
    data_size_t ridx;  // data(row) index
    VAL_T bin;  // bin for this data
    SparsePair(data_size_t r, VAL_T b) : ridx(r), bin(b) {}
  };

  OrderedSparseBin(const SparseBin<VAL_T>* bin_data)
    :bin_data_(bin_data) {
    data_size_t cur_pos = 0;
    data_size_t i_delta = -1;
    while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
      ordered_pair_.emplace_back(cur_pos, static_cast<VAL_T>(0));
    }
    ordered_pair_.shrink_to_fit();
  }

  ~OrderedSparseBin() {
  }

  void Init(const char* used_idices, int num_leaves) override {
    // initialize the leaf information
    leaf_start_ = std::vector<data_size_t>(num_leaves, 0);
    leaf_cnt_ = std::vector<data_size_t>(num_leaves, 0);
    if (used_idices == nullptr) {
      // if using all data, copy all non-zero pair
      data_size_t j = 0;
      data_size_t cur_pos = 0;
      data_size_t i_delta = -1;
      while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
        ordered_pair_[j].ridx = cur_pos;
        ordered_pair_[j].bin = bin_data_->vals_[i_delta];
        ++j;
      }
      leaf_cnt_[0] = static_cast<data_size_t>(j);
    } else {
      // if using part of data(bagging)
      data_size_t j = 0;
      data_size_t cur_pos = 0;
      data_size_t i_delta = -1;
      while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
        if (used_idices[cur_pos]) {
          ordered_pair_[j].ridx = cur_pos;
          ordered_pair_[j].bin = bin_data_->vals_[i_delta];
          ++j;
        }
      }
      leaf_cnt_[0] = j;
    }
  }

  void ConstructHistogram(int leaf, const score_t* gradient, const score_t* hessian,
    HistogramBinEntry* out) const override {
    // get current leaf boundary
    const data_size_t start = leaf_start_[leaf];
    const data_size_t end = start + leaf_cnt_[leaf];
    // use data on current leaf to construct histogram
    for (data_size_t i = start; i < end; ++i) {
      const VAL_T bin = ordered_pair_[i].bin;
      const data_size_t idx = ordered_pair_[i].ridx;
      out[bin].sum_gradients += gradient[idx];
      out[bin].sum_hessians += hessian[idx];
      ++out[bin].cnt;
    }
  }

  void Split(int leaf, int right_leaf, const char* left_indices) override {
    // get current leaf boundary
    const data_size_t l_start = leaf_start_[leaf];
    const data_size_t l_end = l_start + leaf_cnt_[leaf];
    // new left leaf end after split
    data_size_t new_left_end = l_start;

    for (data_size_t i = l_start; i < l_end; ++i) {
      if (left_indices[ordered_pair_[i].ridx]) {
        std::swap(ordered_pair_[new_left_end], ordered_pair_[i]);
        ++new_left_end;
      }
    }

    leaf_start_[right_leaf] = new_left_end;
    leaf_cnt_[leaf] = new_left_end - l_start;
    leaf_cnt_[right_leaf] = l_end - new_left_end;
  }

  /*! \brief Disable copy */
  OrderedSparseBin<VAL_T>& operator=(const OrderedSparseBin<VAL_T>&) = delete;
  /*! \brief Disable copy */
  OrderedSparseBin<VAL_T>(const OrderedSparseBin<VAL_T>&) = delete;

private:
  const SparseBin<VAL_T>* bin_data_;
  /*! \brief Store non-zero pair , group by leaf */
  std::vector<SparsePair> ordered_pair_;
  /*! \brief leaf_start_[i] means data in i-th leaf start from */
  std::vector<data_size_t> leaf_start_;
  /*! \brief leaf_cnt_[i] means number of data in i-th leaf */
  std::vector<data_size_t> leaf_cnt_;
};

template <typename VAL_T>
OrderedBin* SparseBin<VAL_T>::CreateOrderedBin() const {
  return new OrderedSparseBin<VAL_T>(this);
}

}  // namespace LightGBM
#endif   // LightGBM_IO_ORDERED_SPARSE_BIN_HPP_
