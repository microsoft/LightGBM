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
    SparsePair() : ridx(0), bin(0) {}
  };

  OrderedSparseBin(const SparseBin<VAL_T>* bin_data)
    :bin_data_(bin_data) {
    data_size_t cur_pos = 0;
    data_size_t i_delta = -1;
    int non_zero_cnt = 0;
    while (bin_data_->NextNonzero(&i_delta, &cur_pos)) {
      ++non_zero_cnt;
    }
    ordered_pair_.resize(non_zero_cnt);
    leaf_cnt_.push_back(non_zero_cnt);
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

  void ConstructHistogram(int leaf, const score_t* gradient, const score_t* hessian, int num_bin,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const data_size_t start = leaf_start_[leaf];
    const data_size_t end = start + leaf_cnt_[leaf];
    const data_size_t group_rest = (end - start) & 65535;
    const data_size_t rest = (end - start) & 0x7;
    data_size_t i = start;
    for (; i < end - group_rest;) {
      std::vector<HistogramBinEntry> tmp_sumup_buf(num_bin);
      for (data_size_t j = 0; j < 65536; j += 8, i += 8) {
        const VAL_T bin0 = ordered_pair_[i].bin;
        const VAL_T bin1 = ordered_pair_[i + 1].bin;
        const VAL_T bin2 = ordered_pair_[i + 2].bin;
        const VAL_T bin3 = ordered_pair_[i + 3].bin;
        const VAL_T bin4 = ordered_pair_[i + 4].bin;
        const VAL_T bin5 = ordered_pair_[i + 5].bin;
        const VAL_T bin6 = ordered_pair_[i + 6].bin;
        const VAL_T bin7 = ordered_pair_[i + 7].bin;

        const auto g0 = gradient[ordered_pair_[i].ridx];
        const auto h0 = hessian[ordered_pair_[i].ridx];
        const auto g1 = gradient[ordered_pair_[i + 1].ridx];
        const auto h1 = hessian[ordered_pair_[i + 1].ridx];
        const auto g2 = gradient[ordered_pair_[i + 2].ridx];
        const auto h2 = hessian[ordered_pair_[i + 2].ridx];
        const auto g3 = gradient[ordered_pair_[i + 3].ridx];
        const auto h3 = hessian[ordered_pair_[i + 3].ridx];
        const auto g4 = gradient[ordered_pair_[i + 4].ridx];
        const auto h4 = hessian[ordered_pair_[i + 4].ridx];
        const auto g5 = gradient[ordered_pair_[i + 5].ridx];
        const auto h5 = hessian[ordered_pair_[i + 5].ridx];
        const auto g6 = gradient[ordered_pair_[i + 6].ridx];
        const auto h6 = hessian[ordered_pair_[i + 6].ridx];
        const auto g7 = gradient[ordered_pair_[i + 7].ridx];
        const auto h7 = hessian[ordered_pair_[i + 7].ridx];

        AddGradientToHistogram(tmp_sumup_buf.data(), bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                               g0, g1, g2, g3, g4, g5, g6, g7);
        AddHessianToHistogram(tmp_sumup_buf.data(), bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                              h0, h1, h2, h3, h4, h5, h6, h7);
        AddCountToHistogram(tmp_sumup_buf.data(), bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7);
      }
      for (int j = 0; j < num_bin; ++j) {
        out[j].sum_gradients += tmp_sumup_buf[j].sum_gradients;
        out[j].sum_hessians += tmp_sumup_buf[j].sum_hessians;
        out[j].cnt += tmp_sumup_buf[j].cnt;
      }
    }
    // use data on current leaf to construct histogram
    for (; i < end - rest; i += 8) {

      const VAL_T bin0 = ordered_pair_[i].bin;
      const VAL_T bin1 = ordered_pair_[i + 1].bin;
      const VAL_T bin2 = ordered_pair_[i + 2].bin;
      const VAL_T bin3 = ordered_pair_[i + 3].bin;
      const VAL_T bin4 = ordered_pair_[i + 4].bin;
      const VAL_T bin5 = ordered_pair_[i + 5].bin;
      const VAL_T bin6 = ordered_pair_[i + 6].bin;
      const VAL_T bin7 = ordered_pair_[i + 7].bin;

      const auto g0 = gradient[ordered_pair_[i].ridx];
      const auto h0 = hessian[ordered_pair_[i].ridx];
      const auto g1 = gradient[ordered_pair_[i + 1].ridx];
      const auto h1 = hessian[ordered_pair_[i + 1].ridx];
      const auto g2 = gradient[ordered_pair_[i + 2].ridx];
      const auto h2 = hessian[ordered_pair_[i + 2].ridx];
      const auto g3 = gradient[ordered_pair_[i + 3].ridx];
      const auto h3 = hessian[ordered_pair_[i + 3].ridx];
      const auto g4 = gradient[ordered_pair_[i + 4].ridx];
      const auto h4 = hessian[ordered_pair_[i + 4].ridx];
      const auto g5 = gradient[ordered_pair_[i + 5].ridx];
      const auto h5 = hessian[ordered_pair_[i + 5].ridx];
      const auto g6 = gradient[ordered_pair_[i + 6].ridx];
      const auto h6 = hessian[ordered_pair_[i + 6].ridx];
      const auto g7 = gradient[ordered_pair_[i + 7].ridx];
      const auto h7 = hessian[ordered_pair_[i + 7].ridx];

      AddGradientToHistogram(out, bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                             g0, g1, g2, g3, g4, g5, g6, g7);
      AddHessianToHistogram(out, bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                            h0, h1, h2, h3, h4, h5, h6, h7);
      AddCountToHistogram(out, bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7);
    }

    for (; i < end; ++i) {

      const VAL_T bin0 = ordered_pair_[i].bin;

      const auto g0 = gradient[ordered_pair_[i].ridx];
      const auto h0 = hessian[ordered_pair_[i].ridx];

      out[bin0].sum_gradients += g0;
      out[bin0].sum_hessians += h0;
      ++out[bin0].cnt;
    }
  }

  void ConstructHistogram(int leaf, const score_t* gradient, int num_bin,
                          HistogramBinEntry* out) const override {
    // get current leaf boundary
    const data_size_t start = leaf_start_[leaf];
    const data_size_t end = start + leaf_cnt_[leaf];
    const data_size_t group_rest = (end - start) & 65535;
    const data_size_t rest = (end - start) & 0x7;
    data_size_t i = start;
    for (; i < end - group_rest;) {
      std::vector<HistogramBinEntry> tmp_sumup_buf(num_bin);
      for (data_size_t j = 0; j < 65536; j += 8, i += 8) {
        const VAL_T bin0 = ordered_pair_[i].bin;
        const VAL_T bin1 = ordered_pair_[i + 1].bin;
        const VAL_T bin2 = ordered_pair_[i + 2].bin;
        const VAL_T bin3 = ordered_pair_[i + 3].bin;
        const VAL_T bin4 = ordered_pair_[i + 4].bin;
        const VAL_T bin5 = ordered_pair_[i + 5].bin;
        const VAL_T bin6 = ordered_pair_[i + 6].bin;
        const VAL_T bin7 = ordered_pair_[i + 7].bin;

        const auto g0 = gradient[ordered_pair_[i].ridx];
        const auto g1 = gradient[ordered_pair_[i + 1].ridx];
        const auto g2 = gradient[ordered_pair_[i + 2].ridx];
        const auto g3 = gradient[ordered_pair_[i + 3].ridx];
        const auto g4 = gradient[ordered_pair_[i + 4].ridx];
        const auto g5 = gradient[ordered_pair_[i + 5].ridx];
        const auto g6 = gradient[ordered_pair_[i + 6].ridx];
        const auto g7 = gradient[ordered_pair_[i + 7].ridx];

        AddGradientToHistogram(tmp_sumup_buf.data(), bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                               g0, g1, g2, g3, g4, g5, g6, g7);
        AddCountToHistogram(tmp_sumup_buf.data(), bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7);
      }
      for (int j = 0; j < num_bin; ++j) {
        out[j].sum_gradients += tmp_sumup_buf[j].sum_gradients;
        out[j].sum_hessians += tmp_sumup_buf[j].sum_hessians;
        out[j].cnt += tmp_sumup_buf[j].cnt;
      }
    }
    // use data on current leaf to construct histogram
    for (; i < end - rest; i += 8) {

      const VAL_T bin0 = ordered_pair_[i].bin;
      const VAL_T bin1 = ordered_pair_[i + 1].bin;
      const VAL_T bin2 = ordered_pair_[i + 2].bin;
      const VAL_T bin3 = ordered_pair_[i + 3].bin;
      const VAL_T bin4 = ordered_pair_[i + 4].bin;
      const VAL_T bin5 = ordered_pair_[i + 5].bin;
      const VAL_T bin6 = ordered_pair_[i + 6].bin;
      const VAL_T bin7 = ordered_pair_[i + 7].bin;

      const auto g0 = gradient[ordered_pair_[i].ridx];
      const auto g1 = gradient[ordered_pair_[i + 1].ridx];
      const auto g2 = gradient[ordered_pair_[i + 2].ridx];
      const auto g3 = gradient[ordered_pair_[i + 3].ridx];
      const auto g4 = gradient[ordered_pair_[i + 4].ridx];
      const auto g5 = gradient[ordered_pair_[i + 5].ridx];
      const auto g6 = gradient[ordered_pair_[i + 6].ridx];
      const auto g7 = gradient[ordered_pair_[i + 7].ridx];

      AddGradientToHistogram(out, bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7,
                             g0, g1, g2, g3, g4, g5, g6, g7);
      AddCountToHistogram(out, bin0, bin1, bin2, bin3, bin4, bin5, bin6, bin7);
    }

    for (; i < end; ++i) {

      const VAL_T bin0 = ordered_pair_[i].bin;

      const auto g0 = gradient[ordered_pair_[i].ridx];

      out[bin0].sum_gradients += g0;
      ++out[bin0].cnt;
    }
  }

  void Split(int leaf, int right_leaf, const char* is_in_leaf, char mark) override {
    // get current leaf boundary
    const data_size_t l_start = leaf_start_[leaf];
    const data_size_t l_end = l_start + leaf_cnt_[leaf];
    // new left leaf end after split
    data_size_t new_left_end = l_start;

    for (data_size_t i = l_start; i < l_end; ++i) {
      if (is_in_leaf[ordered_pair_[i].ridx] == mark) {
        std::swap(ordered_pair_[new_left_end], ordered_pair_[i]);
        ++new_left_end;
      }
    }

    leaf_start_[right_leaf] = new_left_end;
    leaf_cnt_[leaf] = new_left_end - l_start;
    leaf_cnt_[right_leaf] = l_end - new_left_end;
  }
  data_size_t NonZeroCount(int leaf) const override {
    return static_cast<data_size_t>(leaf_cnt_[leaf]);
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
