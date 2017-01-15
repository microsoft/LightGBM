#ifndef LIGHTGBM_TREELEARNER_SPARSE_BIN_POOL_HPP_
#define LIGHTGBM_TREELEARNER_SPARSE_BIN_POOL_HPP_

#include "feature_histogram.hpp"

#include <LightGBM/dataset.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <cstring>
#include <vector>
#include <chrono>

namespace LightGBM {


class SparseBinPool {
public:
  SparseBinPool(const Dataset* train_data, int num_thread, bool sparse_aware)
    : num_threads_(num_thread), hist_buf(num_thread) {
    if (sparse_aware) {
      for (int i = 0; i < train_data->num_features(); ++i) {
        feature_mapper_.push_back(i);
      } 
    } else {
      // only add sparse feature if not using sparse_aware
      for (int i = 0; i < train_data->num_features(); ++i) {
        if (train_data->FeatureAt(i)->is_sparse()) {
          feature_mapper_.push_back(i);
        }
      }
    }
    total_bin_ = 0;
    total_feature_ = 0;
    bin_boundaries_.push_back(0);
    for (auto fidx : feature_mapper_) {
      total_bin_ += train_data->FeatureAt(fidx)->num_bin();
      ++total_feature_;
      bin_boundaries_.push_back(total_bin_);
    }
    // get col iterator
    std::vector<std::unique_ptr<BinIterator>> iterators(total_feature_);
    for (int i = 0; i < total_feature_; ++i) {
      iterators[i].reset(train_data->FeatureAt(feature_mapper_[i])->bin_data()->GetIterator(0));
    }
    // compression data
    row_boundaries_.clear();
    bins_.clear();
    int non_zero_cnt = 0;
    row_boundaries_.push_back(non_zero_cnt);
    for (int i = 0; i < train_data->num_data(); ++i) {
      for (int j = 0; j < total_feature_; ++j) {
        auto cur_bin = iterators[j]->Get(i);
        if (cur_bin > 0) {
          bins_.push_back(cur_bin + bin_boundaries_[j]);
          ++non_zero_cnt;
        }
      }
      row_boundaries_.push_back(non_zero_cnt);
    }
    row_boundaries_.shrink_to_fit();
    bins_.shrink_to_fit();
#pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num_threads_; ++i) {
      hist_buf[i].resize(total_bin_);
    }
  }

  void Construct(const data_size_t* data_indices, data_size_t num_data,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    FeatureHistogram* histogram_array) {

#pragma omp parallel for num_threads(num_threads_)
    for (int i = 0; i < num_threads_; ++i) {
      std::fill(hist_buf[i].begin(), hist_buf[i].end(), HistogramBinEntry());
    }

    if (data_indices == nullptr) {
      #pragma omp parallel for num_threads(num_threads_) schedule(guided)
      for (data_size_t i = 0; i < num_data; ++i) {
        const int tid = omp_get_thread_num();
        auto buf_ptr = hist_buf[tid].data();
        auto begin = row_boundaries_[i];
        auto end = row_boundaries_[i + 1];
        for (data_size_t j = begin; j < end; ++j) {
          auto cur_bin = bins_[j];
          buf_ptr[cur_bin].sum_gradients += ordered_gradients[i];
          buf_ptr[cur_bin].sum_hessians += ordered_hessians[i];
          ++buf_ptr[cur_bin].cnt;
        }
      }
    } else {
      #pragma omp parallel for num_threads(num_threads_) schedule(guided)
      for (data_size_t i = 0; i < num_data; ++i) {
        const int tid = omp_get_thread_num();
        auto buf_ptr = hist_buf[tid].data();
        auto row_idx = data_indices[i];
        auto begin = row_boundaries_[row_idx];
        auto end = row_boundaries_[row_idx + 1];
        for (data_size_t j = begin; j < end; ++j) {
          auto cur_bin = bins_[j];
          buf_ptr[cur_bin].sum_gradients += ordered_gradients[i];
          buf_ptr[cur_bin].sum_hessians += ordered_hessians[i];
          ++buf_ptr[cur_bin].cnt;
        }
      }
    }
    #pragma omp parallel for num_threads(num_threads_) schedule(guided)
    for (int i = 0; i < total_feature_; ++i) {
      int feat_idx = feature_mapper_[i];
      auto out_bin_data = histogram_array[feat_idx].GetData();
      auto bin_begin = bin_boundaries_[i];
      auto bin_end = bin_boundaries_[i + 1];
      for (int tid = 0; tid < num_threads_; ++tid) {
        auto buf_ptr = hist_buf[tid].data();
        for (int j = bin_begin; j < bin_end; ++j) {
          auto out_idx = j - bin_begin;
          out_bin_data[out_idx].sum_gradients += buf_ptr[j].sum_gradients;
          out_bin_data[out_idx].sum_hessians += buf_ptr[j].sum_hessians;
          out_bin_data[out_idx].cnt += buf_ptr[j].cnt;
        }
      }
    }
  }
private:
  int num_threads_;
  int total_bin_;
  int total_feature_;
  std::vector<int> feature_mapper_;
  std::vector<int> bin_boundaries_;
  std::vector<data_size_t> row_boundaries_;
  std::vector<unsigned int> bins_;
  std::vector<std::vector<HistogramBinEntry>> hist_buf;
  
};



}

#endif 