/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
#define LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/feature_group.h>
#include <LightGBM/tree.h>

#include <fstream>

#include "new_cuda_utils.hpp"

#include <vector>

#define SHRAE_HIST_SIZE (6144 * 2)
#define NUM_DATA_PER_THREAD (400)
#define NUM_THRADS_PER_BLOCK (504)
#define NUM_FEATURE_PER_THREAD_GROUP (28)
#define SUBTRACT_BLOCK_SIZE (1024)
#define FIX_HISTOGRAM_SHARED_MEM_SIZE (1024)
#define FIX_HISTOGRAM_BLOCK_SIZE (512)

namespace LightGBM {

class CUDAHistogramConstructor {
 public:
  CUDAHistogramConstructor(const Dataset* train_data, const int num_leaves, const int num_threads,
    const score_t* cuda_gradients, const score_t* cuda_hessians, const std::vector<uint32_t>& feature_hist_offsets);

  void Init(const Dataset* train_data);

  void ConstructHistogramForLeaf(const int* cuda_smaller_leaf_index, const data_size_t* cuda_num_data_in_smaller_leaf, const int* cuda_larger_leaf_index,
    const data_size_t** cuda_data_indices_in_smaller_leaf, const data_size_t** cuda_data_indices_in_larger_leaf,
    const double* cuda_smaller_leaf_sum_gradients, const double* cuda_smaller_leaf_sum_hessians, hist_t** cuda_smaller_leaf_hist,
    const double* cuda_larger_leaf_sum_gradients, const double* cuda_larger_leaf_sum_hessians, hist_t** cuda_larger_leaf_hist,
    const data_size_t* cuda_leaf_num_data);

  void BeforeTrain();

  const hist_t* cuda_hist() const { return cuda_hist_; }

  hist_t* cuda_hist_pointer() const { return cuda_hist_; }

  hist_t* cuda_hist_pointer() { return cuda_hist_; }

  const uint8_t* cuda_data() const { return cuda_data_; }

  void TestAfterInit() {
    std::vector<uint8_t> test_data(data_.size(), 0);
    CopyFromCUDADeviceToHost(test_data.data(), cuda_data_, data_.size());
    for (size_t i = 0; i < 100; ++i) {
      Log::Warning("CUDAHistogramConstructor::TestAfterInit test_data[%d] = %d", i, test_data[i]);
    }
  }

  void TestAfterConstructHistogram() {
    PrintLastCUDAError();
    std::vector<hist_t> test_hist(num_total_bin_ * 2, 0.0f);
    /*CopyFromCUDADeviceToHost<hist_t>(test_hist.data(), cuda_hist_, static_cast<size_t>(num_total_bin_) * 2);
    for (int i = 0; i < 100; ++i) {
      Log::Warning("bin %d grad %f hess %f", i, test_hist[2 * i], test_hist[2 * i + 1]);
    }*/
    const hist_t* leaf_2_cuda_hist_ptr = cuda_hist_;// + 3 * 2 * num_total_bin_;
    Log::Warning("cuda_hist_ptr = %ld", leaf_2_cuda_hist_ptr);
    CopyFromCUDADeviceToHost<hist_t>(test_hist.data(), leaf_2_cuda_hist_ptr, 2 * num_total_bin_);
    std::ofstream fout("leaf_2_cuda_hist.txt");
    for (int i = 0; i < num_total_bin_; ++i) {
      Log::Warning("bin %d grad %f hess %f", i, test_hist[2 * i], test_hist[2 * i + 1]);
      fout << "bin " << i << " grad " << test_hist[2 * i] << " hess " << test_hist[2 * i + 1] << "\n"; 
    }
    fout.close();
  }

 private:
  void LaunchGetOrderedGradientsKernel(
    const data_size_t num_data_in_leaf,
    const data_size_t** cuda_data_indices_in_leaf);

  void LaunchConstructHistogramKernel(const int* cuda_leaf_index,
    const data_size_t* cuda_smaller_leaf_num_data,
    const data_size_t** cuda_data_indices_in_leaf,
    const data_size_t* cuda_leaf_num_data,
    hist_t** cuda_leaf_hist);

  void LaunchSubtractHistogramKernel(const int* cuda_smaller_leaf_index,
    const int* cuda_larger_leaf_index, const double* smaller_leaf_sum_gradients, const double* smaller_leaf_sum_hessians,
    const double* larger_leaf_sum_gradients, const double* larger_leaf_sum_hessians,
    hist_t** cuda_smaller_leaf_hist, hist_t** cuda_larger_leaf_hist);

  void InitCUDAData(const Dataset* train_data);

  void PushOneData(const uint32_t feature_bin_value, const int feature_group_id, const data_size_t data_index);

  // Host memory
  // data on CPU, stored in row-wise style
  const data_size_t num_data_;
  const int num_features_;
  const int num_leaves_;
  const int num_threads_;
  int num_total_bin_;
  int num_feature_groups_;
  std::vector<uint8_t> data_;
  std::vector<uint32_t> feature_group_bin_offsets_;
  std::vector<uint8_t> feature_mfb_offsets_;
  std::vector<uint32_t> feature_num_bins_;
  std::vector<uint32_t> feature_hist_offsets_;
  std::vector<uint32_t> feature_most_freq_bins_;

  // CUDA memory, held by this object
  uint32_t* cuda_feature_group_bin_offsets_;
  uint8_t* cuda_feature_mfb_offsets_;
  uint32_t* cuda_feature_num_bins_;
  uint32_t* cuda_feature_hist_offsets_;
  uint32_t* cuda_feature_most_freq_bins_;
  hist_t* cuda_hist_;
  int* cuda_num_total_bin_;
  int* cuda_num_feature_groups_;
  uint8_t* cuda_data_;
  int* cuda_num_features_;
  score_t* cuda_ordered_gradients_;
  score_t* cuda_ordered_hessians_;

  // CUDA memory, held by other objects
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_HISTOGRAM_CONSTRUCTOR_HPP_
