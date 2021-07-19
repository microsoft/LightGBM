/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_CENTRALIZED_INFO_HPP_
#define LIGHTGBM_CUDA_CENTRALIZED_INFO_HPP_

#ifdef USE_CUDA

#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>
#include "new_cuda_utils.hpp"

namespace LightGBM {

// maintina centralized information for tree training
// these information are shared by various cuda objects in tree training
class CUDACentralizedInfo {
 public:
  CUDACentralizedInfo(const data_size_t num_data, const int num_leaves, const int num_features);

  void Init(const label_t* labels, const Dataset* train_data);

  void BeforeTrain(const score_t* gradients, const score_t* hessians);

  const data_size_t* cuda_num_data() const { return cuda_num_data_; }

  const int* cuda_num_leaves() const { return cuda_num_leaves_; }

  const int* cuda_num_features() const { return cuda_num_features_; }

  const score_t* cuda_gradients() const { return cuda_gradients_; }

  const score_t* cuda_hessians() const { return cuda_hessians_; }

  const label_t* cuda_labels() const { return cuda_labels_; }

  const data_size_t* cuda_query_boundaries() { return cuda_query_boundaries_; }

  void Test() {
    data_size_t test_num_data = 0;
    int test_num_leaves = 0;
    int test_num_features = 0;

    CopyFromCUDADeviceToHost<data_size_t>(&test_num_data, cuda_num_data_, 1);
    CopyFromCUDADeviceToHost<int>(&test_num_leaves, cuda_num_leaves_, 1);
    CopyFromCUDADeviceToHost<int>(&test_num_features, cuda_num_features_, 1);
    Log::Warning("CUDACentralizedInfo::Test test_num_data = %d", test_num_data);
    Log::Warning("CUDACentralizedInfo::Test test_num_leaves = %d", test_num_leaves);
    Log::Warning("CUDACentralizedInfo::Test test_num_features = %d", test_num_features);
  }

 private:
  // Host memory
  const data_size_t num_data_;
  const int num_leaves_;
  const int num_features_;

  // CUDA memory, held by this object
  data_size_t* cuda_num_data_;
  int* cuda_num_leaves_;
  int* cuda_num_features_;
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
  label_t* cuda_labels_;
  data_size_t* cuda_query_boundaries_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CENTRALIZED_INFO_HPP_
