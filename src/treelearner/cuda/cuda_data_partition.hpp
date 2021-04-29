/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
#define LIGHTGBM_CUDA_DATA_SPLITTER_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include <LightGBM/tree.h>
#include "new_cuda_utils.hpp"

#define FILL_INDICES_BLOCK_SIZE_DATA_PARTITION (1024)

namespace LightGBM {

class CUDADataPartition {
 public:
  CUDADataPartition(const data_size_t num_data, const int num_leaves,
    const data_size_t* cuda_num_data, const int* cuda_num_leaves);

  void Init();

  void BeforeTrain(const data_size_t* data_indices);

  void Split(const int* leaf_id, const int* best_split_feature, const int* best_split_threshold);

  Tree* GetCPUTree();

  const data_size_t* cuda_leaf_num_data_offsets() { return cuda_leaf_num_data_offsets_; }

  void Test() {
    PrintLastCUDAError();
    std::vector<data_size_t> test_data_indices(num_data_, -1);
    CopyFromCUDADeviceToHost<data_size_t>(test_data_indices.data(), cuda_data_indices_, static_cast<size_t>(num_data_));
    for (data_size_t i = 0; i < num_data_; ++i) {
      CHECK_EQ(i, test_data_indices[i]);
    }
    Log::Warning("CUDADataPartition::Test Pass");
  }

  const data_size_t* cuda_leaf_num_data_offsets() const { return cuda_leaf_num_data_offsets_; }

 private:
  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  void LaunchSplitKernel(const int* leaf_id, const int* best_split_feature, const int* best_split_threshold);

  // Host memory
  const data_size_t num_data_;
  const int num_leaves_;

  // CUDA memory, held by this object
  data_size_t* cuda_data_indices_;
  data_size_t* cuda_leaf_num_data_offsets_;

  // CUDA memory, held by other object
  const data_size_t* cuda_num_data_;
  const int* cuda_num_leaves_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_
