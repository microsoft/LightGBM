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

namespace LightGBM {

class CUDADataSplitter {
 public:
  CUDADataSplitter(const data_size_t num_data, const int max_num_leaves);

  void Init();

  void BeforeTrain(const data_size_t* data_indices);

  void Split(const int* leaf_id, const int* best_split_feature, const int* best_split_threshold);

  Tree* GetCPUTree();

  const data_size_t* data_indices() { return cuda_data_indices_; }

  const data_size_t* leaf_num_data_offsets() { return cuda_leaf_num_data_offsets_; }

  const data_size_t* leaf_num_data() { return cuda_leaf_num_data_; }

 private:
  // kernel launch functions
  void LaunchFillDataIndicesBeforeTrain();

  // CPU
  const data_size_t num_data_;
  std::vector<data_size_t> data_indices_;
  const int max_num_leaves_;

  // GPU
  data_size_t* cuda_data_indices_;
  data_size_t* cuda_leaf_num_data_offsets_;
  data_size_t* cuda_leaf_num_data_;

  data_size_t* cuda_num_data_;
  int* cuda_max_num_leaves_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_DATA_SPLITTER_HPP_