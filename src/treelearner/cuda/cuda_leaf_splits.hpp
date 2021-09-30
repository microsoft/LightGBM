/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
#define LIGHTGBM_CUDA_LEAF_SPLITS_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/bin.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>

#define NUM_THRADS_PER_BLOCK_LEAF_SPLITS (1024)
#define NUM_DATA_THREAD_ADD_LEAF_SPLITS (6)

namespace LightGBM {

struct CUDALeafSplitsStruct {
 public:
  int leaf_index;
  double sum_of_gradients;
  double sum_of_hessians;
  data_size_t num_data_in_leaf;
  double gain;
  double leaf_value;
  const data_size_t* data_indices_in_leaf;
  hist_t* hist_in_leaf;
};

class CUDALeafSplits {
 public:
  explicit CUDALeafSplits(const data_size_t num_data);

  ~CUDALeafSplits();

  void Init();

  void InitValues(
    const score_t* cuda_gradients, const score_t* cuda_hessians,
    const data_size_t* cuda_bagging_data_indices,
    const data_size_t* cuda_data_indices_in_leaf, const data_size_t num_used_indices,
    hist_t* cuda_hist_in_leaf, double* root_sum_hessians);

  void InitValues();

  const CUDALeafSplitsStruct* GetCUDAStruct() const { return cuda_struct_; }

  CUDALeafSplitsStruct* GetCUDAStructRef() { return cuda_struct_; }

  void Resize(const data_size_t num_data);

 private:
  void LaunchInitValuesEmptyKernel();

  void LaunchInitValuesKernal(const data_size_t* cuda_bagging_data_indices,
                              const data_size_t* cuda_data_indices_in_leaf,
                              const data_size_t num_used_indices,
                              hist_t* cuda_hist_in_leaf);

  // Host memory
  data_size_t num_data_;
  int num_blocks_init_from_gradients_;

  // CUDA memory, held by this object
  CUDALeafSplitsStruct* cuda_struct_;
  double* cuda_sum_of_gradients_buffer_;
  double* cuda_sum_of_hessians_buffer_;

  // CUDA memory, held by other object
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
