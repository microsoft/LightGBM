/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
#define LIGHTGBM_CUDA_LEAF_SPLITS_HPP_

#ifdef USE_CUDA

#include <LightGBM/utils/log.h>
#include <LightGBM/meta.h>
#include "new_cuda_utils.hpp"

#define INIT_SUM_BLOCK_SIZE_LEAF_SPLITS (6144)
#define NUM_THRADS_PER_BLOCK_LEAF_SPLITS (1024)
#define NUM_DATA_THREAD_ADD_LEAF_SPLITS (6)

namespace LightGBM {

class CUDALeafSplits {
 public:
  CUDALeafSplits();

  void Init();

  void InitValues(const double* cuda_sum_of_gradients, const double* cuda_sum_of_hessians,
    const data_size_t* cuda_num_data_in_leaf, const data_size_t* cuda_data_indices_in_leaf,
    const double* cuda_gain, const double* cuda_leaf_value);

  void InitValues();

 private:
  void LaunchInitValuesKernal();

  // Host memory
  const int num_data_;
  int num_blocks_init_from_gradients_;

  // CUDA memory, held by this object
  double* cuda_sum_of_gradients_;
  double* cuda_sum_of_hessians_;
  data_size_t* cuda_num_data_in_leaf_;
  double* cuda_gain_;
  double* cuda_leaf_value_;

  // CUDA memory, held by other object
  const data_size_t* cuda_data_indices_in_leaf_;
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
  const int* cuda_num_data_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_LEAF_SPLITS_HPP_
