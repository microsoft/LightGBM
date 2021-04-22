/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_LEAF_SPLITS_INIT_HPP_
#define LIGHTGBM_CUDA_LEAF_SPLITS_INIT_HPP_

#ifdef USE_CUDA

#include <LightGBM/utils/log.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/meta.h>

#define INIT_SUM_BLOCK_SIZE (1024)

namespace LightGBM {

class CUDALeafSplitsInit {
 public:
  CUDALeafSplitsInit(const score_t* cuda_gradients, const score_t* cuda_hessians, const data_size_t num_data);

  void Init();

  void Compute();

  const double* smaller_leaf_sum_gradients() { return smaller_leaf_sum_gradients_; }

  const double* smaller_leaf_sum_hessians() { return smaller_leaf_sum_hessians_; }

  const double* larger_leaf_sum_gradients() { return larger_leaf_sum_gradients_; }

  const double* larger_leaf_sum_hessians() { return larger_leaf_sum_hessians_; }

  const int* smaller_leaf_index() { return smaller_leaf_index_; }
  
  const int* larger_leaf_index() { return larger_leaf_index_; }

  void LaunchLeafSplitsInit(const int num_blocks, const int init_sum_block_size,
    const score_t* cuda_gradients, const score_t* cuda_hessians, const data_size_t* num_data,
    double* smaller_leaf_sum_gradients, double* smaller_leaf_sum_hessians);

 protected:
  const score_t* cuda_gradients_;
  const score_t* cuda_hessians_;
  double* smaller_leaf_sum_gradients_;
  double* smaller_leaf_sum_hessians_;
  double host_smaller_leaf_sum_gradients_;
  double host_smaller_leaf_sum_hessians_;
  double* larger_leaf_sum_gradients_;
  double* larger_leaf_sum_hessians_;
  int* smaller_leaf_index_;
  int* larger_leaf_index_;
  int* cuda_num_data_;

  const int num_data_;
  int num_blocks_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_LEAF_SPLITS_INIT_HPP_
