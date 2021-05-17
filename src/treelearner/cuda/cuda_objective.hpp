/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_OBJECTIVE_HPP_

#ifdef USE_CUDA

#include "new_cuda_utils.hpp"
#include <LightGBM/meta.h>

namespace LightGBM {

class CUDAObjective {
 public:
  CUDAObjective(const data_size_t num_data);

  virtual void GetGradients(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) = 0;

 protected:
  const data_size_t num_data_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_OBJECTIVE_HPP_