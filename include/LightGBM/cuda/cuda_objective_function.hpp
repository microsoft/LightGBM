/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/meta.h>

namespace LightGBM {

class CUDAObjectiveInterface {
 public:
  virtual void ConvertOutputCUDA(const data_size_t /*num_data*/, const double* /*input*/, double* /*output*/) const {}
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP

#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_
