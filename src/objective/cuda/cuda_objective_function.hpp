/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/meta.h>

namespace LightGBM {

class CUDAObjectiveInterface {

};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_OBJECTIVE_HPP_
