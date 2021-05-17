/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_objective.hpp"

namespace LightGBM {

CUDAObjective::CUDAObjective(const data_size_t num_data): num_data_(num_data) {}

}  // namespace LightGBM

#endif  // USE_CUDA
