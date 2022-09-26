/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_multiclass_objective.hpp"

#include <string>
#include <vector>

namespace LightGBM {

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const Config& config): CUDAObjectiveInterface<MulticlassSoftmax>(config) {}

CUDAMulticlassSoftmax::CUDAMulticlassSoftmax(const std::vector<std::string>& strs): CUDAObjectiveInterface<MulticlassSoftmax>(strs) {}

CUDAMulticlassSoftmax::~CUDAMulticlassSoftmax() {}

void CUDAMulticlassSoftmax::Init(const Metadata& metadata, data_size_t num_data) {
  CUDAObjectiveInterface<MulticlassSoftmax>::Init(metadata, num_data);
  cuda_softmax_buffer_.Resize(static_cast<size_t>(num_data) * static_cast<size_t>(num_class_));
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
