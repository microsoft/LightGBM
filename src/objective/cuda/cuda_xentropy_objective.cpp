/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_xentropy_objective.hpp"

namespace LightGBM {

CUDACrossEntropy::CUDACrossEntropy(const Config& config): CrossEntropy(config) {}

CUDACrossEntropy::CUDACrossEntropy(const std::vector<std::string>& strs): CrossEntropy(strs) {}

CUDACrossEntropy::~CUDACrossEntropy() {}

void CUDACrossEntropy::Init(const Metadata& metadata, data_size_t num_data) {
  CrossEntropy::Init(metadata, num_data);
  cuda_labels_ = metadata.cuda_metadata()->cuda_label();
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
}

void CUDACrossEntropy::GetGradients(const double* score, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(score, gradients, hessians);
}

}  // namespace LightGBM
