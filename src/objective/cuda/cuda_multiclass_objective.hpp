/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_

#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include <string>
#include <vector>

#include "../multiclass_objective.hpp"

#define GET_GRADIENTS_BLOCK_SIZE_MULTICLASS (1024)

namespace LightGBM {

class CUDAMulticlassSoftmax: public CUDAObjectiveInterface, public MulticlassSoftmax {
 public:
  explicit CUDAMulticlassSoftmax(const Config& config);

  explicit CUDAMulticlassSoftmax(const std::vector<std::string>& strs);

  ~CUDAMulticlassSoftmax();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

  void ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const override;

  std::function<void(data_size_t, const double*, double*)> GetCUDAConvertOutputFunc() const override {
    return [this] (data_size_t num_data, const double* input, double* output) {
      ConvertOutputCUDA(num_data, input, output);
    };
  }

  bool IsCUDAObjective() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const;

  void LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const;

  // CUDA memory, held by other objects
  const label_t* cuda_label_;
  const label_t* cuda_weights_;

  // CUDA memory, held by this object
  CUDAVector<double> cuda_softmax_buffer_;
};


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_MULTICLASS_OBJECTIVE_HPP_
