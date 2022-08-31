/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA_EXP

#define GET_GRADIENTS_BLOCK_SIZE_REGRESSION (1024)

#include <LightGBM/cuda/cuda_objective_function.hpp>
#include "../regression_objective.hpp"

namespace LightGBM {

class CUDARegressionL2loss : public CUDAObjectiveInterface, public RegressionL2loss {
 public:
  explicit CUDARegressionL2loss(const Config& config);

  explicit CUDARegressionL2loss(const std::vector<std::string>& strs);

  ~CUDARegressionL2loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

  void ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const override;

  double BoostFromScore(int) const override;

  void RenewTreeOutputCUDA(const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;

  std::function<void(data_size_t, const double*, double*)> GetCUDAConvertOutputFunc() const override {
    return [this] (data_size_t num_data, const double* input, double* output) {
      ConvertOutputCUDA(num_data, input, output);
    };
  }

  bool IsConstantHessian() const override {
    if (cuda_weights_ == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  bool IsCUDAObjective() const override { return true; }

 protected:
  virtual double LaunchCalcInitScoreKernel() const;

  virtual void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  virtual void LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const;

  virtual void LaunchRenewTreeOutputCUDAKernel(
    const double* /*score*/, const data_size_t* /*data_indices_in_leaf*/, const data_size_t* /*num_data_in_leaf*/,
    const data_size_t* /*data_start_in_leaf*/, const int /*num_leaves*/, double* /*leaf_value*/) const {}

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
  label_t* cuda_trans_label_;
  double* cuda_block_buffer_;
  data_size_t num_get_gradients_blocks_;
  data_size_t num_init_score_blocks_;
};


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
