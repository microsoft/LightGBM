/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA_EXP

#define GET_GRADIENTS_BLOCK_SIZE_REGRESSION (1024)

#include <LightGBM/cuda/cuda_objective_function.hpp>

#include <string>
#include <vector>

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

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
  CUDAVector<label_t> cuda_trans_label_;
  CUDAVector<double> cuda_block_buffer_;
  data_size_t num_get_gradients_blocks_;
  data_size_t num_init_score_blocks_;
};


class CUDARegressionL1loss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionL1loss(const Config& config);

  explicit CUDARegressionL1loss(const std::vector<std::string>& strs);

  ~CUDARegressionL1loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void RenewTreeOutputCUDA(const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;

  bool IsRenewTreeOutput() const override { return true; }

 protected:
  CUDAVector<data_size_t> cuda_data_indices_buffer_;
  CUDAVector<double> cuda_weights_prefix_sum_;
  CUDAVector<double> cuda_weights_prefix_sum_buffer_;
  CUDAVector<double> cuda_residual_buffer_;
  CUDAVector<label_t> cuda_weight_by_leaf_buffer_;
  CUDAVector<label_t> cuda_percentile_result_;

  double LaunchCalcInitScoreKernel() const override;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const;
};


class CUDARegressionHuberLoss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionHuberLoss(const Config& config);

  explicit CUDARegressionHuberLoss(const std::vector<std::string>& strs);

  ~CUDARegressionHuberLoss();

  bool IsRenewTreeOutput() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double alpha_ = 0.0f;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class CUDARegressionFairLoss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionFairLoss(const Config& config);

  explicit CUDARegressionFairLoss(const std::vector<std::string>& strs);

  ~CUDARegressionFairLoss();

  bool IsConstantHessian() const override {
    return false;
  }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double c_ = 0.0f;
};


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_REGRESSION_OBJECTIVE_HPP_
