/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA

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

class CUDARegressionL1loss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionL1loss(const Config& config);

  explicit CUDARegressionL1loss(const std::vector<std::string>& strs);

  ~CUDARegressionL1loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  const char* GetName() const override {
    return "regression_l1";
  }

  bool IsRenewTreeOutput() const override { return true; }

 protected:
  data_size_t* cuda_data_indices_buffer_;
  mutable double* cuda_weights_prefix_sum_;
  double* cuda_weights_prefix_sum_buffer_;
  mutable double* cuda_residual_buffer_;
  mutable label_t* cuda_weight_by_leaf_buffer_;
  label_t* cuda_percentile_result_;

  double LaunchCalcInitScoreKernel() const override;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;
};

class CUDARegressionHuberLoss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionHuberLoss(const Config& config);

  explicit CUDARegressionHuberLoss(const std::vector<std::string>& strs);

  ~CUDARegressionHuberLoss();

  const char* GetName() const override {
    return "huber";
  }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double alpha_ = 0.0f;
};

class CUDARegressionFairLoss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionFairLoss(const Config& config);

  explicit CUDARegressionFairLoss(const std::vector<std::string>& strs);

  ~CUDARegressionFairLoss();

  const char* GetName() const override {
    return "fair";
  }

  bool IsConstantHessian() const override {
    return false;
  }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  const double c_ = 0.0f;
};

class CUDARegressionPoissonLoss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionPoissonLoss(const Config& config);

  explicit CUDARegressionPoissonLoss(const std::vector<std::string>& strs);

  ~CUDARegressionPoissonLoss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  double LaunchCalcInitScoreKernel() const override;

  bool IsConstantHessian() const override {
    return false;
  }

  const char* GetName() const override {
    return "poisson";
  }

 protected:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  void LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const override;

  void LaunchCheckLabelKernel() const;

  const double max_delta_step_ = 0.0f;
  mutable double* cuda_block_buffer_;
};

class CUDARegressionQuantileloss : public CUDARegressionL2loss {
 public:
  explicit CUDARegressionQuantileloss(const Config& config);

  explicit CUDARegressionQuantileloss(const std::vector<std::string>& strs);

  ~CUDARegressionQuantileloss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  const char* GetName() const override {
    return "quantile";
  }

  bool IsRenewTreeOutput() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  double LaunchCalcInitScoreKernel() const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;

  const double alpha_ = 0.0f;
  data_size_t* cuda_data_indices_buffer_;
  mutable double* cuda_weights_prefix_sum_;
  double* cuda_weights_prefix_sum_buffer_;
  mutable double* cuda_residual_buffer_;
  mutable label_t* cuda_weight_by_leaf_buffer_;
  label_t* cuda_percentile_result_;
};

class CUDARegressionMAPELOSS : public CUDARegressionL1loss {
 public:
  explicit CUDARegressionMAPELOSS(const Config& config);

  explicit CUDARegressionMAPELOSS(const std::vector<std::string>& strs);

  ~CUDARegressionMAPELOSS();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  bool IsRenewTreeOutput() const override { return true; }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;

  double LaunchCalcInitScoreKernel() const override;

  void LaunchRenewTreeOutputCUDAKernel(
    const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override;

  void LaunchCalcLabelWeightKernel();

  label_t* cuda_label_weights_;
};

class CUDARegressionGammaLoss : public CUDARegressionPoissonLoss {
 public:
  explicit CUDARegressionGammaLoss(const Config& config);

  explicit CUDARegressionGammaLoss(const std::vector<std::string>& strs);

  ~CUDARegressionGammaLoss();

  const char* GetName() const override {
    return "gamma";
  }

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};

class CUDARegressionTweedieLoss : public CUDARegressionPoissonLoss {
 public:
  explicit CUDARegressionTweedieLoss(const Config& config);

  explicit CUDARegressionTweedieLoss(const std::vector<std::string>& strs);

  ~CUDARegressionTweedieLoss();

  const char* GetName() const override {
    return "tweedie";
  }

 private:
  const double rho_ = 0.0f;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const override;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
