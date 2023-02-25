/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
#define LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/meta.h>

#include <string>
#include <vector>

namespace LightGBM {

template <typename HOST_OBJECTIVE>
class CUDAObjectiveInterface: public HOST_OBJECTIVE {
 public:
  explicit CUDAObjectiveInterface(const Config& config): HOST_OBJECTIVE(config) {}

  explicit CUDAObjectiveInterface(const std::vector<std::string>& strs): HOST_OBJECTIVE(strs) {}

  void Init(const Metadata& metadata, data_size_t num_data) {
    HOST_OBJECTIVE::Init(metadata, num_data);
    cuda_labels_ = metadata.cuda_metadata()->cuda_label();
    cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  }

  virtual const double* ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
    return LaunchConvertOutputCUDAKernel(num_data, input, output);
  }

  double BoostFromScore(int class_id) const override {
    return LaunchCalcInitScoreKernel(class_id);
  }

  bool IsCUDAObjective() const override { return true; }

  void GetGradients(const double* scores, score_t* gradients, score_t* hessians) const override {
    LaunchGetGradientsKernel(scores, gradients, hessians);
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  void RenewTreeOutputCUDA(const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
    const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const override {
    global_timer.Start("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
    LaunchRenewTreeOutputCUDAKernel(score, data_indices_in_leaf, num_data_in_leaf, data_start_in_leaf, num_leaves, leaf_value);
    SynchronizeCUDADevice(__FILE__, __LINE__);
    global_timer.Stop("CUDAObjectiveInterface::LaunchRenewTreeOutputCUDAKernel");
  }

 protected:
  virtual void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const = 0;

  virtual double LaunchCalcInitScoreKernel(const int class_id) const {
    return HOST_OBJECTIVE::BoostFromScore(class_id);
  }

  virtual const double* LaunchConvertOutputCUDAKernel(const data_size_t /*num_data*/, const double* input, double* /*output*/) const { return input; }

  virtual void LaunchRenewTreeOutputCUDAKernel(
    const double* /*score*/, const data_size_t* /*data_indices_in_leaf*/, const data_size_t* /*num_data_in_leaf*/,
    const data_size_t* /*data_start_in_leaf*/, const int /*num_leaves*/, double* /*leaf_value*/) const {}

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
};

}  // namespace LightGBM

#endif  // USE_CUDA

#endif  // LIGHTGBM_CUDA_CUDA_OBJECTIVE_FUNCTION_HPP_
