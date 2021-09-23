/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_CUDA_CUDA_XENTROPY_METRIC_HPP_
#define LIGHTGBM_METRIC_CUDA_CUDA_XENTROPY_METRIC_HPP_

#include "cuda_metric.hpp"
#include "../xentropy_metric.hpp"

#define EVAL_BLOCK_SIZE_XENTROPY_METRIC (1024)

namespace LightGBM {

class CUDACrossEntropyMetric : public CUDAMetricInterface, public CrossEntropyMetric {
 public:
  explicit CUDACrossEntropyMetric(const Config&);

  ~CUDACrossEntropyMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const;

 private:
  void LaunchEvalKernel(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_buffer_;
  double* cuda_sum_loss_;
};

class CUDACrossEntropyLambdaMetric : public CUDAMetricInterface, public CrossEntropyLambdaMetric {
 public:
  explicit CUDACrossEntropyLambdaMetric(const Config&);

  ~CUDACrossEntropyLambdaMetric();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const;

 private:
  void LaunchEvalKernel(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_buffer_;
  double* cuda_sum_loss_;
};

class CUDAKullbackLeiblerDivergence : public CUDAMetricInterface, public KullbackLeiblerDivergence {
 public:
  explicit CUDAKullbackLeiblerDivergence(const Config&);

  ~CUDAKullbackLeiblerDivergence();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  std::vector<double> Eval(const double* score, const ObjectiveFunction* objective) const;

 private:
  void LaunchEvalKernel(const double* score) const;

  const label_t* cuda_label_;
  const label_t* cuda_weights_;
  double* cuda_score_convert_buffer_;
  double* cuda_sum_loss_buffer_;
  double* cuda_sum_loss_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_METRIC_CUDA_CUDA_XENTROPY_METRIC_HPP_
