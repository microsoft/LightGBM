/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_XENTROPY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_XENTROPY_OBJECTIVE_HPP_

#include "../xentropy_objective.hpp"

#define GET_GRADIENTS_BLOCK_SIZE_XENTROPY (1024)

namespace LightGBM {

class CUDACrossEntropy: public CrossEntropy {
 public:
  explicit CUDACrossEntropy(const Config& config);

  explicit CUDACrossEntropy(const std::vector<std::string>& strs);

  ~CUDACrossEntropy();

  virtual void Init(const Metadata& metadata, data_size_t num_data) override;

  double BoostFromScore(int) const override;

  virtual void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  double LaunchCalcInitScoreKernel() const;

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
  double* cuda_reduce_sum_buffer_;
};

class CUDACrossEntropyLambda: public CrossEntropyLambda {
 public:
  explicit CUDACrossEntropyLambda(const Config& config);

  explicit CUDACrossEntropyLambda(const std::vector<std::string>& strs);

  ~CUDACrossEntropyLambda();

  virtual void Init(const Metadata& metadata, data_size_t num_data) override;

  double BoostFromScore(int) const override;

  virtual void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

 private:
  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  double LaunchCalcInitScoreKernel() const;

  const label_t* cuda_labels_;
  const label_t* cuda_weights_;
  double* cuda_reduce_sum_buffer_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_XENTROPY_OBJECTIVE_HPP_
