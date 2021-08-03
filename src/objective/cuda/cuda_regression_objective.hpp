/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_REGRESSION (1024)
#define CALC_INIT_SCORE_BLOCK_SIZE_REGRESSION (1024)
#define NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_REGRESSION (6)

#include "cuda_objective_function.hpp"
#include "../regression_objective.hpp"

namespace LightGBM {

class CUDARegressionL2loss : public CUDAObjectiveInterface, public RegressionL2loss {
 public:
  explicit CUDARegressionL2loss(const Config& config);

  explicit CUDARegressionL2loss(const std::vector<std::string>& strs);

  ~CUDARegressionL2loss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

  double BoostFromScore(int) const override;

 private:
  void LaunchCalcInitScoreKernel() const;

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  const label_t* cuda_labels_;
  // TODO(shiyu1994): add weighted gradients
  const label_t* cuda_weights_;
  double* cuda_boost_from_score_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
