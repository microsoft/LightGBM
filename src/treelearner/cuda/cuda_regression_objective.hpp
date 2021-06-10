/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE (1024)
#define CALC_INIT_SCORE_BLOCK_SIZE (1024)
#define NUM_DATA_THREAD_ADD_CALC_INIT_SCORE (6)

#include "cuda_objective.hpp"

namespace LightGBM {

class CUDARegressionObjective : public CUDAObjective {
 public:
  CUDARegressionObjective(const data_size_t num_data, const label_t* cuda_label);

  void Init() override;

  void CalcInitScore() override;

  const double* cuda_init_score() const override {
    return cuda_init_score_;
  }

  void GetGradients(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) override;

 private:
  void LaunchCalcInitScoreKernel();

  void LaunchGetGradientsKernel(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians);

  const label_t* cuda_labels_;
  double* cuda_init_score_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_REGRESSION_OBJECTIVE_HPP_
