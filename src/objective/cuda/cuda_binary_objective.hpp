/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_BINARY (1024)
#define CALC_INIT_SCORE_BLOCK_SIZE_BINARY (1024)
#define NUM_DATA_THREAD_ADD_CALC_INIT_SCORE_BINARY (6)

#include "cuda_objective_function.hpp"
#include "../binary_objective.hpp"

namespace LightGBM {

class CUDABinaryLogloss : public CUDAObjectiveInterface, public BinaryLogloss {
 public:
  explicit CUDABinaryLogloss(const Config& config,
                             std::function<bool(label_t)> is_pos = nullptr);

  explicit CUDABinaryLogloss(const std::vector<std::string>& strs);

  ~CUDABinaryLogloss();

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* scores, score_t* gradients, score_t* hessians) const override;

  double BoostFromScore(int) const override;

 private:
  void LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const;

  void LaunchBoostFromScoreKernel() const;

  // CUDA memory, held by other objects
  const label_t* cuda_label_;
  const label_t* cuda_weights_;

  // CUDA memory, held by this object
  mutable double* cuda_boost_from_score_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_OBJECTIVE_CUDA_CUDA_BINARY_OBJECTIVE_HPP_
