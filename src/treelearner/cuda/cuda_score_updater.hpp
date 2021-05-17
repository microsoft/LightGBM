/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_NEW_CUDA_SCORE_UPDATER_HPP_
#define LIGHTGBM_NEW_CUDA_SCORE_UPDATER_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include "new_cuda_utils.hpp"

#include <vector>

#define SET_INIT_SCORE_BLOCK_SIZE (1024)

namespace LightGBM {

class CUDAScoreUpdater {
 public:
  CUDAScoreUpdater(const data_size_t num_data);

  void Init();

  void SetInitScore(const double* cuda_init_score);

  void AddScore(const double* cuda_score_to_add);

  const double* cuda_scores() const { return cuda_scores_; }

  double* cuda_score_ref() { return cuda_scores_; }

 private:
  void LaunchSetInitScoreKernel(const double* cuda_init_score);

  void LaunchAddScoreKernel(const double* cuda_scores_to_add);

  const data_size_t num_data_;
  double* cuda_scores_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_SCORE_UPDATER_HPP_
