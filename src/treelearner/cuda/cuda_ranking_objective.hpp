/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define GET_GRADIENTS_BLOCK_SIZE_RANKING_RANKING (128)
#define MAX_NUM_ITEM_IN_QUERY (1024)
#define NUM_QUERY_PER_BLOCK (100)
#define MAX_RANK_LABEL (32)

#include "cuda_objective.hpp"
#include <LightGBM/utils/threading.h>

namespace LightGBM {

class CUDARankingObjective : public CUDAObjective {
 public:
  CUDARankingObjective(
    const data_size_t num_data,
    const label_t* cuda_label,
    const data_size_t* cuda_query_boundaries,
    const data_size_t* cpu_query_boundaries,
    const int num_queries,
    const bool norm,
    const double sigmoid,
    const int truncation_level,
    const label_t* labels,
    const int num_threads);

  void Init() override;

  void CalcInitScore() override;

  const double* cuda_init_score() const override {
    return cuda_init_score_;
  }

  void GetGradients(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) override;

  void TestGlobalArgSort() const override;

 private:

  void LaunchGetGradientsKernel(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians);

  void LaunchCalcInverseMaxDCGKernel();

  void LaunchGlobalArgSort() const;

  // CUDA memory, held by this object
  double* cuda_init_score_;
  double* cuda_lambdas_;
  double* cuda_inverse_max_dcgs_;

  // CUDA memory, held by other objects
  const label_t* cuda_labels_;
  const data_size_t* cuda_query_boundaries_;

  // Host memory
  const int num_queries_;
  const bool norm_;
  const double sigmoid_;
  const int truncation_level_;
  label_t max_label_;
  const int num_threads_;
  int max_items_in_query_aligned_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
