/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_

#ifdef USE_CUDA

#define MAX_NUM_ITEM_IN_QUERY (2048)
#define NUM_QUERY_PER_BLOCK (10)
#define MAX_RANK_LABEL (32)

#include "cuda_objective_function.hpp"
#include "../rank_objective.hpp"
#include <LightGBM/utils/threading.h>

namespace LightGBM {

class CUDALambdarankNDCG : public CUDAObjectiveInterface, public LambdarankNDCG {
 public:
  explicit CUDALambdarankNDCG(const Config& config);

  explicit CUDALambdarankNDCG(const std::vector<std::string>& strs);

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

 private:

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  void LaunchCalcInverseMaxDCGKernel();

  // CUDA memory, held by this object
  double* cuda_lambdas_;
  double* cuda_inverse_max_dcgs_;

  // CUDA memory, held by other objects
  const label_t* cuda_labels_;
  const data_size_t* cuda_query_boundaries_;

  // Host memory
  label_t max_label_;
  int max_items_in_query_aligned_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
