/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
#define LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_

#ifdef USE_CUDA_EXP

#define NUM_QUERY_PER_BLOCK (10)

#include <LightGBM/cuda/cuda_objective_function.hpp>
#include "../rank_objective.hpp"
#include <LightGBM/utils/threading.h>

#include <fstream>

namespace LightGBM {

class CUDALambdarankNDCG : public CUDAObjectiveInterface, public LambdarankNDCG {
 public:
  explicit CUDALambdarankNDCG(const Config& config);

  explicit CUDALambdarankNDCG(const std::vector<std::string>& strs);

  void Init(const Metadata& metadata, data_size_t num_data) override;

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override;

  bool IsCUDAObjective() const override { return true; }

 protected:

  void LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const;

  // CUDA memory, held by this object
  CUDAVector<double> cuda_inverse_max_dcgs_;
  CUDAVector<double> cuda_label_gain_;
  CUDAVector<int> cuda_item_indices_buffer_;

  // CUDA memory, held by other objects
  const label_t* cuda_labels_;
  const data_size_t* cuda_query_boundaries_;

  // Host memory
  int max_items_in_query_aligned_;
};

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
#endif  // LIGHTGBM_NEW_CUDA_RANKING_OBJECTIVE_HPP_
