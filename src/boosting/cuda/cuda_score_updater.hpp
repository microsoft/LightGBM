/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_utils.h>

#include "../score_updater.hpp"

namespace LightGBM {

class CUDAScoreUpdater: public ScoreUpdater {
 public:
  CUDAScoreUpdater(const Dataset* data, int num_tree_per_iteration);

  ~CUDAScoreUpdater();

  inline void AddScore(double val, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, int cur_tree_id) override;

  inline void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) override;

  inline void AddScore(const Tree* tree, const data_size_t* data_indices,
                       data_size_t data_cnt, int cur_tree_id) override;

  inline void MultiplyScore(double val, int cur_tree_id) override;

  inline const double* score() const override { return cuda_score_; } 

  /*! \brief Disable copy */
  CUDAScoreUpdater& operator=(const CUDAScoreUpdater&) = delete;

  CUDAScoreUpdater(const CUDAScoreUpdater&) = delete;

 private:
  void InitCUDA(const size_t total_size);

  void LaunchAddScoreConstantKernel(const double val, const size_t offset);

  void LaunchMultiplyScoreConstantKernel(const double val, const size_t offset);

  double* cuda_score_;

  const int num_threads_per_block_;
};

}  // namespace LightGBM
