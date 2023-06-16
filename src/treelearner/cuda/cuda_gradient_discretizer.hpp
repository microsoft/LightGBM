/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_
#define LIGHTGBM_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_

#ifdef USE_CUDA

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/utils/threading.h>

#include <random>

#include "cuda_leaf_splits.hpp"
#include "../gradient_discretizer.hpp"

namespace LightGBM {

#define CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE (1024)

class CUDAGradientDiscretizer: public GradientDiscretizer {
 public:
  CUDAGradientDiscretizer(int grad_discretize_bins, int num_trees, int random_seed, bool can_lock, bool is_constant_hessian, bool stochastic_roudning):
    GradientDiscretizer(grad_discretize_bins, num_trees, random_seed, can_lock, is_constant_hessian, stochastic_roudning) {
    nccl_comm_ = nullptr;
  }

  void SetNCCL(ncclComm_t* nccl_comm) {
    nccl_comm_ = nccl_comm;
  }

  void DiscretizeGradients(
    const data_size_t num_data,
    const score_t* input_gradients,
    const score_t* input_hessians) override;

  const int32_t* discretized_gradients_and_hessians() const override { return discretized_gradients_and_hessians_.RawData(); }

  const score_t* grad_scale() const override { return grad_max_block_buffer_.RawData(); }

  const score_t* hess_scale() const override { return hess_max_block_buffer_.RawData(); }

  void Init(const data_size_t num_data) override {
    discretized_gradients_and_hessians_.Resize(num_data);
    num_reduce_blocks_ = (num_data + CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE - 1) / CUDA_GRADIENT_DISCRETIZER_BLOCK_SIZE;
    grad_min_block_buffer_.Resize(num_reduce_blocks_);
    grad_max_block_buffer_.Resize(num_reduce_blocks_);
    hess_min_block_buffer_.Resize(num_reduce_blocks_);
    hess_max_block_buffer_.Resize(num_reduce_blocks_);
    random_values_use_start_.Resize(num_trees_);
    gradient_random_values_.Resize(num_data);
    hessian_random_values_.Resize(num_data);
    grad_hess_min_max_.Resize(4);

    std::vector<score_t> gradient_random_values(num_data, 0.0f);
    std::vector<score_t> hessian_random_values(num_data, 0.0f);
    std::vector<int> random_values_use_start(num_trees_, 0);

    const int num_threads = OMP_NUM_THREADS();

    std::mt19937 random_values_use_start_eng = std::mt19937(random_seed_);
    std::uniform_int_distribution<data_size_t> random_values_use_start_dist = std::uniform_int_distribution<data_size_t>(0, num_data);
    for (int tree_index = 0; tree_index < num_trees_; ++tree_index) {
      random_values_use_start[tree_index] = random_values_use_start_dist(random_values_use_start_eng);
    }

    int num_blocks = 0;
    data_size_t block_size = 0;
    Threading::BlockInfo<data_size_t>(num_data, 512, &num_blocks, &block_size);
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int thread_id = 0; thread_id < num_blocks; ++thread_id) {
      const data_size_t start = thread_id * block_size;
      const data_size_t end = std::min(start + block_size, num_data);
      std::mt19937 gradient_random_values_eng(random_seed_ + thread_id);
      std::uniform_real_distribution<double> gradient_random_values_dist(0.0f, 1.0f);
      std::mt19937 hessian_random_values_eng(random_seed_ + thread_id + num_threads);
      std::uniform_real_distribution<double> hessian_random_values_dist(0.0f, 1.0f);
      for (data_size_t i = start; i < end; ++i) {
        gradient_random_values[i] = gradient_random_values_dist(gradient_random_values_eng);
        hessian_random_values[i] = hessian_random_values_dist(hessian_random_values_eng);
      }
    }

    CopyFromHostToCUDADevice<score_t>(gradient_random_values_.RawData(), gradient_random_values.data(), gradient_random_values.size(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<score_t>(hessian_random_values_.RawData(), hessian_random_values.data(), hessian_random_values.size(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(random_values_use_start_.RawData(), random_values_use_start.data(), random_values_use_start.size(), __FILE__, __LINE__);
    iter_ = 0;
  }

 protected:
  mutable CUDAVector<int32_t> discretized_gradients_and_hessians_;
  mutable CUDAVector<score_t> grad_min_block_buffer_;
  mutable CUDAVector<score_t> grad_max_block_buffer_;
  mutable CUDAVector<score_t> hess_min_block_buffer_;
  mutable CUDAVector<score_t> hess_max_block_buffer_;
  mutable CUDAVector<score_t> grad_hess_min_max_;
  CUDAVector<int> random_values_use_start_;
  CUDAVector<score_t> gradient_random_values_;
  CUDAVector<score_t> hessian_random_values_;
  int num_reduce_blocks_;
  ncclComm_t* nccl_comm_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_TREELEARNER_CUDA_CUDA_GRADIENT_DISCRETIZER_HPP_
