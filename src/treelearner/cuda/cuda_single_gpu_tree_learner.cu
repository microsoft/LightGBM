/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"

namespace LightGBM {

__global__ void ReduceLeafStatKernel_SharedMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  extern __shared__ double shared_mem[];
  double* shared_grad_sum = shared_mem;
  double* shared_hess_sum = shared_mem + num_leaves;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    shared_grad_sum[leaf_index] = 0.0f;
    shared_hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(shared_grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(shared_hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, shared_grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, shared_hess_sum[leaf_index]);
  }
}

__global__ void ReduceLeafStatKernel_GlobalMemory(
  const score_t* gradients,
  const score_t* hessians,
  const int num_leaves,
  const data_size_t num_data,
  const int* data_index_to_leaf_index,
  double* leaf_grad_stat_buffer,
  double* leaf_hess_stat_buffer) {
  const size_t offset = static_cast<size_t>(num_leaves) * (blockIdx.x + 1);
  double* grad_sum = leaf_grad_stat_buffer + offset;
  double* hess_sum = leaf_hess_stat_buffer + offset;
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    grad_sum[leaf_index] = 0.0f;
    hess_sum[leaf_index] = 0.0f;
  }
  __syncthreads();
  if (data_index < num_data) {
    const int leaf_index = data_index_to_leaf_index[data_index];
    atomicAdd_block(grad_sum + leaf_index, gradients[data_index]);
    atomicAdd_block(hess_sum + leaf_index, hessians[data_index]);
  }
  __syncthreads();
  for (int leaf_index = static_cast<int>(threadIdx.x); leaf_index < num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
    atomicAdd_system(leaf_grad_stat_buffer + leaf_index, grad_sum[leaf_index]);
    atomicAdd_system(leaf_hess_stat_buffer + leaf_index, hess_sum[leaf_index]);
  }
}

__global__ void CalcRefitLeafOutputKernel(
  const int num_leaves,
  const double* leaf_grad_stat_buffer,
  const double* leaf_hess_stat_buffer,
  const double lambda_l1,
  const bool use_l1,
  const double lambda_l2,
  const double shrinkage_rate,
  const double refit_decay_rate,
  double* leaf_value) {
  const int leaf_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  if (leaf_index < num_leaves) {
    const double sum_gradients = leaf_grad_stat_buffer[leaf_index];
    const double sum_hessians = leaf_hess_stat_buffer[leaf_index];
    const double old_leaf_value = leaf_value[leaf_index];
    double new_leaf_value = CUDABestSplitFinder::CalculateSplittedLeafOutput(sum_gradients, sum_hessians, lambda_l1, use_l1, lambda_l2);
    if (isnan(new_leaf_value)) {
      new_leaf_value = 0.0f;
    } else {
      new_leaf_value *= shrinkage_rate;
    }
    leaf_value[leaf_index] = refit_decay_rate * old_leaf_value + (1.0f - refit_decay_rate) * new_leaf_value;
  }
}

void CUDASingleGPUTreeLearner::LaunchReduceLeafStatKernel(
  const score_t* gradients, const score_t* hessians, const int num_leaves,
  const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const {
  int num_block = (num_data + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  if (num_leaves <= 2048) {
    ReduceLeafStatKernel_SharedMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE, 2 * num_leaves * sizeof(double)>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_, cuda_leaf_hessian_stat_buffer_);
  } else {
    ReduceLeafStatKernel_GlobalMemory<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(
      gradients, hessians, num_leaves, num_data, cuda_data_partition_->cuda_data_index_to_leaf_index(),
      cuda_leaf_gradient_stat_buffer_, cuda_leaf_hessian_stat_buffer_);
  }
  const bool use_l1 = config_->lambda_l1 > 0.0f;
  num_block = (num_leaves + CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE - 1) / CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE;
  CalcRefitLeafOutputKernel<<<num_block, CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE>>>(
    num_leaves, cuda_leaf_gradient_stat_buffer_, cuda_leaf_hessian_stat_buffer_,
    config_->lambda_l1, use_l1, config_->lambda_l2, shrinkage_rate, config_->refit_decay_rate, cuda_leaf_value);
}

}  // namespace LightGBM

#endif  // USE_CUDA
