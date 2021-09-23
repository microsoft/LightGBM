/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_rank_metric.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

template <bool USE_QUERY_WEIGHT, size_t SHARED_MEMORY_SIZE, size_t MAX_NUM_EVAL, bool MAX_ITEM_GREATER_THAN_1024>
__global__ void EvalKernel_NDCG_SharedMemory(
  const double* score,
  const label_t* label,
  const label_t* query_weights,
  const data_size_t* query_boundareis,
  const data_size_t num_queries,
  const data_size_t* eval_at,
  const data_size_t num_eval,
  const double* inverse_max_dcgs,
  const double* label_gains,
  const double* discount,
  double* block_ndcg_buffer) {
  __shared__ uint16_t shared_item_indices[SHARED_MEMORY_SIZE];
  __shared__ score_t shared_item_scores[SHARED_MEMORY_SIZE];
  __shared__ double shared_eval_result[MAX_NUM_EVAL];
  __shared__ data_size_t shared_eval_at[MAX_NUM_EVAL];
  __shared__ double shared_shuffle_buffer[32];
  for (data_size_t eval_index = static_cast<data_size_t>(threadIdx.x); eval_index < num_eval; eval_index += static_cast<data_size_t>(blockDim.x)) {
    shared_eval_at[eval_index] = eval_at[eval_index];
    shared_eval_result[eval_index] = 0.0f;
  }
  __syncthreads();
  const data_size_t start_query_index = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK_METRIC;
  const data_size_t end_query_index = min(start_query_index + NUM_QUERY_PER_BLOCK_METRIC, num_queries);
  for (data_size_t query_index = start_query_index; query_index < end_query_index; ++query_index) {
    const double* inverse_max_dcgs_ptr = inverse_max_dcgs + query_index * num_eval;
    if (inverse_max_dcgs_ptr[0] < 0.0f) {
      for (data_size_t eval_index = static_cast<data_size_t>(threadIdx.x); eval_index < num_eval; eval_index += static_cast<data_size_t>(blockDim.x)) {
        shared_eval_result[eval_index] += 1.0f;
      }
    } else {
      const data_size_t item_start_index = query_boundareis[query_index];
      const data_size_t item_end_index = query_boundareis[query_index + 1];
      const data_size_t num_items = item_end_index - item_start_index;
      const double* score_ptr = score + item_start_index;
      const label_t* label_ptr = label + item_start_index;
      for (data_size_t item_index = static_cast<data_size_t>(threadIdx.x); item_index < num_items; item_index += static_cast<data_size_t>(blockDim.x)) {
        shared_item_scores[item_index] = static_cast<score_t>(score_ptr[item_index]);
        shared_item_indices[item_index] = item_index;
      }
      for (data_size_t item_index = num_items + static_cast<data_size_t>(threadIdx.x); item_index < SHARED_MEMORY_SIZE; item_index += static_cast<data_size_t>(blockDim.x)) {
        shared_item_scores[item_index] = kMinScore;
        shared_item_indices[item_index] = item_index;
      }
      __syncthreads();
      if (MAX_ITEM_GREATER_THAN_1024) {
        if (num_items > 1024) {
          for (data_size_t item_index = num_items + static_cast<data_size_t>(threadIdx.x); item_index < SHARED_MEMORY_SIZE; item_index += static_cast<data_size_t>(blockDim.x)) {
            shared_item_scores[item_index] = kMinScore;
          }
          __syncthreads();
          BitonicArgSort_2048(shared_item_scores, shared_item_indices);
        } else {
          BitonicArgSort_1024(shared_item_scores, shared_item_indices, static_cast<uint16_t>(num_items));
        }
      } else {
        BitonicArgSort_1024(shared_item_scores, shared_item_indices, static_cast<uint16_t>(num_items));
      }
      __syncthreads();
      double thread_eval = 0.0f;
      data_size_t item_index = static_cast<data_size_t>(threadIdx.x);
      for (data_size_t eval_index = 0; eval_index < num_eval; ++eval_index) {
        data_size_t cur_eval_pos = min(num_items, shared_eval_at[eval_index]);
        for (; item_index < cur_eval_pos; item_index += static_cast<data_size_t>(blockDim.x)) {
          const int data_label = static_cast<int>(label_ptr[shared_item_indices[item_index]]);
          thread_eval += label_gains[data_label] * discount[item_index];
        }
        __syncthreads();
        double block_eval = ShuffleReduceSum<double>(thread_eval, shared_shuffle_buffer, blockDim.x);
        if (USE_QUERY_WEIGHT) {
          block_eval *= static_cast<double>(query_weights[query_index]);
        }
        if (threadIdx.x == 0) {
          shared_eval_result[eval_index] += block_eval * inverse_max_dcgs_ptr[eval_index];
        }
      }
      __syncthreads();
    }
  }
  for (data_size_t eval_index = static_cast<data_size_t>(threadIdx.x); eval_index < num_eval; eval_index += static_cast<data_size_t>(blockDim.x)) {
    block_ndcg_buffer[eval_index * gridDim.x + blockIdx.x] = shared_eval_result[eval_index];
  }
}

template <bool USE_QUERY_WEIGHT, size_t MAX_NUM_EVAL>
__global__ void EvalKernel_NDCG_GlobalMemory(
  const double* score,
  const label_t* label,
  const label_t* query_weights,
  const data_size_t* query_boundareis,
  const data_size_t num_queries,
  const data_size_t* eval_at,
  const data_size_t num_eval,
  const double* inverse_max_dcgs,
  const double* label_gains,
  const double* discount,
  double* block_ndcg_buffer,
  const data_size_t* cuda_item_indices_buffer) {
  __shared__ double shared_eval_result[MAX_NUM_EVAL];
  __shared__ data_size_t shared_eval_at[MAX_NUM_EVAL];
  __shared__ double shared_shuffle_buffer[32];
  for (data_size_t eval_index = 0; eval_index < num_eval; eval_index += static_cast<data_size_t>(blockDim.x)) {
    shared_eval_at[eval_index] = eval_at[eval_index];
    shared_eval_result[eval_index] = 0.0f;
  }
  __syncthreads();
  const data_size_t start_query_index = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK_METRIC;
  const data_size_t end_query_index = min(start_query_index + NUM_QUERY_PER_BLOCK_METRIC, num_queries);
  for (data_size_t query_index = start_query_index; query_index < end_query_index; ++query_index) {
    const data_size_t item_start_index = query_boundareis[query_index];
    const data_size_t item_end_index = query_boundareis[query_index + 1];
    const data_size_t num_items = item_end_index - item_start_index;
    const label_t* label_ptr = label + item_start_index;
    const double* inverse_max_dcgs_ptr = inverse_max_dcgs + query_index * num_eval;
    const data_size_t* sorted_item_indices_ptr = cuda_item_indices_buffer + item_start_index;
    double thread_eval = 0.0f;
    data_size_t item_index = static_cast<data_size_t>(threadIdx.x);
    for (data_size_t eval_index = 0; eval_index < num_eval; ++eval_index) {
      data_size_t cur_eval_pos = min(num_items, shared_eval_at[eval_index]);
      for (; item_index < cur_eval_pos; item_index += static_cast<data_size_t>(blockDim.x)) {
        const uint16_t sorted_item_index = sorted_item_indices_ptr[item_index];
        if (static_cast<data_size_t>(sorted_item_index) >= num_items) {
          printf("error sorted_item_index = %d, num_items = %d\n", sorted_item_index, num_items);
        }
        const int data_label = static_cast<int>(label_ptr[sorted_item_indices_ptr[item_index]]);
        thread_eval += label_gains[data_label] * discount[item_index];
      }
      __syncthreads();
      double block_eval = ShuffleReduceSum<double>(thread_eval, shared_shuffle_buffer, blockDim.x);
      if (USE_QUERY_WEIGHT) {
        block_eval *= static_cast<double>(query_weights[query_index]);
      }
      if (threadIdx.x == 0) {
        shared_eval_result[eval_index] += block_eval * inverse_max_dcgs_ptr[eval_index];
      }
    }
    __syncthreads();
  }
  for (data_size_t eval_index = static_cast<data_size_t>(threadIdx.x); eval_index < num_eval; eval_index += static_cast<data_size_t>(blockDim.x)) {
    block_ndcg_buffer[eval_index * gridDim.x + blockIdx.x] = shared_eval_result[eval_index];
  }
}

__global__ void ReduceNDCGFromBlocks(
  const double* block_ndcg_buffer,
  const data_size_t num_eval,
  const int num_blocks,
  double* ndcg_result,
  const double sum_query_weights) {
  __shared__ double shared_mem_buffer[32];
  const data_size_t eval_index = static_cast<data_size_t>(blockIdx.x);
  const double* block_ndcg_buffer_ptr = block_ndcg_buffer + eval_index * num_blocks;
  double thread_sum = 0.0f;
  for (data_size_t block_index = static_cast<data_size_t>(threadIdx.x); block_index < num_blocks; block_index += static_cast<data_size_t>(blockDim.x)) {
    thread_sum += block_ndcg_buffer_ptr[block_index];
  }
  __syncthreads();
  const double block_sum = ShuffleReduceSum<double>(thread_sum, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    ndcg_result[eval_index] = block_sum / sum_query_weights;
  }
}

#define EvalKernel_NDCG_ARGS \
  score, \
  cuda_label_, \
  cuda_query_weights_, \
  cuda_query_boundaries_, \
  num_queries_, \
  cuda_eval_at_, \
  num_eval_, \
  cuda_inverse_max_dcgs_, \
  cuda_label_gain_, \
  cuda_discount_, \
  cuda_block_dcg_buffer_

void CUDANDCGMetric::LaunchEvalKernel(const double* score) const {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK_METRIC - 1) / NUM_QUERY_PER_BLOCK_METRIC;
  if (cuda_query_weights_ == nullptr) {
    if (max_items_in_query_aligned_ <= 1024) {
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_SharedMemory<false, 1024, 32, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_SharedMemory<false, 1024, 256, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_SharedMemory<false, 1024, 1024, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    } else if (max_items_in_query_aligned_ <= 2048) {
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_SharedMemory<false, 2048, 32, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_SharedMemory<false, 2048, 256, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_SharedMemory<false, 2048, 1024, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    } else {
      BitonicArgSortItemsGlobal(score, num_queries_, cuda_query_boundaries_, cuda_item_indices_buffer_);
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_GlobalMemory<false, 32><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_GlobalMemory<false, 256><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_GlobalMemory<false, 1024><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    }
  } else {
    if (max_items_in_query_aligned_ <= 1024) {
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_SharedMemory<true, 1024, 32, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_SharedMemory<true, 1024, 256, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_SharedMemory<true, 1024, 1024, false><<<num_blocks, max_items_in_query_aligned_>>>(EvalKernel_NDCG_ARGS);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    } else if (max_items_in_query_aligned_ <= 2048) {
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_SharedMemory<true, 2048, 32, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_SharedMemory<true, 2048, 256, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_SharedMemory<true, 2048, 1024, true><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    } else {
      BitonicArgSortItemsGlobal(score, num_queries_, cuda_query_boundaries_, cuda_item_indices_buffer_);
      if (num_eval_ <= 32) {
        EvalKernel_NDCG_GlobalMemory<true, 32><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else if (num_eval_ <= 256) {
        EvalKernel_NDCG_GlobalMemory<true, 256><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else if (num_eval_ <= 1024) {
        EvalKernel_NDCG_GlobalMemory<true, 1024><<<num_blocks, EVAL_BLOCK_SIZE_RANK_METRIC>>>(EvalKernel_NDCG_ARGS, cuda_item_indices_buffer_);
      } else {
        Log::Fatal("Number of eval_at %d exceeds the maximum %d for NDCG metric in CUDA version.", num_eval_, 1024);
      }
    }
  }
  ReduceNDCGFromBlocks<<<num_eval_, EVAL_BLOCK_SIZE_RANK_METRIC>>>(
    cuda_block_dcg_buffer_,
    num_eval_,
    num_blocks,
    cuda_ndcg_result_,
    sum_query_weights_);
}

}  // namespace LightGBM
