/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_rank_objective.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <random>
#include <algorithm>

namespace LightGBM {

template <bool MAX_ITEM_GREATER_THAN_1024>
__global__ void GetGradientsKernel_LambdarankNDCG(const double* cuda_scores, const label_t* cuda_labels, const data_size_t num_data,
  const data_size_t num_queries, const data_size_t* cuda_query_boundaries, const double* cuda_inverse_max_dcgs,
  const bool norm, const double sigmoid, const int truncation_level,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  __shared__ score_t shared_scores[MAX_ITEM_GREATER_THAN_1024 ? 2048 : 1024];
  __shared__ uint16_t shared_indices[MAX_ITEM_GREATER_THAN_1024 ? 2048 : 1024];
  __shared__ score_t shared_lambdas[MAX_ITEM_GREATER_THAN_1024 ? 2048 : 1024];
  __shared__ score_t shared_hessians[MAX_ITEM_GREATER_THAN_1024 ? 2048 : 1024];
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  for (data_size_t query_index = query_index_start; query_index < query_index_end; ++query_index) {
    const double inverse_max_dcg = cuda_inverse_max_dcgs[query_index];
    const data_size_t query_start = cuda_query_boundaries[query_index];
    const data_size_t query_end = cuda_query_boundaries[query_index + 1];
    const data_size_t query_item_count = query_end - query_start;
    const double* cuda_scores_pointer = cuda_scores + query_start;
    score_t* cuda_out_gradients_pointer = cuda_out_gradients + query_start;
    score_t* cuda_out_hessians_pointer = cuda_out_hessians + query_start;
    const label_t* cuda_label_pointer = cuda_labels + query_start;
    if (threadIdx.x < query_item_count) {
      shared_scores[threadIdx.x] = cuda_scores_pointer[threadIdx.x];
      shared_indices[threadIdx.x] = static_cast<uint16_t>(threadIdx.x);
      shared_lambdas[threadIdx.x] = 0.0f;
      shared_hessians[threadIdx.x] = 0.0f;
    } else {
      shared_scores[threadIdx.x] = kMinScore;
      shared_indices[threadIdx.x] = static_cast<uint16_t>(threadIdx.x);
    }
    if (MAX_ITEM_GREATER_THAN_1024) {
      if (query_item_count > 1024) {
        const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
        if (threadIdx_x_plus_1024 < query_item_count) {
          shared_scores[threadIdx_x_plus_1024] = cuda_scores_pointer[threadIdx_x_plus_1024];
          shared_indices[threadIdx_x_plus_1024] = static_cast<uint16_t>(threadIdx_x_plus_1024);
          shared_lambdas[threadIdx_x_plus_1024] = 0.0f;
          shared_hessians[threadIdx_x_plus_1024] = 0.0f;
        } else {
          shared_scores[threadIdx_x_plus_1024] = kMinScore;
          shared_indices[threadIdx_x_plus_1024] = static_cast<uint16_t>(threadIdx_x_plus_1024);
        }
      }
    }
    __syncthreads();
    if (MAX_ITEM_GREATER_THAN_1024) {
      if (query_item_count > 1024) {
        BitonicArgSort_2048<score_t, uint16_t, false>(shared_scores, shared_indices);
      } else {
        BitonicArgSort_1024<score_t, uint16_t, false>(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
      }
    } else {
      BitonicArgSort_1024<score_t, uint16_t, false>(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
    }
    __syncthreads();
    // get best and worst score
    const double best_score = shared_scores[shared_indices[0]];
    data_size_t worst_idx = query_item_count - 1;
    if (worst_idx > 0 && shared_scores[shared_indices[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = shared_scores[shared_indices[worst_idx]];
    __shared__ double sum_lambdas;
    if (threadIdx.x == 0) {
      sum_lambdas = 0.0f;
    }
    __syncthreads();
    // start accumulate lambdas by pairs that contain at least one document above truncation level
    const data_size_t num_items_i = min(query_item_count - 1, truncation_level);
    const data_size_t num_j_per_i = query_item_count - 1;
    const data_size_t s = num_j_per_i - num_items_i + 1;
    const data_size_t num_pairs = (num_j_per_i + s) * num_items_i / 2;
    double thread_sum_lambdas = 0.0f;
    for (data_size_t pair_index = static_cast<data_size_t>(threadIdx.x); pair_index < num_pairs; pair_index += static_cast<data_size_t>(blockDim.x)) {
      const double square = 2 * static_cast<double>(pair_index) + s * s - s;
      const double sqrt_result = floor(sqrt(square));
      const data_size_t row_index = static_cast<data_size_t>(floor(sqrt(square - sqrt_result)) + 1 - s);
      const data_size_t i = num_items_i - 1 - row_index;
      const data_size_t j = num_j_per_i - (pair_index - (2 * s + row_index - 1) * row_index / 2);
      if (cuda_label_pointer[shared_indices[i]] != cuda_label_pointer[shared_indices[j]] && shared_scores[shared_indices[j]] != kMinScore) {
        data_size_t high_rank, low_rank;
        if (cuda_label_pointer[shared_indices[i]] > cuda_label_pointer[shared_indices[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = shared_indices[high_rank];
        const int high_label = static_cast<int>(cuda_label_pointer[high]);
        const double high_score = shared_scores[high];
        const double high_label_gain = static_cast<double>((1 << high_label) - 1);
        const double high_discount = log2(2.0f + high_rank);
        const data_size_t low = shared_indices[low_rank];
        const int low_label = static_cast<int>(cuda_label_pointer[low]);
        const double low_score = shared_scores[low];
        const double low_label_gain = static_cast<double>((1 << low_label) - 1);
        const double low_discount = log2(2.0f + low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = 1.0f / (1.0f + exp(sigmoid * delta_score));
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid * delta_pair_NDCG;
        p_hessian *= sigmoid * sigmoid * delta_pair_NDCG;
        atomicAdd_block(shared_lambdas + low, -static_cast<score_t>(p_lambda));
        atomicAdd_block(shared_hessians + low, static_cast<score_t>(p_hessian));
        atomicAdd_block(shared_lambdas + high, static_cast<score_t>(p_lambda));
        atomicAdd_block(shared_hessians + high, static_cast<score_t>(p_hessian));
        // lambda is negative, so use minus to accumulate
        thread_sum_lambdas -= 2 * p_lambda;
      }
    }
    atomicAdd_block(&sum_lambdas, thread_sum_lambdas);
    __syncthreads();
    if (norm && sum_lambdas > 0) {
      const double norm_factor = log2(1 + sum_lambdas) / sum_lambdas;
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x] * norm_factor);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x] * norm_factor);
      }
      if (MAX_ITEM_GREATER_THAN_1024) {
        if (query_item_count > 1024) {
          const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
          if (threadIdx_x_plus_1024 < static_cast<unsigned int>(query_item_count)) {
            cuda_out_gradients_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_lambdas[threadIdx_x_plus_1024] * norm_factor);
            cuda_out_hessians_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_hessians[threadIdx_x_plus_1024] * norm_factor);
          }
        }
      }
    } else {
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x]);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x]);
      }
      if (MAX_ITEM_GREATER_THAN_1024) {
        if (query_item_count > 1024) {
          const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
          if (threadIdx_x_plus_1024 < static_cast<unsigned int>(query_item_count)) {
            cuda_out_gradients_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_lambdas[threadIdx_x_plus_1024]);
            cuda_out_hessians_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_hessians[threadIdx_x_plus_1024]);
          }
        }
      }
    }
    __syncthreads();
  }
}

__global__ void GetGradientsKernel_LambdarankNDCG_Sorted(
  const double* cuda_scores, const int* cuda_item_indices_buffer, const label_t* cuda_labels, const data_size_t num_data,
  const data_size_t num_queries, const data_size_t* cuda_query_boundaries, const double* cuda_inverse_max_dcgs,
  const bool norm, const double sigmoid, const int truncation_level,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  for (data_size_t query_index = query_index_start; query_index < query_index_end; ++query_index) {
    const double inverse_max_dcg = cuda_inverse_max_dcgs[query_index];
    const data_size_t query_start = cuda_query_boundaries[query_index];
    const data_size_t query_end = cuda_query_boundaries[query_index + 1];
    const data_size_t query_item_count = query_end - query_start;
    const double* cuda_scores_pointer = cuda_scores + query_start;
    const int* cuda_item_indices_buffer_pointer = cuda_item_indices_buffer + query_start;
    score_t* cuda_out_gradients_pointer = cuda_out_gradients + query_start;
    score_t* cuda_out_hessians_pointer = cuda_out_hessians + query_start;
    const label_t* cuda_label_pointer = cuda_labels + query_start;
    // get best and worst score
    const double best_score = cuda_scores_pointer[cuda_item_indices_buffer_pointer[0]];
    data_size_t worst_idx = query_item_count - 1;
    if (worst_idx > 0 && cuda_scores_pointer[cuda_item_indices_buffer_pointer[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = cuda_scores_pointer[cuda_item_indices_buffer_pointer[worst_idx]];
    __shared__ double sum_lambdas;
    if (threadIdx.x == 0) {
      sum_lambdas = 0.0f;
    }
    for (int item_index = static_cast<int>(threadIdx.x); item_index < query_item_count; item_index += static_cast<int>(blockDim.x)) {
      cuda_out_gradients_pointer[item_index] = 0.0f;
      cuda_out_hessians_pointer[item_index] = 0.0f;
    }
    __syncthreads();
    // start accumulate lambdas by pairs that contain at least one document above truncation level
    const data_size_t num_items_i = min(query_item_count - 1, truncation_level);
    const data_size_t num_j_per_i = query_item_count - 1;
    const data_size_t s = num_j_per_i - num_items_i + 1;
    const data_size_t num_pairs = (num_j_per_i + s) * num_items_i / 2;
    double thread_sum_lambdas = 0.0f;
    for (data_size_t pair_index = static_cast<data_size_t>(threadIdx.x); pair_index < num_pairs; pair_index += static_cast<data_size_t>(blockDim.x)) {
      const double square = 2 * static_cast<double>(pair_index) + s * s - s;
      const double sqrt_result = floor(sqrt(square));
      const data_size_t row_index = static_cast<data_size_t>(floor(sqrt(square - sqrt_result)) + 1 - s);
      const data_size_t i = num_items_i - 1 - row_index;
      const data_size_t j = num_j_per_i - (pair_index - (2 * s + row_index - 1) * row_index / 2);
      if (j > i) {
        // skip pairs with the same labels
        if (cuda_label_pointer[cuda_item_indices_buffer_pointer[i]] != cuda_label_pointer[cuda_item_indices_buffer_pointer[j]] && cuda_scores_pointer[cuda_item_indices_buffer_pointer[j]] != kMinScore) {
          data_size_t high_rank, low_rank;
          if (cuda_label_pointer[cuda_item_indices_buffer_pointer[i]] > cuda_label_pointer[cuda_item_indices_buffer_pointer[j]]) {
            high_rank = i;
            low_rank = j;
          } else {
            high_rank = j;
            low_rank = i;
          }
          const data_size_t high = cuda_item_indices_buffer_pointer[high_rank];
          const int high_label = static_cast<int>(cuda_label_pointer[high]);
          const double high_score = cuda_scores_pointer[high];
          const double high_label_gain = static_cast<double>((1 << high_label) - 1);
          const double high_discount = log2(2.0f + high_rank);
          const data_size_t low = cuda_item_indices_buffer_pointer[low_rank];
          const int low_label = static_cast<int>(cuda_label_pointer[low]);
          const double low_score = cuda_scores_pointer[low];
          const double low_label_gain = static_cast<double>((1 << low_label) - 1);
          const double low_discount = log2(2.0f + low_rank);

          const double delta_score = high_score - low_score;

          // get dcg gap
          const double dcg_gap = high_label_gain - low_label_gain;
          // get discount of this pair
          const double paired_discount = fabs(high_discount - low_discount);
          // get delta NDCG
          double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
          // regular the delta_pair_NDCG by score distance
          if (norm && best_score != worst_score) {
            delta_pair_NDCG /= (0.01f + fabs(delta_score));
          }
          // calculate lambda for this pair
          double p_lambda = 1.0f / (1.0f + exp(sigmoid * delta_score));
          double p_hessian = p_lambda * (1.0f - p_lambda);
          // update
          p_lambda *= -sigmoid * delta_pair_NDCG;
          p_hessian *= sigmoid * sigmoid * delta_pair_NDCG;
          atomicAdd_block(cuda_out_gradients_pointer + low, -static_cast<score_t>(p_lambda));
          atomicAdd_block(cuda_out_hessians_pointer + low, static_cast<score_t>(p_hessian));
          atomicAdd_block(cuda_out_gradients_pointer + high, static_cast<score_t>(p_lambda));
          atomicAdd_block(cuda_out_hessians_pointer + high, static_cast<score_t>(p_hessian));
          // lambda is negative, so use minus to accumulate
          thread_sum_lambdas -= 2 * p_lambda;
        }
      }
    }
    atomicAdd_block(&sum_lambdas, thread_sum_lambdas);
    __syncthreads();
    if (norm && sum_lambdas > 0) {
      const double norm_factor = log2(1 + sum_lambdas) / sum_lambdas;
      for (int item_index = static_cast<int>(threadIdx.x); item_index < query_item_count; item_index += static_cast<int>(blockDim.x)) {
        cuda_out_gradients_pointer[item_index] *= norm_factor;
        cuda_out_hessians_pointer[item_index] *= norm_factor;
      }
    }
    __syncthreads();
  }
}

void CUDALambdarankNDCG::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK - 1) / NUM_QUERY_PER_BLOCK;
  if (max_items_in_query_aligned_ <= 1024) {
    GetGradientsKernel_LambdarankNDCG<false><<<num_blocks, max_items_in_query_aligned_>>>(score, cuda_labels_, num_data_,
      num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
      norm_, sigmoid_, truncation_level_,
      gradients, hessians);
  } else if (max_items_in_query_aligned_ <= 2048) {
    GetGradientsKernel_LambdarankNDCG<true><<<num_blocks, 1024>>>(score, cuda_labels_, num_data_,
      num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
      norm_, sigmoid_, truncation_level_,
      gradients, hessians);
  } else {
    BitonicArgSortItemsGlobal(score, num_queries_, cuda_query_boundaries_, cuda_item_indices_buffer_);
    GetGradientsKernel_LambdarankNDCG_Sorted<<<num_blocks, 1024>>>(score, cuda_item_indices_buffer_, cuda_labels_, num_data_,
      num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
      norm_, sigmoid_, truncation_level_,
      gradients, hessians);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

// TODO(shiyu1994): calc inverse max dcg kernel should consider items >= 1024
__global__ void CalcInverseMaxDCGKernel(
  const data_size_t* cuda_query_boundaries,
  const label_t* cuda_labels,
  const int truncation_level,
  const data_size_t num_queries,
  double* cuda_inverse_max_dcgs) {
  __shared__ uint32_t label_sum[MAX_RANK_LABEL];
  __shared__ uint16_t label_pos[MAX_RANK_LABEL + 1];
  __shared__ uint16_t shared_buffer[32];
  uint16_t pos = 0;
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  for (data_size_t query_index = query_index_start; query_index < query_index_end; ++query_index) {
    const data_size_t query_start = cuda_query_boundaries[query_index];
    const data_size_t query_end = cuda_query_boundaries[query_index + 1];
    const data_size_t query_count = query_end - query_start;
    if (threadIdx.x < MAX_RANK_LABEL) {
      label_sum[threadIdx.x] = 0;
    }
    __syncthreads();
    const label_t* label_pointer = cuda_labels + query_start;
    if (threadIdx.x < static_cast<unsigned int>(query_count)) {
      atomicAdd_system(label_sum + (MAX_RANK_LABEL - 1 - static_cast<size_t>(label_pointer[threadIdx.x])), 1);
    }
    __syncthreads();
    if (threadIdx.x < MAX_RANK_LABEL) {
      pos = label_sum[threadIdx.x];
    }
    __syncthreads();
    pos = ShufflePrefixSumExclusive<uint16_t>(pos, shared_buffer);
    label_pos[threadIdx.x] = pos;
    __syncthreads();
    __shared__ double gain;
    if (threadIdx.x == 0) {
      gain = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x < MAX_RANK_LABEL && label_sum[threadIdx.x] > 0) {
      const uint16_t start_pos = label_pos[threadIdx.x];
      const uint16_t end_pos = min(label_pos[threadIdx.x + 1], truncation_level);
      double label_gain = 0.0f;
      for (uint16_t k = start_pos; k < end_pos; ++k) {
        label_gain += ((1 << (MAX_RANK_LABEL - 1 - threadIdx.x)) - 1) / log(2.0f + k);
      }
      atomicAdd_system(&gain, label_gain);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (gain > 0.0f) {
        cuda_inverse_max_dcgs[query_index] = 1.0f / gain;
      } else {
        cuda_inverse_max_dcgs[query_index] = 0.0f;
      }
    }
    __syncthreads();
  }
}

void CUDALambdarankNDCG::LaunchCalcInverseMaxDCGKernel() {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK - 1) / NUM_QUERY_PER_BLOCK;
  CalcInverseMaxDCGKernel<<<num_blocks, MAX_RANK_LABEL>>>(
    cuda_query_boundaries_,
    cuda_labels_,
    truncation_level_,
    num_queries_,
    cuda_inverse_max_dcgs_);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
