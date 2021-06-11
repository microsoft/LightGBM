/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_ranking_objective.hpp"

namespace LightGBM {

__device__ void ArgSort(const double* scores, uint16_t* indices, const uint16_t num_items) {
  uint16_t num_items_aligned = 1;
  uint16_t num_items_ref = num_items - 1;
  uint16_t depth = 1;
  while (num_items_ref > 0) {
    num_items_aligned <<= 1;
    num_items_ref >>= 1;
    ++depth;
  }
  for (uint16_t outer_depth = depth - 1; outer_depth >= 1; --outer_depth) {
    const uint16_t outer_segment_length = 1 << (depth - outer_depth);
    const uint16_t outer_segment_index = threadIdx.x / outer_segment_length;
    const bool ascending = (outer_segment_index % 2 == 0);
    for (uint16_t inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const uint16_t segment_length = 1 << (depth - inner_depth);
      const uint16_t half_segment_length = segment_length >> 1;
      const uint16_t half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligned) {
        if (half_segment_index % 2 == 0) {
          const uint16_t index_to_compare = threadIdx.x + half_segment_length;
          if (ascending) {
            if (scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) {
              const uint16_t index = indices[threadIdx.x];
              indices[threadIdx.x] = indices[index_to_compare];
              indices[index_to_compare] = index;
            }
          } else {
            if (scores[indices[threadIdx.x]] < scores[indices[index_to_compare]]) {
              const uint16_t index = indices[threadIdx.x];
              indices[threadIdx.x] = indices[index_to_compare];
              indices[index_to_compare] = index;
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ void GetGradientsKernel_Ranking(const double* cuda_scores, const label_t* cuda_labels, const data_size_t num_data,
  const data_size_t num_queries, const data_size_t* cuda_query_boundaries, const double* cuda_inverse_max_dcgs,
  const bool norm, const double sigmoid, const int truncation_level,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  __shared__ double shared_scores[MAX_NUM_ITEM_IN_QUERY];
  __shared__ uint16_t shared_indices[MAX_NUM_ITEM_IN_QUERY];
  __shared__ double shared_lambdas[MAX_NUM_ITEM_IN_QUERY];
  __shared__ double shared_hessians[MAX_NUM_ITEM_IN_QUERY];
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  const double min_score = -100000000000.0f;
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
      shared_scores[threadIdx.x] = min_score;
      shared_indices[threadIdx.x] = 0;
    }
    __syncthreads();
    ArgSort(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
    // get best and worst score
    const double best_score = shared_scores[shared_indices[0]];
    data_size_t worst_idx = query_item_count - 1;
    if (worst_idx > 0 && shared_scores[shared_indices[worst_idx]] == min_score) {
      worst_idx -= 1;
    }
    const double worst_score = shared_scores[shared_indices[worst_idx]];
    double sum_lambdas = 0.0;
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < query_item_count - 1 && i < truncation_level; ++i) {
      if (shared_scores[shared_indices[i]] == min_score) { continue; }
      if (threadIdx.x > static_cast<unsigned int>(i) && threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        const data_size_t j = static_cast<data_size_t>(threadIdx.x);
        // skip pairs with the same labels
        if (cuda_label_pointer[shared_indices[i]] != cuda_label_pointer[shared_indices[j]] && shared_scores[shared_indices[j]] != min_score) {
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
          atomicAdd_system(shared_lambdas + low, -static_cast<score_t>(p_lambda));
          atomicAdd_system(shared_hessians + low, static_cast<score_t>(p_hessian));
          atomicAdd_system(shared_lambdas + high, static_cast<score_t>(p_lambda));
          atomicAdd_system(shared_hessians + high, static_cast<score_t>(p_hessian));
          // lambda is negative, so use minus to accumulate
          atomicAdd_system(&sum_lambdas, -2 * p_lambda);
        }
      }
    }
    __syncthreads();
    if (norm && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < query_item_count; ++i) {
        cuda_out_gradients_pointer[i] = static_cast<score_t>(shared_lambdas[i] * norm_factor);
        cuda_out_hessians_pointer[i] = static_cast<score_t>(shared_hessians[i] * norm_factor);
      }
    }
  }
}

void CUDARankingObjective::LaunchGetGradientsKernel(const double* cuda_scores, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK - 1) / NUM_QUERY_PER_BLOCK;
  GetGradientsKernel_Ranking<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_RANKING_RANKING>>>(cuda_scores, cuda_labels_, num_data_,
    num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
    norm_, sigmoid_, truncation_level_,
    cuda_out_gradients, cuda_out_hessians);
}

__device__ void PrefixSumBankConflict(uint16_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  __syncthreads();
  for (int d = (n >> 1); d > 0; d >>= 1) {
    if (threadIdx_x < d) {
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      elements[dst_pos] += elements[src_pos];
    }
    offset <<= 1;
    __syncthreads();
  }
  if (threadIdx_x == 0) {
    elements[n - 1] = 0; 
  }
  __syncthreads();
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    if (threadIdx_x < d) {
      const unsigned int dst_pos = offset * (2 * threadIdx_x + 2) - 1;
      const unsigned int src_pos = offset * (2 * threadIdx_x + 1) - 1;
      const uint32_t src_val = elements[src_pos];
      elements[src_pos] = elements[dst_pos];
      elements[dst_pos] += src_val;
    }
    __syncthreads();
  }
}

__global__ void CalcInverseMaxDCGKernel(
  const data_size_t* cuda_query_boundaries,
  const label_t* cuda_labels,
  const int truncation_level,
  const data_size_t num_queries,
  double* cuda_inverse_max_dcgs) {
  __shared__ uint32_t label_sum[MAX_RANK_LABEL];
  __shared__ uint16_t label_pos[MAX_RANK_LABEL];
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
      atomicAdd_system(label_sum + static_cast<size_t>(label_pointer[threadIdx.x]), 1);
    }
    __syncthreads();
    if (threadIdx.x < MAX_RANK_LABEL) {
      if (label_sum[threadIdx.x] > 0) {
        label_pos[threadIdx.x] = 1;
      } else {
        label_pos[threadIdx.x] = 0;
      }
    }
    __syncthreads();
    PrefixSumBankConflict(label_pos, MAX_RANK_LABEL);
    double gain = 0.0f;
    if (threadIdx.x < MAX_RANK_LABEL && label_sum[threadIdx.x] > 0) {
      const double label_gain = (1 << threadIdx.x - 1) / log2(2.0f + label_pos[threadIdx.x]);
      atomicAdd_system(&gain, label_gain);
    }
    __syncthreads();
    if (gain > 0.0f) {
      cuda_inverse_max_dcgs[query_index] = 1.0f / gain;
    } else {
      cuda_inverse_max_dcgs[query_index] = 0.0f;
    }
  }
}

void CUDARankingObjective::LaunchCalcInverseMaxDCGKernel() {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK - 1) / NUM_QUERY_PER_BLOCK;
  CalcInverseMaxDCGKernel<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_RANKING_RANKING>>>(
    cuda_query_boundaries_,
    cuda_labels_,
    truncation_level_,
    num_queries_,
    cuda_inverse_max_dcgs_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
