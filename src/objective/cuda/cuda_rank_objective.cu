/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_rank_objective.hpp"

namespace LightGBM {

__device__ void ArgSort(const score_t* scores, uint16_t* indices, const uint16_t num_items) {
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
    const bool ascending = (outer_segment_index % 2 > 0);
    for (uint16_t inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const uint16_t segment_length = 1 << (depth - inner_depth);
      const uint16_t half_segment_length = segment_length >> 1;
      const uint16_t half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligned) {
        if (half_segment_index % 2 == 0) {
          const uint16_t index_to_compare = threadIdx.x + half_segment_length;
          if ((scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) == ascending) {
            const uint16_t index = indices[threadIdx.x];
            indices[threadIdx.x] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ void ArgSort_Partial(const score_t* scores, uint16_t* indices, const uint16_t num_items, const bool outer_decending) {
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
    const bool ascending = outer_decending ? (outer_segment_index % 2 > 0) : (outer_segment_index % 2 == 0);
    for (uint16_t inner_depth = outer_depth; inner_depth < depth; ++inner_depth) {
      const uint16_t segment_length = 1 << (depth - inner_depth);
      const uint16_t half_segment_length = segment_length >> 1;
      const uint16_t half_segment_index = threadIdx.x / half_segment_length;
      if (threadIdx.x < num_items_aligned) {
        if (half_segment_index % 2 == 0) {
          const uint16_t index_to_compare = threadIdx.x + half_segment_length;
          if ((scores[indices[threadIdx.x]] > scores[indices[index_to_compare]]) == ascending) {
            const uint16_t index = indices[threadIdx.x];
            indices[threadIdx.x] = indices[index_to_compare];
            indices[index_to_compare] = index;
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ void ArgSort_2048(const score_t* scores, uint16_t* indices, const uint16_t num_items) {
  const uint16_t depth = 11;
  const uint16_t half_num_items_aligned = 1024;
  ArgSort_Partial(scores, indices, half_num_items_aligned, true);
  ArgSort_Partial(scores + half_num_items_aligned, indices + half_num_items_aligned, half_num_items_aligned, false);
  const unsigned int index_to_compare = threadIdx.x + half_num_items_aligned;
  if (scores[indices[index_to_compare]] > scores[indices[threadIdx.x]]) {
    const uint16_t temp_index = indices[index_to_compare];
    indices[index_to_compare] = indices[threadIdx.x];
    indices[threadIdx.x] = temp_index;
  }
  __syncthreads();
  for (uint16_t inner_depth = 1; inner_depth < depth; ++inner_depth) {
    const uint16_t segment_length = 1 << (depth - inner_depth);
    const uint16_t half_segment_length = segment_length >> 1;
    const uint16_t half_segment_index = threadIdx.x / half_segment_length;
    if (threadIdx.x < half_num_items_aligned) {
      if (half_segment_index % 2 == 0) {
        const uint16_t index_to_compare = threadIdx.x + half_segment_length;
        if (scores[indices[threadIdx.x]] < scores[indices[index_to_compare]]) {
          const uint16_t index = indices[threadIdx.x];
          indices[threadIdx.x] = indices[index_to_compare];
          indices[index_to_compare] = index;
        }
      }
    }
    __syncthreads();
  }
  const score_t* scores_ptr = scores + half_num_items_aligned;
  uint16_t* indices_ptr = indices + half_num_items_aligned;
  for (uint16_t inner_depth = 1; inner_depth < depth; ++inner_depth) {
    const uint16_t segment_length = 1 << (depth - inner_depth);
    const uint16_t half_segment_length = segment_length >> 1;
    const uint16_t half_segment_index = threadIdx.x / half_segment_length;
    if (threadIdx.x < half_num_items_aligned) {
      if (half_segment_index % 2 == 0) {
        const uint16_t index_to_compare = threadIdx.x + half_segment_length;
        if (scores_ptr[indices_ptr[threadIdx.x]] < scores_ptr[indices_ptr[index_to_compare]]) {
          const uint16_t index = indices_ptr[threadIdx.x];
          indices_ptr[threadIdx.x] = indices_ptr[index_to_compare];
          indices_ptr[index_to_compare] = index;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void GetGradientsKernel_Ranking(const double* cuda_scores, const label_t* cuda_labels, const data_size_t num_data,
  const data_size_t num_queries, const data_size_t* cuda_query_boundaries, const double* cuda_inverse_max_dcgs,
  const bool norm, const double sigmoid, const int truncation_level,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  __shared__ score_t shared_scores[MAX_NUM_ITEM_IN_QUERY];
  __shared__ uint16_t shared_indices[MAX_NUM_ITEM_IN_QUERY];
  __shared__ score_t shared_lambdas[MAX_NUM_ITEM_IN_QUERY];
  __shared__ score_t shared_hessians[MAX_NUM_ITEM_IN_QUERY];
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  const double min_score = kMinScore;
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
      shared_indices[threadIdx.x] = static_cast<uint16_t>(threadIdx.x);
    }
    __syncthreads();
    ArgSort(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
    __syncthreads();
    // get best and worst score
    const double best_score = shared_scores[shared_indices[0]];
    data_size_t worst_idx = query_item_count - 1;
    if (worst_idx > 0 && shared_scores[shared_indices[worst_idx]] == min_score) {
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
    const data_size_t num_pairs = num_items_i * num_j_per_i;
    const data_size_t num_pairs_per_thread = (num_pairs + blockDim.x - 1) / blockDim.x;
    const data_size_t thread_start = static_cast<data_size_t>(threadIdx.x) * num_pairs_per_thread;
    const data_size_t thread_end = min(thread_start + num_pairs_per_thread, num_pairs);
    for (data_size_t pair_index = thread_start; pair_index < thread_end; ++pair_index) {
      const data_size_t i = pair_index / num_j_per_i;
      const data_size_t j = pair_index % num_j_per_i + 1;
      if (j > i) {
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
          atomicAdd_block(shared_lambdas + low, -static_cast<score_t>(p_lambda));
          atomicAdd_block(shared_hessians + low, static_cast<score_t>(p_hessian));
          atomicAdd_block(shared_lambdas + high, static_cast<score_t>(p_lambda));
          atomicAdd_block(shared_hessians + high, static_cast<score_t>(p_hessian));
          // lambda is negative, so use minus to accumulate
          atomicAdd_block(&sum_lambdas, -2 * p_lambda);
        }
      }
    }
    __syncthreads();
    if (norm && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x] * norm_factor);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x] * norm_factor);
      }
    } else {
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x]);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x]);
      }
    }
    __syncthreads();
  }
}

__global__ void GetGradientsKernel_Ranking_2048(const double* cuda_scores, const label_t* cuda_labels, const data_size_t num_data,
  const data_size_t num_queries, const data_size_t* cuda_query_boundaries, const double* cuda_inverse_max_dcgs,
  const bool norm, const double sigmoid, const int truncation_level,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  __shared__ score_t shared_scores[MAX_NUM_ITEM_IN_QUERY];
  __shared__ uint16_t shared_indices[MAX_NUM_ITEM_IN_QUERY];
  __shared__ score_t shared_lambdas[MAX_NUM_ITEM_IN_QUERY];
  __shared__ score_t shared_hessians[MAX_NUM_ITEM_IN_QUERY];
  const data_size_t query_index_start = static_cast<data_size_t>(blockIdx.x) * NUM_QUERY_PER_BLOCK;
  const data_size_t query_index_end = min(query_index_start + NUM_QUERY_PER_BLOCK, num_queries);
  const double min_score = kMinScore;
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
      shared_indices[threadIdx.x] = static_cast<uint16_t>(threadIdx.x);
    }
    if (query_item_count > 1024) {
      const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
      if (threadIdx_x_plus_1024 < query_item_count) {
        shared_scores[threadIdx_x_plus_1024] = cuda_scores_pointer[threadIdx_x_plus_1024];
        shared_indices[threadIdx_x_plus_1024] = static_cast<uint16_t>(threadIdx_x_plus_1024);
        shared_lambdas[threadIdx_x_plus_1024] = 0.0f;
        shared_hessians[threadIdx_x_plus_1024] = 0.0f;
      } else {
        shared_scores[threadIdx_x_plus_1024] = min_score;
        shared_indices[threadIdx_x_plus_1024] = static_cast<uint16_t>(threadIdx_x_plus_1024);
      }
    }
    __syncthreads();
    if (query_item_count > 1024) {
      ArgSort_2048(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
    } else {
      ArgSort(shared_scores, shared_indices, static_cast<uint16_t>(query_item_count));
    }
    __syncthreads();
    // get best and worst score
    const double best_score = shared_scores[shared_indices[0]];
    data_size_t worst_idx = query_item_count - 1;
    if (worst_idx > 0 && shared_scores[shared_indices[worst_idx]] == min_score) {
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
    const data_size_t num_pairs = num_items_i * num_j_per_i;
    const data_size_t num_pairs_per_thread = (num_pairs + blockDim.x - 1) / blockDim.x;
    const data_size_t thread_start = static_cast<data_size_t>(threadIdx.x) * num_pairs_per_thread;
    const data_size_t thread_end = min(thread_start + num_pairs_per_thread, num_pairs);
    double thread_sum_lambdas = 0.0f;
    for (data_size_t pair_index = thread_start; pair_index < thread_end; ++pair_index) {
      const data_size_t i = pair_index / num_j_per_i;
      const data_size_t j = pair_index % num_j_per_i + 1;
      if (j > i) {
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
          atomicAdd_block(shared_lambdas + low, -static_cast<score_t>(p_lambda));
          atomicAdd_block(shared_hessians + low, static_cast<score_t>(p_hessian));
          atomicAdd_block(shared_lambdas + high, static_cast<score_t>(p_lambda));
          atomicAdd_block(shared_hessians + high, static_cast<score_t>(p_hessian));
          // lambda is negative, so use minus to accumulate
          thread_sum_lambdas -= 2 * p_lambda;
        }
      }
    }
    atomicAdd_block(&sum_lambdas, thread_sum_lambdas);
    __syncthreads();
    if (norm && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x] * norm_factor);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x] * norm_factor);
      }
      if (query_item_count > 1024) {
        const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
        if (threadIdx_x_plus_1024 < static_cast<unsigned int>(query_item_count)) {
          cuda_out_gradients_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_lambdas[threadIdx_x_plus_1024] * norm_factor);
          cuda_out_hessians_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_hessians[threadIdx_x_plus_1024] * norm_factor);
        }
      }
    } else {
      if (threadIdx.x < static_cast<unsigned int>(query_item_count)) {
        cuda_out_gradients_pointer[threadIdx.x] = static_cast<score_t>(shared_lambdas[threadIdx.x]);
        cuda_out_hessians_pointer[threadIdx.x] = static_cast<score_t>(shared_hessians[threadIdx.x]);
      }
      if (query_item_count > 1024) {
        const unsigned int threadIdx_x_plus_1024 = threadIdx.x + 1024;
        if (threadIdx_x_plus_1024 < static_cast<unsigned int>(query_item_count)) {
          cuda_out_gradients_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_lambdas[threadIdx_x_plus_1024]);
          cuda_out_hessians_pointer[threadIdx_x_plus_1024] = static_cast<score_t>(shared_hessians[threadIdx_x_plus_1024]);
        }
      }
    }
    __syncthreads();
  }
}

void CUDALambdarankNDCG::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_queries_ + NUM_QUERY_PER_BLOCK - 1) / NUM_QUERY_PER_BLOCK;
  if (max_items_in_query_aligned_ <= 1024) {
    GetGradientsKernel_Ranking<<<num_blocks, max_items_in_query_aligned_>>>(score, cuda_labels_, num_data_,
      num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
      norm_, sigmoid_, truncation_level_,
      gradients, hessians);
  } else if (max_items_in_query_aligned_ <= 2048) {
    GetGradientsKernel_Ranking_2048<<<num_blocks, 1024>>>(score, cuda_labels_, num_data_,
      num_queries_, cuda_query_boundaries_, cuda_inverse_max_dcgs_,
      norm_, sigmoid_, truncation_level_,
      gradients, hessians);
  } else {
    Log::Fatal("Too large max_items_in_query_aligned_ = %d", max_items_in_query_aligned_);
  }
}

__device__ void PrefixSumBankConflict(uint16_t* elements, unsigned int n) {
  unsigned int offset = 1;
  unsigned int threadIdx_x = threadIdx.x;
  const uint16_t last_element = elements[n - 1];
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
  if (threadIdx.x == 0) {
    elements[n] = elements[n - 1] + last_element;
  }
  __syncthreads();
}

__global__ void CalcInverseMaxDCGKernel(
  const data_size_t* cuda_query_boundaries,
  const label_t* cuda_labels,
  const int truncation_level,
  const data_size_t num_queries,
  double* cuda_inverse_max_dcgs) {
  __shared__ uint32_t label_sum[MAX_RANK_LABEL];
  __shared__ uint16_t label_pos[MAX_RANK_LABEL + 1];
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
      label_pos[threadIdx.x] = label_sum[threadIdx.x];
    }
    __syncthreads();
    PrefixSumBankConflict(label_pos, MAX_RANK_LABEL);
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
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

}  // namespace LightGBM

#endif  // USE_CUDA
