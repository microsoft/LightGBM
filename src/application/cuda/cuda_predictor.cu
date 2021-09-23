/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"

namespace LightGBM {

template <bool IS_CSR, bool PREDICT_LEAF_INDEX>
__global__ void PredictKernel(const data_size_t num_data,
                              const int num_feature,
                              const int* num_leaves,
                              const int** left_child,
                              const int** right_child,
                              const double** threshold,
                              const int8_t** decision_type,
                              const double** leaf_value,
                              const int** split_feature_index,
                              const int num_trees,
                              double* data,
                              double* cuda_result_buffer) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const unsigned int thread_index = threadIdx.x;
  double* data_pointer = nullptr;
  if (data_index < num_data) {
    data_pointer = data + data_index * num_feature;
    if (!PREDICT_LEAF_INDEX) {
      cuda_result_buffer[data_index] = 0.0f;
    }
  }
  __shared__ double shared_tree_threshold[CUDA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int shared_tree_left_child[CUDA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int shared_tree_right_child[CUDA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int8_t shared_tree_decision_type[CUDA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ double shared_tree_leaf_value[CUDA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int shared_tree_split_feature_index[CUDA_PREDICTOR_MAX_TREE_SIZE];
  for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
    const int tree_num_leaves = num_leaves[tree_index];
    const int* tree_left_child = left_child[tree_index];
    const int* tree_right_child = right_child[tree_index];
    const double* tree_threshold = threshold[tree_index];
    const double* tree_leaf_value = leaf_value[tree_index];
    const int8_t* tree_decision_type = decision_type[tree_index];
    const int* tree_split_feature_index = split_feature_index[tree_index];
    for (int leaf_index = static_cast<int>(thread_index); leaf_index < tree_num_leaves; leaf_index += static_cast<int>(blockDim.x)) {
      shared_tree_threshold[leaf_index] = tree_threshold[leaf_index];
      shared_tree_left_child[leaf_index] = tree_left_child[leaf_index];
      shared_tree_right_child[leaf_index] = tree_right_child[leaf_index];
      shared_tree_leaf_value[leaf_index] = tree_leaf_value[leaf_index];
      shared_tree_decision_type[leaf_index] = tree_decision_type[leaf_index];
      shared_tree_split_feature_index[leaf_index] = tree_split_feature_index[leaf_index];
    }
    __syncthreads();
    if (data_index < num_data) {
      int node = 0;
      while (node >= 0) {
        const double node_threshold = shared_tree_threshold[node];
        const int node_split_feature_index = shared_tree_split_feature_index[node];
        const int8_t node_decision_type = shared_tree_decision_type[node];
        double value = data_pointer[node_split_feature_index];
        uint8_t missing_type = GetMissingTypeCUDA(node_decision_type);
        if (isnan(value) && missing_type != MissingType::NaN) {
          value = 0.0f;
        }
        if ((missing_type == MissingType::Zero && IsZeroCUDA(value)) ||
            (missing_type == MissingType::NaN && isnan(value))) {
          if (GetDecisionTypeCUDA(node_decision_type, kDefaultLeftMask)) {
            node = shared_tree_left_child[node];
          } else {
            node = shared_tree_right_child[node];
          }
        } else {
          if (value <= node_threshold) {
            node = shared_tree_left_child[node];
          } else {
            node = shared_tree_right_child[node];
          }
        }
      }
      if (PREDICT_LEAF_INDEX) {
        cuda_result_buffer[data_index * num_trees + tree_index] = ~node;
      } else {
        cuda_result_buffer[data_index] += shared_tree_leaf_value[~node];
      }
    }
    __syncthreads();
  }
}

#define PREDICT_KERNEL_ARGS \
  num_data, \
  num_feature_, \
  cuda_tree_num_leaves_, \
  cuda_left_child_, \
  cuda_right_child_, \
  cuda_threshold_, \
  cuda_decision_type_, \
  cuda_leaf_value_, \
  cuda_split_feature_index_, \
  num_iteration_, \
  cuda_data_, \
  cuda_result_buffer_

void CUDAPredictor::LaunchPredictKernelAsync(const data_size_t num_data, const bool is_csr) {
  const int num_blocks = (num_data + CUAA_PREDICTOR_PREDICT_BLOCK_SIZE - 1) / CUAA_PREDICTOR_PREDICT_BLOCK_SIZE;
  if (is_csr) {
    if (predict_leaf_index_) {
      PredictKernel<true, true><<<num_blocks, CUAA_PREDICTOR_PREDICT_BLOCK_SIZE, 0, cuda_stream_>>>(PREDICT_KERNEL_ARGS);
    } else {
      PredictKernel<true, false><<<num_blocks, CUAA_PREDICTOR_PREDICT_BLOCK_SIZE, 0, cuda_stream_>>>(PREDICT_KERNEL_ARGS);
    }
  } else {
    if (predict_leaf_index_) {
      PredictKernel<false, true><<<num_blocks, CUAA_PREDICTOR_PREDICT_BLOCK_SIZE, 0, cuda_stream_>>>(PREDICT_KERNEL_ARGS);
    } else {
      PredictKernel<false, false><<<num_blocks, CUAA_PREDICTOR_PREDICT_BLOCK_SIZE, 0, cuda_stream_>>>(PREDICT_KERNEL_ARGS);
    }
  }
  if (!is_raw_score_ && !predict_leaf_index_) {
    cuda_convert_output_function_(num_data, cuda_result_buffer_, cuda_result_buffer_);
  }
}

#undef PREDICT_KERNEL_ARGS

}  // namespace LightGBM
