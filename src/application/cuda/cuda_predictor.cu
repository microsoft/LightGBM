/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"

namespace LightGBM {

__global__ void PredictKernel(const data_size_t num_data,
                              const int num_feature,
                              const int* feature_index,
                              const double* feature_value,
                              const data_size_t* row_ptr,
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
    const data_size_t offset = row_ptr[data_index];
    data_pointer = data + offset;
    for (int i = 0; i < num_feature; ++i) {
      data_pointer[i] = 0.0f;
    }
    const data_size_t num_value = row_ptr[data_index + 1] - offset;
    const int* data_feature_index = feature_index + offset;
    const double* data_feature_value = feature_value + offset;
    for (int value_index = 0; value_index < num_value; ++value_index) {
      data_pointer[data_feature_index[value_index]] = data_feature_value[value_index];
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
      cuda_result_buffer[data_index] += shared_tree_leaf_value[~node];
    }
  }
}

void CUDAPredictor::LaunchPredictKernel(const data_size_t num_data) {
  const int num_blocks = (num_data + CUAA_PREDICTOR_PREDICT_BLOCK_SIZE - 1) / CUAA_PREDICTOR_PREDICT_BLOCK_SIZE;
  PredictKernel<<<num_blocks, CUAA_PREDICTOR_PREDICT_BLOCK_SIZE>>>(
    num_data,
    num_feature_,
    cuda_predict_feature_index_,
    cuda_predict_feature_value_,
    cuda_predict_row_ptr_,
    cuda_tree_num_leaves_,
    cuda_left_child_,
    cuda_right_child_,
    cuda_threshold_,
    cuda_decision_type_,
    cuda_leaf_value_,
    cuda_split_feature_index_,
    num_trees_,
    cuda_data_,
    cuda_result_buffer_);
}

}  // namespace LightGBM
