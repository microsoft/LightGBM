/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#ifdef USE_CUDA_EXP

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

__device__ void SetDecisionTypeCUDA(int8_t* decision_type, bool input, int8_t mask) {
  if (input) {
    (*decision_type) |= mask;
  } else {
    (*decision_type) &= (127 - mask);
  }
}

__device__ void SetMissingTypeCUDA(int8_t* decision_type, int8_t input) {
  (*decision_type) &= 3;
  (*decision_type) |= (input << 2);
}

__device__ bool GetDecisionTypeCUDA(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

__device__ int8_t GetMissingTypeCUDA(int8_t decision_type) {
  return (decision_type >> 2) & 3;
}

__device__ bool IsZeroCUDA(double fval) {
  return (fval >= -kZeroThreshold && fval <= kZeroThreshold);
}

__global__ void SplitKernel(  // split information
                            const int leaf_index,
                            const int real_feature_index,
                            const double real_threshold,
                            const MissingType missing_type,
                            const CUDASplitInfo* cuda_split_info,
                            // tree structure
                            const int num_leaves,
                            int* leaf_parent,
                            int* leaf_depth,
                            int* left_child,
                            int* right_child,
                            int* split_feature_inner,
                            int* split_feature,
                            float* split_gain,
                            double* internal_weight,
                            double* internal_value,
                            data_size_t* internal_count,
                            double* leaf_weight,
                            double* leaf_value,
                            data_size_t* leaf_count,
                            int8_t* decision_type,
                            uint32_t* threshold_in_bin,
                            double* threshold) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = leaf_weight[leaf_index];
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], false, kCategoricalMask);
    SetDecisionTypeCUDA(&decision_type[new_node_index], cuda_split_info->default_left, kDefaultLeftMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = cuda_split_info->threshold;
  } else if (thread_index == 14) {
    threshold[new_node_index] = real_threshold;
  }
}

void CUDATree::LaunchSplitKernel(const int leaf_index,
                                 const int real_feature_index,
                                 const double real_threshold,
                                 const MissingType missing_type,
                                 const CUDASplitInfo* cuda_split_info) {
  SplitKernel<<<3, 5, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    real_threshold,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_,
    cuda_leaf_depth_,
    cuda_left_child_,
    cuda_right_child_,
    cuda_split_feature_inner_,
    cuda_split_feature_,
    cuda_split_gain_,
    cuda_internal_weight_,
    cuda_internal_value_,
    cuda_internal_count_,
    cuda_leaf_weight_,
    cuda_leaf_value_,
    cuda_leaf_count_,
    cuda_decision_type_,
    cuda_threshold_in_bin_,
    cuda_threshold_);
}

__global__ void SplitCategoricalKernel(  // split information
  const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  // tree structure
  const int num_leaves,
  int* leaf_parent,
  int* leaf_depth,
  int* left_child,
  int* right_child,
  int* split_feature_inner,
  int* split_feature,
  float* split_gain,
  double* internal_weight,
  double* internal_value,
  data_size_t* internal_count,
  double* leaf_weight,
  double* leaf_value,
  data_size_t* leaf_count,
  int8_t* decision_type,
  uint32_t* threshold_in_bin,
  double* threshold,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len,
  int num_cat,
  int* cuda_cat_boundaries,
  int* cuda_cat_boundaries_inner) {
  const int new_node_index = num_leaves - 1;
  const int thread_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int parent_index = leaf_parent[leaf_index];
  if (thread_index == 0) {
    if (parent_index >= 0) {
      // if cur node is left child
      if (left_child[parent_index] == ~leaf_index) {
        left_child[parent_index] = new_node_index;
      } else {
        right_child[parent_index] = new_node_index;
      }
    }
    left_child[new_node_index] = ~leaf_index;
    right_child[new_node_index] = ~num_leaves;
    leaf_parent[leaf_index] = new_node_index;
    leaf_parent[num_leaves] = new_node_index;
  } else if (thread_index == 1) {
    // add new node
    split_feature_inner[new_node_index] = cuda_split_info->inner_feature_index;
  } else if (thread_index == 2) {
    split_feature[new_node_index] = real_feature_index;
  } else if (thread_index == 3) {
    split_gain[new_node_index] = static_cast<float>(cuda_split_info->gain);
  } else if (thread_index == 4) {
    // save current leaf value to internal node before change
    internal_weight[new_node_index] = leaf_weight[leaf_index];
    leaf_weight[leaf_index] = cuda_split_info->left_sum_hessians;
  } else if (thread_index == 5) {
    internal_value[new_node_index] = leaf_value[leaf_index];
    leaf_value[leaf_index] = isnan(cuda_split_info->left_value) ? 0.0f : cuda_split_info->left_value;
  } else if (thread_index == 6) {
    internal_count[new_node_index] = cuda_split_info->left_count + cuda_split_info->right_count;
  } else if (thread_index == 7) {
    leaf_count[leaf_index] = cuda_split_info->left_count;
  } else if (thread_index == 8) {
    leaf_value[num_leaves] = isnan(cuda_split_info->right_value) ? 0.0f : cuda_split_info->right_value;
  } else if (thread_index == 9) {
    leaf_weight[num_leaves] = cuda_split_info->right_sum_hessians;
  } else if (thread_index == 10) {
    leaf_count[num_leaves] = cuda_split_info->right_count;
  } else if (thread_index == 11) {
    // update leaf depth
    leaf_depth[num_leaves] = leaf_depth[leaf_index] + 1;
    leaf_depth[leaf_index]++;
  } else if (thread_index == 12) {
    decision_type[new_node_index] = 0;
    SetDecisionTypeCUDA(&decision_type[new_node_index], true, kCategoricalMask);
    SetMissingTypeCUDA(&decision_type[new_node_index], static_cast<int8_t>(missing_type));
  } else if (thread_index == 13) {
    threshold_in_bin[new_node_index] = num_cat;
  } else if (thread_index == 14) {
    threshold[new_node_index] = num_cat;
  } else if (thread_index == 15) {
    if (num_cat == 0) {
      cuda_cat_boundaries[num_cat] = 0;
    }
    cuda_cat_boundaries[num_cat + 1] = cuda_cat_boundaries[num_cat] + cuda_bitset_len;
  } else if (thread_index == 16) {
    if (num_cat == 0) {
      cuda_cat_boundaries_inner[num_cat] = 0;
    }
    cuda_cat_boundaries_inner[num_cat + 1] = cuda_cat_boundaries_inner[num_cat] + cuda_bitset_inner_len;
  }
}

void CUDATree::LaunchSplitCategoricalKernel(const int leaf_index,
  const int real_feature_index,
  const MissingType missing_type,
  const CUDASplitInfo* cuda_split_info,
  size_t cuda_bitset_len,
  size_t cuda_bitset_inner_len) {
  SplitCategoricalKernel<<<3, 6, 0, cuda_stream_>>>(
    // split information
    leaf_index,
    real_feature_index,
    missing_type,
    cuda_split_info,
    // tree structure
    num_leaves_,
    cuda_leaf_parent_,
    cuda_leaf_depth_,
    cuda_left_child_,
    cuda_right_child_,
    cuda_split_feature_inner_,
    cuda_split_feature_,
    cuda_split_gain_,
    cuda_internal_weight_,
    cuda_internal_value_,
    cuda_internal_count_,
    cuda_leaf_weight_,
    cuda_leaf_value_,
    cuda_leaf_count_,
    cuda_decision_type_,
    cuda_threshold_in_bin_,
    cuda_threshold_,
    cuda_bitset_len,
    cuda_bitset_inner_len,
    num_cat_,
    cuda_cat_boundaries_.RawData(),
    cuda_cat_boundaries_inner_.RawData());
}

__global__ void ShrinkageKernel(const double rate, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] *= rate;
  }
}

void CUDATree::LaunchShrinkageKernel(const double rate) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  ShrinkageKernel<<<num_blocks, num_threads_per_block>>>(rate, cuda_leaf_value_, num_leaves_);
}

__global__ void AddBiasKernel(const double val, double* cuda_leaf_value, const int num_leaves) {
  const int leaf_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (leaf_index < num_leaves) {
    cuda_leaf_value[leaf_index] += val;
  }
}

void CUDATree::LaunchAddBiasKernel(const double val) {
  const int num_threads_per_block = 1024;
  const int num_blocks = (num_leaves_ + num_threads_per_block - 1) / num_threads_per_block;
  AddBiasKernel<<<num_blocks, num_threads_per_block>>>(val, cuda_leaf_value_, num_leaves_);
}

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
