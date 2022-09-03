/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include <algorithm>

#include "cuda_binary_objective.hpp"

namespace LightGBM {

template <bool IS_OVA, bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_1_BinaryLogloss(const label_t* cuda_labels, const data_size_t num_data, double* out_cuda_sum_labels,
                                                     double* out_cuda_sum_weights, const label_t* cuda_weights, const int ova_class_id) {
  __shared__ double shared_buffer[32];
  const uint32_t mask = 0xffffffff;
  const uint32_t warpLane = threadIdx.x % warpSize;
  const uint32_t warpID = threadIdx.x / warpSize;
  const uint32_t num_warp = blockDim.x / warpSize;
  const data_size_t index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double label_value = 0.0;
  double weight_value = 0.0;
  if (index < num_data) {
    if (USE_WEIGHT) {
      const label_t cuda_label = cuda_labels[index];
      const double sample_weight = cuda_weights[index];
      const label_t label = IS_OVA ? (static_cast<int>(cuda_label) == ova_class_id ? 1 : 0) : (cuda_label > 0 ? 1 : 0);
      label_value = label * sample_weight;
      weight_value = sample_weight;
    } else {
      const label_t cuda_label = cuda_labels[index];
      label_value = IS_OVA ? (static_cast<int>(cuda_label) == ova_class_id ? 1 : 0) : (cuda_label > 0 ? 1 : 0);
    }
  }
  for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
    label_value += __shfl_down_sync(mask, label_value, offset);
  }
  if (warpLane == 0) {
    shared_buffer[warpID] = label_value;
  }
  __syncthreads();
  if (warpID == 0) {
    label_value = (warpLane < num_warp ? shared_buffer[warpLane] : 0);
    for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
      label_value += __shfl_down_sync(mask, label_value, offset);
    }
  }
  __syncthreads();
  if (USE_WEIGHT) {
    for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
      weight_value += __shfl_down_sync(mask, weight_value, offset);
    }
    if (warpLane == 0) {
      shared_buffer[warpID] = weight_value;
    }
    __syncthreads();
    if (warpID == 0) {
      weight_value = (warpLane < num_warp ? shared_buffer[warpLane] : 0);
      for (uint32_t offset = warpSize / 2; offset >= 1; offset >>= 1) {
        weight_value += __shfl_down_sync(mask, weight_value, offset);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd_system(out_cuda_sum_labels, label_value);
    if (USE_WEIGHT) {
      atomicAdd_system(out_cuda_sum_weights, weight_value);
    }
  }
}

template <bool USE_WEIGHT>
__global__ void BoostFromScoreKernel_2_BinaryLogloss(double* out_cuda_sum_labels, double* out_cuda_sum_weights,
                                                     const data_size_t num_data, const double sigmoid) {
  const double suml = *out_cuda_sum_labels;
  const double sumw = USE_WEIGHT ? *out_cuda_sum_weights : static_cast<double>(num_data);
  double pavg = suml / sumw;
  pavg = min(pavg, 1.0 - kEpsilon);
  pavg = max(pavg, kEpsilon);
  const double init_score = log(pavg / (1.0f - pavg)) / sigmoid;
  *out_cuda_sum_weights = pavg;
  *out_cuda_sum_labels = init_score;
}

void CUDABinaryLogloss::LaunchBoostFromScoreKernel() const {
  const int num_blocks = (num_data_ + CALC_INIT_SCORE_BLOCK_SIZE_BINARY - 1) / CALC_INIT_SCORE_BLOCK_SIZE_BINARY;
  if (ova_class_id_ == -1) {
    if (cuda_weights_ == nullptr) {
      BoostFromScoreKernel_1_BinaryLogloss<false, false><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
        (cuda_label_, num_data_, cuda_boost_from_score_, cuda_sum_weights_, cuda_weights_, ova_class_id_);
    } else {
      BoostFromScoreKernel_1_BinaryLogloss<false, true><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
        (cuda_label_, num_data_, cuda_boost_from_score_, cuda_sum_weights_, cuda_weights_, ova_class_id_);
    }
  } else {
    if (cuda_weights_ == nullptr) {
      BoostFromScoreKernel_1_BinaryLogloss<true, false><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
        (cuda_label_, num_data_, cuda_boost_from_score_, cuda_sum_weights_, cuda_weights_, ova_class_id_);
    } else {
      BoostFromScoreKernel_1_BinaryLogloss<true, true><<<num_blocks, CALC_INIT_SCORE_BLOCK_SIZE_BINARY>>>
        (cuda_label_, num_data_, cuda_boost_from_score_, cuda_sum_weights_, cuda_weights_, ova_class_id_);
    }
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (cuda_weights_ == nullptr) {
    BoostFromScoreKernel_2_BinaryLogloss<false><<<1, 1>>>(cuda_boost_from_score_, cuda_sum_weights_, num_data_, sigmoid_);
  } else {
    BoostFromScoreKernel_2_BinaryLogloss<true><<<1, 1>>>(cuda_boost_from_score_, cuda_sum_weights_, num_data_, sigmoid_);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

template <bool USE_LABEL_WEIGHT, bool USE_WEIGHT, bool IS_OVA>
__global__ void GetGradientsKernel_BinaryLogloss(const double* cuda_scores, const label_t* cuda_labels,
  const double* cuda_label_weights, const label_t* cuda_weights, const int ova_class_id,
  const double sigmoid, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const label_t cuda_label = static_cast<int>(cuda_labels[data_index]);
    const int label = IS_OVA ? (cuda_label == ova_class_id ? 1 : -1) : (cuda_label > 0 ? 1 : -1);
    const double response = -label * sigmoid / (1.0f + exp(label * sigmoid * cuda_scores[data_index]));
    const double abs_response = fabs(response);
    if (!USE_WEIGHT) {
      if (USE_LABEL_WEIGHT) {
        const double label_weight = cuda_label_weights[label];
        cuda_out_gradients[data_index] = static_cast<score_t>(response * label_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * label_weight);
      } else {
        cuda_out_gradients[data_index] = static_cast<score_t>(response);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response));
      }
    } else {
      const double sample_weight = cuda_weights[data_index];
      if (USE_LABEL_WEIGHT) {
        const double label_weight = cuda_label_weights[label];
        cuda_out_gradients[data_index] = static_cast<score_t>(response * label_weight * sample_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * label_weight * sample_weight);
      } else {
        cuda_out_gradients[data_index] = static_cast<score_t>(response * sample_weight);
        cuda_out_hessians[data_index] = static_cast<score_t>(abs_response * (sigmoid - abs_response) * sample_weight);
      }
    }
  }
}

#define GetGradientsKernel_BinaryLogloss_ARGS \
  scores, \
  cuda_label_, \
  cuda_label_weights_, \
  cuda_weights_, \
  ova_class_id_, \
  sigmoid_, \
  num_data_, \
  gradients, \
  hessians

void CUDABinaryLogloss::LaunchGetGradientsKernel(const double* scores, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  if (ova_class_id_ == -1) {
    if (cuda_label_weights_ == nullptr) {
      if (cuda_weights_ == nullptr) {
        GetGradientsKernel_BinaryLogloss<false, false, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      } else {
        GetGradientsKernel_BinaryLogloss<false, true, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      }
    } else {
      if (cuda_weights_ == nullptr) {
        GetGradientsKernel_BinaryLogloss<true, false, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      } else {
        GetGradientsKernel_BinaryLogloss<true, true, false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      }
    }
  } else {
    if (cuda_label_weights_ == nullptr) {
      if (cuda_weights_ == nullptr) {
        GetGradientsKernel_BinaryLogloss<false, false, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      } else {
        GetGradientsKernel_BinaryLogloss<false, true, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      }
    } else {
      if (cuda_weights_ == nullptr) {
        GetGradientsKernel_BinaryLogloss<true, false, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      } else {
        GetGradientsKernel_BinaryLogloss<true, true, true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(GetGradientsKernel_BinaryLogloss_ARGS);
      }
    }
  }
}

#undef GetGradientsKernel_BinaryLogloss_ARGS

__global__ void ConvertOutputCUDAKernel_BinaryLogloss(const double sigmoid, const data_size_t num_data, const double* input, double* output) {
  const data_size_t data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    output[data_index] = 1.0f / (1.0f + exp(-sigmoid * input[data_index]));
  }
}

void CUDABinaryLogloss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  ConvertOutputCUDAKernel_BinaryLogloss<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(sigmoid_, num_data, input, output);
}

__global__ void ResetOVACUDALableKernel(
  const int ova_class_id,
  const data_size_t num_data,
  label_t* cuda_label) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const int int_label = static_cast<int>(cuda_label[data_index]);
    cuda_label[data_index] = (int_label == ova_class_id ? 1.0f : 0.0f);
  }
}

void CUDABinaryLogloss::LaunchResetOVACUDALableKernel() const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_BINARY - 1) / GET_GRADIENTS_BLOCK_SIZE_BINARY;
  ResetOVACUDALableKernel<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_BINARY>>>(ova_class_id_, num_data_, cuda_ova_label_);
}

}  // namespace LightGBM

#endif  // USE_CUDA_EXP
