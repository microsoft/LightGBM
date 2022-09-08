/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA_EXP

#include "cuda_regression_objective.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

double CUDARegressionL2loss::LaunchCalcInitScoreKernel() const {
  double label_sum = 0.0f, weight_sum = 0.0f;
  if (cuda_weights_ == nullptr) {
    ShuffleReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
    weight_sum = static_cast<double>(num_data_);
  } else {
    ShuffleReduceDotProdGlobal<label_t, double>(cuda_labels_, cuda_weights_, static_cast<size_t>(num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&label_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
    ShuffleReduceSumGlobal<label_t, double>(cuda_weights_, static_cast<size_t>(num_data_), cuda_block_buffer_.RawData());
    CopyFromCUDADeviceToHost<double>(&weight_sum, cuda_block_buffer_.RawData(), 1, __FILE__, __LINE__);
  }
  return label_sum / weight_sum;
}

__global__ void ConvertOutputCUDAKernel_Regression(const bool sqrt, const data_size_t num_data, const double* input, double* output) {
  const int data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    if (sqrt) {
      const double sign = input[data_index] >= 0.0f ? 1 : -1;
      output[data_index] = sign * input[data_index] * input[data_index];
    } else {
      output[data_index] = input[data_index];
    }
  }
}

void CUDARegressionL2loss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  ConvertOutputCUDAKernel_Regression<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(sqrt_, num_data, input, output);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL2(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(cuda_scores[data_index] - cuda_labels[data_index]) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionL2loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL2<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL2<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}


double CUDARegressionL1loss::LaunchCalcInitScoreKernel() const {
  const double alpha = 0.5f;
  if (cuda_weights_ == nullptr) {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, false>(
      cuda_labels_, nullptr, cuda_data_indices_buffer_.RawData(), nullptr, nullptr, alpha, num_data_, cuda_percentile_result_.RawData());
  } else {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, true>(
      cuda_labels_, cuda_weights_, cuda_data_indices_buffer_.RawData(), cuda_weights_prefix_sum_.RawData(),
      cuda_weights_prefix_sum_buffer_.RawData(), alpha, num_data_, cuda_percentile_result_.RawData());
  }
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHost<label_t>(&percentile_result, cuda_percentile_result_.RawData(), 1, __FILE__, __LINE__);
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return static_cast<label_t>(percentile_result);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionL1(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f)) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionL1loss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionL1<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionL1<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, gradients, hessians);
  }
}

template <bool USE_WEIGHT>
__global__ void RenewTreeOutputCUDAKernel_RegressionL1(
  const double* score,
  const label_t* label,
  const label_t* weight,
  double* residual_buffer,
  label_t* weight_by_leaf,
  double* weight_prefix_sum_buffer,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  data_size_t* data_indices_buffer,
  double* leaf_value) {
  const int leaf_index = static_cast<int>(blockIdx.x);
  const data_size_t data_start = data_start_in_leaf[leaf_index];
  const data_size_t num_data = num_data_in_leaf[leaf_index];
  data_size_t* data_indices_buffer_pointer = data_indices_buffer + data_start;
  const label_t* weight_by_leaf_pointer = weight_by_leaf + data_start;
  double* weight_prefix_sum_buffer_pointer = weight_prefix_sum_buffer + data_start;
  const double* residual_buffer_pointer = residual_buffer + data_start;
  const double alpha = 0.5f;
  for (data_size_t inner_data_index = data_start + static_cast<data_size_t>(threadIdx.x); inner_data_index < data_start + num_data; inner_data_index += static_cast<data_size_t>(blockDim.x)) {
    const data_size_t data_index = data_indices_in_leaf[inner_data_index];
    const label_t data_label = label[data_index];
    const double data_score = score[data_index];
    residual_buffer[inner_data_index] = static_cast<double>(data_label) - data_score;
    if (USE_WEIGHT) {
      weight_by_leaf[inner_data_index] = weight[data_index];
    }
  }
  __syncthreads();
  const double renew_leaf_value = PercentileDevice<double, data_size_t, label_t, double, false, USE_WEIGHT>(
    residual_buffer_pointer, weight_by_leaf_pointer, data_indices_buffer_pointer,
    weight_prefix_sum_buffer_pointer, alpha, num_data);
  if (threadIdx.x == 0) {
    leaf_value[leaf_index] = renew_leaf_value;
  }
}

void CUDARegressionL1loss::LaunchRenewTreeOutputCUDAKernel(
  const double* score,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  const int num_leaves,
  double* leaf_value) const {
  if (cuda_weights_ == nullptr) {
    RenewTreeOutputCUDAKernel_RegressionL1<false><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 2>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value);
  } else {
    RenewTreeOutputCUDAKernel_RegressionL1<true><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_.RawData(),
      cuda_weight_by_leaf_buffer_.RawData(),
      cuda_weights_prefix_sum_.RawData(),
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_.RawData(),
      leaf_value);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}


template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Huber(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double alpha, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      if (fabs(diff) <= alpha) {
        cuda_out_gradients[data_index] = static_cast<score_t>(diff);
      } else {
        const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
        cuda_out_gradients[data_index] = static_cast<score_t>(sign * alpha);
      }
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      if (fabs(diff) <= alpha) {
        cuda_out_gradients[data_index] = static_cast<score_t>(diff) * weight;
      } else {
        const score_t sign = static_cast<score_t>((diff > 0.0f) - (diff < 0.0f));
        cuda_out_gradients[data_index] = static_cast<score_t>(sign * alpha) * weight;
      }
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionHuberLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Huber<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, alpha_, gradients, hessians);
  } else {
    GetGradientsKernel_Huber<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, alpha_, gradients, hessians);
  }
}


template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Fair(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double c, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(c * diff / (fabs(diff) + c));
      cuda_out_hessians[data_index] = static_cast<score_t>(c * c / ((fabs(diff) + c) * (fabs(diff) + c)));
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(c * diff / (fabs(diff) + c) * weight);
      cuda_out_hessians[data_index] = static_cast<score_t>(c * c / ((fabs(diff) + c) * (fabs(diff) + c)) * weight);
    }
  }
}

void CUDARegressionFairLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Fair<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, c_, gradients, hessians);
  } else {
    GetGradientsKernel_Fair<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, c_, gradients, hessians);
  }
}


}  // namespace LightGBM

#endif  // USE_CUDA_EXP
