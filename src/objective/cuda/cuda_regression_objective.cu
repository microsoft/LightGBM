/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_regression_objective.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

double CUDARegressionL2loss::LaunchCalcInitScoreKernel() const {
  double label_sum = 0.0f, weight_sum = 0.0f;
  ReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_);
  CopyFromCUDADeviceToHostOuter<double>(&label_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);
  if (cuda_weights_ == nullptr) {
    weight_sum = static_cast<double>(num_data_);
  } else {
    ReduceSumGlobal<label_t, double>(cuda_weights_, static_cast<size_t>(num_data_), cuda_block_buffer_);
    CopyFromCUDADeviceToHostOuter<double>(&weight_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);
  }
  return label_sum / weight_sum;
}

// TODO(shiyu1994): try to use global kernels as class methods
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
  const double alpha = 0.9f;
  if (cuda_weights_ == nullptr) {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, false>(
      cuda_labels_, nullptr, cuda_data_indices_buffer_, nullptr, nullptr, alpha, num_data_, cuda_percentile_result_);
  } else {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, true>(
      cuda_labels_, cuda_weights_, cuda_data_indices_buffer_, cuda_weights_prefix_sum_, cuda_weights_prefix_sum_buffer_, alpha, num_data_, cuda_percentile_result_);
  }
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHostOuter<label_t>(&percentile_result, cuda_percentile_result_, 1, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
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
  // TODO(shiyu1994): replace this bitonic sort based percentile method with a more efficient one 
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
      cuda_residual_buffer_,
      cuda_weight_by_leaf_buffer_,
      cuda_weights_prefix_sum_,
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_,
      leaf_value);
  } else {
    RenewTreeOutputCUDAKernel_RegressionL1<true><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_,
      cuda_weight_by_leaf_buffer_,
      cuda_weights_prefix_sum_,
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_,
      leaf_value);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
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

void CUDARegressionPoissonLoss::LaunchCheckLabelKernel() const {
  ReduceSumGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_);
  double label_sum = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&label_sum, cuda_block_buffer_, 1, __FILE__, __LINE__);

  ReduceMinGlobal<label_t, double>(cuda_labels_, static_cast<size_t>(num_data_), cuda_block_buffer_);
  double label_min = 0.0f;
  CopyFromCUDADeviceToHostOuter<double>(&label_min, cuda_block_buffer_, 1, __FILE__, __LINE__);

  if (label_min < 0.0f) {
    Log::Fatal("[%s]: at least one target label is negative", GetName());
  }
  if (label_sum == 0.0f) {
    Log::Fatal("[%s]: sum of labels is zero", GetName());
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Poisson(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double max_delta_step, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(exp(cuda_scores[data_index]) - cuda_labels[data_index]);
      cuda_out_hessians[data_index] = static_cast<score_t>(std::exp(cuda_scores[data_index] + max_delta_step));
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(exp(cuda_scores[data_index]) - cuda_labels[data_index]) * weight;
      cuda_out_hessians[data_index] = static_cast<score_t>(std::exp(cuda_scores[data_index] + max_delta_step)) * weight;
    }
  }
}

void CUDARegressionPoissonLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Poisson<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, max_delta_step_, gradients, hessians);
  } else {
    GetGradientsKernel_Poisson<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, max_delta_step_, gradients, hessians);
  }
}

// TODO(shiyu1994): try to use global kernels as class methods
__global__ void ConvertOutputCUDAKernel_Regression_Poissson(const bool sqrt, const data_size_t num_data, const double* input, double* output) {
  const int data_index = static_cast<data_size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (data_index < num_data) {
    output[data_index] = exp(input[data_index]);
  }
}

void CUDARegressionPoissonLoss::LaunchConvertOutputCUDAKernel(const data_size_t num_data, const double* input, double* output) const {
  const int num_blocks = (num_data + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  ConvertOutputCUDAKernel_Regression_Poissson<<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(sqrt_, num_data, input, output);
}

double CUDARegressionQuantileloss::LaunchCalcInitScoreKernel() const {
  if (cuda_weights_ == nullptr) {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, false>(
      cuda_labels_, nullptr, cuda_data_indices_buffer_, nullptr, nullptr, alpha_, num_data_, cuda_percentile_result_);
  } else {
    PercentileGlobal<label_t, data_size_t, label_t, double, false, true>(
      cuda_labels_, cuda_weights_, cuda_data_indices_buffer_, cuda_weights_prefix_sum_, cuda_weights_prefix_sum_buffer_, alpha_, num_data_, cuda_percentile_result_);
  }
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHostOuter<label_t>(&percentile_result, cuda_percentile_result_, 1, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  return static_cast<label_t>(percentile_result);
}

template <bool USE_WEIGHT>
__global__ void RenewTreeOutputCUDAKernel_RegressionQuantile(
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
  double* leaf_value,
  const double alpha) {
  const int leaf_index = static_cast<int>(blockIdx.x);
  const data_size_t data_start = data_start_in_leaf[leaf_index];
  const data_size_t num_data = num_data_in_leaf[leaf_index];
  data_size_t* data_indices_buffer_pointer = data_indices_buffer + data_start;
  const label_t* weight_by_leaf_pointer = weight_by_leaf + data_start;
  double* weight_prefix_sum_buffer_pointer = weight_prefix_sum_buffer + data_start;
  const double* residual_buffer_pointer = residual_buffer + data_start;
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
  // TODO(shiyu1994): replace this bitonic sort based percentile method with a more efficient one 
  const double renew_leaf_value = PercentileDevice<double, data_size_t, label_t, double, false, USE_WEIGHT>(
    residual_buffer_pointer, weight_by_leaf_pointer, data_indices_buffer_pointer,
    weight_prefix_sum_buffer_pointer, alpha, num_data);
  if (threadIdx.x == 0) {
    leaf_value[leaf_index] = renew_leaf_value;
  }
}

void CUDARegressionQuantileloss::LaunchRenewTreeOutputCUDAKernel(
  const double* score, const data_size_t* data_indices_in_leaf, const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf, const int num_leaves, double* leaf_value) const {
  if (cuda_weights_ == nullptr) {
    RenewTreeOutputCUDAKernel_RegressionQuantile<false><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 2>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_,
      cuda_weight_by_leaf_buffer_,
      cuda_weights_prefix_sum_,
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_,
      leaf_value,
      alpha_);
  } else {
    RenewTreeOutputCUDAKernel_RegressionQuantile<true><<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
      score,
      cuda_labels_,
      cuda_weights_,
      cuda_residual_buffer_,
      cuda_weight_by_leaf_buffer_,
      cuda_weights_prefix_sum_,
      data_indices_in_leaf,
      num_data_in_leaf,
      data_start_in_leaf,
      cuda_data_indices_buffer_,
      leaf_value,
      alpha_);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionQuantile(const double* cuda_scores, const label_t* cuda_labels,
  const label_t* cuda_weights, const data_size_t num_data, const double alpha,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      if (diff >= 0.0f) {
        cuda_out_gradients[data_index] = (1.0f - alpha);
      } else {
        cuda_out_gradients[data_index] = -alpha;
      }
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      if (diff >= 0.0f) {
        cuda_out_gradients[data_index] = (1.0f - alpha) * weight;
      } else {
        cuda_out_gradients[data_index] = -alpha * weight;
      }
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionQuantileloss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionQuantile<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, alpha_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionQuantile<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, alpha_, gradients, hessians);
  }
}

template <bool USE_WEIGHT>
__global__ void CalcLabelWeightKernel(
  const label_t* cuda_labels,
  const label_t* cuda_weights,
  const data_size_t num_data,
  label_t* label_weights
) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    const label_t label = cuda_labels[data_index];
    if (!USE_WEIGHT) {
      label_weights[data_index] = 1.0f / max(1.0f, fabs(label));
    } else {
      const label_t weight = cuda_weights[data_index];
      label_weights[data_index] = 1.0f / max(1.0f, fabs(label)) * weight;
    }
  }
}

void CUDARegressionMAPELOSS::LaunchCalcLabelWeightKernel() {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    CalcLabelWeightKernel<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(cuda_labels_, cuda_weights_, num_data_, cuda_label_weights_);
  } else {
    CalcLabelWeightKernel<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(cuda_labels_, cuda_weights_, num_data_, cuda_label_weights_);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_RegressionMAPELOSS(const double* cuda_scores, const label_t* cuda_labels,
  const label_t* cuda_weights, const label_t* cuda_label_weights, const data_size_t num_data,
  score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    const double diff = cuda_scores[data_index] - static_cast<double>(cuda_labels[data_index]);
    const label_t label_weight = cuda_label_weights[data_index];
    const double sign = static_cast<double>((diff > 0) - (diff < 0));
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(sign * label_weight);
      cuda_out_hessians[data_index] = 1.0f;
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(sign * label_weight) * weight;
      cuda_out_hessians[data_index] = weight;
    }
  }
}

void CUDARegressionMAPELOSS::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_RegressionMAPELOSS<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, cuda_label_weights_, num_data_, gradients, hessians);
  } else {
    GetGradientsKernel_RegressionMAPELOSS<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, cuda_label_weights_, num_data_, gradients, hessians);
  }
}

double CUDARegressionMAPELOSS::LaunchCalcInitScoreKernel() const {
  PercentileGlobal<label_t, data_size_t, label_t, double, false, true>(
    cuda_labels_, cuda_label_weights_, cuda_data_indices_buffer_,
    cuda_weights_prefix_sum_, cuda_weights_prefix_sum_buffer_, 0.5f, num_data_, cuda_percentile_result_);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  label_t percentile_result = 0.0f;
  CopyFromCUDADeviceToHostOuter<label_t>(&percentile_result, cuda_percentile_result_, 1, __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  return static_cast<label_t>(percentile_result);
}

__global__ void RenewTreeOutputCUDAKernel_RegressionMAPE(
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
    weight_by_leaf[inner_data_index] = weight[data_index];
  }
  __syncthreads();
  // TODO(shiyu1994): replace this bitonic sort based percentile method with a more efficient one 
  const double renew_leaf_value = PercentileDevice<double, data_size_t, label_t, double, false, true>(
    residual_buffer_pointer, weight_by_leaf_pointer, data_indices_buffer_pointer,
    weight_prefix_sum_buffer_pointer, alpha, num_data);
  if (threadIdx.x == 0) {
    leaf_value[leaf_index] = renew_leaf_value;
  }
}

void CUDARegressionMAPELOSS::LaunchRenewTreeOutputCUDAKernel(
  const double* score,
  const data_size_t* data_indices_in_leaf,
  const data_size_t* num_data_in_leaf,
  const data_size_t* data_start_in_leaf,
  const int num_leaves,
  double* leaf_value) const {
  RenewTreeOutputCUDAKernel_RegressionMAPE<<<num_leaves, GET_GRADIENTS_BLOCK_SIZE_REGRESSION / 4>>>(
    score,
    cuda_labels_,
    cuda_label_weights_,
    cuda_residual_buffer_,
    cuda_weight_by_leaf_buffer_,
    cuda_weights_prefix_sum_,
    data_indices_in_leaf,
    num_data_in_leaf,
    data_start_in_leaf,
    cuda_data_indices_buffer_,
    leaf_value);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Gamma(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double max_delta_step, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(1.0 - cuda_labels[data_index] / exp(cuda_scores[data_index]));
      cuda_out_hessians[data_index] = static_cast<score_t>(cuda_labels[data_index] / exp(cuda_scores[data_index]));
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(1.0 - cuda_labels[data_index] / exp(cuda_scores[data_index])) * weight;
      cuda_out_hessians[data_index] = static_cast<score_t>(cuda_labels[data_index] / exp(cuda_scores[data_index])) * weight;
    }
  }
}

void CUDARegressionGammaLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Gamma<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, max_delta_step_, gradients, hessians);
  } else {
    GetGradientsKernel_Gamma<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, max_delta_step_, gradients, hessians);
  }
}

template <bool USE_WEIGHT>
__global__ void GetGradientsKernel_Tweedie(const double* cuda_scores, const label_t* cuda_labels, const label_t* cuda_weights, const data_size_t num_data,
  const double rho, score_t* cuda_out_gradients, score_t* cuda_out_hessians) {
  const data_size_t data_index = static_cast<data_size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (data_index < num_data) {
    if (!USE_WEIGHT) {
      cuda_out_gradients[data_index] = static_cast<score_t>(-cuda_labels[data_index] * exp((1 - rho) * cuda_scores[data_index]) + exp((2 - rho) * cuda_scores[data_index]));
      cuda_out_hessians[data_index] = static_cast<score_t>(-cuda_labels[data_index] * (1 - rho) * exp((1 - rho) * cuda_scores[data_index]) +
        (2 - rho) * exp((2 - rho) * cuda_scores[data_index]));
    } else {
      const score_t weight = static_cast<score_t>(cuda_weights[data_index]);
      cuda_out_gradients[data_index] = static_cast<score_t>(-cuda_labels[data_index] * exp((1 - rho) * cuda_scores[data_index]) +
        exp((2 - rho) * cuda_scores[data_index])) * weight;
      cuda_out_hessians[data_index] = static_cast<score_t>(-cuda_labels[data_index] * (1 - rho) * exp((1 - rho) * cuda_scores[data_index]) +
        (2 - rho) * exp((2 - rho) * cuda_scores[data_index])) * weight;
    }
  }
}

void CUDARegressionTweedieLoss::LaunchGetGradientsKernel(const double* score, score_t* gradients, score_t* hessians) const {
  const int num_blocks = (num_data_ + GET_GRADIENTS_BLOCK_SIZE_REGRESSION - 1) / GET_GRADIENTS_BLOCK_SIZE_REGRESSION;
  if (cuda_weights_ == nullptr) {
    GetGradientsKernel_Tweedie<false><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, nullptr, num_data_, rho_, gradients, hessians);
  } else {
    GetGradientsKernel_Tweedie<true><<<num_blocks, GET_GRADIENTS_BLOCK_SIZE_REGRESSION>>>(score, cuda_labels_, cuda_weights_, num_data_, rho_, gradients, hessians);
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
