/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_multiclass_metric.hpp"
#include <LightGBM/cuda/cuda_algorithms.hpp>

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator, bool USE_WEIGHT>
__global__ void EvalKernel_MulticlassPointWiseLoss(const double* score,
                            const label_t* label,
                            const label_t* weights,
                            const data_size_t num_data,
                            const double sum_weight,
                            double* cuda_sum_loss_buffer,
                            const int num_classes,
                            const int multi_error_top_k) {
  // assert that warpSize == 32 and maximum number of threads per block is 1024
  __shared__ double shared_buffer[32];
  const int data_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const double* score_ptr = score + data_index * num_classes;
  const double pointwise_loss = data_index < num_data ?
    (USE_WEIGHT ? CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score_ptr, num_classes, multi_error_top_k) * weights[data_index] :
                  CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score_ptr, num_classes, multi_error_top_k)) :
                  0.0f;
  const double loss = ShuffleReduceSum<double>(pointwise_loss, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = loss;
  }
}

template <typename CUDAPointWiseLossCalculator>
__global__ void ReduceLossKernel_Multiclass(const double* cuda_sum_loss_buffer, const data_size_t num_blocks, double* out_loss) {
  __shared__ double shared_buffer[32];
  double thread_sum_loss = 0.0f;
  for (int block_index = static_cast<int>(threadIdx.x); block_index < num_blocks; block_index += static_cast<int>(blockDim.x)) {
    thread_sum_loss += cuda_sum_loss_buffer[block_index];
  }
  const double sum_loss = ShuffleReduceSum<double>(thread_sum_loss, shared_buffer, static_cast<size_t>(num_blocks));
  if (threadIdx.x == 0) {
    *out_loss = sum_loss;
  }
}

template <typename CUDAPointWiseLossCalculator>
void CUDAMulticlassMetric<CUDAPointWiseLossCalculator>::LaunchEvalKernelInner(const double* score) const {
  const data_size_t num_blocks = (MulticlassMetric<CUDAPointWiseLossCalculator>::num_data_ + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_MulticlassPointWiseLoss<CUDAPointWiseLossCalculator, false><<<num_blocks, EVAL_BLOCK_SIZE_MULTICLASS_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_,
      this->num_class_,
      this->config_.multi_error_top_k);
  } else {
    EvalKernel_MulticlassPointWiseLoss<CUDAPointWiseLossCalculator, true><<<num_blocks, EVAL_BLOCK_SIZE_MULTICLASS_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_,
      this->num_class_,
      this->config_.multi_error_top_k);
  }
  ReduceLossKernel_Multiclass<CUDAPointWiseLossCalculator><<<1, EVAL_BLOCK_SIZE_MULTICLASS_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <>
void CUDAMulticlassMetric<CUDAMultiErrorMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDAMulticlassMetric<CUDAMultiSoftmaxLoglossMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

__global__ void EvalKernel_AucMuInner(
  const data_size_t* cuda_class_start,
  const data_size_t* cuda_class_size,
  const data_size_t* cuda_sorted_indices,
  const double* cuda_class_data_weights,
  double* cuda_dist,
  data_size_t* cuda_sorted_indices_by_dist) {
  
}

__global__ void EvalKernel_AucMuWriteDist(
  const data_size_t i_class_start,
  const data_size_t i_class_size,
  const data_size_t j_class_start,
  const data_size_t j_class_size,
  const data_size_t* cuda_sorted_indices,
  const double* cuda_class_data_weights,
  const double* cuda_curr_v,
  const double* score,
  const data_size_t max_pair_buffer_size,
  const data_size_t num_data,
  const int num_class,
  double* cuda_dist) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  // put the dist of class j in the front
  const data_size_t data_index_in_class = data_index < j_class_size ? data_index : data_index - j_class_size;
  const data_size_t class_start = data_index < j_class_size ? j_class_size : i_class_start;
  const data_size_t class_size = data_index < j_class_size ? j_class_size : i_class_size;
  const data_size_t* sorted_indices_in_class = cuda_sorted_indices + class_start;
  const data_size_t a = sorted_indices_in_class[data_index_in_class];
  double v_a = 0.0f;
  for (int m = 0; m < num_class; ++m) {
    v_a += cuda_curr_v[m] * score[num_data * m + a];
  }
  const double t1 = cuda_curr_v[i] - cuda_curr_v[j];
  cuda_dist[data_index] = v_a * t1;
}

__global__ void BitonicArgSortGlobal_AucMu(
  const double* dist,
  data_size_t* out_data_indices,
  const data_size_t num_data) {
  int max_depth = 1;
  int len_to_shift = static_cast<int>(num_data) - 1;
  while (len_to_shift > 0) {
    ++max_depth;
    len_to_shift >>= 1;
  }
  const int num_blocks = (static_cast<int>(num_data) + BITONIC_SORT_NUM_ELEMENTS - 1) / BITONIC_SORT_NUM_ELEMENTS;
  BitonicArgSortGlobalKernel<double, data_size_t, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(dist, out_data_indices, static_cast<int>(num_data));
  for (int depth = max_depth - 11; depth >= 1; --depth) {
    const int segment_length = (1 << (max_depth - depth));
    int half_segment_length = (segment_length >> 1);
    {
      BitonicArgCompareKernel<double, data_size_t, ASCENDING, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        dist, out_data_indices, half_segment_length, segment_length, static_cast<int>(num_data));
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicArgCompareKernel<double, data_size_t, ASCENDING, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        dist, out_data_indices, half_segment_length, segment_length, static_cast<int>(num_data));
      half_segment_length >>= 1;
    }
    BitonicArgSortMergeKernel<double, data_size_t, ASCENDING><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
      dist, out_data_indices, segment_length, static_cast<int>(len));
  }
}

template <bool USE_WEIGHT, bool IS_POS>
__global__ void GenAucMuPosPrefixSumWithinBlock(
  const data_size_t* sorted_data_indices_global,
  const data_size_t* sorted_data_indices_two_class,
  const data_size_t i_class_size,
  const data_size_t j_class_size,
  const data_size_t i_class_start,
  const data_size_t j_class_start,
  const label_t* cuda_weights,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer) {
  __shared__ double shared_buffer[GLOBAL_PREFIX_SUM_BLOCK_SIZE + 1];
  const data_size_t inner_data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double pos = 0.0f;
  if (inner_data_index < j_class_size + i_class_size) {
    const data_size_t data_index_two_class = sorted_data_indices[inner_data_index];
    const bool is_pos_class = (data_index_two_class < j_class_size);
    if (USE_WEIGHT) {
      const data_size_t data_index_one_class = (is_pos_class ? data_index_two_class : data_index_two_class - j_class_size);
      const data_size_t data_index_global = (is_pos_class ? sorted_data_indices_global[j_class_start + data_index_one_class] :
        sorted_data_indices_global[i_class_start + data_index_one_class]);
      pos = ((is_pos_class == IS_POS) ? cuda_weights[data_index_global] : 0.0f);
    } else {
      pos = ((is_pos_class == IS_POS) ? 1.0f : 0.0f);
    }
  }
  shared_buffer[threadIdx.x] = pos;
  __syncthreads();
  PrefixSum<double>(shared_buffer, blockDim.x);
  if (inner_data_index < j_class_size + i_class_size) {
    sum_pos_buffer[inner_data_index] = shared_buffer[threadIdx.x + 1];
  }
  if (threadIdx.x == 0) {
    block_sum_pos_buffer[blockidx.x + 1] = shared_buffer[blockDim.x];
  }
}

__global__ void GenAucMuPosPrefixSum(
  const data_size_t* sorted_data_indices_global,
  const data_size_t* sorted_data_indices_two_class,
  const data_size_t i_class_size,
  const data_size_t j_class_size,
  const data_size_t i_class_start,
  const data_size_t j_class_start,
  const label_t* cuda_weights,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer) {
  const data_size_t num_data = i_class_size + j_class_size;
  const int num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GenAucMuPosPrefixSumWithinBlock<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    sorted_data_indices_global,
    sorted_data_indices_two_class,
    i_class_size,
    j_class_size,
    i_class_start,
    j_class_start,
    cuda_weights,
    sum_pos_buffer,
    block_sum_pos_buffer);
  GlobalInclusivePrefixSumReduceBlockKernel<double><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_sum_pos_buffer, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseKernel<double><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_sum_pos_buffer, sum_pos_buffer, num_data);
}

__global__ void GenAucMuMark(
  const double* dist,
  const data_size_t* sorted_data_indices,
  const data_size_t num_data,
  data_size_t* threshold_mark,
  data_size_t* block_mark_buffer,
  uint16_t* block_mark_first_zero) {
  const data_size_t num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GlobalGenAUCMarkKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(dist, sorted_data_indices, threshold_mark, block_mark_buffer, block_mark_first_zero, num_data);
  GlobalInclusivePrefixSumReduceBlockZeroOutKernel<<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_mark_buffer, block_mark_first_zero, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseGenAUCMarkKernel<<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_mark_buffer, threshold_mark, block_mark_first_zero, num_data);
}

template <bool USE_WEIGHT>
__global__ void EvalKernel_AucMu(
  const data_size_t* cuda_class_start,
  const data_size_t* cuda_class_size,
  const data_size_t* cuda_sorted_indices,
  const double* cuda_class_data_weights,
  const double* cuda_curr_v,
  const double* score,
  const data_size_t max_pair_buffer_size,
  const data_size_t num_data,
  const int num_class,
  const label_t* cuda_weights,
  double* cuda_dist,
  data_size_t* cuda_sorted_indices_by_dist,
  data_size_t* cuda_threshold_mark,
  data_size_t* cuda_block_threshold_mark_buffer,
  uint16_t* cuda_block_mark_first_zero,
  double* sum_pos_buffer,
  double* block_sum_pos_buffer) {
  const int pair_index = static_cast<int>(blockIdx.x);
  const double index_2 = 2 * static_cast<double>(pair_index);
  const int sqrt_round = static_cast<int>(sqrt(index_2));
  const int i_p = static_cast<int>(sqrt(index_2 - static_cast<double>(sqrt_round) + 1));
  const int j_p = pair_index - ((i_p + 1) * i_p / 2);
  const int i = num_class - 2 - i_p;
  const int j = j_p + i + 1;
  const data_size_t num_data_in_pair = cuda_class_size[i] + cuda_class_size[j];
  const int num_blocks = (num_data_in_pair + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  const data_size_t i_class_start = cuda_class_start[i];
  const data_size_t j_class_start = cuda_class_start[j];
  const data_size_t i_class_size = cuda_class_size[i];
  const data_size_t j_class_size = cuda_class_size[j];
  double* cuda_dist_ptr = cuda_dist + pair_index * max_pair_buffer_size;
  data_size_t* cuda_sorted_indices_by_dist_ptr = cuda_sorted_indices_by_dist + pair_index * max_pair_buffer_size;
  const double* cuda_curr_v_ptr = cuda_curr_v + pair_index * num_class;
  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);
  EvalKernel_AucMuWriteDist<<<num_blocks, EVAL_BLOCK_SIZE_MULTICLASS_METRIC, 0, cuda_stream>>>(
    i_class_start,
    i_class_size,
    j_class_start,
    j_class_size,
    cuda_sorted_indices,
    cuda_class_data_weights,
    cuda_curr_v_ptr,
    score,
    max_pair_buffer_size,
    num_data,
    num_class,
    cuda_dist_ptr);
  BitonicArgSortGlobal_AucMu<<<1, 1, 0, cuda_stream>>>(
    cuda_dist_ptr,
    cuda_sorted_indices_by_dist_ptr,
    i_class_size + j_class_size);
  GenAucMuPosPrefixSum<<<1, 1, 0, cuda_stream>>>(
    cuda_sorted_indices,
    cuda_sorted_indices_by_dist,
    i_class_size,
    j_class_size,
    i_class_start,
    j_class_start,
    cuda_weights,
    sum_pos_buffer,
    block_sum_pos_buffer);
  GenAucMuMark<<<1, 1, 0, cuda_stream>>>(
    cuda_dist_ptr,
    cuda_sorted_indices_by_dist_ptr,
    cuda_threshold_mark,
    cuda_block_threshold_mark_buffer,
    cuda_block_mark_first_zero);
  
}

void CUDAAucMuMetric::LaunchEvalKernel(const double* score) const {
  const int num_class_pair = (num_class_ - 1) * num_class_ / 2;
  EvalKernel_AucMu<<<num_class_pair, 1>>>(
    cuda_class_start_,
    cuda_class_size_,
    cuda_sorted_indices,
    cuda_class_data_weights_,
    cuda_dist_,
    cuda_sorted_indices_by_dist);
}

}  // namespace LightGBM
