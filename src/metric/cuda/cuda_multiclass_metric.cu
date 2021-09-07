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
  const int i,
  const int j,
  double* cuda_dist) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < j_class_size + i_class_size) {
    // put the dist of class j in the front
    const data_size_t data_index_in_class = data_index < j_class_size ? data_index : data_index - j_class_size;
    const data_size_t class_start = data_index < j_class_size ? j_class_start : i_class_start;
    const data_size_t* sorted_indices_in_class = cuda_sorted_indices + class_start;
    const data_size_t a = sorted_indices_in_class[data_index_in_class];
    double v_a = 0.0f;
    for (int m = 0; m < num_class; ++m) {
      v_a += cuda_curr_v[m] * score[num_data * m + a];
    }
    const double t1 = cuda_curr_v[i] - cuda_curr_v[j];
    cuda_dist[data_index] = v_a * t1;
  }
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
      BitonicArgCompareKernel<double, data_size_t, true, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        dist, out_data_indices, half_segment_length, segment_length, static_cast<int>(num_data));
      half_segment_length >>= 1;
    }
    for (int inner_depth = depth + 1; inner_depth <= max_depth - 11; ++inner_depth) {
      BitonicArgCompareKernel<double, data_size_t, true, false><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
        dist, out_data_indices, half_segment_length, segment_length, static_cast<int>(num_data));
      half_segment_length >>= 1;
    }
    BitonicArgSortMergeKernel<double, data_size_t, true><<<num_blocks, BITONIC_SORT_NUM_ELEMENTS>>>(
      dist, out_data_indices, segment_length, static_cast<int>(num_data));
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
    const data_size_t data_index_two_class = sorted_data_indices_two_class[inner_data_index];
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
    block_sum_pos_buffer[blockIdx.x + 1] = shared_buffer[blockDim.x];
  }
}

template <bool USE_WEIGHT, bool IS_POS>
__global__ void GenAucMuPosPrefixSum(
  const data_size_t* sorted_data_indices_global,
  const data_size_t* sorted_data_indices_two_class,
  const data_size_t i_class_size,
  const data_size_t j_class_size,
  const data_size_t i_class_start,
  const data_size_t j_class_start,
  const label_t* cuda_weights,
  double* prefix_sum_result,
  double* block_buffer) {
  const data_size_t num_data = i_class_size + j_class_size;
  const int num_blocks = (num_data + GLOBAL_PREFIX_SUM_BLOCK_SIZE - 1) / GLOBAL_PREFIX_SUM_BLOCK_SIZE;
  GenAucMuPosPrefixSumWithinBlock<USE_WEIGHT, IS_POS><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    sorted_data_indices_global,
    sorted_data_indices_two_class,
    i_class_size,
    j_class_size,
    i_class_start,
    j_class_start,
    cuda_weights,
    prefix_sum_result,
    block_buffer);
  GlobalInclusivePrefixSumReduceBlockKernel<double><<<1, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, num_blocks);
  GlobalInclusivePrefixSumAddBlockBaseKernel<double><<<num_blocks, GLOBAL_PREFIX_SUM_BLOCK_SIZE>>>(
    block_buffer, prefix_sum_result, num_data);
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
__global__ void CalcAucMuArea(
  const double* block_sum_pos_buffer,
  const data_size_t* sorted_data_indices_global,
  const data_size_t* sorted_data_indices_two_class,
  const data_size_t* threshold_mark,
  const label_t* cuda_weights,
  const data_size_t num_data,
  const data_size_t i_class_start,
  const data_size_t j_class_size,
  double* block_buffer) {
  __shared__ double shared_mem_buffer[32];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  double area = 0.0f;
  if (data_index < num_data) {
    const data_size_t data_index_two_class = sorted_data_indices_two_class[data_index];
    if (data_index_two_class >= j_class_size) {
      const data_size_t data_index_global = sorted_data_indices_global[i_class_start + data_index_two_class - j_class_size];
      const double num_j = block_sum_pos_buffer[data_index];
      if (USE_WEIGHT) {
        const double curr_weight = static_cast<double>(cuda_weights[data_index_global]);
        if (threshold_mark[data_index] > 0) {
          const data_size_t prev_data_index = data_index - threshold_mark[data_index] - 1;
          const double prev_sum_pos = prev_data_index < 0 ? 0.0f : block_sum_pos_buffer[prev_data_index];
          const double num_curr_j = block_sum_pos_buffer[data_index] - prev_sum_pos;
          area = curr_weight * (num_j - 0.5f * num_curr_j);
        } else {
          area = curr_weight * num_j;
        }
      } else {
        if (threshold_mark[data_index] > 0) {
          const data_size_t prev_data_index = data_index - threshold_mark[data_index] - 1;
          const double prev_sum_pos = prev_data_index < 0 ? 0.0f : block_sum_pos_buffer[prev_data_index];
          const double num_curr_j = block_sum_pos_buffer[data_index] - prev_sum_pos;
          area = num_j - 0.5f * num_curr_j;
        } else {
          area = num_j;
        }
      }
    }
  }
  const double block_area = ShuffleReduceSum<double>(area, shared_mem_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    block_buffer[blockIdx.x] = block_area;
  }
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
  double* block_sum_pos_buffer,
  double* reduce_ans_buffer) {
  const int pair_index = static_cast<int>(blockIdx.x);
  const double index_2 = 2.0f * static_cast<double>(pair_index);
  const int sqrt_round = static_cast<int>(sqrt(index_2));
  const int i_p = (pair_index == 0) ? 0 : static_cast<int>(sqrt(index_2 - static_cast<double>(sqrt_round) + 1));
  const int j_p = pair_index - ((i_p + 1) * i_p / 2);
  const int i = num_class - 2 - i_p;
  const int j = j_p + i + 1;
  const data_size_t i_class_size = cuda_class_size[i];
  const data_size_t j_class_size = cuda_class_size[j];
  const data_size_t num_data_in_pair = i_class_size + j_class_size;
  const int num_blocks = (num_data_in_pair + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  const int num_blocks_for_offset = (max_pair_buffer_size + EVAL_BLOCK_SIZE_MULTICLASS_METRIC - 1) / EVAL_BLOCK_SIZE_MULTICLASS_METRIC;
  const data_size_t i_class_start = cuda_class_start[i];
  const data_size_t j_class_start = cuda_class_start[j];
  double* cuda_dist_ptr = cuda_dist + pair_index * max_pair_buffer_size;
  data_size_t* cuda_sorted_indices_by_dist_ptr = cuda_sorted_indices_by_dist + pair_index * max_pair_buffer_size;
  const double* cuda_curr_v_ptr = cuda_curr_v + pair_index * num_class;
  double* sum_pos_buffer_ptr = sum_pos_buffer + pair_index * max_pair_buffer_size;
  double* block_sum_pos_buffer_ptr = block_sum_pos_buffer + pair_index * (num_blocks_for_offset + 1);
  data_size_t* cuda_threshold_mark_ptr = cuda_threshold_mark + pair_index * max_pair_buffer_size;
  data_size_t* cuda_block_threshold_mark_buffer_ptr = cuda_block_threshold_mark_buffer + pair_index * (num_blocks_for_offset + 1);
  uint16_t* cuda_block_mark_first_zero_ptr = cuda_block_mark_first_zero + pair_index * (num_blocks_for_offset + 1);
  cudaStream_t cuda_stream;
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
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
    i,
    j,
    cuda_dist_ptr);
  BitonicArgSortGlobal_AucMu<<<1, 1, 0, cuda_stream>>>(
    cuda_dist_ptr,
    cuda_sorted_indices_by_dist_ptr,
    num_data_in_pair);
  GenAucMuPosPrefixSum<USE_WEIGHT, true><<<1, 1, 0, cuda_stream>>>(
    cuda_sorted_indices,
    cuda_sorted_indices_by_dist_ptr,
    i_class_size,
    j_class_size,
    i_class_start,
    j_class_start,
    cuda_weights,
    sum_pos_buffer_ptr,
    block_sum_pos_buffer_ptr);
  GenAucMuMark<<<1, 1, 0, cuda_stream>>>(
    cuda_dist_ptr,
    cuda_sorted_indices_by_dist_ptr,
    num_data_in_pair,
    cuda_threshold_mark_ptr,
    cuda_block_threshold_mark_buffer_ptr,
    cuda_block_mark_first_zero_ptr);
  CalcAucMuArea<USE_WEIGHT><<<1, 1, 0, cuda_stream>>>(
    block_sum_pos_buffer_ptr,
    cuda_sorted_indices,
    cuda_sorted_indices_by_dist_ptr,
    cuda_threshold_mark_ptr,
    cuda_weights,
    num_data_in_pair,
    i_class_start,
    j_class_size,
    block_sum_pos_buffer_ptr);
  BlockReduceSum<double><<<1, EVAL_BLOCK_SIZE_MULTICLASS_METRIC, 0, cuda_stream>>>(block_sum_pos_buffer_ptr, num_blocks);
  if (USE_WEIGHT) {
    reduce_ans_buffer[pair_index] = block_sum_pos_buffer_ptr[0] / cuda_class_data_weights[i] / cuda_class_data_weights[j];
  } else {
    reduce_ans_buffer[pair_index] = block_sum_pos_buffer_ptr[0] / static_cast<double>(cuda_class_size[i]) / static_cast<double>(cuda_class_size[j]);
  }
  cudaStreamDestroy(cuda_stream);
}

void CUDAAucMuMetric::LaunchEvalKernel(const double* score) const {
  const int num_class_pair = (num_class_ - 1) * num_class_ / 2;
  if (cuda_weights_ == nullptr) {
    EvalKernel_AucMu<false><<<num_class_pair, 1>>>(
      cuda_class_start_,
      cuda_class_size_,
      cuda_sorted_indices_,
      cuda_class_data_weights_,
      cuda_curr_v_,
      score,
      max_pair_buffer_size_,
      num_data_,
      num_class_,
      cuda_weights_,
      cuda_dist_,
      cuda_sorted_indices_by_dist_,
      cuda_threshold_mark_,
      cuda_block_mark_buffer_,
      cuda_block_mark_first_zero_,
      cuda_sum_pos_buffer_,
      cuda_reduce_block_buffer_,
      cuda_reduce_ans_buffer_);
  } else {
    EvalKernel_AucMu<true><<<num_class_pair, 1>>>(
      cuda_class_start_,
      cuda_class_size_,
      cuda_sorted_indices_,
      cuda_class_data_weights_,
      cuda_curr_v_,
      score,
      max_pair_buffer_size_,
      num_data_,
      num_class_,
      cuda_weights_,
      cuda_dist_,
      cuda_sorted_indices_by_dist_,
      cuda_threshold_mark_,
      cuda_block_mark_buffer_,
      cuda_block_mark_first_zero_,
      cuda_sum_pos_buffer_,
      cuda_reduce_block_buffer_,
      cuda_reduce_ans_buffer_);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  BlockReduceSum<double><<<1, EVAL_BLOCK_SIZE_MULTICLASS_METRIC>>>(cuda_reduce_ans_buffer_, num_class_pair);
}

}  // namespace LightGBM
