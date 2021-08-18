/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include "cuda_binary_metric.hpp"

#include <chrono>

namespace LightGBM {

template <typename CUDAPointWiseLossCalculator, bool USE_WEIGHT>
__global__ void EvalKernel_BinaryPointWiseLoss(const double* score,
                           const label_t* label,
                           const label_t* weights,
                           const data_size_t num_data,
                           const double sum_weight,
                           double* cuda_sum_loss_buffer) {
  // assert that warpSize == 32 and maximum number of threads per block is 1024
  __shared__ double shared_buffer[32];
  const int data_index = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const double pointwise_loss = data_index < num_data ?
    (USE_WEIGHT ? CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index]) * weights[data_index] :
                  CUDAPointWiseLossCalculator::LossOnPointCUDA(label[data_index], score[data_index])) :
                  0.0f;
  const double loss = ShuffleReduceSum<double>(pointwise_loss, shared_buffer, blockDim.x);
  if (threadIdx.x == 0) {
    cuda_sum_loss_buffer[blockIdx.x] = loss;
  }
}

__global__ void ReduceLossKernel(const double* cuda_sum_loss_buffer, const data_size_t num_blocks, double* out_loss) {
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
void CUDABinaryMetric<CUDAPointWiseLossCalculator>::LaunchEvalKernelInner(const double* score) const {
  const data_size_t num_blocks = (BinaryMetric<CUDAPointWiseLossCalculator>::num_data_ + EVAL_BLOCK_SIZE_BINARY_METRIC - 1) / EVAL_BLOCK_SIZE_BINARY_METRIC;
  if (cuda_weights_ == nullptr) {
    EvalKernel_BinaryPointWiseLoss<CUDAPointWiseLossCalculator, false><<<num_blocks, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_);
  } else {
    EvalKernel_BinaryPointWiseLoss<CUDAPointWiseLossCalculator, true><<<num_blocks, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(
      score, cuda_label_, cuda_weights_,
      this->num_data_,
      this->sum_weights_,
      cuda_sum_loss_buffer_);
  }
  ReduceLossKernel<<<1, EVAL_BLOCK_SIZE_BINARY_METRIC>>>(cuda_sum_loss_buffer_, num_blocks, cuda_sum_loss_);
}

template <>
void CUDABinaryMetric<CUDABinaryLoglossMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

template <>
void CUDABinaryMetric<CUDABinaryErrorMetric>::LaunchEvalKernel(const double* score) const {
  LaunchEvalKernelInner(score);
}

void CUDAAUCMetric::LaunchEvalKernel(const double* score) const {
  int num_pos = 0;
  for (int data_index = 0; data_index < num_data_; ++data_index) {
    num_pos += static_cast<int>(label_[data_index]);
  }
  Log::Warning("sum_pos = %d", num_pos);
  BitonicArgSortGlobal<double, data_size_t, false>(score, cuda_indices_buffer_, static_cast<size_t>(num_data_));
  std::vector<data_size_t> host_sorted_indices(num_data_, 0);
  std::vector<double> host_score(num_data_, 0.0f);
  CopyFromCUDADeviceToHostOuter<data_size_t>(host_sorted_indices.data(), cuda_indices_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(host_score.data(), score, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  //Log::Warning("host_sorted_indices[%d] = %d, host_score[%d] = %f", 0, host_sorted_indices[0], host_score[host_sorted_indices[0]]);
  for (int i = 0; i < num_data_ - 1; ++i) {
    //Log::Warning("host_sorted_indices[%d] = %d, host_score[%d] = %f", i + 1, host_sorted_indices[i + 1], host_sorted_indices[i + 1], host_score[host_sorted_indices[i + 1]]);
    CHECK_GE(host_score[host_sorted_indices[i]], host_score[host_sorted_indices[i + 1]]);
  }
  SetCUDAMemoryOuter<double>(cuda_block_sum_pos_buffer_, 0, 1, __FILE__, __LINE__);
  if (cuda_weights_ == nullptr) {
    GlobalGenAUCPosNegSum<false, true>(cuda_label_, cuda_weights_, cuda_indices_buffer_, cuda_sum_pos_buffer_, cuda_block_sum_pos_buffer_, num_data_);
    std::vector<double> host_cuda_sum_pos_buffer(num_data_);
    CopyFromCUDADeviceToHostOuter<double>(host_cuda_sum_pos_buffer.data(), cuda_sum_pos_buffer_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
    double cur_sum_pos = 0.0f;
    for (data_size_t data_index = 0; data_index < num_data_; ++data_index) {
      cur_sum_pos += static_cast<double>(label_[host_sorted_indices[data_index]] > 0);
      CHECK_EQ(cur_sum_pos, host_cuda_sum_pos_buffer[data_index]);
    }
  } else {
    GlobalGenAUCPosNegSum<true, false>(cuda_label_, cuda_weights_, cuda_indices_buffer_, cuda_sum_pos_buffer_, cuda_block_sum_pos_buffer_, num_data_);
    Log::Fatal("CUDA AUC with weights is not supported.");
  }
  GloblGenAUCMark(score, cuda_indices_buffer_, cuda_threshold_mark_, cuda_block_threshold_mark_buffer_, cuda_block_mark_first_zero_, num_data_);
  std::vector<data_size_t> host_threshold_mask(num_data_, 0);
  CopyFromCUDADeviceToHostOuter<data_size_t>(host_threshold_mask.data(), cuda_threshold_mark_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  for (int i = 0; i < num_data_; ++i) {
    //Log::Warning("host_threshold_mask[%d] = %d", i, host_threshold_mask[i]);
    const bool is_valid = i == 0 || host_threshold_mask[i] == 0 || (host_threshold_mask[i] == host_threshold_mask[i - 1] + 1);
    if (!is_valid) {
      Log::Warning("host_threshold_mask[%d] = %d, host_threshold_mask[%d] = %d", i, host_threshold_mask[i], i - 1, host_threshold_mask[i - 1]);
    }
    CHECK(is_valid);
    if (i > 0) {
      const bool should_increase = (host_score[host_sorted_indices[i]] == host_score[host_sorted_indices[i - 1]]);
      if (should_increase) {
        CHECK_EQ(host_threshold_mask[i], host_threshold_mask[i - 1] + 1);
      } else {
        CHECK_EQ(host_threshold_mask[i], 0);
      }
    }
  }
  GlobalCalcAUC(cuda_sum_pos_buffer_, cuda_threshold_mark_, num_data_, cuda_block_sum_pos_buffer_);
}

void CUDAAUCMetric::TestCUDABitonicSortForQueryItems() const {
  int num_queries = 1000;
  std::vector<int> items_per_query(num_queries + 1, 0);
  std::vector<double> item_scores;
  const int max_item_per_query = 5000;
  std::vector<double> num_item_probs(max_item_per_query, 1.0f / max_item_per_query);
  std::discrete_distribution<int> num_item_distribution(num_item_probs.begin(), num_item_probs.end());
  std::uniform_real_distribution<double> score_dist;
  const int num_threads = OMP_NUM_THREADS();
  std::vector<std::mt19937> thread_random_engines(num_threads);
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    thread_random_engines[thread_index] = std::mt19937(thread_index);
  }
  int num_total_items = 0;
  #pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:num_total_items)
  for (int query_index = 0; query_index < num_queries; ++query_index) {
    const int thread_index = omp_get_thread_num();
    items_per_query[query_index + 1] = num_item_distribution(thread_random_engines[thread_index]);
    num_total_items += items_per_query[query_index + 1];
  }
  for (int query_index = 0; query_index < num_queries; ++query_index) {
    items_per_query[query_index + 1] += items_per_query[query_index];
  }
  item_scores.resize(num_total_items, 0.0f);
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int item_index = 0; item_index < num_total_items; ++item_index) {
    const int thread_index = omp_get_thread_num();
    item_scores[item_index] = score_dist(thread_random_engines[thread_index]);
  }
  double* cuda_score = nullptr;
  data_size_t* cuda_query_boundaries = nullptr;
  data_size_t* cuda_out_indices = nullptr;
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_score, item_scores.data(), item_scores.size(), __FILE__, __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_query_boundaries, items_per_query.data(), items_per_query.size(), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_out_indices, item_scores.size(), __FILE__, __LINE__);
  const auto start = std::chrono::steady_clock::now();
  BitonicArgSortItemsGlobal(cuda_score, num_queries, cuda_query_boundaries, cuda_out_indices);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  const auto end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> duration = static_cast<std::chrono::duration<double>>(end - start);
  Log::Warning("bitonic arg sort items global time = %f", duration.count());
  std::vector<int> sorted_item_indices(item_scores.size());
  CopyFromCUDADeviceToHostOuter<int>(sorted_item_indices.data(), cuda_out_indices, item_scores.size(), __FILE__, __LINE__);
  std::vector<int> host_sorted_item_indices(item_scores.size());
  PrintLastCUDAErrorOuter(__FILE__, __LINE__);
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int i = 0; i < num_queries; ++i) {
    const int query_start = items_per_query[i];
    const int query_end = items_per_query[i + 1];
    for (int j = query_start; j < query_end; ++j) {
      host_sorted_item_indices[j] = j - query_start;
    }
    std::sort(host_sorted_item_indices.data() + query_start, host_sorted_item_indices.data() + query_end, [&item_scores, query_start] (int a, int b) {
      return item_scores[query_start + a] > item_scores[query_start + b];
    });
  }
  for (int query_index = 0; query_index < num_queries; ++query_index) {
    const int query_start = items_per_query[query_index];
    const int query_end = items_per_query[query_index + 1];
    for (int item_index = query_start; item_index < query_end; ++item_index) {
      const double cuda_item_score = item_scores[query_start + sorted_item_indices[item_index]];
      const double host_item_score = item_scores[query_start + host_sorted_item_indices[item_index]];
      if (cuda_item_score != host_item_score) {
        Log::Warning("item_index = %d, query_start = %d, cuda_item_score = %f, host_item_score = %f, sorted_item_indices = %d",
                      item_index, query_start, cuda_item_score, host_item_score, sorted_item_indices[item_index]);
      }
    }
  }
  Log::Warning("bitonic argsort items test pass");
  std::vector<double> copied_scores = item_scores;
  const std::vector<double> const_copied_scores = item_scores;
  std::vector<data_size_t> host_indices(item_scores.size());
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (data_size_t data_index = 0; data_index < static_cast<data_size_t>(host_indices.size()); ++data_index) {
    host_indices[data_index] = data_index;
  }
  data_size_t* cuda_indices = nullptr;
  AllocateCUDAMemoryOuter<data_size_t>(&cuda_indices, item_scores.size(), __FILE__, __LINE__);
  BitonicArgSortGlobal<double, data_size_t, true>(cuda_score, cuda_indices, host_indices.size());
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  PrintLastCUDAErrorOuter(__FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<data_size_t>(host_indices.data(), cuda_indices, host_indices.size(), __FILE__, __LINE__);
  std::vector<double> host_cuda_score(item_scores.size());
  CopyFromCUDADeviceToHostOuter<double>(host_cuda_score.data(), cuda_score, item_scores.size(), __FILE__, __LINE__);
  for (size_t i = 0; i < host_indices.size() - 1; ++i) {
    const data_size_t index_1 = host_indices[i];
    const data_size_t index_2 = host_indices[i + 1];
    const double score_1 = host_cuda_score[index_1];
    const double score_2 = host_cuda_score[index_2];
    if (score_1 > score_2) {
      Log::Warning("error in argsort score_1 = %.20f, score_2 = %.20f", score_1, score_2);
      break;
    }
  }
  std::vector<double> host_sort_item_scores = item_scores;
  std::sort(host_sort_item_scores.begin(), host_sort_item_scores.end());
  double* new_cuda_score = nullptr;
  InitCUDAMemoryFromHostMemoryOuter<double>(&new_cuda_score, item_scores.data(), item_scores.size(), __FILE__, __LINE__);
  BitonicSortGlobal<double, true>(cuda_score, item_scores.size());
  BitonicSortGlobal<double, true>(new_cuda_score, item_scores.size());
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  PrintLastCUDAErrorOuter(__FILE__, __LINE__);
  CopyFromCUDADeviceToHostOuter<double>(item_scores.data(), cuda_score, item_scores.size(), __FILE__, __LINE__);
  std::vector<double> cuda_score_sorted(item_scores.size());
  CopyFromCUDADeviceToHostOuter<double>(cuda_score_sorted.data(), new_cuda_score, cuda_score_sorted.size(), __FILE__, __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  for (size_t i = 0; i < item_scores.size() - 1; ++i) {
    const double score_1 = item_scores[i];
    const double score_2 = item_scores[i + 1];
    if (score_1 > score_2) {
      Log::Warning("error in sort score_1 = %.20f, score_2 = %.20f", score_1, score_2);
      break;
    }
  }
  for (size_t i = 0; i < cuda_score_sorted.size() - 1; ++i) {
    const double score_1 = cuda_score_sorted[i];
    const double score_2 = cuda_score_sorted[i + 1];
    if (score_1 > score_2) {
      Log::Warning("error in new sort score_1 = %.20f, score_2 = %.20f", score_1, score_2);
      break;
    }
  }
  {
    const int num_test_int = 2508113;
    std::vector<int> random_ints(num_test_int, 0);
    Threading::For<int>(0, num_test_int, 512, [&random_ints, &thread_random_engines, &num_item_distribution] (int thread_index, int start, int end) {
      for (int data_index = start; data_index < end; ++data_index) {
        random_ints[data_index] = num_item_distribution(thread_random_engines[thread_index]);
      }
    });
    int* cuda_rand_ints = nullptr;
    InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_rand_ints, random_ints.data(), random_ints.size(), __FILE__, __LINE__);
    BitonicSortGlobal<int, true>(cuda_rand_ints, random_ints.size());
    CopyFromCUDADeviceToHostOuter<int>(random_ints.data(), cuda_rand_ints, random_ints.size(), __FILE__, __LINE__);
    /*const int segment_length = 1024;
    const int num_segments = (num_test_int + segment_length - 1) / segment_length;
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
      const int segment_start = segment_index * segment_length;
      const int segment_end = std::min(segment_start + segment_length, num_test_int);
      const bool ascending = (segment_index % 2 == 0);
      for (int data_index = segment_start; data_index < segment_end - 1; ++data_index) {
        const int value_1 = random_ints[data_index];
        const int value_2 = random_ints[data_index + 1];
        if (ascending) {
          if (value_1 > value_2) {
            Log::Warning("data_index = %d error in first stage of bitonic sort value_1 = %d, value_2 = %d", data_index, value_1, value_2);
          }
        } else {
          if (value_1 < value_2) {
            Log::Warning("data_index = %d error in first stage of bitonic sort value_1 = %d, value_2 = %d", data_index, value_1, value_2);
          }
        }
      } 
    }*/
    for (int i = 0; i < num_test_int - 1; ++i) {
      const int value_1 = random_ints[i];
      const int value_2 = random_ints[i + 1];
      if (value_1 > value_2) {
        Log::Warning("error in int value_1 = %d, value_2 = %d", value_1, value_2);
        break;
      }
    }
    Log::Warning("test int global sort passed");
  }
  {
    const int num_test_double = 2508113;
    std::vector<double> random_double(num_test_double, 0);
    Threading::For<int>(0, num_test_double, 512, [&random_double, &thread_random_engines, &score_dist] (int thread_index, int start, int end) {
      for (int data_index = start; data_index < end; ++data_index) {
        if (data_index % 10 == 0) {
          random_double[data_index] = random_double[data_index / 10];
        } else {
          random_double[data_index] = score_dist(thread_random_engines[thread_index]);
        }
      }
    });
    double* cuda_rand_double = nullptr;
    for (int i = 0; i < num_test_double; ++i) {
      CHECK_GE(random_double[i], 0.0f);
      CHECK_LE(random_double[i], 1.0f);
    }
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_rand_double, random_double.data(), random_double.size(), __FILE__, __LINE__);
    BitonicSortGlobal<double, true>(cuda_rand_double, random_double.size());
    CopyFromCUDADeviceToHostOuter<double>(random_double.data(), cuda_rand_double, random_double.size(), __FILE__, __LINE__);
    /*const int segment_length = 1024;
    const int num_segments = (num_test_double + segment_length - 1) / segment_length;
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
      const int segment_start = segment_index * segment_length;
      const int segment_end = std::min(segment_start + segment_length, num_test_double);
      const bool ascending = (segment_index % 2 == 0);
      for (int data_index = segment_start; data_index < segment_end - 1; ++data_index) {
        const double value_1 = random_double[data_index];
        const double value_2 = random_double[data_index + 1];
        if (ascending) {
          if (value_1 > value_2) {
            Log::Warning("data_index = %d error in first stage of bitonic sort value_1 = %f, value_2 = %f", data_index, value_1, value_2);
          }
        } else {
          if (value_1 < value_2) {
            Log::Warning("data_index = %d error in first stage of bitonic sort value_1 = %f, value_2 = %f", data_index, value_1, value_2);
          }
        }
      } 
    }*/
    for (int i = 0; i < num_test_double - 1; ++i) {
      const double value_1 = random_double[i];
      const double value_2 = random_double[i + 1];
      if (value_1 > value_2) {
        Log::Warning("error in double value_1 = %.20f, value_2 = %.20f", value_1, value_2);
        break;
      }
    }
    for (int i = 0; i < num_test_double; ++i) {
      CHECK_GE(random_double[i], 0.0f);
      CHECK_LE(random_double[i], 1.0f);
    }
    Log::Warning("test doublecd  global sort passed");
  }
  {
    double* cuda_copied_scores = nullptr;
    std::vector<double> host_copied_scores = const_copied_scores;
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_copied_scores, copied_scores.data(), copied_scores.size(), __FILE__, __LINE__);
    BitonicSortGlobal<double, false>(cuda_copied_scores, copied_scores.size());
    std::sort(host_copied_scores.begin(), host_copied_scores.end(), [] (double a, double b) { return a > b; });
    CopyFromCUDADeviceToHostOuter<double>(copied_scores.data(), cuda_copied_scores, copied_scores.size(), __FILE__, __LINE__);
    for (int i = 0; i < copied_scores.size(); ++i) {
      const double host_value = host_copied_scores[i];
      const double cuda_value = copied_scores[i];
      const double host_sort_value = host_sort_item_scores[i];
      if (host_value != cuda_value) {
        Log::Warning("error in sort item scores %f vs %f", host_value, cuda_value);
        break;
      }
    }
  }
  {
    double* cuda_copied_scores = nullptr;
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_copied_scores, const_copied_scores.data(), const_copied_scores.size(), __FILE__, __LINE__);
    data_size_t* cuda_indices = nullptr;
    AllocateCUDAMemoryOuter<data_size_t>(&cuda_indices, const_copied_scores.size(), __FILE__, __LINE__);
    BitonicArgSortGlobal<double, data_size_t, false>(cuda_copied_scores, cuda_indices, const_copied_scores.size());
    std::vector<data_size_t> host_indices(const_copied_scores.size());
    const int num_threads = OMP_NUM_THREADS();
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (data_size_t i = 0; i < static_cast<data_size_t>(const_copied_scores.size()); ++i) {
      host_indices[i] = i;
    }
    std::sort(host_indices.begin(), host_indices.end(),
      [&const_copied_scores] (data_size_t a, data_size_t b) { return const_copied_scores[a] > const_copied_scores[b]; });
    std::vector<data_size_t> host_cuda_indices(const_copied_scores.size());
    SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
    CopyFromCUDADeviceToHostOuter<data_size_t>(host_cuda_indices.data(), cuda_indices, const_copied_scores.size(), __FILE__, __LINE__);
    /*const int segment_length = 1024;
    const int num_segments = (static_cast<int>(const_copied_scores.size()) + segment_length - 1) / segment_length;
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
      const int segment_start = segment_index * segment_length;
      const int segment_end = std::min(segment_start + segment_length, static_cast<int>(const_copied_scores.size()));
      const bool ascending = (segment_index % 2 == 1);
      for (data_size_t data_index = segment_start; data_index < segment_end - 1; ++data_index) {
        const data_size_t index_1 = host_cuda_indices[data_index];
        const data_size_t index_2 = host_cuda_indices[data_index + 1];
        const double value_1 = const_copied_scores[index_1];
        const double value_2 = const_copied_scores[index_2];
        if (ascending) {
          if (value_1 > value_2) {
            Log::Warning("error ascending = %d, index_1 = %d, index_2 = %d, value_1 = %f, value_2 = %f", static_cast<int>(ascending),
              index_1, index_2, value_1, value_2);
          }
        } else {
          if (value_1 < value_2) {
            Log::Warning("error ascending = %d, index_1 = %d, index_2 = %d, value_1 = %f, value_2 = %f", static_cast<int>(ascending),
              index_1, index_2, value_1, value_2);
          }
        }
      }
    }*/
    BitonicSortGlobal<double, false>(cuda_copied_scores, const_copied_scores.size());
    std::vector<double> host_cuda_sort_scores(const_copied_scores.size());
    CopyFromCUDADeviceToHostOuter<double>(host_cuda_sort_scores.data(), cuda_copied_scores, const_copied_scores.size(), __FILE__, __LINE__);
    for (int i = 0; i < static_cast<int>(const_copied_scores.size()); ++i) {
      const double sort_score = host_cuda_sort_scores[i];
      const double argsort_score = const_copied_scores[host_cuda_indices[i]];
      if (sort_score != argsort_score) {
        Log::Warning("error sort_score = %.20f, argsort_score = %.20f", sort_score, argsort_score);
      }
      CHECK_EQ(sort_score, argsort_score);
    }
    for (int i = 0; i < copied_scores.size(); ++i) {
      const data_size_t host_index = host_indices[i];
      const data_size_t cuda_index = host_cuda_indices[i];
      const double host_value = const_copied_scores[host_index];
      const double cuda_value = const_copied_scores[cuda_index];
      if (host_index != cuda_index) {
        Log::Warning("i = %d error in arg sort scores %d vs %d, host_value = %f, cuda_value = %f", i, host_index, cuda_index, host_value, cuda_value);
        break;
      }
    }
  }
  {
    std::vector<double> item_argsort_scores = const_copied_scores;
    std::vector<data_size_t> item_argsort_query_boundaries{0, static_cast<data_size_t>(item_scores.size())};
    data_size_t* cuda_query_boundaries = nullptr;
    InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_query_boundaries, item_argsort_query_boundaries.data(), item_argsort_query_boundaries.size(), __FILE__, __LINE__);
    double* cuda_argsort_scores = nullptr;
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_argsort_scores, item_argsort_scores.data(), item_argsort_scores.size(), __FILE__, __LINE__);
    data_size_t* out_indices = nullptr;
    AllocateCUDAMemoryOuter<data_size_t>(&out_indices, item_argsort_scores.size(), __FILE__, __LINE__);
    BitonicArgSortItemsGlobal(cuda_argsort_scores, 1, cuda_query_boundaries, out_indices);
    std::vector<data_size_t> cuda_sort_indices(item_argsort_scores.size());
    CopyFromCUDADeviceToHostOuter<data_size_t>(cuda_sort_indices.data(), out_indices, cuda_sort_indices.size(), __FILE__, __LINE__);

    double* cuda_sort_scores = nullptr;
    InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_sort_scores, const_copied_scores.data(), const_copied_scores.size(), __FILE__, __LINE__);
    BitonicSortGlobal<double, false>(cuda_sort_scores, const_copied_scores.size());
    std::vector<double> cuda_sort_scores_to_host(const_copied_scores.size());
    CopyFromCUDADeviceToHostOuter<double>(cuda_sort_scores_to_host.data(), cuda_sort_scores, const_copied_scores.size(), __FILE__, __LINE__);

    std::vector<double> host_sort_result = const_copied_scores;
    Log::Warning("num scores = %d", const_copied_scores.size());
    std::sort(host_sort_result.begin(), host_sort_result.end(), [] (const double a, const double b) { return a > b; });
    for (data_size_t i = 0; i < static_cast<data_size_t>(const_copied_scores.size()); ++i) {
      CHECK_EQ(host_sort_result[i], const_copied_scores[cuda_sort_indices[i]]);
    }
    Log::Warning("bitonic item arg sort items success");
    for (data_size_t i = 0; i < static_cast<data_size_t>(const_copied_scores.size()); ++i) {
      CHECK_EQ(host_sort_result[i], cuda_sort_scores_to_host[i]);
    }
    Log::Warning("bitonic sort items success");
  }
}

}  // namespace LightGBM
