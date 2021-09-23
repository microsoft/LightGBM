/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_score_updater.hpp"

namespace LightGBM {

CUDAScoreUpdater::CUDAScoreUpdater(const Dataset* data, int num_tree_per_iteration):
  ScoreUpdater(data, num_tree_per_iteration), num_threads_per_block_(1024) {
  num_data_ = data->num_data();
  int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration;
  InitCUDA(total_size);
  has_init_score_ = false;
  const double* init_score = data->metadata().init_score();
  // if exists initial score, will start from it
  if (init_score != nullptr) {
    if ((data->metadata().num_init_score() % num_data_) != 0
        || (data->metadata().num_init_score() / num_data_) != num_tree_per_iteration) {
      Log::Fatal("Number of class for initial score error");
    }
    has_init_score_ = true;
    CopyFromHostToCUDADeviceOuter(cuda_score_, init_score, total_size, __FILE__, __LINE__);
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDAScoreUpdater::InitCUDA(const size_t total_size) {
  AllocateCUDAMemoryOuter<double>(&cuda_score_, total_size, __FILE__, __LINE__);
}

CUDAScoreUpdater::~CUDAScoreUpdater() {
  DeallocateCUDAMemoryOuter<double>(&cuda_score_, __FILE__, __LINE__);
}

inline void CUDAScoreUpdater::AddScore(double val, int cur_tree_id) {
  Common::FunctionTimer fun_timer("CUDAScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  LaunchAddScoreConstantKernel(val, offset);
}

inline void CUDAScoreUpdater::AddScore(const Tree* tree, int cur_tree_id) {
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  tree->AddPredictionToScore(data_, num_data_, cuda_score_ + offset);
}

inline void CUDAScoreUpdater::AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  tree_learner->AddPredictionToScore(tree, cuda_score_ + offset);
  std::vector<double> class_train_score(num_data_, 0.0f);
  CopyFromCUDADeviceToHostOuter<double>(class_train_score.data(), cuda_score_ + offset, num_data_, __FILE__, __LINE__);
  const int num_threads = OMP_NUM_THREADS();
  std::vector<double> thread_max_abs_train_score(num_threads, 0.0f);
  Threading::For<data_size_t>(0, num_data_, 512,
    [&thread_max_abs_train_score, &class_train_score] (int thread_index, data_size_t start, data_size_t end) {
      for (data_size_t data_index = start; data_index < end; ++data_index) {
        if (std::fabs(class_train_score[data_index]) > std::fabs(thread_max_abs_train_score[thread_index])) {
          thread_max_abs_train_score[thread_index] = class_train_score[data_index];
        }
      }
    });
  double max_abs_train_score = 0.0f;
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    if (std::fabs(thread_max_abs_train_score[thread_index]) > std::fabs(max_abs_train_score)) {
      max_abs_train_score = thread_max_abs_train_score[thread_index];
    }
  }
  Log::Warning("class %d max_abs_train_score = %f", cur_tree_id, max_abs_train_score);
}

inline void CUDAScoreUpdater::AddScore(const Tree* tree, const data_size_t* data_indices,
                      data_size_t data_cnt, int cur_tree_id) {
  // TODO(shiyu1994): bagging is not supported yet
  Common::FunctionTimer fun_timer("ScoreUpdater::AddScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  tree->AddPredictionToScore(data_, data_indices, data_cnt, cuda_score_ + offset);
}

inline void CUDAScoreUpdater::MultiplyScore(double val, int cur_tree_id) {
  Common::FunctionTimer fun_timer("CUDAScoreUpdater::MultiplyScore", global_timer);
  const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
  LaunchMultiplyScoreConstantKernel(val, offset);
}

}  // namespace LightGBM
