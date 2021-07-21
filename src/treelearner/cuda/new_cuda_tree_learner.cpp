/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "new_cuda_tree_learner.hpp"

#include <LightGBM/cuda/cuda_tree.hpp>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/feature_group.h>

namespace LightGBM {

NewCUDATreeLearner::NewCUDATreeLearner(const Config* config): SerialTreeLearner(config) {}

NewCUDATreeLearner::~NewCUDATreeLearner() {}

void NewCUDATreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // use the first gpu by now
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  const label_t* labels = train_data->metadata().label();
  cuda_centralized_info_.reset(new CUDACentralizedInfo(num_data_, this->config_->num_leaves, num_features_));
  cuda_centralized_info_->Init(labels, train_data_);
  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_, 0, cuda_centralized_info_->cuda_num_data()));
  cuda_smaller_leaf_splits_->Init();
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_, -1, cuda_centralized_info_->cuda_num_data()));
  cuda_larger_leaf_splits_->Init();
  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, this->config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf));
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, share_state_->feature_hist_offsets().back(), this->config_->num_leaves, num_threads_,
    cuda_centralized_info_->cuda_num_data(),
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->Init();
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), this->config_->num_leaves,
    this->config_->lambda_l1, this->config_->lambda_l2, this->config_->min_data_in_leaf,
    this->config_->min_sum_hessian_in_leaf, this->config_->min_gain_to_split,
    cuda_centralized_info_->cuda_num_features()));
  cuda_best_split_finder_->Init();

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);
}

void NewCUDATreeLearner::BeforeTrain() {
  cuda_data_partition_->BeforeTrain(nullptr);
  global_timer.Start("CUDACentralizedInfo::BeforeTrain");
  global_timer.Stop("CUDACentralizedInfo::BeforeTrain");
  cuda_smaller_leaf_splits_->InitValues(
    gradients_,
    hessians_,
    cuda_data_partition_->cuda_data_indices(),
    cuda_histogram_constructor_->cuda_hist_pointer(),
    &leaf_sum_hessians_[0]);
  cuda_larger_leaf_splits_->InitValues();
  cuda_histogram_constructor_->BeforeTrain(gradients_, hessians_);
  cuda_best_split_finder_->BeforeTrain();
  leaf_num_data_[0] = num_data_;
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

void NewCUDATreeLearner::FindBestSplits(const Tree* /*tree*/) {}

void NewCUDATreeLearner::ConstructHistograms(const std::vector<int8_t>& /*is_feature_used*/,
  bool /*use_subtract*/) {}

void NewCUDATreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t>& /*is_feature_used*/,
  bool /*use_subtract*/, const Tree* /*tree*/) {}

void NewCUDATreeLearner::Split(Tree* /*tree*/, int /*best_leaf*/,
  int* /*left_leaf*/, int* /*right_leaf*/) {}

void NewCUDATreeLearner::AddPredictionToScore(const Tree* /*tree*/, double* out_score) const {
  cuda_data_partition_->UpdateTrainScore(config_->learning_rate, out_score);
}

Tree* NewCUDATreeLearner::BuildTree(const int num_leaves) {
  std::unique_ptr<Tree> tree(new Tree(config_->num_leaves, false, false));
  std::vector<int> leaf_index(config_->num_leaves);
  std::vector<int> inner_feature_index(config_->num_leaves);
  std::vector<uint32_t> threshold(config_->num_leaves);
  std::vector<double> left_output(config_->num_leaves);
  std::vector<double> right_output(config_->num_leaves);
  std::vector<data_size_t> left_count(config_->num_leaves);
  std::vector<data_size_t> right_count(config_->num_leaves);
  std::vector<double> left_sum_hessian(config_->num_leaves);
  std::vector<double> right_sum_hessian(config_->num_leaves);
  std::vector<double> gain(config_->num_leaves);
  std::vector<uint8_t> default_left(config_->num_leaves);
  CopyFromCUDADeviceToHost<int>(leaf_index.data(), cuda_data_partition_->tree_split_leaf_index(), config_->num_leaves);
  CopyFromCUDADeviceToHost<int>(inner_feature_index.data(), cuda_data_partition_->tree_inner_feature_index(), config_->num_leaves);
  CopyFromCUDADeviceToHost<uint32_t>(threshold.data(), cuda_data_partition_->tree_threshold(), config_->num_leaves);
  CopyFromCUDADeviceToHost<double>(left_output.data(), cuda_data_partition_->tree_left_output(), config_->num_leaves);
  CopyFromCUDADeviceToHost<double>(right_output.data(), cuda_data_partition_->tree_right_output(), config_->num_leaves);
  CopyFromCUDADeviceToHost<data_size_t>(left_count.data(), cuda_data_partition_->tree_left_count(), config_->num_leaves);
  CopyFromCUDADeviceToHost<data_size_t>(right_count.data(), cuda_data_partition_->tree_right_count(), config_->num_leaves);
  CopyFromCUDADeviceToHost<double>(left_sum_hessian.data(), cuda_data_partition_->tree_left_sum_hessian(), config_->num_leaves);
  CopyFromCUDADeviceToHost<double>(right_sum_hessian.data(), cuda_data_partition_->tree_right_sum_hessian(), config_->num_leaves);
  CopyFromCUDADeviceToHost<double>(gain.data(), cuda_data_partition_->tree_gain(), config_->num_leaves);
  CopyFromCUDADeviceToHost<uint8_t>(default_left.data(), cuda_data_partition_->tree_default_left(), config_->num_leaves);
  for (int i = 0; i < num_leaves - 1; ++i) {
    tree->Split(
      leaf_index[i],
      inner_feature_index[i],
      train_data_->RealFeatureIndex(inner_feature_index[i]),
      threshold[i],
      train_data_->RealThreshold(inner_feature_index[i], threshold[i]),
      left_output[i],
      right_output[i],
      left_count[i],
      right_count[i],
      left_sum_hessian[i],
      right_sum_hessian[i],
      gain[i],
      train_data_->FeatureBinMapper(inner_feature_index[i])->missing_type(),
      static_cast<bool>(default_left[i]));
  }
  return tree.release();
}

Tree* NewCUDATreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  const auto start = std::chrono::steady_clock::now();
  auto before_train_start = std::chrono::steady_clock::now();
  global_timer.Start("NewCUDATreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("NewCUDATreeLearner::BeforeTrain");
  auto before_train_end = std::chrono::steady_clock::now();
  double construct_histogram_time = 0.0f;
  double find_best_split_time = 0.0f;
  double find_best_split_from_all_leaves_time = 0.0f;
  double split_data_indices_time = 0.0f;
  int num_leaves = 1;
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    global_timer.Start("NewCUDATreeLearner::ConstructHistogramForLeaf");
    auto start = std::chrono::steady_clock::now();
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      cuda_data_partition_->cuda_leaf_num_data(),
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf);
    auto end = std::chrono::steady_clock::now();
    auto duration = static_cast<std::chrono::duration<double>>(end - start);
    global_timer.Stop("NewCUDATreeLearner::ConstructHistogramForLeaf");
    construct_histogram_time += duration.count();
    global_timer.Start("NewCUDATreeLearner::FindBestSplitsForLeaf");
    start = std::chrono::steady_clock::now();
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      smaller_leaf_index_, larger_leaf_index_,
      num_data_in_smaller_leaf, num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf);
    end = std::chrono::steady_clock::now();
    duration = static_cast<std::chrono::duration<double>>(end - start);
    global_timer.Stop("NewCUDATreeLearner::FindBestSplitsForLeaf");
    find_best_split_time += duration.count();
    start = std::chrono::steady_clock::now();
    global_timer.Start("NewCUDATreeLearner::FindBestFromAllSplits");
    cuda_best_split_finder_->FindBestFromAllSplits(cuda_data_partition_->cuda_cur_num_leaves(),
      smaller_leaf_index_, larger_leaf_index_,
      &leaf_best_split_feature_, &leaf_best_split_threshold_, &leaf_best_split_default_left_, &best_leaf_index_);
    global_timer.Stop("NewCUDATreeLearner::FindBestFromAllSplits");
    end = std::chrono::steady_clock::now();
    duration = static_cast<std::chrono::duration<double>>(end - start);
    find_best_split_from_all_leaves_time += duration.count();

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    global_timer.Start("NewCUDATreeLearner::Split");
    start = std::chrono::steady_clock::now();
    cuda_data_partition_->Split(cuda_best_split_finder_->cuda_best_leaf(),
      cuda_best_split_finder_->cuda_leaf_best_split_feature(),
      cuda_best_split_finder_->cuda_leaf_best_split_info(),
      cuda_smaller_leaf_splits_->GetCUDAStructRef(),
      cuda_larger_leaf_splits_->GetCUDAStructRef(),
      &leaf_num_data_,
      &leaf_data_start_,
      &leaf_sum_hessians_,
      leaf_best_split_feature_,
      leaf_best_split_threshold_,
      leaf_best_split_default_left_,
      &smaller_leaf_index_,
      &larger_leaf_index_,
      best_leaf_index_,
      num_leaves);
    end = std::chrono::steady_clock::now();
    duration = static_cast<std::chrono::duration<double>>(end - start);
    global_timer.Stop("NewCUDATreeLearner::Split");
    split_data_indices_time += duration.count();
    ++num_leaves;
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  const auto end = std::chrono::steady_clock::now();
  const double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  const auto build_tree_start = std::chrono::steady_clock::now();
  std::unique_ptr<Tree> tree(BuildTree(num_leaves));
  const auto build_tree_end = std::chrono::steady_clock::now();
  const auto build_tre_duration = (static_cast<std::chrono::duration<double>>(build_tree_end - build_tree_start)).count();
  Log::Warning("Train time %f", duration);
  Log::Warning("before train time %f", static_cast<std::chrono::duration<double>>(before_train_end - before_train_start).count());
  Log::Warning("construct histogram time %f", construct_histogram_time);
  Log::Warning("find best split time %f", find_best_split_time);
  Log::Warning("find best split time from all leaves %f", find_best_split_from_all_leaves_time);
  Log::Warning("split data indices time %f", split_data_indices_time);
  Log::Warning("build tree time %f", build_tre_duration);
  tree.reset(new CUDATree(tree.get()));
  return tree.release();
}

void NewCUDATreeLearner::ResetTrainingData(const Dataset* /*train_data*/,
                         bool /*is_constant_hessian*/) {}

void NewCUDATreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* /*used_indices*/, data_size_t /*num_data*/) {}

}  // namespace LightGBM

#endif  // USE_CUDA
