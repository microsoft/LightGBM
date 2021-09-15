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
#include <LightGBM/objective_function.h>

namespace LightGBM {

NewCUDATreeLearner::NewCUDATreeLearner(const Config* config): SerialTreeLearner(config) {}

NewCUDATreeLearner::~NewCUDATreeLearner() {}

void NewCUDATreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  // use the first gpu by now
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  const int gpu_device_id = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
  CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_device_id));
  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->Init();
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->Init();
  cuda_histogram_constructor_.reset(new CUDAHistogramConstructor(train_data_, this->config_->num_leaves, num_threads_,
    share_state_->feature_hist_offsets(),
    config_->min_data_in_leaf, config_->min_sum_hessian_in_leaf, config_->gpu_device_id));
  cuda_histogram_constructor_->Init(train_data_, share_state_.get());

  cuda_data_partition_.reset(new CUDADataPartition(
    train_data_, share_state_->feature_hist_offsets().back(), this->config_->num_leaves, num_threads_,
    cuda_histogram_constructor_->cuda_hist_pointer()));
  cuda_data_partition_->Init();
  cuda_best_split_finder_.reset(new CUDABestSplitFinder(cuda_histogram_constructor_->cuda_hist(),
    train_data_, this->share_state_->feature_hist_offsets(), config_));
  cuda_best_split_finder_->Init();

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  AllocateCUDAMemoryOuter<double>(&cuda_add_train_score_, static_cast<size_t>(num_data_), __FILE__, __LINE__);
  add_train_score_.resize(num_data_, 0.0f);
}

void NewCUDATreeLearner::BeforeTrain() {
  cuda_data_partition_->BeforeTrain();
  cuda_smaller_leaf_splits_->InitValues(
    gradients_,
    hessians_,
    cuda_data_partition_->cuda_data_indices(),
    cuda_data_partition_->root_num_data(),
    cuda_histogram_constructor_->cuda_hist_pointer(),
    &leaf_sum_hessians_[0]);
  leaf_num_data_[0] = cuda_data_partition_->root_num_data();
  cuda_larger_leaf_splits_->InitValues();
  cuda_histogram_constructor_->BeforeTrain(gradients_, hessians_);
  cuda_best_split_finder_->BeforeTrain();
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

void NewCUDATreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  CHECK(tree->is_cuda_tree());
  const CUDATree* cuda_tree = reinterpret_cast<const CUDATree*>(tree);
  cuda_data_partition_->UpdateTrainScore(cuda_tree->cuda_leaf_value(), cuda_add_train_score_);
  CopyFromCUDADeviceToHostOuter<double>(add_train_score_.data(),
    cuda_add_train_score_, static_cast<size_t>(cuda_data_partition_->root_num_data()), __FILE__, __LINE__);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t data_index = 0; data_index < cuda_data_partition_->root_num_data(); ++data_index) {
    out_score[data_index] += add_train_score_[data_index];
  }
  OMP_THROW_EX();
}

Tree* NewCUDATreeLearner::Train(const score_t* gradients,
  const score_t* hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  global_timer.Start("NewCUDATreeLearner::BeforeTrain");
  BeforeTrain();
  global_timer.Stop("NewCUDATreeLearner::BeforeTrain");
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features, config_->linear_tree, config_->gpu_device_id));
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    global_timer.Start("NewCUDATreeLearner::ConstructHistogramForLeaf");
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    cuda_histogram_constructor_->ConstructHistogramForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf);
    global_timer.Stop("NewCUDATreeLearner::ConstructHistogramForLeaf");
    global_timer.Start("NewCUDATreeLearner::FindBestSplitsForLeaf");
    cuda_best_split_finder_->FindBestSplitsForLeaf(
      cuda_smaller_leaf_splits_->GetCUDAStruct(),
      cuda_larger_leaf_splits_->GetCUDAStruct(),
      smaller_leaf_index_, larger_leaf_index_,
      num_data_in_smaller_leaf, num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf, sum_hessians_in_larger_leaf);
    global_timer.Stop("NewCUDATreeLearner::FindBestSplitsForLeaf");
    global_timer.Start("NewCUDATreeLearner::FindBestFromAllSplits");
    const CUDASplitInfo* best_split_info = nullptr;
    if (larger_leaf_index_ >= 0) {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        &leaf_best_split_feature_[larger_leaf_index_],
        &leaf_best_split_threshold_[larger_leaf_index_],
        &leaf_best_split_default_left_[larger_leaf_index_],
        &best_leaf_index_);
    } else {
      best_split_info = cuda_best_split_finder_->FindBestFromAllSplits(
        tree->num_leaves(),
        smaller_leaf_index_,
        larger_leaf_index_,
        &leaf_best_split_feature_[smaller_leaf_index_],
        &leaf_best_split_threshold_[smaller_leaf_index_],
        &leaf_best_split_default_left_[smaller_leaf_index_],
        nullptr,
        nullptr,
        nullptr,
        &best_leaf_index_);
    }
    global_timer.Stop("NewCUDATreeLearner::FindBestFromAllSplits");

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    global_timer.Start("NewCUDATreeLearner::Split");
    int right_leaf_index = tree->Split(best_leaf_index_,
                                       train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
                                       train_data_->RealThreshold(leaf_best_split_feature_[best_leaf_index_],
                                       leaf_best_split_threshold_[best_leaf_index_]),
                                       train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
                                       best_split_info);

    cuda_data_partition_->Split(best_split_info,
                                best_leaf_index_,
                                right_leaf_index,
                                leaf_best_split_feature_[best_leaf_index_],
                                leaf_best_split_threshold_[best_leaf_index_],
                                leaf_best_split_default_left_[best_leaf_index_],
                                leaf_num_data_[best_leaf_index_],
                                leaf_data_start_[best_leaf_index_],
                                cuda_smaller_leaf_splits_->GetCUDAStructRef(),
                                cuda_larger_leaf_splits_->GetCUDAStructRef(),
                                &leaf_num_data_[best_leaf_index_],
                                &leaf_num_data_[right_leaf_index],
                                &leaf_data_start_[best_leaf_index_],
                                &leaf_data_start_[right_leaf_index],
                                &leaf_sum_hessians_[best_leaf_index_],
                                &leaf_sum_hessians_[right_leaf_index]);
    smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
    larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);
    global_timer.Stop("NewCUDATreeLearner::Split");
  }
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  tree->ToHost();
  return tree.release();
}

void NewCUDATreeLearner::ResetTrainingData(const Dataset* /*train_data*/,
                         bool /*is_constant_hessian*/) {}

void NewCUDATreeLearner::SetBaggingData(const Dataset* subset,
  const data_size_t* used_indices, data_size_t num_data) {
  cuda_data_partition_->SetUsedDataIndices(used_indices, num_data);
  if (subset == nullptr) {
    
  }
}

void NewCUDATreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                         data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  CHECK(tree->is_cuda_tree());
  CUDATree* cuda_tree = reinterpret_cast<CUDATree*>(tree);
  SerialTreeLearner::RenewTreeOutput(tree, obj, residual_getter, total_num_data, bag_indices, bag_cnt);
  cuda_tree->SyncLeafOutputFromHostToCUDA();
}

void NewCUDATreeLearner::AfterTrain() {
  cuda_data_partition_->SetUseBagging(false);
}

}  // namespace LightGBM

#endif  // USE_CUDA
