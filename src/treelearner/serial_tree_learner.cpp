/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include "serial_tree_learner.h"

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/common.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>

#include "cost_effective_gradient_boosting.hpp"

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> init_train_time;
std::chrono::duration<double, std::milli> init_split_time;
std::chrono::duration<double, std::milli> hist_time;
std::chrono::duration<double, std::milli> find_split_time;
std::chrono::duration<double, std::milli> split_time;
std::chrono::duration<double, std::milli> ordered_bin_time;
std::chrono::duration<double, std::milli> refit_leaves_time;
#endif  // TIMETAG

double EPS = 1e-12;

SerialTreeLearner::SerialTreeLearner(const Config* config)
  :config_(config) {
  random_ = Random(config_->feature_fraction_seed);
  #pragma omp parallel
  #pragma omp master
  {
    num_threads_ = omp_get_num_threads();
  }
}

SerialTreeLearner::~SerialTreeLearner() {
  #ifdef TIMETAG
  Log::Info("SerialTreeLearner::init_train costs %f", init_train_time * 1e-3);
  Log::Info("SerialTreeLearner::init_split costs %f", init_split_time * 1e-3);
  Log::Info("SerialTreeLearner::hist_build costs %f", hist_time * 1e-3);
  Log::Info("SerialTreeLearner::find_split costs %f", find_split_time * 1e-3);
  Log::Info("SerialTreeLearner::split costs %f", split_time * 1e-3);
  Log::Info("SerialTreeLearner::ordered_bin costs %f", ordered_bin_time * 1e-3);
  Log::Info("SerialTreeLearner::refit_leaves costs %f", refit_leaves_time * 1e-3);
  #endif
}

void SerialTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  is_constant_hessian_ = is_constant_hessian;
  int max_cache_size = 0;
  // Get the max size of pool
  if (config_->histogram_pool_size <= 0) {
    max_cache_size = config_->num_leaves;
  } else {
    size_t total_histogram_size = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureNumBin(i);
    }
    max_cache_size = static_cast<int>(config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
  }
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, config_->num_leaves);

  histogram_pool_.DynamicChangeSize(train_data_, config_, max_cache_size, config_->num_leaves);
  // push split information for all leaves
  best_split_per_leaf_.resize(config_->num_leaves);

  // when the monotone precise mode is enabled, we need to store
  // more constraints; hence the constructors are different
  if (config_->monotone_precise_mode) {
    constraints_per_leaf_.resize(config_->num_leaves,
                                 Constraints(num_features_));
  } else {
    constraints_per_leaf_.resize(config_->num_leaves,
                                 Constraints());
  }
  splits_per_leaf_.resize(config_->num_leaves*train_data_->num_features());

  // get ordered bin
  train_data_->CreateOrderedBins(&ordered_bins_);

  // check existing for ordered bin
  for (int i = 0; i < static_cast<int>(ordered_bins_.size()); ++i) {
    if (ordered_bins_[i] != nullptr) {
      has_ordered_bin_ = true;
      break;
    }
  }
  // initialize splits for leaf
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, config_->num_leaves));
  is_feature_used_.resize(num_features_);
  valid_feature_indices_ = train_data_->ValidFeatureIndices();
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), static_cast<char>(0));
    ordered_bin_indices_.clear();
    for (int i = 0; i < static_cast<int>(ordered_bins_.size()); i++) {
      if (ordered_bins_[i] != nullptr) {
        ordered_bin_indices_.push_back(i);
      }
    }
  }
  Log::Info("Number of data: %d, number of used features: %d", num_data_, num_features_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }

  dummy_min_constraints.resize(num_threads_);
  min_constraints.resize(num_threads_);
  dummy_max_constraints.resize(num_threads_);
  max_constraints.resize(num_threads_);

  thresholds_min_constraints.resize(num_threads_);
  thresholds_max_constraints.resize(num_threads_);

  features.resize(num_threads_);
  is_in_right_split.resize(num_threads_);
  thresholds.resize(num_threads_);

  // the number 32 has no real meaning here, but during our experiments,
  // we found that the number of constraints per feature was well below 32, so by
  // allocating this space, we may save some time because we won't have to allocate it later
  int space_to_reserve = 32;
  if (!config_->monotone_precise_mode) {
    space_to_reserve = 1;
  }

  for (int i = 0; i < num_threads_; ++i) {
    dummy_min_constraints[i].reserve(space_to_reserve);
    min_constraints[i].reserve(space_to_reserve);
    dummy_max_constraints[i].reserve(space_to_reserve);
    max_constraints[i].reserve(space_to_reserve);

    thresholds_min_constraints[i].reserve(space_to_reserve);
    thresholds_max_constraints[i].reserve(space_to_reserve);

    if (!config_->monotone_constraints.empty()) {
      // the number 100 has no real meaning here, same as before
      features[i].reserve(std::max(100, config_->max_depth));
      is_in_right_split[i].reserve(std::max(100, config_->max_depth));
      thresholds[i].reserve(std::max(100, config_->max_depth));
    }

    InitializeConstraints(i);
  }
}

void SerialTreeLearner::ResetTrainingData(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK(num_features_ == train_data_->num_features());

  // get ordered bin
  train_data_->CreateOrderedBins(&ordered_bins_);

  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), static_cast<char>(0));
  }
  if (cegb_ != nullptr) {
    cegb_->Init();
  }
}

void SerialTreeLearner::ResetConfig(const Config* config) {
  if (config_->num_leaves != config->num_leaves) {
    config_ = config;
    int max_cache_size = 0;
    // Get the max size of pool
    if (config->histogram_pool_size <= 0) {
      max_cache_size = config_->num_leaves;
    } else {
      size_t total_histogram_size = 0;
      for (int i = 0; i < train_data_->num_features(); ++i) {
        total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureNumBin(i);
      }
      max_cache_size = static_cast<int>(config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
    }
    // at least need 2 leaves
    max_cache_size = std::max(2, max_cache_size);
    max_cache_size = std::min(max_cache_size, config_->num_leaves);
    histogram_pool_.DynamicChangeSize(train_data_, config_, max_cache_size, config_->num_leaves);

    // push split information for all leaves
    best_split_per_leaf_.resize(config_->num_leaves);
    data_partition_->ResetLeaves(config_->num_leaves);
  } else {
    config_ = config;
  }
  histogram_pool_.ResetConfig(config_);
  if (CostEfficientGradientBoosting::IsEnable(config_)) {
    cegb_.reset(new CostEfficientGradientBoosting(this));
    cegb_->Init();
  }
}

Tree* SerialTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian, const Json& forced_split_json) {
  gradients_ = gradients;
  hessians_ = hessians;
  is_constant_hessian_ = is_constant_hessian;
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // some initial works before training
  BeforeTrain();

  #ifdef TIMETAG
  init_train_time += std::chrono::steady_clock::now() - start_time;
  #endif

  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves));
  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;

  int init_splits = 0;
  bool aborted_last_force_split = false;
  if (!forced_split_json.is_null()) {
    init_splits = ForceSplits(tree.get(), forced_split_json, &left_leaf,
                              &right_leaf, &cur_depth, &aborted_last_force_split);
  }

  for (int split = init_splits; split < config_->num_leaves - 1; ++split) {
    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif
    // some initial works before finding best split
    if (!aborted_last_force_split && BeforeFindBestSplit(tree.get(), left_leaf, right_leaf)) {
      #ifdef TIMETAG
      init_split_time += std::chrono::steady_clock::now() - start_time;
      #endif
      // find best threshold for every feature
      FindBestSplits(tree.get());
    } else if (aborted_last_force_split) {
      aborted_last_force_split = false;
    }

    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
      break;
    }
    #ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
    #endif
    // split tree with best leaf
    Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
    #ifdef TIMETAG
    split_time += std::chrono::steady_clock::now() - start_time;
    #endif
    cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));
  }
    #ifdef TIMETAG
      start_time = std::chrono::steady_clock::now();
    #endif
      // when the monotone precise mode is enabled, some splits might unconstrain leaves in other branches
      // if these leaves are not split before the tree is being fully built, then it might be possible to
      // move their internal value (because they have been unconstrained) to achieve a better gain
      if (config_->monotone_precise_mode) {
        ReFitLeaves(tree.get());
      }
    #ifdef TIMETAG
      refit_leaves_time += std::chrono::steady_clock::now() - start_time;
    #endif
  Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
}

void SerialTreeLearner::ReFitLeaves(Tree *tree) {
  CHECK(data_partition_->num_leaves() >= tree->num_leaves());
  bool might_be_something_to_update = true;
  std::vector<double> sum_grad(tree->num_leaves(), 0.0f);
  std::vector<double> sum_hess(tree->num_leaves(), kEpsilon);
  OMP_INIT_EX();
  // first we need to compute gradients and hessians for each leaf
#pragma omp parallel for schedule(static)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    OMP_LOOP_EX_BEGIN();
    if (!tree->leaf_is_in_monotone_subtree(i)) {
      continue;
    }
    data_size_t cnt_leaf_data = 0;
    auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
    for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
      auto idx = tmp_idx[j];
      sum_grad[i] += gradients_[idx];
      sum_hess[i] += hessians_[idx];
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  while (might_be_something_to_update) {
    might_be_something_to_update = false;
    // this loop can't be multi-threaded easily because we could break
    // monotonicity in the tree
    for (int i = 0; i < tree->num_leaves(); ++i) {
      if (!tree->leaf_is_in_monotone_subtree(i)) {
        continue;
      }
      // we compute the constraints, and we only need one min and one max constraint this time
      // because we are not going to split the leaf, we may just change its value
      ComputeConstraintsPerThreshold(-1, tree, ~i, 0, false);
      double min_constraint = min_constraints[0][0];
      double max_constraint = max_constraints[0][0];
#ifdef DEBUG
      CHECK(tree->LeafOutput(i) >= min_constraint);
      CHECK(tree->LeafOutput(i) <= max_constraint);
#endif
      double new_constrained_output =
          FeatureHistogram::CalculateSplittedLeafOutput(
              sum_grad[i], sum_hess[i], config_->lambda_l1, config_->lambda_l2,
              config_->max_delta_step, min_constraint, max_constraint);
      double old_output = tree->LeafOutput(i);
      // a more accurate value may not immediately result in a loss reduction because of the shrinkage rate
      if (fabs(old_output - new_constrained_output) > EPS) {
        might_be_something_to_update = true;
        tree->SetLeafOutput(i, new_constrained_output);
      }

      // we reset the constraints
      min_constraints[0][0] = -std::numeric_limits<double>::max();
      max_constraints[0][0] = std::numeric_limits<double>::max();
      thresholds[0].clear();
      is_in_right_split[0].clear();
      features[0].clear();
    }
  }
}

Tree* SerialTreeLearner::FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t *hessians) const {
  auto tree = std::unique_ptr<Tree>(new Tree(*old_tree));
  CHECK(data_partition_->num_leaves() >= tree->num_leaves());
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < tree->num_leaves(); ++i) {
    OMP_LOOP_EX_BEGIN();
    data_size_t cnt_leaf_data = 0;
    auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
    double sum_grad = 0.0f;
    double sum_hess = kEpsilon;
    for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
      auto idx = tmp_idx[j];
      sum_grad += gradients[idx];
      sum_hess += hessians[idx];
    }
    double output = FeatureHistogram::CalculateSplittedLeafOutput(sum_grad, sum_hess,
                                                                  config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    auto old_leaf_output = tree->LeafOutput(i);
    auto new_leaf_output = output * tree->shrinkage();
    tree->SetLeafOutput(i, config_->refit_decay_rate * old_leaf_output + (1.0 - config_->refit_decay_rate) * new_leaf_output);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  return tree.release();
}

Tree* SerialTreeLearner::FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred, const score_t* gradients, const score_t *hessians) {
  data_partition_->ResetByLeafPred(leaf_pred, old_tree->num_leaves());
  return FitByExistingTree(old_tree, gradients, hessians);
}

std::vector<int8_t> SerialTreeLearner::GetUsedFeatures(bool is_tree_level) {
  std::vector<int8_t> ret(num_features_, 1);
  if (config_->feature_fraction >= 1.0f && is_tree_level) {
    return ret;
  }
  if (config_->feature_fraction_bynode >= 1.0f && !is_tree_level) {
    return ret;
  }
  std::memset(ret.data(), 0, sizeof(int8_t) * num_features_);
  const int min_used_features = std::min(2, static_cast<int>(valid_feature_indices_.size()));
  if (is_tree_level) {
    int used_feature_cnt = static_cast<int>(std::round(valid_feature_indices_.size() * config_->feature_fraction));
    used_feature_cnt = std::max(used_feature_cnt, min_used_features);
    used_feature_indices_ = random_.Sample(static_cast<int>(valid_feature_indices_.size()), used_feature_cnt);
    int omp_loop_size = static_cast<int>(used_feature_indices_.size());
    #pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
    for (int i = 0; i < omp_loop_size; ++i) {
      int used_feature = valid_feature_indices_[used_feature_indices_[i]];
      int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
      CHECK(inner_feature_index >= 0);
      ret[inner_feature_index] = 1;
    }
  } else if (used_feature_indices_.size() <= 0) {
    int used_feature_cnt = static_cast<int>(std::round(valid_feature_indices_.size() * config_->feature_fraction_bynode));
    used_feature_cnt = std::max(used_feature_cnt, min_used_features);
    auto sampled_indices = random_.Sample(static_cast<int>(valid_feature_indices_.size()), used_feature_cnt);
    int omp_loop_size = static_cast<int>(sampled_indices.size());
    #pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
    for (int i = 0; i < omp_loop_size; ++i) {
      int used_feature = valid_feature_indices_[sampled_indices[i]];
      int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
      CHECK(inner_feature_index >= 0);
      ret[inner_feature_index] = 1;
    }
  } else {
    int used_feature_cnt = static_cast<int>(std::round(used_feature_indices_.size() * config_->feature_fraction_bynode));
    used_feature_cnt = std::max(used_feature_cnt, min_used_features);
    auto sampled_indices = random_.Sample(static_cast<int>(used_feature_indices_.size()), used_feature_cnt);
    int omp_loop_size = static_cast<int>(sampled_indices.size());
    #pragma omp parallel for schedule(static, 512) if (omp_loop_size >= 1024)
    for (int i = 0; i < omp_loop_size; ++i) {
      int used_feature = valid_feature_indices_[used_feature_indices_[sampled_indices[i]]];
      int inner_feature_index = train_data_->InnerFeatureIndex(used_feature);
      CHECK(inner_feature_index >= 0);
      ret[inner_feature_index] = 1;
    }
  }
  return ret;
}

void SerialTreeLearner::BeforeTrain() {
  // reset histogram pool
  histogram_pool_.ResetMap();

  if (config_->feature_fraction < 1.0f) {
    is_feature_used_ = GetUsedFeatures(true);
  } else {
    #pragma omp parallel for schedule(static, 512) if (num_features_ >= 1024)
    for (int i = 0; i < num_features_; ++i) {
      is_feature_used_[i] = 1;
    }
  }

  // initialize data partition
  data_partition_->Init();

  // reset the splits for leaves
  for (int i = 0; i < config_->num_leaves; ++i) {
    best_split_per_leaf_[i].Reset();
    constraints_per_leaf_[i].Reset();
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);

  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_, 0.);
  }

  larger_leaf_splits_->Init();

  // if has ordered bin, need to initialize the ordered bin
  if (has_ordered_bin_) {
    #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
    #endif
    if (data_partition_->leaf_count(0) == num_data_) {
      // use all data, pass nullptr
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        ordered_bins_[ordered_bin_indices_[i]]->Init(nullptr, config_->num_leaves);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      // bagging, only use part of data

      // mark used data
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(0);
      data_size_t end = begin + data_partition_->leaf_count(0);
      #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 1;
      }
      OMP_INIT_EX();
      // initialize ordered bin
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
        OMP_LOOP_EX_BEGIN();
        ordered_bins_[ordered_bin_indices_[i]]->Init(is_data_in_leaf_.data(), config_->num_leaves);
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 0;
      }
    }
    #ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
    #endif
  }
}

bool SerialTreeLearner::BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  // check depth of current leaf
  if (config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
  data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
  // no enough data to continue
  if (num_data_in_right_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)
      && num_data_in_left_child < static_cast<data_size_t>(config_->min_data_in_leaf * 2)) {
    best_split_per_leaf_[left_leaf].gain = kMinScore;
    if (right_leaf >= 0) {
      best_split_per_leaf_[right_leaf].gain = kMinScore;
    }
    return false;
  }
  parent_leaf_histogram_array_ = nullptr;
  // only have root
  if (right_leaf < 0) {
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    larger_leaf_histogram_array_ = nullptr;
  } else if (num_data_in_left_child < num_data_in_right_child) {
    // put parent(left) leaf's histograms into larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Move(left_leaf, right_leaf);
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
  } else {
    // put parent(left) leaf's histograms to larger leaf's histograms
    if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { parent_leaf_histogram_array_ = larger_leaf_histogram_array_; }
    histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
  }
  // split for the ordered bin
  if (has_ordered_bin_ && right_leaf >= 0) {
    #ifdef TIMETAG
    auto start_time = std::chrono::steady_clock::now();
    #endif
    // mark data that at left-leaf
    const data_size_t* indices = data_partition_->indices();
    const auto left_cnt = data_partition_->leaf_count(left_leaf);
    const auto right_cnt = data_partition_->leaf_count(right_leaf);
    char mark = 1;
    data_size_t begin = data_partition_->leaf_begin(left_leaf);
    data_size_t end = begin + left_cnt;
    if (left_cnt > right_cnt) {
      begin = data_partition_->leaf_begin(right_leaf);
      end = begin + right_cnt;
      mark = 0;
    }
    #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 1;
    }
    OMP_INIT_EX();
    // split the ordered bin
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(ordered_bin_indices_.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      ordered_bins_[ordered_bin_indices_[i]]->Split(left_leaf, right_leaf, is_data_in_leaf_.data(), mark);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    #pragma omp parallel for schedule(static, 512) if (end - begin >= 1024)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 0;
    }
    #ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
    #endif
  }
  return true;
}

void SerialTreeLearner::FindBestSplits(const Tree* tree) {
  std::vector<int8_t> is_feature_used(num_features_, 0);
  #pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = parent_leaf_histogram_array_ != nullptr;
  ConstructHistograms(is_feature_used, use_subtract);
  FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree);
}

void SerialTreeLearner::ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  train_data_->ConstructHistograms(is_feature_used,
                                   smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
                                   smaller_leaf_splits_->LeafIndex(),
                                   &ordered_bins_, gradients_, hessians_,
                                   ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                   ptr_smaller_leaf_hist_data);

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
                                     larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
                                     larger_leaf_splits_->LeafIndex(),
                                     &ordered_bins_, gradients_, hessians_,
                                     ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                     ptr_larger_leaf_hist_data);
  }
  #ifdef TIMETAG
  hist_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

void SerialTreeLearner::FindBestSplitsFromHistograms(
    const std::vector<int8_t> &is_feature_used, bool use_subtract,
    const Tree *tree) {
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  std::vector<int8_t> smaller_node_used_features(num_features_, 1);
  std::vector<int8_t> larger_node_used_features(num_features_, 1);
  if (config_->feature_fraction_bynode < 1.0f) {
    smaller_node_used_features = GetUsedFeatures(false);
    larger_node_used_features = GetUsedFeatures(false);
  }
  OMP_INIT_EX();
  // find splits
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) { continue; }
    const int tid = omp_get_thread_num();

    train_data_->FixHistogram(feature_index,
                              smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                              smaller_leaf_splits_->num_data_in_leaf(),
                              smaller_leaf_histogram_array_[feature_index].RawData());
    int real_fidx = train_data_->RealFeatureIndex(feature_index);

    ComputeBestSplitForFeature(smaller_leaf_splits_->sum_gradients(),
                               smaller_leaf_splits_->sum_hessians(),
                               smaller_leaf_splits_->num_data_in_leaf(),
                               feature_index, smaller_leaf_histogram_array_,
                               smaller_best, smaller_leaf_splits_->LeafIndex(),
                               smaller_leaf_splits_->depth(), tid, real_fidx,
                               tree);
    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                larger_leaf_splits_->num_data_in_leaf(),
                                larger_leaf_histogram_array_[feature_index].RawData());
    }

    ComputeBestSplitForFeature(larger_leaf_splits_->sum_gradients(),
                               larger_leaf_splits_->sum_hessians(),
                               larger_leaf_splits_->num_data_in_leaf(),
                               feature_index, larger_leaf_histogram_array_,
                               larger_best, larger_leaf_splits_->LeafIndex(),
                               larger_leaf_splits_->depth(), tid, real_fidx,
                               tree);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  int leaf = smaller_leaf_splits_->LeafIndex();
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
    best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  }
  #ifdef TIMETAG
  find_split_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

int32_t SerialTreeLearner::ForceSplits(Tree* tree, const Json& forced_split_json, int* left_leaf,
                                       int* right_leaf, int *cur_depth,
                                       bool *aborted_last_force_split) {
  int32_t result_count = 0;
  // start at root leaf
  *left_leaf = 0;
  std::queue<std::pair<Json, int>> q;
  Json left = forced_split_json;
  Json right;
  bool left_smaller = true;
  std::unordered_map<int, SplitInfo> forceSplitMap;
  q.push(std::make_pair(forced_split_json, *left_leaf));
  while (!q.empty()) {
    // before processing next node from queue, store info for current left/right leaf
    // store "best split" for left and right, even if they might be overwritten by forced split
    if (BeforeFindBestSplit(tree, *left_leaf, *right_leaf)) {
      FindBestSplits(tree);
    }
    // then, compute own splits
    SplitInfo left_split;
    SplitInfo right_split;

    if (!left.is_null()) {
      const int left_feature = left["feature"].int_value();
      const double left_threshold_double = left["threshold"].number_value();
      const int left_inner_feature_index = train_data_->InnerFeatureIndex(left_feature);
      const uint32_t left_threshold = train_data_->BinThreshold(
              left_inner_feature_index, left_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? smaller_leaf_histogram_array_ : larger_leaf_histogram_array_;
      auto left_leaf_splits = (left_smaller) ? smaller_leaf_splits_.get() : larger_leaf_splits_.get();
      leaf_histogram_array[left_inner_feature_index].GatherInfoForThreshold(
              left_leaf_splits->sum_gradients(),
              left_leaf_splits->sum_hessians(),
              left_threshold,
              left_leaf_splits->num_data_in_leaf(),
              &left_split);
      left_split.feature = left_feature;
      forceSplitMap[*left_leaf] = left_split;
      if (left_split.gain < 0) {
        forceSplitMap.erase(*left_leaf);
      }
    }

    if (!right.is_null()) {
      const int right_feature = right["feature"].int_value();
      const double right_threshold_double = right["threshold"].number_value();
      const int right_inner_feature_index = train_data_->InnerFeatureIndex(right_feature);
      const uint32_t right_threshold = train_data_->BinThreshold(
              right_inner_feature_index, right_threshold_double);
      auto leaf_histogram_array = (left_smaller) ? larger_leaf_histogram_array_ : smaller_leaf_histogram_array_;
      auto right_leaf_splits = (left_smaller) ? larger_leaf_splits_.get() : smaller_leaf_splits_.get();
      leaf_histogram_array[right_inner_feature_index].GatherInfoForThreshold(
        right_leaf_splits->sum_gradients(),
        right_leaf_splits->sum_hessians(),
        right_threshold,
        right_leaf_splits->num_data_in_leaf(),
        &right_split);
      right_split.feature = right_feature;
      forceSplitMap[*right_leaf] = right_split;
      if (right_split.gain < 0) {
        forceSplitMap.erase(*right_leaf);
      }
    }

    std::pair<Json, int> pair = q.front();
    q.pop();
    int current_leaf = pair.second;
    // split info should exist because searching in bfs fashion - should have added from parent
    if (forceSplitMap.find(current_leaf) == forceSplitMap.end()) {
        *aborted_last_force_split = true;
        break;
    }
    SplitInfo current_split_info = forceSplitMap[current_leaf];
    const int inner_feature_index = train_data_->InnerFeatureIndex(
            current_split_info.feature);
    // we want to know if the feature has to be monotone
    bool feature_is_monotone = false;
    if (!config_->monotone_constraints.empty()) {
        feature_is_monotone = config_->monotone_constraints[inner_feature_index] != 0;
    }
    auto threshold_double = train_data_->RealThreshold(
            inner_feature_index, current_split_info.threshold);

    // split tree, will return right leaf
    *left_leaf = current_leaf;
    if (train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin) {
      *right_leaf = tree->Split(current_leaf,
                                inner_feature_index,
                                current_split_info.feature,
                                current_split_info.threshold,
                                threshold_double,
                                static_cast<double>(current_split_info.left_output),
                                static_cast<double>(current_split_info.right_output),
                                static_cast<data_size_t>(current_split_info.left_count),
                                static_cast<data_size_t>(current_split_info.right_count),
                                static_cast<double>(current_split_info.left_sum_hessian),
                                static_cast<double>(current_split_info.right_sum_hessian),
                                static_cast<float>(current_split_info.gain),
                                train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
                                current_split_info.default_left,
                                feature_is_monotone);
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             &current_split_info.threshold, 1,
                             current_split_info.default_left, *right_leaf);
    } else {
      std::vector<uint32_t> cat_bitset_inner = Common::ConstructBitset(
              current_split_info.cat_threshold.data(), current_split_info.num_cat_threshold);
      std::vector<int> threshold_int(current_split_info.num_cat_threshold);
      for (int i = 0; i < current_split_info.num_cat_threshold; ++i) {
        threshold_int[i] = static_cast<int>(train_data_->RealThreshold(
                    inner_feature_index, current_split_info.cat_threshold[i]));
      }
      std::vector<uint32_t> cat_bitset = Common::ConstructBitset(
              threshold_int.data(), current_split_info.num_cat_threshold);
      *right_leaf = tree->SplitCategorical(current_leaf,
                                           inner_feature_index,
                                           current_split_info.feature,
                                           cat_bitset_inner.data(),
                                           static_cast<int>(cat_bitset_inner.size()),
                                           cat_bitset.data(),
                                           static_cast<int>(cat_bitset.size()),
                                           static_cast<double>(current_split_info.left_output),
                                           static_cast<double>(current_split_info.right_output),
                                           static_cast<data_size_t>(current_split_info.left_count),
                                           static_cast<data_size_t>(current_split_info.right_count),
                                           static_cast<double>(current_split_info.left_sum_hessian),
                                           static_cast<double>(current_split_info.right_sum_hessian),
                                           static_cast<float>(current_split_info.gain),
                                           train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
                                           feature_is_monotone);
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
                             current_split_info.default_left, *right_leaf);
    }

    int depth = tree->leaf_depth(*left_leaf);
    #ifdef DEBUG
    CHECK(depth == tree->leaf_depth(*right_leaf));
    #endif
    if (current_split_info.left_count < current_split_info.right_count) {
      left_smaller = true;
      smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                 current_split_info.left_sum_gradient,
                                 current_split_info.left_sum_hessian, depth);
      larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                current_split_info.right_sum_gradient,
                                current_split_info.right_sum_hessian, depth);
    } else {
      left_smaller = false;
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 current_split_info.right_sum_gradient,
                                 current_split_info.right_sum_hessian, depth);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                current_split_info.left_sum_gradient,
                                current_split_info.left_sum_hessian, depth);
    }

    left = Json();
    right = Json();
    if ((pair.first).object_items().count("left") > 0) {
      left = (pair.first)["left"];
      if (left.object_items().count("feature") > 0 && left.object_items().count("threshold") > 0) {
        q.push(std::make_pair(left, *left_leaf));
      }
    }
    if ((pair.first).object_items().count("right") > 0) {
      right = (pair.first)["right"];
      if (right.object_items().count("feature") > 0 && right.object_items().count("threshold") > 0) {
        q.push(std::make_pair(right, *right_leaf));
      }
    }
    result_count++;
    *(cur_depth) = std::max(*(cur_depth), tree->leaf_depth(*left_leaf));
  }
  return result_count;
}

void SerialTreeLearner::Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
  double previous_leaf_output = tree->LeafOutput(best_leaf);
  const int inner_feature_index = train_data_->InnerFeatureIndex(best_split_info.feature);
  if (cegb_ != nullptr) {
    cegb_->UpdateLeafBestSplits(tree, best_leaf, &best_split_info, &best_split_per_leaf_);
  }
  // left = parent
  *left_leaf = best_leaf;
  bool is_numerical_split = train_data_->FeatureBinMapper(inner_feature_index)->bin_type() == BinType::NumericalBin;
  if (is_numerical_split) {
    auto threshold_double = train_data_->RealThreshold(inner_feature_index, best_split_info.threshold);
    // split tree, will return right leaf
    *right_leaf = tree->Split(best_leaf,
                              inner_feature_index,
                              best_split_info.feature,
                              best_split_info.threshold,
                              threshold_double,
                              static_cast<double>(best_split_info.left_output),
                              static_cast<double>(best_split_info.right_output),
                              static_cast<data_size_t>(best_split_info.left_count),
                              static_cast<data_size_t>(best_split_info.right_count),
                              static_cast<double>(best_split_info.left_sum_hessian),
                              static_cast<double>(best_split_info.right_sum_hessian),
                              static_cast<float>(best_split_info.gain),
                              train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
                              best_split_info.default_left,
                              best_split_info.monotone_type != 0);
    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           &best_split_info.threshold, 1, best_split_info.default_left, *right_leaf);
  } else {
    std::vector<uint32_t> cat_bitset_inner = Common::ConstructBitset(best_split_info.cat_threshold.data(), best_split_info.num_cat_threshold);
    std::vector<int> threshold_int(best_split_info.num_cat_threshold);
    for (int i = 0; i < best_split_info.num_cat_threshold; ++i) {
      threshold_int[i] = static_cast<int>(train_data_->RealThreshold(inner_feature_index, best_split_info.cat_threshold[i]));
    }
    std::vector<uint32_t> cat_bitset = Common::ConstructBitset(threshold_int.data(), best_split_info.num_cat_threshold);
    *right_leaf = tree->SplitCategorical(best_leaf,
                                         inner_feature_index,
                                         best_split_info.feature,
                                         cat_bitset_inner.data(),
                                         static_cast<int>(cat_bitset_inner.size()),
                                         cat_bitset.data(),
                                         static_cast<int>(cat_bitset.size()),
                                         static_cast<double>(best_split_info.left_output),
                                         static_cast<double>(best_split_info.right_output),
                                         static_cast<data_size_t>(best_split_info.left_count),
                                         static_cast<data_size_t>(best_split_info.right_count),
                                         static_cast<double>(best_split_info.left_sum_hessian),
                                         static_cast<double>(best_split_info.right_sum_hessian),
                                         static_cast<float>(best_split_info.gain),
                                         train_data_->FeatureBinMapper(inner_feature_index)->missing_type(),
                                         best_split_info.monotone_type != 0);
    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()), best_split_info.default_left, *right_leaf);
  }

  #ifdef DEBUG
  CHECK(best_split_info.left_count == data_partition_->leaf_count(best_leaf));
  #endif
  // init the leaves that used on next iteration
  int depth = tree->leaf_depth(*left_leaf);
  #ifdef DEBUG
  CHECK(depth == tree->leaf_depth(*right_leaf));
  #endif
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian, depth);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                              best_split_info.right_sum_gradient,
                              best_split_info.right_sum_hessian, depth);
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                               best_split_info.right_sum_gradient,
                               best_split_info.right_sum_hessian, depth);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                              best_split_info.left_sum_gradient,
                              best_split_info.left_sum_hessian, depth);
  }

  // when the monotone precise mode is disabled it is very easy to compute the constraints of
  // the children of a leaf, but when it is enabled, one needs to go through the tree to do so,
  // and it is done directly before computing best splits
  if (!config_->monotone_precise_mode) {
    constraints_per_leaf_[*right_leaf] = constraints_per_leaf_[*left_leaf];
    if (is_numerical_split) {
      // depending on the monotone type we set constraints on the future splits
      // these constraints may be updated later in the algorithm
      if (best_split_info.monotone_type < 0) {
        constraints_per_leaf_[*left_leaf]
            .SetMinConstraint(best_split_info.right_output);
        constraints_per_leaf_[*right_leaf]
            .SetMaxConstraint(best_split_info.left_output);
      } else if (best_split_info.monotone_type > 0) {
        constraints_per_leaf_[*left_leaf]
            .SetMaxConstraint(best_split_info.right_output);
        constraints_per_leaf_[*right_leaf]
            .SetMinConstraint(best_split_info.left_output);
      }
    }
  }

  // if there is a monotone split above, we need to make sure the new
  // values don't clash with existing constraints in the subtree,
  // and if they do, the existing splits need to be updated
  if (tree->leaf_is_in_monotone_subtree(*right_leaf)) {
    GoUpToFindLeavesToUpdate(tree, tree->leaf_parent(*right_leaf),
                             inner_feature_index, best_split_info,
                             previous_leaf_output, best_split_info.threshold);
  }
}

// this function is only used if the monotone precise mode is enabled
// it computes the constraints for a given leaf and a given feature
// (there can be many constraints because the constraints can depend on thresholds)
void SerialTreeLearner::ComputeConstraintsPerThreshold(
    int feature, const Tree *tree, int node_idx, unsigned int tid,
    bool per_threshold, bool compute_min, bool compute_max, uint32_t it_start,
    uint32_t it_end) {
  int parent_idx = (node_idx < 0) ? tree->leaf_parent(~node_idx)
                                  : tree->node_parent(node_idx);

  if (parent_idx != -1) {
    int inner_feature = tree->split_feature_inner(parent_idx);
    int8_t monotone_type = train_data_->FeatureMonotone(inner_feature);
    bool is_right_split = tree->right_child(parent_idx) == node_idx;
    bool split_contains_new_information = true;
    bool is_split_numerical = (train_data_->FeatureBinMapper(inner_feature)
                                   ->bin_type()) == BinType::NumericalBin;
    uint32_t threshold = tree->threshold_in_bin(parent_idx);

    // when we go up, we can get more information about the position of the original leaf
    // so the starting and ending thresholds can be updated, which will save some time later
    if ((feature == inner_feature) && is_split_numerical) {
      if (is_right_split) {
        it_start = std::max(threshold, it_start);
      } else {
        it_end = std::min(threshold + 1, it_end);
      }
#ifdef DEBUG
      CHECK(it_start < it_end);
#endif
    }

    // only branches that contain leaves that are contiguous to the original leaf need to be visited
    for (unsigned int i = 0; i < features[tid].size(); ++i) {
      if (features[tid][i] == inner_feature && is_split_numerical &&
          is_in_right_split[tid][i] == is_right_split) {
        split_contains_new_information = false;
        break;
      }
    }

    if (split_contains_new_information) {
      if (monotone_type != 0) {
        int left_child_idx = tree->left_child(parent_idx);
        int right_child_idx = tree->right_child(parent_idx);
        bool left_child_is_curr_idx = (left_child_idx == node_idx);

        bool take_min = (monotone_type < 0) ? left_child_is_curr_idx
                                            : !left_child_is_curr_idx;
        if ((take_min && compute_min) || (!take_min && compute_max)) {
          int node_idx_to_pass =
              (left_child_is_curr_idx) ? right_child_idx : left_child_idx;

          // we go down in the opposite branch to see if some
          // constraints that would apply to the original leaf can be found
          ComputeConstraintsPerThresholdInSubtree(
              feature, inner_feature, tree, node_idx_to_pass, take_min,
              it_start, it_end, features[tid], thresholds[tid],
              is_in_right_split[tid], tid, per_threshold);
        }
      }

      is_in_right_split[tid].push_back(is_right_split);
      thresholds[tid].push_back(threshold);
      features[tid].push_back(inner_feature);
    }

    // we keep going up the tree to find constraints that could come from somewhere else
    if (parent_idx != 0) {
      ComputeConstraintsPerThreshold(feature, tree, parent_idx, tid,
                                     per_threshold, compute_min, compute_max,
                                     it_start, it_end);
    }
  }
}

// this function checks if the original leaf and the children of the node that is
// currently being visited are contiguous, and if so, the children should be visited too
std::pair<bool, bool> SerialTreeLearner::ShouldKeepGoingLeftRight(
    const Tree *tree, int node_idx, const std::vector<int> &features,
    const std::vector<uint32_t> &thresholds,
    const std::vector<bool> &is_in_right_split) {
  int inner_feature = tree->split_feature_inner(node_idx);
  uint32_t threshold = tree->threshold_in_bin(node_idx);
  bool is_split_numerical = train_data_->FeatureBinMapper(inner_feature)
                                ->bin_type() == BinType::NumericalBin;

  bool keep_going_right = true;
  bool keep_going_left = true;
  // we check if the left and right node are contiguous with the original leaf
  // if so we should keep going down these nodes to update constraints
  for (unsigned int i = 0; i < features.size(); ++i) {
    if (features[i] == inner_feature) {
      if (is_split_numerical) {
        if (threshold >= thresholds[i] && !is_in_right_split[i]) {
          keep_going_right = false;
        }
        if (threshold <= thresholds[i] && is_in_right_split[i]) {
          keep_going_left = false;
        }
      }
    }
  }
  return std::pair<bool, bool>(keep_going_left, keep_going_right);
}

// this function is called only when computing constraints when the monotone
// precise mode is set to true
// it makes sure that it is worth it to visit a branch, as it could
// not contain any relevant constraint (for example if the a branch
// with bigger values is also constraining the original leaf, then
// it is useless to visit the branch with smaller values)
std::pair<bool, bool> SerialTreeLearner::LeftRightContainsRelevantInformation(
    bool maximum, int inner_feature, bool split_feature_is_inner_feature) {
  if (split_feature_is_inner_feature) {
    return std::pair<bool, bool>(true, true);
  }
  int8_t monotone_type = train_data_->FeatureMonotone(inner_feature);
  if (monotone_type == 0) {
    return std::pair<bool, bool>(true, true);
  }
  if ((monotone_type == -1 && maximum) || (monotone_type == 1 && !maximum)) {
    return std::pair<bool, bool>(true, false);
  }
  if ((monotone_type == 1 && maximum) || (monotone_type == -1 && !maximum)) {
    return std::pair<bool, bool>(false, true);
  }
}

// at any point in time, for an index i, the constraint constraint[i] has to be valid on
// [threshold[i]: threshold[i + 1]) (or [threshold[i]: +inf) if i is the last index of the array)
// therefore, when a constraint is added on a leaf, it must be done very carefully
void SerialTreeLearner::UpdateConstraints(
    std::vector<std::vector<double> > &constraints,
    std::vector<std::vector<uint32_t> > &thresholds, double extremum,
    uint32_t it_start, uint32_t it_end, int split_feature, int tid,
    bool maximum) {
  bool start_done = false;
  bool end_done = false;
  // one must always keep track of the previous constraint
  // for example when adding a constraints cstr2 on thresholds [1:2),
  // on an existing constraints cstr1 on thresholds [0, +inf),
  // the thresholds and constraints must become
  // [0, 1, 2] and  [cstr1, cstr2, cstr1]
  // so since we loop through thresholds only once,
  // the previous constraint that still applies needs to be recorded
  double previous_constraint;
  double current_constraint;
  for (unsigned int i = 0; i < thresholds[tid].size();) {
    current_constraint = constraints[tid][i];
    // this is the easy case when the thresholds match
    if (thresholds[tid][i] == it_start) {
      constraints[tid][i] = (maximum) ? std::max(extremum, constraints[tid][i])
                                      : std::min(extremum, constraints[tid][i]);
      start_done = true;
    }
    if (thresholds[tid][i] > it_start) {
      // existing constraint is updated if there is a need for it
      if (thresholds[tid][i] < it_end) {
        constraints[tid][i] = (maximum)
                                  ? std::max(extremum, constraints[tid][i])
                                  : std::min(extremum, constraints[tid][i]);
      }
      // when thresholds don't match, a new threshold
      // and a new constraint may need to be inserted
      if (!start_done) {
        start_done = true;
        if ((maximum && extremum > previous_constraint) ||
            (!maximum && extremum < previous_constraint)) {
          constraints[tid].insert(constraints[tid].begin() + i, extremum);
          thresholds[tid].insert(thresholds[tid].begin() + i, it_start);
          i += 1;
        }
      }
    }
    // easy case when the thresholds match again
    if (thresholds[tid][i] == it_end) {
      end_done = true;
      i += 1;
      break;
    }
    // if they don't then, the previous constraint needs to be added back where the current one ends
    if (thresholds[tid][i] > it_end) {
      if (i != 0 && previous_constraint != constraints[tid][i - 1]) {
        constraints[tid]
            .insert(constraints[tid].begin() + i, previous_constraint);
        thresholds[tid].insert(thresholds[tid].begin() + i, it_end);
      }
      end_done = true;
      i += 1;
      break;
    }
    // If 2 successive constraints are the same then the second one may as well be deleted
    if (i != 0 && constraints[tid][i] == constraints[tid][i - 1]) {
      constraints[tid].erase(constraints[tid].begin() + i);
      thresholds[tid].erase(thresholds[tid].begin() + i);
      previous_constraint = current_constraint;
      i -= 1;
    }
    previous_constraint = current_constraint;
    i += 1;
  }
  // if the loop didn't get to an index greater than it_start, it needs to be added at the end
  if (!start_done) {
    if ((maximum && extremum > constraints[tid].back()) ||
        (!maximum && extremum < constraints[tid].back())) {
      constraints[tid].push_back(extremum);
      thresholds[tid].push_back(it_start);
    } else {
      end_done = true;
    }
  }
  // if we didn't get to an index after it_end, then the previous constraint needs to be set back
  // unless it_end goes up to the last bin of the feature
  if (!end_done &&
      static_cast<int>(it_end) != train_data_->NumBin(split_feature) &&
      previous_constraint != constraints[tid].back()) {
    constraints[tid].push_back(previous_constraint);
    thresholds[tid].push_back(it_end);
  }
}

// this function goes down in a subtree to find the constraints that would apply
void SerialTreeLearner::ComputeConstraintsPerThresholdInSubtree(
    int split_feature, int monotone_feature, const Tree *tree, int node_idx,
    bool maximum, uint32_t it_start, uint32_t it_end,
    const std::vector<int> &features, const std::vector<uint32_t> &thresholds,
    const std::vector<bool> &is_in_right_split, unsigned int tid,
    bool per_threshold) {
  bool is_original_split_numerical =
      train_data_->FeatureBinMapper(split_feature)->bin_type() ==
      BinType::NumericalBin;
  double extremum;
  // if we just got to a leaf, then we update
  // the constraints using the leaf value
  if (node_idx < 0) {
    extremum = tree->LeafOutput(~node_idx);
#ifdef DEBUG
    CHECK(it_start < it_end);
#endif
    // if the constraints per threshold are needed then monotone
    // precise mode is enabled and we are not refitting leaves
    if (per_threshold && is_original_split_numerical) {
      std::vector<std::vector<double> > &constraints =
          (maximum) ? min_constraints : max_constraints;
      std::vector<std::vector<uint32_t> > &thresholds =
          (maximum) ? thresholds_min_constraints : thresholds_max_constraints;
      UpdateConstraints(constraints, thresholds, extremum, it_start, it_end,
                        split_feature, tid, maximum);
    } else { // otherwise the constraints can be updated just by performing a min / max
      if (maximum) {
        min_constraints[tid][0] = std::max(min_constraints[tid][0], extremum);
      } else {
        max_constraints[tid][0] = std::min(max_constraints[tid][0], extremum);
      }
    }
  }
  // if the function got to a node, it keeps going down the tree
  else {
    // check if the children are contiguous to the original leaf
    std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
        tree, node_idx, features, thresholds, is_in_right_split);
    int inner_feature = tree->split_feature_inner(node_idx);
    uint32_t threshold = tree->threshold_in_bin(node_idx);

    bool split_feature_is_inner_feature = (inner_feature == split_feature);
    bool split_feature_is_monotone_feature =
        (monotone_feature == split_feature);
    // it is made sure that both children contain values that could potentially
    // help determine the true constraints for the original leaf
    std::pair<bool, bool> left_right_contain_relevant_information =
        LeftRightContainsRelevantInformation(
            maximum, inner_feature, split_feature_is_inner_feature &&
                                        !split_feature_is_monotone_feature);
    // if a child does not contain relevant information compared to the other child,
    // and if the other child is not contiguous, then we still need to go down the first child
    if (keep_going_left_right.first &&
        (left_right_contain_relevant_information.first ||
         !keep_going_left_right.second)) {
      uint32_t new_it_end =
          (split_feature_is_inner_feature && is_original_split_numerical)
              ? std::min(threshold + 1, it_end)
              : it_end;
      ComputeConstraintsPerThresholdInSubtree(
          split_feature, monotone_feature, tree, tree->left_child(node_idx),
          maximum, it_start, new_it_end, features, thresholds,
          is_in_right_split, tid, per_threshold);
    }
    if (keep_going_left_right.second &&
        (left_right_contain_relevant_information.second ||
         !keep_going_left_right.first)) {
      uint32_t new_it_start =
          (split_feature_is_inner_feature && is_original_split_numerical)
              ? std::max(threshold + 1, it_start)
              : it_start;
      ComputeConstraintsPerThresholdInSubtree(
          split_feature, monotone_feature, tree, tree->right_child(node_idx),
          maximum, new_it_start, it_end, features, thresholds,
          is_in_right_split, tid, per_threshold);
    }
  }
}

void SerialTreeLearner::RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                                        data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    const data_size_t* bag_mapper = nullptr;
    if (total_num_data != num_data_) {
      CHECK(bag_cnt == num_data_);
      bag_mapper = bag_indices;
    }
    std::vector<int> n_nozeroworker_perleaf(tree->num_leaves(), 1);
    int num_machines = Network::num_machines();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      const double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto index_mapper = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      if (cnt_leaf_data > 0) {
        // bag_mapper[index_mapper[i]]
        const double new_output = obj->RenewTreeOutput(output, residual_getter, index_mapper, bag_mapper, cnt_leaf_data);
        tree->SetLeafOutput(i, new_output);
      } else {
        CHECK(num_machines > 1);
        tree->SetLeafOutput(i, 0.0);
        n_nozeroworker_perleaf[i] = 0;
      }
    }
    if (num_machines > 1) {
      std::vector<double> outputs(tree->num_leaves());
      for (int i = 0; i < tree->num_leaves(); ++i) {
        outputs[i] = static_cast<double>(tree->LeafOutput(i));
      }
      outputs = Network::GlobalSum(&outputs);
      n_nozeroworker_perleaf = Network::GlobalSum(&n_nozeroworker_perleaf);
      for (int i = 0; i < tree->num_leaves(); ++i) {
        tree->SetLeafOutput(i, outputs[i] / n_nozeroworker_perleaf[i]);
      }
    }
  }
}

// this function goes through the tree to find how the split that was just made is
// going to affect other leaves
void SerialTreeLearner::GoDownToFindLeavesToUpdate(
    const Tree *tree, int node_idx, const std::vector<int> &features,
    const std::vector<uint32_t> &thresholds,
    const std::vector<bool> &is_in_right_split, int maximum, int split_feature,
    const SplitInfo &split_info, double previous_leaf_output,
    bool use_left_leaf, bool use_right_leaf, uint32_t split_threshold) {
  if (node_idx < 0) {
    int leaf_idx = ~node_idx;

    // if leaf is at max depth then there is no need to update it
    int max_depth = config_->max_depth;
    if (tree->leaf_depth(leaf_idx) >= max_depth && max_depth > 0) {
      return;
    }

    // splits that are not to be used shall not be updated
    if (best_split_per_leaf_[leaf_idx].gain == kMinScore) {
      return;
    }

    std::pair<double, double> min_max_constraints;
    bool something_changed;
    if (use_right_leaf && use_left_leaf) {
      min_max_constraints =
          std::minmax(split_info.right_output, split_info.left_output);
    } else if (use_right_leaf && !use_left_leaf) {
      min_max_constraints = std::pair<double, double>(split_info.right_output,
                                                      split_info.right_output);
    } else {
      min_max_constraints = std::pair<double, double>(split_info.left_output,
                                                      split_info.left_output);
    }

#ifdef DEBUG
    if (maximum) {
      CHECK(min_max_constraints.first >= tree->LeafOutput(leaf_idx));
    } else {
      CHECK(min_max_constraints.second <= tree->LeafOutput(leaf_idx));
    }
#endif

    if (!config_->monotone_precise_mode) {
      if (!maximum) {
        something_changed =
            constraints_per_leaf_[leaf_idx]
                .SetMinConstraintAndReturnChange(min_max_constraints.second);
      } else {
        something_changed =
            constraints_per_leaf_[leaf_idx]
                .SetMaxConstraintAndReturnChange(min_max_constraints.first);
      }
      if (!something_changed) {
        return;
      }
    } else {
      if (!maximum) {
        // both functions need to be called in this order
        // because they modify the struct
        something_changed =
            constraints_per_leaf_[leaf_idx]
                .CrossesMinConstraint(min_max_constraints.second);
        something_changed = constraints_per_leaf_[leaf_idx]
                                .IsInMinConstraints(previous_leaf_output) ||
                            something_changed;
      } else {
        // both functions need to be called in this order
        // because they modify the struct
        something_changed =
            constraints_per_leaf_[leaf_idx]
                .CrossesMaxConstraint(min_max_constraints.first);
        something_changed = constraints_per_leaf_[leaf_idx]
                                .IsInMaxConstraints(previous_leaf_output) ||
                            something_changed;
      }
      // if constraints have changed, then best splits need to be updated
      // otherwise, we can just continue and go to the next split
      if (!something_changed) {
        return;
      }
    }
    UpdateBestSplitsFromHistograms(best_split_per_leaf_[leaf_idx], leaf_idx,
                                   tree->leaf_depth(leaf_idx), tree);
  } else {
    // check if the children are contiguous with the original leaf
    std::pair<bool, bool> keep_going_left_right = ShouldKeepGoingLeftRight(
        tree, node_idx, features, thresholds, is_in_right_split);
    int inner_feature = tree->split_feature_inner(node_idx);
    uint32_t threshold = tree->threshold_in_bin(node_idx);
    bool is_split_numerical = train_data_->FeatureBinMapper(inner_feature)
                                  ->bin_type() == BinType::NumericalBin;
    bool use_left_leaf_for_update = true;
    bool use_right_leaf_for_update = true;
    if (is_split_numerical && inner_feature == split_feature) {
      if (threshold >= split_threshold) {
        use_left_leaf_for_update = false;
      }
      if (threshold <= split_threshold) {
        use_right_leaf_for_update = false;
      }
    }

    if (keep_going_left_right.first) {
      GoDownToFindLeavesToUpdate(
          tree, tree->left_child(node_idx), features, thresholds,
          is_in_right_split, maximum, split_feature, split_info,
          previous_leaf_output, use_left_leaf,
          use_right_leaf_for_update && use_right_leaf, split_threshold);
    }
    if (keep_going_left_right.second) {
      GoDownToFindLeavesToUpdate(
          tree, tree->right_child(node_idx), features, thresholds,
          is_in_right_split, maximum, split_feature, split_info,
          previous_leaf_output, use_left_leaf_for_update && use_left_leaf,
          use_right_leaf, split_threshold);
    }
  }
}

// this function goes through the tree to find how the split that
// has just been performed is going to affect the constraints of other leaves
void SerialTreeLearner::GoUpToFindLeavesToUpdate(
    const Tree *tree, int node_idx, std::vector<int> &features,
    std::vector<uint32_t> &thresholds, std::vector<bool> &is_in_right_split,
    int split_feature, const SplitInfo &split_info, double previous_leaf_output,
    uint32_t split_threshold) {
  int parent_idx = tree->node_parent(node_idx);
  if (parent_idx != -1) {
    int inner_feature = tree->split_feature_inner(parent_idx);
    int8_t monotone_type = train_data_->FeatureMonotone(inner_feature);
    bool is_right_split = tree->right_child(parent_idx) == node_idx;
    bool split_contains_new_information = true;
    bool is_split_numerical = train_data_->FeatureBinMapper(inner_feature)
                                  ->bin_type() == BinType::NumericalBin;

    // only branches containing leaves that are contiguous to the original leaf need to be updated
    for (unsigned int i = 0; i < features.size(); ++i) {
      if ((features[i] == inner_feature && is_split_numerical) &&
          (is_in_right_split[i] == is_right_split)) {
        split_contains_new_information = false;
        break;
      }
    }

    if (split_contains_new_information) {
      if (monotone_type != 0) {
        int left_child_idx = tree->left_child(parent_idx);
        int right_child_idx = tree->right_child(parent_idx);
        bool left_child_is_curr_idx = (left_child_idx == node_idx);
        int node_idx_to_pass =
            (left_child_is_curr_idx) ? right_child_idx : left_child_idx;
        bool take_min = (monotone_type < 0) ? left_child_is_curr_idx
                                            : !left_child_is_curr_idx;

        GoDownToFindLeavesToUpdate(tree, node_idx_to_pass, features, thresholds,
                                   is_in_right_split, take_min, split_feature,
                                   split_info, previous_leaf_output, true, true,
                                   split_threshold);
      }

      is_in_right_split.push_back(tree->right_child(parent_idx) == node_idx);
      thresholds.push_back(tree->threshold_in_bin(parent_idx));
      features.push_back(tree->split_feature_inner(parent_idx));
    }

    if (parent_idx != 0) {
      GoUpToFindLeavesToUpdate(tree, parent_idx, features, thresholds,
                               is_in_right_split, split_feature, split_info,
                               previous_leaf_output, split_threshold);
    }
  }
}

// this function updates the best split for each leaf
// it is called only when monotone constraints exist
void SerialTreeLearner::UpdateBestSplitsFromHistograms(SplitInfo &split,
                                                       int leaf, int depth,
                                                       const Tree *tree) {
  std::vector<SplitInfo> bests(num_threads_);
  std::vector<bool> should_split_be_worse(num_threads_, false);

  // the feature histogram is retrieved
  FeatureHistogram *histogram_array_;
  histogram_pool_.Get(leaf, &histogram_array_);

  OMP_INIT_EX();
#pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    // the feature that are supposed to be used are computed
    if (!is_feature_used_[feature_index])
      continue;
    if (!histogram_array_[feature_index].is_splittable()) {
      constraints_per_leaf_[leaf].are_actual_constraints_worse[feature_index] =
          false;
      continue;
    }

    // loop through the features to find the best one just like in the
    // FindBestSplitsFromHistograms function
    const int tid = omp_get_thread_num();
    int real_fidx = train_data_->RealFeatureIndex(feature_index);

    // if the monotone precise mode is disabled or if the constraints have to be updated,
    // but are not exclusively worse, then we update the constraints and the best split
    if (!config_->monotone_precise_mode ||
        (constraints_per_leaf_[leaf].ToBeUpdated(feature_index) &&
         !constraints_per_leaf_[leaf]
              .AreActualConstraintsWorse(feature_index))) {

      ComputeBestSplitForFeature(
          split.left_sum_gradient + split.right_sum_gradient,
          split.left_sum_hessian + split.right_sum_hessian,
          split.left_count + split.right_count, feature_index, histogram_array_,
          bests, leaf, depth, tid, real_fidx, tree, true);
    } else {
      if (cegb_->splits_per_leaf_[leaf * train_data_->num_features() + feature_index] >
          bests[tid]) {
        bests[tid] = cegb_->splits_per_leaf_
            [leaf * train_data_->num_features() + feature_index];
        should_split_be_worse[tid] =
            constraints_per_leaf_[leaf]
                .AreActualConstraintsWorse(feature_index);
      }
    }

    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  // if the best split that has been found previously actually doesn't have the true constraints
  // but worse ones that were not computed before to optimize the computation time,
  // then we update every split and every constraints that should be updated
  if (should_split_be_worse[best_idx]) {
    std::fill(bests.begin(), bests.end(), SplitInfo());
#pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
    for (int feature_index = 0; feature_index < num_features_;
         ++feature_index) {
      OMP_LOOP_EX_BEGIN();
      if (!is_feature_used_[feature_index])
        continue;
      if (!histogram_array_[feature_index].is_splittable()) {
        continue;
      }

      const int tid = omp_get_thread_num();
      int real_fidx = train_data_->RealFeatureIndex(feature_index);

      if (constraints_per_leaf_[leaf]
              .AreActualConstraintsWorse(feature_index)) {
#ifdef DEBUG
        CHECK(config_->monotone_precise_mode);
        CHECK((constraints_per_leaf_[leaf].ToBeUpdated(feature_index)));
#endif

        ComputeBestSplitForFeature(
            split.left_sum_gradient + split.right_sum_gradient,
            split.left_sum_hessian + split.right_sum_hessian,
            split.left_count + split.right_count, feature_index,
            histogram_array_, bests, leaf, depth, tid, real_fidx, tree, true);
      } else {
#ifdef DEBUG
        CHECK(!constraints_per_leaf_[leaf].ToBeUpdated(feature_index));
#endif
        if (cegb_->splits_per_leaf_
                [leaf * train_data_->num_features() + feature_index] >
            bests[tid]) {
          bests[tid] = cegb_->splits_per_leaf_
              [leaf * train_data_->num_features() + feature_index];
        }
      }

      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    best_idx = ArrayArgs<SplitInfo>::ArgMax(bests);
  }

  // note: the gains may differ for the same set of constraints due to the non-deterministic OMP reduction.
  split = bests[best_idx];
}

// this function computes the best split for a given leaf and a given feature
void SerialTreeLearner::ComputeBestSplitForFeature(
    double sum_gradient, double sum_hessian, data_size_t num_data,
    int feature_index, FeatureHistogram *histogram_array_,
    std::vector<SplitInfo> &bests, int leaf_index, int depth, const int tid,
    int real_fidx, const Tree *tree, bool update) {

  // if this is not a subtree stemming from a monotone split, then no constraint apply
  if (tree->leaf_is_in_monotone_subtree(leaf_index)) {
    if (config_->monotone_precise_mode) {

      ComputeConstraintsPerThreshold(
          feature_index, tree, ~leaf_index, tid, config_->monotone_precise_mode,
          constraints_per_leaf_[leaf_index].MinToBeUpdated(feature_index) ||
              !update,
          constraints_per_leaf_[leaf_index].MaxToBeUpdated(feature_index) ||
              !update);

      if (!constraints_per_leaf_[leaf_index].MinToBeUpdated(feature_index) &&
          update) {
        min_constraints[tid] =
            constraints_per_leaf_[leaf_index].min_constraints[feature_index];
        thresholds_min_constraints[tid] =
            constraints_per_leaf_[leaf_index].min_thresholds[feature_index];
      } else {
        constraints_per_leaf_[leaf_index].min_constraints[feature_index] =
            min_constraints[tid];
        constraints_per_leaf_[leaf_index].min_thresholds[feature_index] =
            thresholds_min_constraints[tid];
      }

      if (!constraints_per_leaf_[leaf_index].MaxToBeUpdated(feature_index) &&
          update) {
        max_constraints[tid] =
            constraints_per_leaf_[leaf_index].max_constraints[feature_index];
        thresholds_max_constraints[tid] =
            constraints_per_leaf_[leaf_index].max_thresholds[feature_index];
      } else {
        constraints_per_leaf_[leaf_index].max_constraints[feature_index] =
            max_constraints[tid];
        constraints_per_leaf_[leaf_index].max_thresholds[feature_index] =
            thresholds_max_constraints[tid];
      }

      dummy_min_constraints[tid] = min_constraints[tid];
      dummy_max_constraints[tid] = max_constraints[tid];
    }
    if (!config_->monotone_precise_mode) {
      dummy_min_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].min_constraints[0][0];
      dummy_max_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].max_constraints[0][0];

      min_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].min_constraints[0][0];
      max_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].max_constraints[0][0];

      thresholds_min_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].min_thresholds[0][0];
      thresholds_max_constraints[tid][0] =
          constraints_per_leaf_[leaf_index].max_thresholds[0][0];
    }
  }

#ifdef DEBUG
  CHECK(dummy_min_constraints[tid] == min_constraints[tid]);
  CHECK(dummy_max_constraints[tid] == max_constraints[tid]);
  for (const auto &x : max_constraints[tid]) {
    CHECK(tree->LeafOutput(leaf_index) <= EPS + x);
    CHECK(x > -std::numeric_limits<double>::max());
  }
  for (const auto &x : dummy_min_constraints[tid]) {
    CHECK(tree->LeafOutput(leaf_index) + EPS >= x);
    CHECK(x < std::numeric_limits<double>::max());
  }
#endif

  SplitInfo new_split;
  histogram_array_[feature_index].FindBestThreshold(
      sum_gradient, sum_hessian, num_data, &new_split, min_constraints[tid],
      dummy_min_constraints[tid], max_constraints[tid],
      dummy_max_constraints[tid], thresholds_min_constraints[tid],
      thresholds_max_constraints[tid]);

  if (tree->leaf_is_in_monotone_subtree(leaf_index)) {
    InitializeConstraints(tid);
  }

  new_split.feature = real_fidx;
  if (cegb_ != nullptr) {
      new_split.gain -= cegb_->DetlaGain(feature_index, real_fidx, leaf_index, num_data, new_split);
  }


  if (new_split.monotone_type != 0) {
    double penalty =
        ComputeMonotoneSplitGainPenalty(depth, config_->monotone_penalty);
    new_split.gain *= penalty;
  }

  if (new_split > bests[tid]) {
    bests[tid] = new_split;
  }

  if (config_->monotone_precise_mode &&
      tree->leaf_is_in_monotone_subtree(leaf_index)) {
    constraints_per_leaf_[leaf_index].ResetUpdates(feature_index);
  }

#ifdef DEBUG
  ComputeConstraintsPerThreshold(-1, tree, ~leaf_index, tid, false);
  double min_constraint = min_constraints[tid][0];
  double max_constraint = max_constraints[tid][0];
  CHECK(tree->LeafOutput(leaf_index) >= min_constraint);
  CHECK(tree->LeafOutput(leaf_index) <= max_constraint);

  min_constraints[tid][0] = -std::numeric_limits<double>::max();
  max_constraints[tid][0] = std::numeric_limits<double>::max();
  thresholds[tid].clear();
  is_in_right_split[tid].clear();
  features[tid].clear();
#endif
}

// initializing constraints is just writing that the constraints should +/- inf from threshold 0
void SerialTreeLearner::InitializeConstraints(unsigned int tid) {
  thresholds[tid].clear();
  is_in_right_split[tid].clear();
  features[tid].clear();

  thresholds_min_constraints[tid].resize(1);
  thresholds_max_constraints[tid].resize(1);

  dummy_min_constraints[tid].resize(1);
  min_constraints[tid].resize(1);
  dummy_max_constraints[tid].resize(1);
  max_constraints[tid].resize(1);

  dummy_min_constraints[tid][0] = -std::numeric_limits<double>::max();
  min_constraints[tid][0] = -std::numeric_limits<double>::max();
  dummy_max_constraints[tid][0] = std::numeric_limits<double>::max();
  max_constraints[tid][0] = std::numeric_limits<double>::max();

  thresholds_min_constraints[tid][0] = 0;
  thresholds_max_constraints[tid][0] = 0;
}

double SerialTreeLearner::ComputeMonotoneSplitGainPenalty(int depth,
                                                          double penalization,
                                                          double epsilon) {
  if (penalization >= depth + 1.) {
    return epsilon;
  }
  if (penalization <= 1.) {
    return 1. - penalization / pow(2., depth) + epsilon;
  }
  return 1. - pow(2, penalization - 1. - depth) + epsilon;
}

}  // namespace LightGBM
