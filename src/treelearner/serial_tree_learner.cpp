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
#endif  // TIMETAG

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
      FindBestSplits();
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
  Log::Debug("Trained a tree with leaves = %d and max_depth = %d", tree->num_leaves(), cur_depth);
  return tree.release();
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
  }

  // Sumup for root
  if (data_partition_->leaf_count(0) == num_data_) {
    // use all data
    smaller_leaf_splits_->Init(gradients_, hessians_);

  } else {
    // use bagging, only use part of data
    smaller_leaf_splits_->Init(0, data_partition_.get(), gradients_, hessians_);
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

void SerialTreeLearner::FindBestSplits() {
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
  FindBestSplitsFromHistograms(is_feature_used, use_subtract);
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

void SerialTreeLearner::FindBestSplitsFromHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) {
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
    SplitInfo smaller_split;
    train_data_->FixHistogram(feature_index,
                              smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                              smaller_leaf_splits_->num_data_in_leaf(),
                              smaller_leaf_histogram_array_[feature_index].RawData());
    int real_fidx = train_data_->RealFeatureIndex(feature_index);
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      smaller_leaf_splits_->min_constraint(),
      smaller_leaf_splits_->max_constraint(),
      &smaller_split);
    smaller_split.feature = real_fidx;
    if (cegb_ != nullptr) {
      smaller_split.gain -= cegb_->DetlaGain(feature_index, real_fidx, smaller_leaf_splits_->LeafIndex(), smaller_leaf_splits_->num_data_in_leaf(), smaller_split);
    }
    if (smaller_split > smaller_best[tid] && smaller_node_used_features[feature_index]) {
      smaller_best[tid] = smaller_split;
    }
    // only has root leaf
    if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                larger_leaf_splits_->num_data_in_leaf(),
                                larger_leaf_histogram_array_[feature_index].RawData());
    }
    SplitInfo larger_split;
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      larger_leaf_splits_->sum_gradients(),
      larger_leaf_splits_->sum_hessians(),
      larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->min_constraint(),
      larger_leaf_splits_->max_constraint(),
      &larger_split);
    larger_split.feature = real_fidx;
    if (cegb_ != nullptr) {
      larger_split.gain -= cegb_->DetlaGain(feature_index, real_fidx, larger_leaf_splits_->LeafIndex(), larger_leaf_splits_->num_data_in_leaf(), larger_split);
    }
    if (larger_split > larger_best[tid] && larger_node_used_features[feature_index]) {
      larger_best[tid] = larger_split;
    }
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
      FindBestSplits();
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
                                current_split_info.default_left);
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
                                           train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
      data_partition_->Split(current_leaf, train_data_, inner_feature_index,
                             cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()),
                             current_split_info.default_left, *right_leaf);
    }

    if (current_split_info.left_count < current_split_info.right_count) {
      left_smaller = true;
      smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                 current_split_info.left_sum_gradient,
                                 current_split_info.left_sum_hessian);
      larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                current_split_info.right_sum_gradient,
                                current_split_info.right_sum_hessian);
    } else {
      left_smaller = false;
      smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                                 current_split_info.right_sum_gradient, current_split_info.right_sum_hessian);
      larger_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                                current_split_info.left_sum_gradient, current_split_info.left_sum_hessian);
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
                              best_split_info.default_left);
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
                                         train_data_->FeatureBinMapper(inner_feature_index)->missing_type());
    data_partition_->Split(best_leaf, train_data_, inner_feature_index,
                           cat_bitset_inner.data(), static_cast<int>(cat_bitset_inner.size()), best_split_info.default_left, *right_leaf);
  }

  #ifdef DEBUG
  CHECK(best_split_info.left_count == data_partition_->leaf_count(best_leaf));
  #endif
  auto p_left = smaller_leaf_splits_.get();
  auto p_right = larger_leaf_splits_.get();
  // init the leaves that used on next iteration
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
    p_right = smaller_leaf_splits_.get();
    p_left = larger_leaf_splits_.get();
  }
  p_left->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);
  p_right->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);
  if (is_numerical_split) {
    double mid = (best_split_info.left_output + best_split_info.right_output) / 2.0f;
    if (best_split_info.monotone_type < 0) {
      p_left->SetValueConstraint(mid, best_split_info.max_constraint);
      p_right->SetValueConstraint(best_split_info.min_constraint, mid);
    } else if (best_split_info.monotone_type > 0) {
      p_left->SetValueConstraint(best_split_info.min_constraint, mid);
      p_right->SetValueConstraint(mid, best_split_info.max_constraint);
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

}  // namespace LightGBM
