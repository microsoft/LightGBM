#include "serial_tree_learner.h"

#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

#ifdef TIMETAG
std::chrono::duration<double, std::milli> init_train_time;
std::chrono::duration<double, std::milli> init_split_time;
std::chrono::duration<double, std::milli> hist_time;
std::chrono::duration<double, std::milli> find_split_time;
std::chrono::duration<double, std::milli> split_time;
std::chrono::duration<double, std::milli> ordered_bin_time;
#endif // TIMETAG

SerialTreeLearner::SerialTreeLearner(const TreeConfig* tree_config)
  :tree_config_(tree_config){
  random_ = Random(tree_config_->feature_fraction_seed);
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

void SerialTreeLearner::Init(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();
  int max_cache_size = 0;
  // Get the max size of pool
  if (tree_config_->histogram_pool_size <= 0) {
    max_cache_size = tree_config_->num_leaves;
  } else {
    size_t total_histogram_size = 0;
    for (int i = 0; i < train_data_->num_features(); ++i) {
      total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureNumBin(i);
    }
    max_cache_size = static_cast<int>(tree_config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
  }
  // at least need 2 leaves
  max_cache_size = std::max(2, max_cache_size);
  max_cache_size = std::min(max_cache_size, tree_config_->num_leaves);

  histogram_pool_.DynamicChangeSize(train_data_, tree_config_, max_cache_size, tree_config_->num_leaves);
  // push split information for all leaves
  best_split_per_leaf_.resize(tree_config_->num_leaves);
  
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
  smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));
  larger_leaf_splits_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));

  // initialize data partition
  data_partition_.reset(new DataPartition(num_data_, tree_config_->num_leaves));
  is_feature_used_.resize(num_features_);
  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), 0);
    order_bin_indices_.clear();
    for (int i = 0; i < static_cast<int>(ordered_bins_.size()); i++) {
      if (ordered_bins_[i] != nullptr) {
        order_bin_indices_.push_back(i);
      }
    }
  }
  Log::Info("Number of data: %d, number of used features: %d", num_data_, num_features_);
}

void SerialTreeLearner::ResetTrainingData(const Dataset* train_data) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  num_features_ = train_data_->num_features();

  // get ordered bin
  train_data_->CreateOrderedBins(&ordered_bins_);

  has_ordered_bin_ = false;
  // check existing for ordered bin
  for (int i = 0; i < static_cast<int>(ordered_bins_.size()); ++i) {
    if (ordered_bins_[i] != nullptr) {
      has_ordered_bin_ = true;
      break;
    }
  }
  // initialize splits for leaf
  smaller_leaf_splits_->ResetNumData(num_data_);
  larger_leaf_splits_->ResetNumData(num_data_);

  // initialize data partition
  data_partition_->ResetNumData(num_data_);

  is_feature_used_.resize(num_features_);

  // initialize ordered gradients and hessians
  ordered_gradients_.resize(num_data_);
  ordered_hessians_.resize(num_data_);
  // if has ordered bin, need to allocate a buffer to fast split
  if (has_ordered_bin_) {
    is_data_in_leaf_.resize(num_data_);
    std::fill(is_data_in_leaf_.begin(), is_data_in_leaf_.end(), 0);
    order_bin_indices_.clear();
    for (int i = 0; i < static_cast<int>(ordered_bins_.size()); i++) {
      if (ordered_bins_[i] != nullptr) {
        order_bin_indices_.push_back(i);
      }
    }
  }

}

void SerialTreeLearner::ResetConfig(const TreeConfig* tree_config) {
  if (tree_config_->num_leaves != tree_config->num_leaves) {
    tree_config_ = tree_config;
    int max_cache_size = 0;
    // Get the max size of pool
    if (tree_config->histogram_pool_size <= 0) {
      max_cache_size = tree_config_->num_leaves;
    } else {
      size_t total_histogram_size = 0;
      for (int i = 0; i < train_data_->num_features(); ++i) {
        total_histogram_size += sizeof(HistogramBinEntry) * train_data_->FeatureNumBin(i);
      }
      max_cache_size = static_cast<int>(tree_config_->histogram_pool_size * 1024 * 1024 / total_histogram_size);
    }
    // at least need 2 leaves
    max_cache_size = std::max(2, max_cache_size);
    max_cache_size = std::min(max_cache_size, tree_config_->num_leaves);
    histogram_pool_.DynamicChangeSize(train_data_, tree_config_, max_cache_size, tree_config_->num_leaves);

    // push split information for all leaves
    best_split_per_leaf_.resize(tree_config_->num_leaves);
    data_partition_->ResetLeaves(tree_config_->num_leaves);
  } else {
    tree_config_ = tree_config;
  }

  histogram_pool_.ResetConfig(tree_config_);
}

Tree* SerialTreeLearner::Train(const score_t* gradients, const score_t *hessians) {
  gradients_ = gradients;
  hessians_ = hessians;

#ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
#endif
  // some initial works before training
  BeforeTrain();

#ifdef TIMETAG
  init_train_time += std::chrono::steady_clock::now() - start_time;
#endif

  auto tree = std::unique_ptr<Tree>(new Tree(tree_config_->num_leaves));
  // save pointer to last trained tree
  last_trained_tree_ = tree.get();
  // root leaf
  int left_leaf = 0;
  int cur_depth = 1;
  // only root leaf can be splitted on first time
  int right_leaf = -1;
  for (int split = 0; split < tree_config_->num_leaves - 1; ++split) {
#ifdef TIMETAG
    start_time = std::chrono::steady_clock::now();
#endif
    // some initial works before finding best split
    if (BeforeFindBestSplit(left_leaf, right_leaf)) {
#ifdef TIMETAG
      init_split_time += std::chrono::steady_clock::now() - start_time;
#endif
      // find best threshold for every feature
      FindBestThresholds();
      // find best split from all features
      FindBestSplitsForLeaves();
    }
    // Get a leaf with max split gain
    int best_leaf = static_cast<int>(ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_));
    // Get split information for best leaf
    const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];
    // cannot split, quit
    if (best_leaf_SplitInfo.gain <= 0.0) {
      Log::Info("No further splits with positive gain, best gain: %f", best_leaf_SplitInfo.gain);
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
  Log::Info("Trained a tree with leaves=%d and max_depth=%d", tree->num_leaves(), cur_depth);
  return tree.release();
}

void SerialTreeLearner::BeforeTrain() {

  // reset histogram pool
  histogram_pool_.ResetMap();
  int used_feature_cnt = static_cast<int>(num_features_*tree_config_->feature_fraction);

  if (used_feature_cnt < num_features_) {
    // initialize used features
    std::memset(is_feature_used_.data(), 0, sizeof(int8_t) * num_features_);
    // Get used feature at current tree
    auto used_feature_indices = random_.Sample(num_features_, used_feature_cnt);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(used_feature_indices.size()); ++i) {
      is_feature_used_[used_feature_indices[i]] = 1;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_features_; ++i) {
      is_feature_used_[i] = 1;
    }
  }

  // initialize data partition
  data_partition_->Init();

  // reset the splits for leaves
  for (int i = 0; i < tree_config_->num_leaves; ++i) {
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
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(order_bin_indices_.size()); ++i) {
        ordered_bins_[order_bin_indices_[i]]->Init(nullptr, tree_config_->num_leaves);
      }
    } else {
      // bagging, only use part of data

      // mark used data
      const data_size_t* indices = data_partition_->indices();
      data_size_t begin = data_partition_->leaf_begin(0);
      data_size_t end = begin + data_partition_->leaf_count(0);
      #pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 1;
      }
      // initialize ordered bin
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < static_cast<int>(order_bin_indices_.size()); ++i) {
        ordered_bins_[order_bin_indices_[i]]->Init(is_data_in_leaf_.data(), tree_config_->num_leaves);
      }
#pragma omp parallel for schedule(static)
      for (data_size_t i = begin; i < end; ++i) {
        is_data_in_leaf_[indices[i]] = 0;
      }
    }
#ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
#endif
  }
}

bool SerialTreeLearner::BeforeFindBestSplit(int left_leaf, int right_leaf) {
  // check depth of current leaf
  if (tree_config_->max_depth > 0) {
    // only need to check left leaf, since right leaf is in same level of left leaf
    if (last_trained_tree_->leaf_depth(left_leaf) >= tree_config_->max_depth) {
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
  if (num_data_in_right_child < static_cast<data_size_t>(tree_config_->min_data_in_leaf * 2)
    && num_data_in_left_child < static_cast<data_size_t>(tree_config_->min_data_in_leaf * 2)) {
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
    #pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 1;
    }
    // split the ordered bin
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(order_bin_indices_.size()); ++i) {
      ordered_bins_[order_bin_indices_[i]]->Split(left_leaf, right_leaf, is_data_in_leaf_.data(), mark);
    }
#pragma omp parallel for schedule(static)
    for (data_size_t i = begin; i < end; ++i) {
      is_data_in_leaf_[indices[i]] = 0;
    }
#ifdef TIMETAG
    ordered_bin_time += std::chrono::steady_clock::now() - start_time;
#endif
  }
  return true;
}

void SerialTreeLearner::FindBestThresholds() {
#ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
#endif
  std::vector<int8_t> is_feature_used(num_features_, 0);
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index]) continue;
    if (parent_leaf_histogram_array_ != nullptr 
        && !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }
  bool use_subtract = true;
  if (parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }
  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  train_data_->ConstructHistograms(is_feature_used,
    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
    smaller_leaf_splits_->LeafIndex(),
    ordered_bins_, gradients_, hessians_,
    ordered_gradients_.data(), ordered_hessians_.data(),
    ptr_smaller_leaf_hist_data);

  if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
    // construct larger leaf
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
      larger_leaf_splits_->LeafIndex(),
      ordered_bins_, gradients_, hessians_,
      ordered_gradients_.data(), ordered_hessians_.data(),
      ptr_larger_leaf_hist_data);
  }
#ifdef TIMETAG
  hist_time += std::chrono::steady_clock::now() - start_time;
#endif
#ifdef TIMETAG
  start_time = std::chrono::steady_clock::now();
#endif
  std::vector<SplitInfo> smaller_best(num_threads_);
  std::vector<SplitInfo> larger_best(num_threads_);
  // find splits
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used[feature_index]) { continue; }
    const int tid = omp_get_thread_num();
    SplitInfo smaller_split;
    train_data_->FixHistogram(feature_index, 
      smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      smaller_leaf_histogram_array_[feature_index].RawData());

    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      &smaller_split);
    if (smaller_split.gain > smaller_best[tid].gain) {
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
      &larger_split);
    if (larger_split.gain > larger_best[tid].gain) {
      larger_best[tid] = larger_split;
    }
  }

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

void SerialTreeLearner::FindBestSplitsForLeaves() {

}


void SerialTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
  // left = parent
  *left_leaf = best_Leaf;
  // split tree, will return right leaf
  *right_leaf = tree->Split(best_Leaf, best_split_info.feature,
    train_data_->FeatureBinMapper(best_split_info.feature)->bin_type(),
    best_split_info.threshold,
    train_data_->RealFeatureIndex(best_split_info.feature),
    train_data_->RealThreshold(best_split_info.feature, best_split_info.threshold),
    static_cast<double>(best_split_info.left_output),
    static_cast<double>(best_split_info.right_output),
    static_cast<data_size_t>(best_split_info.left_count),
    static_cast<data_size_t>(best_split_info.right_count),
    static_cast<double>(best_split_info.gain));
  // split data partition
  data_partition_->Split(best_Leaf, train_data_, best_split_info.feature, 
                         best_split_info.threshold, *right_leaf);

  // init the leaves that used on next iteration
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(),
                               best_split_info.left_sum_gradient,
                               best_split_info.left_sum_hessian);
    larger_leaf_splits_->Init(*right_leaf, data_partition_.get(),
                               best_split_info.right_sum_gradient,
                               best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
    larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
  }
}


}  // namespace LightGBM
