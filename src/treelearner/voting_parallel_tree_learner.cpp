#include "parallel_tree_learner.h"

#include <LightGBM/utils/common.h>

#include <cstring>

#include <tuple>
#include <vector>

namespace LightGBM {

VotingParallelTreeLearner::VotingParallelTreeLearner(const TreeConfig* tree_config)
  :SerialTreeLearner(tree_config) {
  top_k_ = tree_config_->top_k;
}

void VotingParallelTreeLearner::Init(const Dataset* train_data) {
  SerialTreeLearner::Init(train_data);
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // limit top k
  if (top_k_ > num_features_) {
    top_k_ = num_features_;
  }
  // get max bin
  int max_bin = 0;
  for (int i = 0; i < num_features_; ++i) {
    if (max_bin < train_data_->FeatureAt(i)->num_bin()) {
      max_bin = train_data_->FeatureAt(i)->num_bin();
    }
  }
  // calculate buffer size
  size_t buffer_size = 2 * top_k_ * std::max(max_bin * sizeof(HistogramBinEntry), sizeof(SplitInfo) * num_machines_);
  // left and right on same time, so need double size
  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  smaller_is_feature_aggregated_.resize(num_features_);
  larger_is_feature_aggregated_.resize(num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  smaller_buffer_read_start_pos_.resize(num_features_);
  larger_buffer_read_start_pos_.resize(num_features_);
  global_data_count_in_leaf_.resize(tree_config_->num_leaves);

  smaller_leaf_splits_global_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));
  larger_leaf_splits_global_.reset(new LeafSplits(train_data_->num_features(), train_data_->num_data()));

  local_tree_config_ = *tree_config_;
  local_tree_config_.min_data_in_leaf /= num_machines_;
  local_tree_config_.min_sum_hessian_in_leaf /= num_machines_;

  histogram_pool_.ResetConfig(&local_tree_config_, train_data_->num_features());

  // initialize histograms for global
  smaller_leaf_histogram_array_global_.reset(new FeatureHistogram[num_features_]);
  larger_leaf_histogram_array_global_.reset(new FeatureHistogram[num_features_]);
  for (int j = 0; j < num_features_; ++j) {
    smaller_leaf_histogram_array_global_[j].Init(train_data_->FeatureAt(j), j, tree_config_);
    larger_leaf_histogram_array_global_[j].Init(train_data_->FeatureAt(j), j, tree_config_);
  }
}

void VotingParallelTreeLearner::ResetConfig(const TreeConfig* tree_config) {
  SerialTreeLearner::ResetConfig(tree_config);

  local_tree_config_ = *tree_config_;
  local_tree_config_.min_data_in_leaf /= num_machines_;
  local_tree_config_.min_sum_hessian_in_leaf /= num_machines_;

  histogram_pool_.ResetConfig(&local_tree_config_, train_data_->num_features());
  global_data_count_in_leaf_.resize(tree_config_->num_leaves);

  for (int j = 0; j < num_features_; ++j) {
    smaller_leaf_histogram_array_global_[j].ResetConfig(tree_config_);
    larger_leaf_histogram_array_global_[j].ResetConfig(tree_config_);
  }
}

void VotingParallelTreeLearner::BeforeTrain() {
  SerialTreeLearner::BeforeTrain();
  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(smaller_leaf_splits_->num_data_in_leaf(), smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians());
  int size = sizeof(std::tuple<data_size_t, double, double>);
  std::memcpy(input_buffer_.data(), &data, size);

  Network::Allreduce(input_buffer_.data(), size, size, output_buffer_.data(), [](const char *src, char *dst, int len) {
    int used_size = 0;
    int type_size = sizeof(std::tuple<data_size_t, double, double>);
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  });

  std::memcpy(&data, output_buffer_.data(), size);

  // set global sumup info
  smaller_leaf_splits_global_->Init(std::get<1>(data), std::get<2>(data));
  larger_leaf_splits_global_->Init();
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

bool VotingParallelTreeLearner::BeforeFindBestSplit(int left_leaf, int right_leaf) {
  if (SerialTreeLearner::BeforeFindBestSplit(left_leaf, right_leaf)) {
    data_size_t num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
    data_size_t num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);
    if (right_leaf < 0) {
      return true;
    } else if (num_data_in_left_child < num_data_in_right_child) {
      // get local sumup
      smaller_leaf_splits_->Init(left_leaf, data_partition_.get(), gradients_, hessians_);
      larger_leaf_splits_->Init(right_leaf, data_partition_.get(), gradients_, hessians_);
    } else {
      // get local sumup
      smaller_leaf_splits_->Init(right_leaf, data_partition_.get(), gradients_, hessians_);
      larger_leaf_splits_->Init(left_leaf, data_partition_.get(), gradients_, hessians_);
    }
    return true;
  } else {
    return false;
  }
}

void VotingParallelTreeLearner::GlobalVoting(int leaf_idx, const std::vector<SplitInfo>& splits, std::vector<int>* out) {
  out->clear();
  if (leaf_idx < 0) {
    return;
  }
  // get mean number on machines
  score_t mean_num_data = GetGlobalDataCountInLeaf(leaf_idx) / static_cast<score_t>(num_machines_);
  std::vector<SplitInfo> feature_best_split(num_features_, SplitInfo());
  for (auto & split : splits) {
    int fid = split.feature;
    if (fid < 0) {
      continue;
    }
    // weighted gain
    double gain = split.gain * (split.left_count + split.right_count) / mean_num_data;
    if (gain > feature_best_split[fid].gain) {
      feature_best_split[fid] = split;
      feature_best_split[fid].gain = gain;
    }
  }
  // get top k
  std::vector<SplitInfo> top_k_splits;
  ArrayArgs<SplitInfo>::MaxK(feature_best_split, top_k_, &top_k_splits);
  for (auto& split : top_k_splits) {
    if (split.gain == kMinScore || split.feature == -1) {
      continue;
    }
    out->push_back(split.feature);
  }
}

void VotingParallelTreeLearner::CopyLocalHistogram(const std::vector<int>& smaller_top_features, const std::vector<int>& larger_top_features) {
  for (int i = 0; i < num_features_; ++i) {
    smaller_is_feature_aggregated_[i] = false;
    larger_is_feature_aggregated_[i] = false;
  }
  size_t total_num_features = smaller_top_features.size() + larger_top_features.size();
  size_t average_feature = (total_num_features + num_machines_ - 1) / num_machines_;
  size_t used_num_features = 0, smaller_idx = 0, larger_idx = 0;
  block_start_[0] = 0;
  reduce_scatter_size_ = 0;
  // Copy histogram to buffer, and Get local aggregate features
  for (int i = 0; i < num_machines_; ++i) {
    size_t cur_size = 0, cur_used_features = 0;
    size_t cur_total_feature = std::min(average_feature, total_num_features - used_num_features);
    // copy histograms.
    while (cur_used_features < cur_total_feature) {
      // copy smaller leaf histograms first
      if (smaller_idx < smaller_top_features.size()) {
        int fid = smaller_top_features[smaller_idx];
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          smaller_is_feature_aggregated_[fid] = true;
          smaller_buffer_read_start_pos_[fid] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, smaller_leaf_histogram_array_[fid].HistogramData(), smaller_leaf_histogram_array_[fid].SizeOfHistgram());
        cur_size += smaller_leaf_histogram_array_[fid].SizeOfHistgram();
        reduce_scatter_size_ += smaller_leaf_histogram_array_[fid].SizeOfHistgram();
        ++smaller_idx;
      }
      if (cur_used_features >= cur_total_feature) {
        break;
      }
      // then copy larger leaf histograms
      if (larger_idx < larger_top_features.size()) {
        int fid = larger_top_features[larger_idx];
        ++cur_used_features;
        // mark local aggregated feature
        if (i == rank_) {
          larger_is_feature_aggregated_[fid] = true;
          larger_buffer_read_start_pos_[fid] = static_cast<int>(cur_size);
        }
        // copy
        std::memcpy(input_buffer_.data() + reduce_scatter_size_, larger_leaf_histogram_array_[fid].HistogramData(), larger_leaf_histogram_array_[fid].SizeOfHistgram());
        cur_size += larger_leaf_histogram_array_[fid].SizeOfHistgram();
        reduce_scatter_size_ += larger_leaf_histogram_array_[fid].SizeOfHistgram();
        ++larger_idx;
      }
    }
    used_num_features += cur_used_features;
    block_len_[i] = static_cast<int>(cur_size);
    if (i < num_machines_ - 1) {
      block_start_[i + 1] = block_start_[i] + block_len_[i];
    }
  }
}

void VotingParallelTreeLearner::FindBestThresholds() {
  // use local data to find local best splits
  SerialTreeLearner::FindBestThresholds();
  std::vector<SplitInfo> smaller_top_k_splits, larger_top_k_splits;
  // local voting
  ArrayArgs<SplitInfo>::MaxK(smaller_leaf_splits_->BestSplitPerFeature(), top_k_, &smaller_top_k_splits);
  ArrayArgs<SplitInfo>::MaxK(larger_leaf_splits_->BestSplitPerFeature(), top_k_, &larger_top_k_splits);
  // gather
  int offset = 0;
  for (int i = 0; i < top_k_; ++i) {
    std::memcpy(input_buffer_.data() + offset, &smaller_top_k_splits[i], sizeof(SplitInfo));
    offset += sizeof(SplitInfo);
    std::memcpy(input_buffer_.data() + offset, &larger_top_k_splits[i], sizeof(SplitInfo));
    offset += sizeof(SplitInfo);
  }
  Network::Allgather(input_buffer_.data(), offset, output_buffer_.data());
  // get all top-k from all machines
  std::vector<SplitInfo> smaller_top_k_splits_global;
  std::vector<SplitInfo> larger_top_k_splits_global;
  offset = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (int j = 0; j < top_k_; ++j) {
      smaller_top_k_splits_global.push_back(SplitInfo());
      std::memcpy(&smaller_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(SplitInfo));
      offset += sizeof(SplitInfo);
      larger_top_k_splits_global.push_back(SplitInfo());
      std::memcpy(&larger_top_k_splits_global.back(), output_buffer_.data() + offset, sizeof(SplitInfo));
      offset += sizeof(SplitInfo);
    }
  }
  // global voting
  std::vector<int> smaller_top_features, larger_top_features;
  GlobalVoting(smaller_leaf_splits_->LeafIndex(), smaller_top_k_splits_global, &smaller_top_features);
  GlobalVoting(larger_leaf_splits_->LeafIndex(), larger_top_k_splits_global, &larger_top_features);
  // copy local histgrams to buffer
  CopyLocalHistogram(smaller_top_features, larger_top_features);

  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, block_start_.data(), block_len_.data(),
                         output_buffer_.data(), &HistogramBinEntry::SumReducer);
  // find best split from local aggregated histograms
  #pragma omp parallel for schedule(guided)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {

    if (smaller_is_feature_aggregated_[feature_index]) {
      smaller_leaf_histogram_array_global_[feature_index].SetSumup(
                                   GetGlobalDataCountInLeaf(smaller_leaf_splits_global_->LeafIndex()),
                                                            smaller_leaf_splits_global_->sum_gradients(),
                                                            smaller_leaf_splits_global_->sum_hessians());
      // restore from buffer
      smaller_leaf_histogram_array_global_[feature_index].FromMemory(
                                   output_buffer_.data() + smaller_buffer_read_start_pos_[feature_index]);
      // find best threshold
      smaller_leaf_histogram_array_global_[feature_index].FindBestThreshold(
                                   &smaller_leaf_splits_global_->BestSplitPerFeature()[feature_index]);
    }

    if (larger_is_feature_aggregated_[feature_index]) {
      larger_leaf_histogram_array_global_[feature_index].SetSumup(GetGlobalDataCountInLeaf(larger_leaf_splits_global_->LeafIndex()),
                                                                  larger_leaf_splits_global_->sum_gradients(), larger_leaf_splits_global_->sum_hessians());
      // restore from buffer
      larger_leaf_histogram_array_global_[feature_index].FromMemory(output_buffer_.data() + larger_buffer_read_start_pos_[feature_index]);
      // find best threshold
      larger_leaf_histogram_array_global_[feature_index].FindBestThreshold(&larger_leaf_splits_global_->BestSplitPerFeature()[feature_index]);
    }
  }

}

void VotingParallelTreeLearner::FindBestSplitsForLeaves() {
  int smaller_best_feature = -1, larger_best_feature = -1;
  // find local best
  SplitInfo smaller_best, larger_best;
  std::vector<double> gains;
  for (size_t i = 0; i < smaller_leaf_splits_global_->BestSplitPerFeature().size(); ++i) {
    gains.push_back(smaller_leaf_splits_global_->BestSplitPerFeature()[i].gain);
  }
  smaller_best_feature = static_cast<int>(ArrayArgs<double>::ArgMax(gains));
  smaller_best = smaller_leaf_splits_global_->BestSplitPerFeature()[smaller_best_feature];

  if (larger_leaf_splits_global_->LeafIndex() >= 0) {
    gains.clear();
    for (size_t i = 0; i < larger_leaf_splits_global_->BestSplitPerFeature().size(); ++i) {
      gains.push_back(larger_leaf_splits_global_->BestSplitPerFeature()[i].gain);
    }
    larger_best_feature = static_cast<int>(ArrayArgs<double>::ArgMax(gains));
    larger_best = larger_leaf_splits_global_->BestSplitPerFeature()[larger_best_feature];
  }
  // sync global best info
  std::memcpy(input_buffer_.data(), &smaller_best, sizeof(SplitInfo));
  std::memcpy(input_buffer_.data() + sizeof(SplitInfo), &larger_best, sizeof(SplitInfo));

  Network::Allreduce(input_buffer_.data(), sizeof(SplitInfo) * 2, sizeof(SplitInfo), output_buffer_.data(), &SplitInfo::MaxReducer);

  std::memcpy(&smaller_best, output_buffer_.data(), sizeof(SplitInfo));
  std::memcpy(&larger_best, output_buffer_.data() + sizeof(SplitInfo), sizeof(SplitInfo));

  // copy back
  best_split_per_leaf_[smaller_leaf_splits_global_->LeafIndex()] = smaller_best;
  if (larger_best.feature >= 0 && larger_leaf_splits_global_->LeafIndex() >= 0) {
    best_split_per_leaf_[larger_leaf_splits_global_->LeafIndex()] = larger_best;
  }
}

void VotingParallelTreeLearner::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  SerialTreeLearner::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo& best_split_info = best_split_per_leaf_[best_Leaf];
  // set the global number of data for leaves
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
  // init the global sumup info
  if (best_split_info.left_count < best_split_info.right_count) {
    smaller_leaf_splits_global_->Init(*left_leaf, data_partition_.get(),
                                      best_split_info.left_sum_gradient,
                                      best_split_info.left_sum_hessian);
    larger_leaf_splits_global_->Init(*right_leaf, data_partition_.get(),
                                     best_split_info.right_sum_gradient,
                                     best_split_info.right_sum_hessian);
  } else {
    smaller_leaf_splits_global_->Init(*right_leaf, data_partition_.get(),
                                      best_split_info.right_sum_gradient,
                                      best_split_info.right_sum_hessian);
    larger_leaf_splits_global_->Init(*left_leaf, data_partition_.get(),
                                     best_split_info.left_sum_gradient,
                                     best_split_info.left_sum_hessian);
  }
}

}  // namespace FTLBoost
