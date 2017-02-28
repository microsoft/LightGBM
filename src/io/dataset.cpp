#include <LightGBM/dataset.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/utils/array_args.h>

#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <limits>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

namespace LightGBM {

const char* Dataset::binary_file_token = "______LightGBM_Binary_File_Token______\n";

Dataset::Dataset() {
  data_filename_ = "noname";
  num_data_ = 0;
}

Dataset::Dataset(data_size_t num_data) {
  data_filename_ = "noname";
  num_data_ = num_data;
  metadata_.Init(num_data_, NO_SPECIFIC, NO_SPECIFIC);
}

Dataset::~Dataset() {
}

std::vector<std::vector<int>> NoGroup(
  const std::vector<int>& used_features) {
  std::vector<std::vector<int>> features_in_group;
  features_in_group.resize(used_features.size());
  for (size_t i = 0; i < used_features.size(); ++i) {
    features_in_group[i].emplace_back(used_features[i]);
  }
  return features_in_group;
}

int GetConfilctCount(const std::vector<bool>& mark, const std::vector<int>& indices, int max_cnt) {
  int ret = 0;
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    if (mark[indices[i]]) {
      ++ret;
      if (ret > max_cnt) {
        return -1;
      }
    }
  }
  return ret;
}
void MarkUsed(std::vector<bool>& mark, const std::vector<int>& indices) {
  for (size_t i = 0; i < indices.size(); ++i) {
    mark[indices[i]] = true;
  }
}


std::vector<std::vector<int>> FindGroups(const std::vector<int>& fnd_order,
  const std::vector<std::vector<int>>& sample_indices,
  size_t total_sample_cnt,
  data_size_t max_error_cnt,
  data_size_t filter_cnt,
  data_size_t num_data) {
  std::vector<std::vector<int>> features_in_group;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<int> group_conflict_cnt;
  std::vector<size_t> group_non_zero_cnt;
  for (auto fidx : fnd_order) {
    const size_t cur_non_zero_cnt = sample_indices[fidx].size();
    bool need_new_group = true;
    for (size_t gid = 0; gid < features_in_group.size(); ++gid) {
      const int rest_max_cnt = max_error_cnt - group_conflict_cnt[gid];
      int cnt = GetConfilctCount(conflict_marks[gid], sample_indices[fidx], rest_max_cnt);
      if (cnt >= 0 && cnt <= rest_max_cnt) {
        data_size_t rest_non_zero_data = static_cast<data_size_t>(
          static_cast<double>(cur_non_zero_cnt - cnt) * num_data / total_sample_cnt);
        if (rest_non_zero_data < filter_cnt) { continue; }
        need_new_group = false;
        features_in_group[gid].push_back(fidx);
        group_conflict_cnt[gid] += cnt;
        group_non_zero_cnt[gid] += cur_non_zero_cnt - cnt;
        MarkUsed(conflict_marks[gid], sample_indices[fidx]);
        break;
      }
    }
    if (need_new_group) {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      group_conflict_cnt.push_back(0);
      conflict_marks.emplace_back(total_sample_cnt, false);
      MarkUsed(conflict_marks.back(), sample_indices[fidx]);
      group_non_zero_cnt.emplace_back(cur_non_zero_cnt);
    }
  }
  return features_in_group;
}

std::vector<std::vector<int>> FastFeatureBundling(
  std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
  const std::vector<std::vector<int>>& sample_indices,
  size_t total_sample_cnt,
  const std::vector<int>& used_features,
  double max_conflict_rate,
  data_size_t num_data,
  data_size_t min_data,
  bool is_enable_sparse) {
  // filter is based on sampling data, so decrease its range
  const data_size_t filter_cnt = static_cast<data_size_t>(static_cast<double>(0.95 * min_data) / num_data * total_sample_cnt);
  const data_size_t max_error_cnt = static_cast<data_size_t>(total_sample_cnt * max_conflict_rate);
  int cur_used_feature_cnt = 0;
  std::vector<size_t> feature_non_zero_cnt;
  // put dense feature first
  for (auto fidx : used_features) {
    feature_non_zero_cnt.emplace_back(sample_indices[fidx].size());
    ++cur_used_feature_cnt;
  }
  // sort by non zero cnt
  std::vector<int> sorted_idx;
  for (int i = 0; i < cur_used_feature_cnt; ++i) {
    sorted_idx.emplace_back(i);
  }
  // sort by non zero cnt, bigger first
  std::sort(sorted_idx.begin(), sorted_idx.end(),
    [&feature_non_zero_cnt](int a, int b) {
    return feature_non_zero_cnt[a] > feature_non_zero_cnt[b];
  });

  std::vector<int> feature_order_by_cnt;
  for (auto sidx : sorted_idx) {
    feature_order_by_cnt.push_back(used_features[sidx]);
  }
  // use this to discover continous one-hot coding
  auto features_in_group = FindGroups(used_features, sample_indices, total_sample_cnt, max_error_cnt, filter_cnt, num_data);
  // use this to discover hidden combine
  auto group2 = FindGroups(feature_order_by_cnt, sample_indices, total_sample_cnt, max_error_cnt, filter_cnt, num_data);
  if (features_in_group.size() > group2.size()) {
    features_in_group = group2;
  }
  Log::Info("feature clustering: merge %d features into %d groups", sample_indices.size(), features_in_group.size());
  std::vector<std::vector<int>> ret;
  for (size_t i = 0; i < features_in_group.size(); ++i) {
    if (features_in_group[i].size() <= 1 || features_in_group[i].size() >= 5) {
      ret.push_back(features_in_group[i]);
    } else {
      int cnt_non_zero = 0;
      for (size_t j = 0; j < features_in_group[i].size(); ++j) {
        const int fidx = features_in_group[i][j];
        cnt_non_zero += static_cast<int>(num_data * (1.0f - bin_mappers[fidx]->sparse_rate()));
      }
      double sparse_rate = 1.0f - static_cast<double>(cnt_non_zero) / (num_data);
      // take apart small sparse group, due it will not gain on speed 
      if (sparse_rate >= BinMapper::kSparseThreshold && is_enable_sparse) {
        for (size_t j = 0; j < features_in_group[i].size(); ++j) {
          const int fidx = features_in_group[i][j];
          ret.emplace_back();
          ret.back().push_back(fidx);
        }
      } else {
        ret.push_back(features_in_group[i]);
      }
    }
  }
  // shuffle groups
  int num_group = static_cast<int>(ret.size());
  Log::Info("feature clustering: real groups %d", num_group);
  Random tmp_rand(12);
  for (int i = 0; i < num_group - 1; ++i) {
    int j = tmp_rand.NextShort(i + 1, num_group);
    std::swap(ret[i], ret[j]);
  }
  return ret;
}

std::vector<std::vector<int>> AdjacentBundling(
  std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
  const std::vector<std::vector<int>>& sample_indices,
  size_t total_sample_cnt,
  const std::vector<int>& used_features,
  double max_conflict_rate,
  data_size_t num_data,
  bool is_enable_sparse) {

  // filter is based on sampling data, so decrease its range
  const data_size_t max_error_cnt = static_cast<data_size_t>(total_sample_cnt * max_conflict_rate);

  std::vector<std::vector<int>> features_in_group;
  std::vector<bool> conflict_marks(total_sample_cnt, false);
  int group_conflict_cnt = 0;
  for (auto fidx : used_features) {
    const int rest_max_cnt = max_error_cnt - group_conflict_cnt;
    int cnt = GetConfilctCount(conflict_marks, sample_indices[fidx], rest_max_cnt);
    if (!features_in_group.empty() && cnt >= 0 && cnt <= rest_max_cnt) {
      features_in_group.back().push_back(fidx);
      group_conflict_cnt += cnt;
      MarkUsed(conflict_marks, sample_indices[fidx]);
    } else {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      group_conflict_cnt = 0;
      conflict_marks = std::vector<bool>(total_sample_cnt, false);
      MarkUsed(conflict_marks, sample_indices[fidx]);
    }
  }
  Log::Info("feature clustering: merge %d features into %d groups", sample_indices.size(), features_in_group.size());
  std::vector<std::vector<int>> ret;
  for (size_t i = 0; i < features_in_group.size(); ++i) {
    if (features_in_group[i].size() <= 1 || features_in_group[i].size() >= 5) {
      ret.push_back(features_in_group[i]);
    } else {
      int cnt_non_zero = 0;
      for (size_t j = 0; j < features_in_group[i].size(); ++j) {
        const int fidx = features_in_group[i][j];
        cnt_non_zero += static_cast<int>(num_data * (1.0f - bin_mappers[fidx]->sparse_rate()));
      }
      double sparse_rate = 1.0f - static_cast<double>(cnt_non_zero) / (num_data);
      // take apart small sparse group, due it will not gain on speed 
      if (sparse_rate >= BinMapper::kSparseThreshold && is_enable_sparse) {
        for (size_t j = 0; j < features_in_group[i].size(); ++j) {
          const int fidx = features_in_group[i][j];
          ret.emplace_back();
          ret.back().push_back(fidx);
        }
      } else {
        ret.push_back(features_in_group[i]);
      }
    }
  }
  // shuffle groups
  int num_group = static_cast<int>(ret.size());
  Log::Info("feature clustering: real groups %d", num_group);
  Random tmp_rand(12);
  for (int i = 0; i < num_group - 1; ++i) {
    int j = tmp_rand.NextShort(i + 1, num_group);
    std::swap(ret[i], ret[j]);
  }
  return ret;
}

void Dataset::Construct(
  std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
  const std::vector<std::vector<int>>& sample_indices,
  size_t total_sample_cnt,
  const IOConfig& io_config) {
  num_total_features_ = static_cast<int>(bin_mappers.size());
  // get num_features
  std::vector<int> used_features;
  for (int i = 0; i < static_cast<int>(bin_mappers.size()); ++i) {
    if (bin_mappers[i] != nullptr && !bin_mappers[i]->is_trival()) {
      used_features.emplace_back(i);
    } 
  }

  auto features_in_group = NoGroup(used_features);
  std::chrono::duration<double, std::milli> bundling_time_;
  if (io_config.enable_bundle) {
    auto start_time = std::chrono::steady_clock::now();
    if (!io_config.adjacent_bundle) {
      features_in_group = FastFeatureBundling(bin_mappers,
        sample_indices, total_sample_cnt,
        used_features, io_config.max_conflict_rate,
        num_data_, io_config.min_data_in_leaf, io_config.is_enable_sparse);
    } else {
      Log::Info("using adjacent bundling");
      features_in_group = AdjacentBundling(bin_mappers,
        sample_indices, total_sample_cnt,
        used_features, io_config.max_conflict_rate,
        num_data_, io_config.is_enable_sparse);
    }
    bundling_time_ += std::chrono::steady_clock::now() - start_time;
    Log::Info("Cost %f seconds for bundling", bundling_time_* 1e-3);
  }

  num_features_ = 0;
  for (const auto& fs : features_in_group) {
    num_features_ += static_cast<int>(fs.size());
  }
  int cur_fidx = 0;
  used_feature_map_ = std::vector<int>(num_total_features_, -1);
  num_groups_ = static_cast<int>(features_in_group.size());
  real_feature_idx_.resize(num_features_);
  feature2group_.resize(num_features_);
  feature2subfeature_.resize(num_features_);
  for (int i = 0; i < num_groups_; ++i) {
    auto cur_features = features_in_group[i];
    int cur_cnt_features = static_cast<int>(cur_features.size());
    // get bin_mappers
    std::vector<std::unique_ptr<BinMapper>> cur_bin_mappers;
    for (int j = 0; j < cur_cnt_features; ++j) {
      int real_fidx = cur_features[j];
      used_feature_map_[real_fidx] = cur_fidx;
      real_feature_idx_[cur_fidx] = real_fidx;
      feature2group_[cur_fidx] = i;
      feature2subfeature_[cur_fidx] = j;
      cur_bin_mappers.emplace_back(bin_mappers[real_fidx].release());
      ++cur_fidx;
    }
    feature_groups_.emplace_back(std::unique_ptr<FeatureGroup>(
      new FeatureGroup(cur_cnt_features, cur_bin_mappers, num_data_, io_config.is_enable_sparse)));
  }
  feature_groups_.shrink_to_fit();
  group_bin_boundaries_.clear();
  uint64_t num_total_bin = 0;
  group_bin_boundaries_.push_back(num_total_bin);
  for (int i = 0; i < num_groups_; ++i) {
    num_total_bin += feature_groups_[i]->num_total_bin_;
    group_bin_boundaries_.push_back(num_total_bin);
  }
  int last_group = 0;
  group_feature_start_.reserve(num_groups_);
  group_feature_cnt_.reserve(num_groups_);
  group_feature_start_.push_back(0);
  group_feature_cnt_.push_back(1);
  for (int i = 1; i < num_features_; ++i) {
    const int group = feature2group_[i];
    if (group == last_group) {
      group_feature_cnt_.back() = group_feature_cnt_.back() + 1;
    } else {
      group_feature_start_.push_back(i);
      group_feature_cnt_.push_back(1);
      last_group = group;
    }
  }
}

void Dataset::FinishLoad() {
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_groups_; ++i) {
    feature_groups_[i]->bin_data_->FinishLoad();
  }
}

void Dataset::CopyFeatureMapperFrom(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = dataset->num_groups_;
  bool is_enable_sparse = false;
  for (int i = 0; i < num_groups_; ++i) {
    if (dataset->feature_groups_[i]->is_sparse_) {
      is_enable_sparse = true;
      break;
    }
  }
  // copy feature bin mapper data
  for (int i = 0; i < num_groups_; ++i) {
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    for (int j = 0; j < dataset->feature_groups_[i]->num_feature_; ++j) {
      bin_mappers.emplace_back(new BinMapper(*(dataset->feature_groups_[i]->bin_mappers_[j])));
    }
    feature_groups_.emplace_back(new FeatureGroup(
      dataset->feature_groups_[i]->num_feature_,
      bin_mappers,
      num_data_,
      is_enable_sparse));
  }
  feature_groups_.shrink_to_fit();
  used_feature_map_ = dataset->used_feature_map_;
  num_total_features_ = dataset->num_total_features_;
  feature_names_ = dataset->feature_names_;
  label_idx_ = dataset->label_idx_;
  real_feature_idx_ = dataset->real_feature_idx_;
  feature2group_ = dataset->feature2group_;
  feature2subfeature_ = dataset->feature2subfeature_;
  group_bin_boundaries_ = dataset->group_bin_boundaries_;
  group_feature_start_ = dataset->group_feature_start_;
  group_feature_cnt_ = dataset->group_feature_cnt_;
}

void Dataset::CreateValid(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = num_features_;
  bool is_enable_sparse = true;
  feature2group_.clear();
  feature2subfeature_.clear();
  // copy feature bin mapper data
  for (int i = 0; i < num_features_; ++i) {
    std::vector<std::unique_ptr<BinMapper>> bin_mappers;
    bin_mappers.emplace_back(new BinMapper(*(dataset->FeatureBinMapper(i))));
    feature_groups_.emplace_back(new FeatureGroup(
      1,
      bin_mappers,
      num_data_,
      is_enable_sparse));
    feature2group_.push_back(i);
    feature2subfeature_.push_back(0);
  }

  feature_groups_.shrink_to_fit();
  used_feature_map_ = dataset->used_feature_map_;
  num_total_features_ = dataset->num_total_features_;
  feature_names_ = dataset->feature_names_;
  label_idx_ = dataset->label_idx_;
  real_feature_idx_ = dataset->real_feature_idx_;
  group_bin_boundaries_.clear();
  uint64_t num_total_bin = 0;
  group_bin_boundaries_.push_back(num_total_bin);
  for (int i = 0; i < num_groups_; ++i) {
    num_total_bin += feature_groups_[i]->num_total_bin_;
    group_bin_boundaries_.push_back(num_total_bin);
  }
  int last_group = 0;
  group_feature_start_.reserve(num_groups_);
  group_feature_cnt_.reserve(num_groups_);
  group_feature_start_.push_back(0);
  group_feature_cnt_.push_back(1);
  for (int i = 1; i < num_features_; ++i) {
    const int group = feature2group_[i];
    if (group == last_group) {
      group_feature_cnt_.back() = group_feature_cnt_.back() + 1;
    } else {
      group_feature_start_.push_back(i);
      group_feature_cnt_.push_back(1);
      last_group = group;
    }
  }
}

void Dataset::ReSize(data_size_t num_data) {
  if (num_data_ != num_data) {
    num_data_ = num_data;
#pragma omp parallel for schedule(static)
    for (int group = 0; group < num_groups_; ++group) {
      feature_groups_[group]->bin_data_->ReSize(num_data_);
    }
  }
}

void Dataset::CopySubset(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data) {
  CHECK(num_used_indices == num_data_);
#pragma omp parallel for schedule(static)
  for (int group = 0; group < num_groups_; ++group) {
    feature_groups_[group]->CopySubset(fullset->feature_groups_[group].get(), used_indices, num_used_indices);
  }
  if (need_meta_data) {
    metadata_.Init(fullset->metadata_, used_indices, num_used_indices);
  }
}

bool Dataset::SetFloatField(const char* field_name, const float* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    metadata_.SetLabel(field_data, num_element);
  } else if (name == std::string("weight") || name == std::string("weights")) {
    metadata_.SetWeights(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::SetDoubleField(const char* field_name, const double* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("init_score")) {
    metadata_.SetInitScore(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::SetIntField(const char* field_name, const int* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    metadata_.SetQuery(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetFloatField(const char* field_name, data_size_t* out_len, const float** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    *out_ptr = metadata_.label();
    *out_len = num_data_;
  } else if (name == std::string("weight") || name == std::string("weights")) {
    *out_ptr = metadata_.weights();
    *out_len = num_data_;
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetDoubleField(const char* field_name, data_size_t* out_len, const double** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("init_score")) {
    *out_ptr = metadata_.init_score();
    *out_len = static_cast<data_size_t>(metadata_.num_init_score());
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetIntField(const char* field_name, data_size_t* out_len, const int** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    *out_ptr = metadata_.query_boundaries();
    *out_len = metadata_.num_queries() + 1;
  } else {
    return false;
  }
  return true;
}

void Dataset::SaveBinaryFile(const char* bin_filename) {
  if (bin_filename != nullptr
    && std::string(bin_filename) == std::string(data_filename_)) {
    Log::Warning("Bianry file %s already existed", bin_filename);
    return;
  }
  // if not pass a filename, just append ".bin" of original file
  std::string bin_filename_str(data_filename_);
  if (bin_filename == nullptr || bin_filename[0] == '\0') {
    bin_filename_str.append(".bin");
    bin_filename = bin_filename_str.c_str();
  }
  bool is_file_existed = false;
  FILE* file;
#ifdef _MSC_VER
  fopen_s(&file, bin_filename, "rb");
#else
  file = fopen(bin_filename, "rb");
#endif

  if (file != NULL) {
    is_file_existed = true;
    Log::Warning("File %s existed, cannot save binary to it", bin_filename);
    fclose(file);
  }

  if (!is_file_existed) {
#ifdef _MSC_VER
    fopen_s(&file, bin_filename, "wb");
#else
    file = fopen(bin_filename, "wb");
#endif
    if (file == NULL) {
      Log::Fatal("Cannot write binary data to %s ", bin_filename);
    }
    Log::Info("Saving data to binary file %s", bin_filename);
    size_t size_of_token = std::strlen(binary_file_token);
    fwrite(binary_file_token, sizeof(char), size_of_token, file);
    // get size of header
    size_t size_of_header = sizeof(num_data_) + sizeof(num_features_) + sizeof(num_total_features_)
      + sizeof(int) * num_total_features_ + sizeof(num_groups_)
      + 3 * sizeof(int) * num_features_ + sizeof(uint64_t) * (num_groups_ + 1) + 2 * sizeof(int) * num_groups_;
    // size of feature names
    for (int i = 0; i < num_total_features_; ++i) {
      size_of_header += feature_names_[i].size() + sizeof(int);
    }
    fwrite(&size_of_header, sizeof(size_of_header), 1, file);
    // write header
    fwrite(&num_data_, sizeof(num_data_), 1, file);
    fwrite(&num_features_, sizeof(num_features_), 1, file);
    fwrite(&num_total_features_, sizeof(num_total_features_), 1, file);
    fwrite(used_feature_map_.data(), sizeof(int), num_total_features_, file);
    fwrite(&num_groups_, sizeof(num_groups_), 1, file);
    fwrite(real_feature_idx_.data(), sizeof(int), num_features_, file);
    fwrite(feature2group_.data(), sizeof(int), num_features_, file);
    fwrite(feature2subfeature_.data(), sizeof(int), num_features_, file);
    fwrite(group_bin_boundaries_.data(), sizeof(uint64_t), num_groups_ + 1, file);
    fwrite(group_feature_start_.data(), sizeof(int), num_groups_, file);
    fwrite(group_feature_cnt_.data(), sizeof(int), num_groups_, file);

    // write feature names
    for (int i = 0; i < num_total_features_; ++i) {
      int str_len = static_cast<int>(feature_names_[i].size());
      fwrite(&str_len, sizeof(int), 1, file);
      const char* c_str = feature_names_[i].c_str();
      fwrite(c_str, sizeof(char), str_len, file);
    }

    // get size of meta data
    size_t size_of_metadata = metadata_.SizesInByte();
    fwrite(&size_of_metadata, sizeof(size_of_metadata), 1, file);
    // write meta data
    metadata_.SaveBinaryToFile(file);

    // write feature data
    for (int i = 0; i < num_groups_; ++i) {
      // get size of feature
      size_t size_of_feature = feature_groups_[i]->SizesInByte();
      fwrite(&size_of_feature, sizeof(size_of_feature), 1, file);
      // write feature
      feature_groups_[i]->SaveBinaryToFile(file);
    }
    fclose(file);
  }
}

void Dataset::ConstructHistograms(
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* data_indices, data_size_t num_data,
  int leaf_idx,
  std::vector<std::unique_ptr<OrderedBin>>& ordered_bins,
  const score_t* gradients, const score_t* hessians,
  score_t* ordered_gradients, score_t* ordered_hessians,
  HistogramBinEntry* hist_data) const {

  if (leaf_idx < 0 || num_data <= 0 || hist_data == nullptr) {
    return;
  }
  auto ptr_ordered_grad = gradients;
  auto ptr_ordered_hess = hessians;
  if (data_indices != nullptr && num_data < num_data_) {
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data; ++i) {
      ordered_gradients[i] = gradients[data_indices[i]];
      ordered_hessians[i] = hessians[data_indices[i]];
    }
    ptr_ordered_grad = ordered_gradients;
    ptr_ordered_hess = ordered_hessians;
  }

#pragma omp parallel for schedule(static)
  for (int group = 0; group < num_groups_; ++group) {
    bool is_groud_used = false;
    const int f_cnt = group_feature_cnt_[group];
    for (int j = 0; j < f_cnt; ++j) {
      const int fidx = group_feature_start_[group] + j;
      if (is_feature_used[fidx]) {
        is_groud_used = true;
        break;
      }
    }
    if (!is_groud_used) { continue; }
    // feature is not used
    auto data_ptr = hist_data + group_bin_boundaries_[group];
    const int num_bin = feature_groups_[group]->num_total_bin_;
    std::memset(data_ptr + 1, 0, (num_bin - 1) * sizeof(HistogramBinEntry));
    // construct histograms for smaller leaf
    if (ordered_bins[group] == nullptr) {
      // if not use ordered bin
      feature_groups_[group]->bin_data_->ConstructHistogram(
        data_indices,
        num_data,
        ptr_ordered_grad,
        ptr_ordered_hess,
        data_ptr);
    } else {
      // used ordered bin
      ordered_bins[group]->ConstructHistogram(leaf_idx,
        gradients,
        hessians,
        data_ptr);
    }
  }
}

void Dataset::FixHistogram(int feature_idx, double sum_gradient, double sum_hessian, data_size_t num_data,
  HistogramBinEntry* data) const {
  const int group = feature2group_[feature_idx];
  const int sub_feature = feature2subfeature_[feature_idx];
  const BinMapper* bin_mapper = feature_groups_[group]->bin_mappers_[sub_feature].get();
  const int default_bin = bin_mapper->GetDefaultBin();
  if (default_bin > 0) {
    const int num_bin = bin_mapper->num_bin();
    data[default_bin].sum_gradients = sum_gradient;
    data[default_bin].sum_hessians = sum_hessian;
    data[default_bin].cnt = num_data;
    for (int i = 0; i < num_bin; ++i) {
      if (i != default_bin) {
        data[default_bin].sum_gradients -= data[i].sum_gradients;
        data[default_bin].sum_hessians -= data[i].sum_hessians;
        data[default_bin].cnt -= data[i].cnt;
      }
    }
  }
}

}  // namespace LightGBM
