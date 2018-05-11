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
  is_finish_load_ = false;
}

Dataset::Dataset(data_size_t num_data) {
  CHECK(num_data > 0);
  data_filename_ = "noname";
  num_data_ = num_data;
  metadata_.Init(num_data_, NO_SPECIFIC, NO_SPECIFIC);
  is_finish_load_ = false;
  group_bin_boundaries_.push_back(0);
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

int GetConfilctCount(const std::vector<bool>& mark, const int* indices, int num_indices, int max_cnt) {
  int ret = 0;
  for (int i = 0; i < num_indices; ++i) {
    if (mark[indices[i]]) {
      ++ret;
      if (ret > max_cnt) {
        return -1;
      }
    }
  }
  return ret;
}
void MarkUsed(std::vector<bool>& mark, const int* indices, int num_indices) {
  for (int i = 0; i < num_indices; ++i) {
    mark[indices[i]] = true;
  }
}

std::vector<std::vector<int>> FindGroups(const std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                         const std::vector<int>& find_order,
                                         int** sample_indices,
                                         const int* num_per_col,
                                         size_t total_sample_cnt,
                                         data_size_t max_error_cnt,
                                         data_size_t filter_cnt,
                                         data_size_t num_data,
                                         bool is_use_gpu) {
  const int max_search_group = 100;
  const int gpu_max_bin_per_group = 256;
  Random rand(num_data);
  std::vector<std::vector<int>> features_in_group;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<int> group_conflict_cnt;
  std::vector<size_t> group_non_zero_cnt;
  std::vector<int> group_num_bin;

  for (auto fidx : find_order) {
    const size_t cur_non_zero_cnt = num_per_col[fidx];
    bool need_new_group = true;
    std::vector<int> available_groups;
    for (int gid = 0; gid < static_cast<int>(features_in_group.size()); ++gid) {
      if (group_non_zero_cnt[gid] + cur_non_zero_cnt <= total_sample_cnt + max_error_cnt){
        if (!is_use_gpu || group_num_bin[gid] + bin_mappers[fidx]->num_bin() + (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0)
            <= gpu_max_bin_per_group) {
          available_groups.push_back(gid);
        }
      }
    }
    std::vector<int> search_groups;
    if (!available_groups.empty()) {
      int last = static_cast<int>(available_groups.size()) - 1;
      auto indices = rand.Sample(last, std::min(last, max_search_group - 1));
      search_groups.push_back(available_groups.back());
      for (auto idx : indices) {
        search_groups.push_back(available_groups[idx]);
      }
    }
    for (auto gid : search_groups) {
      const int rest_max_cnt = max_error_cnt - group_conflict_cnt[gid];
      int cnt = GetConfilctCount(conflict_marks[gid], sample_indices[fidx], num_per_col[fidx], rest_max_cnt);
      if (cnt >= 0 && cnt <= rest_max_cnt) {
        data_size_t rest_non_zero_data = static_cast<data_size_t>(
          static_cast<double>(cur_non_zero_cnt - cnt) * num_data / total_sample_cnt);
        if (rest_non_zero_data < filter_cnt) { continue; }
        need_new_group = false;
        features_in_group[gid].push_back(fidx);
        group_conflict_cnt[gid] += cnt;
        group_non_zero_cnt[gid] += cur_non_zero_cnt - cnt;
        MarkUsed(conflict_marks[gid], sample_indices[fidx], num_per_col[fidx]);
        if (is_use_gpu) {
          group_num_bin[gid] += bin_mappers[fidx]->num_bin() + (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0);
        }
        break;
      }
    }
    if (need_new_group) {
      features_in_group.emplace_back();
      features_in_group.back().push_back(fidx);
      group_conflict_cnt.push_back(0);
      conflict_marks.emplace_back(total_sample_cnt, false);
      MarkUsed(conflict_marks.back(), sample_indices[fidx], num_per_col[fidx]);
      group_non_zero_cnt.emplace_back(cur_non_zero_cnt);
      if (is_use_gpu) {
        group_num_bin.push_back(1 + bin_mappers[fidx]->num_bin() + (bin_mappers[fidx]->GetDefaultBin() == 0 ? -1 : 0));
      }
    }
  }
  return features_in_group;
}

std::vector<std::vector<int>> FastFeatureBundling(std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
                                                  int** sample_indices,
                                                  const int* num_per_col,
                                                  size_t total_sample_cnt,
                                                  const std::vector<int>& used_features,
                                                  double max_conflict_rate,
                                                  data_size_t num_data,
                                                  data_size_t min_data,
                                                  double sparse_threshold,
                                                  bool is_enable_sparse,
                                                  bool is_use_gpu) {
  // filter is based on sampling data, so decrease its range
  const data_size_t filter_cnt = static_cast<data_size_t>(static_cast<double>(0.95 * min_data) / num_data * total_sample_cnt);
  const data_size_t max_error_cnt = static_cast<data_size_t>(total_sample_cnt * max_conflict_rate);
  int cur_used_feature_cnt = 0;
  std::vector<size_t> feature_non_zero_cnt;
  // put dense feature first
  for (auto fidx : used_features) {
    feature_non_zero_cnt.emplace_back(num_per_col[fidx]);
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
  auto features_in_group = FindGroups(bin_mappers, used_features, sample_indices, num_per_col, total_sample_cnt, max_error_cnt, filter_cnt, num_data, is_use_gpu);
  auto group2 = FindGroups(bin_mappers, feature_order_by_cnt, sample_indices, num_per_col, total_sample_cnt, max_error_cnt, filter_cnt, num_data, is_use_gpu);
  if (features_in_group.size() > group2.size()) {
    features_in_group = group2;
  }
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
      if (sparse_rate >= sparse_threshold && is_enable_sparse) {
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
  Random tmp_rand(12);
  for (int i = 0; i < num_group - 1; ++i) {
    int j = tmp_rand.NextShort(i + 1, num_group);
    std::swap(ret[i], ret[j]);
  }
  return ret;
}

void Dataset::Construct(
  std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
  int** sample_non_zero_indices,
  const int* num_per_col,
  size_t total_sample_cnt,
  const IOConfig& io_config) {

  num_total_features_ = static_cast<int>(bin_mappers.size());
  sparse_threshold_ = io_config.sparse_threshold;
  // get num_features
  std::vector<int> used_features;
  for (int i = 0; i < static_cast<int>(bin_mappers.size()); ++i) {
    if (bin_mappers[i] != nullptr && !bin_mappers[i]->is_trival()) {
      used_features.emplace_back(i);
    }
  }
  if (used_features.empty()) {
    Log::Fatal("Cannot construct Dataset since there are not useful features.\n"
               "It should be at least two unique rows.\n"
               "If the num_row (num_data) is small, you can set min_data=1 and min_data_in_bin=1 to fix this.\n"
               "Otherwise please make sure you are using the right dataset");
  }
  auto features_in_group = NoGroup(used_features);

  if (io_config.enable_bundle) {
    features_in_group = FastFeatureBundling(bin_mappers,
                                            sample_non_zero_indices, num_per_col, total_sample_cnt,
                                            used_features, io_config.max_conflict_rate,
                                            num_data_, io_config.min_data_in_leaf,
                                            sparse_threshold_, io_config.is_enable_sparse, io_config.device_type == std::string("gpu"));
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
      new FeatureGroup(cur_cnt_features, cur_bin_mappers, num_data_, sparse_threshold_,
                       io_config.is_enable_sparse)));
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

  if (!io_config.monotone_constraints.empty()) {
    CHECK(static_cast<size_t>(num_total_features_) == io_config.monotone_constraints.size());
    monotone_types_.resize(num_features_);
    for (int i = 0; i < num_total_features_; ++i) {
      int inner_fidx = InnerFeatureIndex(i);
      if (inner_fidx >= 0) {
        monotone_types_[inner_fidx] = io_config.monotone_constraints[i];
      }
    }
    if (ArrayArgs<int8_t>::CheckAllZero(monotone_types_)) {
      monotone_types_.clear();
    }
  }
}

void Dataset::FinishLoad() {
  if (is_finish_load_) { return; }
  OMP_INIT_EX();
  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_groups_; ++i) {
    OMP_LOOP_EX_BEGIN();
    feature_groups_[i]->bin_data_->FinishLoad();
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  is_finish_load_ = true;
}

void Dataset::CopyFeatureMapperFrom(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = dataset->num_groups_;
  sparse_threshold_ = dataset->sparse_threshold_;
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
      dataset->feature_groups_[i]->is_sparse_));
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
  monotone_types_ = dataset->monotone_types_;
}

void Dataset::CreateValid(const Dataset* dataset) {
  feature_groups_.clear();
  num_features_ = dataset->num_features_;
  num_groups_ = num_features_;
  sparse_threshold_ = dataset->sparse_threshold_;
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
      dataset->sparse_threshold_,
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
  monotone_types_ = dataset->monotone_types_;
}

void Dataset::ReSize(data_size_t num_data) {
  if (num_data_ != num_data) {
    num_data_ = num_data;
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static)
    for (int group = 0; group < num_groups_; ++group) {
      OMP_LOOP_EX_BEGIN();
      feature_groups_[group]->bin_data_->ReSize(num_data_);
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
  }
}

void Dataset::CopySubset(const Dataset* fullset, const data_size_t* used_indices, data_size_t num_used_indices, bool need_meta_data) {
  CHECK(num_used_indices == num_data_);
  OMP_INIT_EX();
  #pragma omp parallel for schedule(static)
  for (int group = 0; group < num_groups_; ++group) {
    OMP_LOOP_EX_BEGIN();
    feature_groups_[group]->CopySubset(fullset->feature_groups_[group].get(), used_indices, num_used_indices);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
  if (need_meta_data) {
    metadata_.Init(fullset->metadata_, used_indices, num_used_indices);
  }
  is_finish_load_ = true;
}

bool Dataset::SetFloatField(const char* field_name, const float* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    #ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
    #else
    metadata_.SetLabel(field_data, num_element);
    #endif
  } else if (name == std::string("weight") || name == std::string("weights")) {
    #ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
    #else
    metadata_.SetWeights(field_data, num_element);
    #endif
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
    #ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
    #else
    *out_ptr = metadata_.label();
    *out_len = num_data_;
    #endif
  } else if (name == std::string("weight") || name == std::string("weights")) {
    #ifdef LABEL_T_USE_DOUBLE
    Log::Fatal("Don't support LABEL_T_USE_DOUBLE");
    #else
    *out_ptr = metadata_.weights();
    *out_len = num_data_;
    #endif
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
      && std::string(bin_filename) == data_filename_) {
    Log::Warning("Bianry file %s already exists", bin_filename);
    return;
  }
  // if not pass a filename, just append ".bin" of original file
  std::string bin_filename_str(data_filename_);
  if (bin_filename == nullptr || bin_filename[0] == '\0') {
    bin_filename_str.append(".bin");
    bin_filename = bin_filename_str.c_str();
  }
  bool is_file_existed = false;

  if (VirtualFileWriter::Exists(bin_filename)) {
    is_file_existed = true;
    Log::Warning("File %s exists, cannot save binary to it", bin_filename);
  }

  if (!is_file_existed) {
    auto writer = VirtualFileWriter::Make(bin_filename);
    if (!writer->Init()) {
      Log::Fatal("Cannot write binary data to %s ", bin_filename);
    }
    Log::Info("Saving data to binary file %s", bin_filename);
    size_t size_of_token = std::strlen(binary_file_token);
    writer->Write(binary_file_token, size_of_token);
    // get size of header
    size_t size_of_header = sizeof(num_data_) + sizeof(num_features_) + sizeof(num_total_features_)
      + sizeof(int) * num_total_features_ + sizeof(label_idx_) + sizeof(num_groups_)
      + 3 * sizeof(int) * num_features_ + sizeof(uint64_t) * (num_groups_ + 1) + 2 * sizeof(int) * num_groups_ + sizeof(int8_t) * num_features_;
    // size of feature names
    for (int i = 0; i < num_total_features_; ++i) {
      size_of_header += feature_names_[i].size() + sizeof(int);
    }
    writer->Write(&size_of_header, sizeof(size_of_header));
    // write header
    writer->Write(&num_data_, sizeof(num_data_));
    writer->Write(&num_features_, sizeof(num_features_));
    writer->Write(&num_total_features_, sizeof(num_total_features_));
    writer->Write(&label_idx_, sizeof(label_idx_));
    writer->Write(used_feature_map_.data(), sizeof(int) * num_total_features_);
    writer->Write(&num_groups_, sizeof(num_groups_));
    writer->Write(real_feature_idx_.data(), sizeof(int) * num_features_);
    writer->Write(feature2group_.data(), sizeof(int) * num_features_);
    writer->Write(feature2subfeature_.data(), sizeof(int) * num_features_);
    writer->Write(group_bin_boundaries_.data(), sizeof(uint64_t) * (num_groups_ + 1));
    writer->Write(group_feature_start_.data(), sizeof(int) * num_groups_);
    writer->Write(group_feature_cnt_.data(), sizeof(int) * num_groups_);
    if (monotone_types_.empty()) {
      ArrayArgs<int8_t>::Assign(&monotone_types_, 0, num_features_);
    }
    writer->Write(monotone_types_.data(), sizeof(int8_t) * num_features_);
    if (ArrayArgs<int8_t>::CheckAllZero(monotone_types_)) {
      monotone_types_.clear();
    }
    // write feature names
    for (int i = 0; i < num_total_features_; ++i) {
      int str_len = static_cast<int>(feature_names_[i].size());
      writer->Write(&str_len, sizeof(int));
      const char* c_str = feature_names_[i].c_str();
      writer->Write(c_str, sizeof(char) * str_len);
    }

    // get size of meta data
    size_t size_of_metadata = metadata_.SizesInByte();
    writer->Write(&size_of_metadata, sizeof(size_of_metadata));
    // write meta data
    metadata_.SaveBinaryToFile(writer.get());

    // write feature data
    for (int i = 0; i < num_groups_; ++i) {
      // get size of feature
      size_t size_of_feature = feature_groups_[i]->SizesInByte();
      writer->Write(&size_of_feature, sizeof(size_of_feature));
      // write feature
      feature_groups_[i]->SaveBinaryToFile(writer.get());
    }
  }
}

void Dataset::ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                                  const data_size_t* data_indices, data_size_t num_data,
                                  int leaf_idx,
                                  std::vector<std::unique_ptr<OrderedBin>>& ordered_bins,
                                  const score_t* gradients, const score_t* hessians,
                                  score_t* ordered_gradients, score_t* ordered_hessians,
                                  bool is_constant_hessian,
                                  HistogramBinEntry* hist_data) const {

  if (leaf_idx < 0 || num_data < 0 || hist_data == nullptr) {
    return;
  }

  std::vector<int> used_group;
  used_group.reserve(num_groups_);
  for (int group = 0; group < num_groups_; ++group) {
    const int f_cnt = group_feature_cnt_[group];
    for (int j = 0; j < f_cnt; ++j) {
      const int fidx = group_feature_start_[group] + j;
      if (is_feature_used[fidx]) {
        break;
      }
    }
    used_group.push_back(group);
  }
  int num_used_group = static_cast<int>(used_group.size());
  auto ptr_ordered_grad = gradients;
  auto ptr_ordered_hess = hessians;
  if (data_indices != nullptr && num_data < num_data_) {
    if (!is_constant_hessian) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
        ordered_hessians[i] = hessians[data_indices[i]];
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        ordered_gradients[i] = gradients[data_indices[i]];
      }
    }
    ptr_ordered_grad = ordered_gradients;
    ptr_ordered_hess = ordered_hessians;
    if (!is_constant_hessian) {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int gi = 0; gi < num_used_group; ++gi) {
        OMP_LOOP_EX_BEGIN();
        int group = used_group[gi];
        // feature is not used
        auto data_ptr = hist_data + group_bin_boundaries_[group];
        const int num_bin = feature_groups_[group]->num_total_bin_;
        std::memset((void*)(data_ptr + 1), 0, (num_bin - 1) * sizeof(HistogramBinEntry));
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
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int gi = 0; gi < num_used_group; ++gi) {
        OMP_LOOP_EX_BEGIN();
        int group = used_group[gi];
        // feature is not used
        auto data_ptr = hist_data + group_bin_boundaries_[group];
        const int num_bin = feature_groups_[group]->num_total_bin_;
        std::memset((void*)(data_ptr + 1), 0, (num_bin - 1) * sizeof(HistogramBinEntry));
        // construct histograms for smaller leaf
        if (ordered_bins[group] == nullptr) {
          // if not use ordered bin
          feature_groups_[group]->bin_data_->ConstructHistogram(
            data_indices,
            num_data,
            ptr_ordered_grad,
            data_ptr);
        } else {
          // used ordered bin
          ordered_bins[group]->ConstructHistogram(leaf_idx,
                                                  gradients,
                                                  data_ptr);
        }
        // fixed hessian.
        for (int i = 0; i < num_bin; ++i) {
          data_ptr[i].sum_hessians = data_ptr[i].cnt * hessians[0];
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    }
  } else {
    if (!is_constant_hessian) {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int gi = 0; gi < num_used_group; ++gi) {
        OMP_LOOP_EX_BEGIN();
        int group = used_group[gi];
        // feature is not used
        auto data_ptr = hist_data + group_bin_boundaries_[group];
        const int num_bin = feature_groups_[group]->num_total_bin_;
        std::memset((void*)(data_ptr + 1), 0, (num_bin - 1) * sizeof(HistogramBinEntry));
        // construct histograms for smaller leaf
        if (ordered_bins[group] == nullptr) {
          // if not use ordered bin
          feature_groups_[group]->bin_data_->ConstructHistogram(
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
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
    } else {
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static)
      for (int gi = 0; gi < num_used_group; ++gi) {
        OMP_LOOP_EX_BEGIN();
        int group = used_group[gi];
        // feature is not used
        auto data_ptr = hist_data + group_bin_boundaries_[group];
        const int num_bin = feature_groups_[group]->num_total_bin_;
        std::memset((void*)(data_ptr + 1), 0, (num_bin - 1) * sizeof(HistogramBinEntry));
        // construct histograms for smaller leaf
        if (ordered_bins[group] == nullptr) {
          // if not use ordered bin
          feature_groups_[group]->bin_data_->ConstructHistogram(
            num_data,
            ptr_ordered_grad,
            data_ptr);
        } else {
          // used ordered bin
          ordered_bins[group]->ConstructHistogram(leaf_idx,
                                                  gradients,
                                                  data_ptr);
        }
        // fixed hessian.
        for (int i = 0; i < num_bin; ++i) {
          data_ptr[i].sum_hessians = data_ptr[i].cnt * hessians[0];
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
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
