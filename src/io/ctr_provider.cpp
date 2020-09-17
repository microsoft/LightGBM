/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include <LightGBM/ctr_provider.hpp>

#include <set>

namespace LightGBM {

void CTRProvider::GenTrainingDataFoldID() {
  std::vector<std::mt19937> mt_generators;
  for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      mt_generators.emplace_back(config_.seed + thread_id);
  }
  const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
  std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
  training_data_fold_id_.clear();
  training_data_fold_id_.resize(num_data_, -1);
  const int block_size = (num_data_ + num_threads_ - 1) / num_threads_;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    int block_start = block_size * thread_id;
    int block_end = std::min(block_start + block_size, num_data_);
    std::mt19937& generator = mt_generators[thread_id];
    for(int index = block_start; index < block_end; ++index) {
      training_data_fold_id_[index] = fold_distribution(generator);
    }
  }
}

std::string CTRProvider::DumpModelInfo() const {
  std::stringstream str_buf;
  if (cat_converters_.size() > 0) {
    for (const int cat_fid : categorical_features_) {
      str_buf << cat_fid;
      for (const auto& pair : label_info_.at(cat_fid).at(config_.num_ctr_folds)) {
        str_buf << " " << pair.first << ":" << pair.second << ":" << count_info_.at(cat_fid).at(config_.num_ctr_folds).at(pair.first);
      }
      str_buf << "@";
    }
    for (const auto& cat_converter : cat_converters_) {
      str_buf << cat_converter->DumpToString() << " ";
    }
  }
  return str_buf.str();
}

CTRProvider* CTRProvider::RecoverFromModelString(const std::string model_string) {
  return new CTRProvider(model_string);
} 

std::unordered_map<int, std::unordered_map<int, double>> CTRProvider::RecoverCTRValues(const std::string str) {
  std::stringstream sin(str);
  std::unordered_map<int, std::unordered_map<int, double>> ctr_values;
  int fid = 0;
  int cat_feature_value = 0;
  double ctr_value = 0.0;
  while(sin >> fid) {
    while(sin.get() != ' ') {
      sin >> cat_feature_value;
      CHECK(sin.get() == ':');
      sin >> ctr_value;  
      ctr_values[fid][cat_feature_value] = ctr_value;
    }
  }
  return ctr_values;
}

void CTRProvider::ExpandCountEncodings(std::vector<std::vector<int>>& sampled_non_missing_data_indices,
    std::vector<std::vector<double>>& sampled_non_missing_feature_values,
    std::unordered_set<int>& ignored_features) {
  
  sampled_non_missing_data_indices.resize(num_total_features_);
  sampled_non_missing_feature_values.resize(num_total_features_);
  std::vector<size_t> old_feature_sample_size(num_original_features_, 0);
  for (const int fid : categorical_features_) {
    old_feature_sample_size[fid] = sampled_non_missing_feature_values[fid].size();
    CHECK(old_feature_sample_size[fid] == sampled_non_missing_data_indices[fid].size());
  }
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
    const int fid = categorical_features_[i];
    for (const auto& cat_converter : cat_converters_) {
      const int convert_fid = cat_converter->GetConvertFid(fid);
      if (convert_fid >= num_original_features_) {
        auto& count_feature_values = sampled_non_missing_feature_values[convert_fid];
        auto& count_feature_indices = sampled_non_missing_data_indices[convert_fid];
        count_feature_values.resize(old_feature_sample_size[fid]);
        count_feature_indices.resize(old_feature_sample_size[fid]);
      }
    }
  }
  for (const int fid : categorical_features_) {
    if (ignored_features.count(fid) > 0) {
      for (const auto& cat_converter : cat_converters_) {
        ignored_features.insert(cat_converter->GetConvertFid(fid));
      }
    }
  }
}

void CTRProvider::SyncCTRStat(std::vector<std::unordered_map<int, label_t>>& fold_label_sum,
    std::vector<std::unordered_map<int, int>>& fold_total_count, const int num_machines) const {
  if(num_machines > 1) {
    //CHECK(Network::num_machines() == config_.num_machines);
    std::string ctr_stat_string;
    for(int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
      ctr_stat_string += DumpDictToString(fold_label_sum[fold_id], ' ') + "@";
      ctr_stat_string += DumpDictToString(fold_total_count[fold_id], ' ') + "@";
    }
    const size_t max_ctr_values_string_size = Network::GlobalSyncUpByMax(ctr_stat_string.size()) + 1;
    std::vector<char> input_buffer(max_ctr_values_string_size), output_buffer(max_ctr_values_string_size * num_machines);
    std::memcpy(input_buffer.data(), ctr_stat_string.c_str(), ctr_stat_string.size() * sizeof(char));
    input_buffer[ctr_stat_string.size()] = '\0';

    Network::Allgather(input_buffer.data(), sizeof(char) * max_ctr_values_string_size, output_buffer.data());

    int feature_value = 0;
    int count_value = 0;
    label_t label_sum = 0;

    for(int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
      fold_label_sum[fold_id].clear();
      fold_total_count[fold_id].clear();
    }

    size_t cur_str_pos = 0;
    int check_num_machines = 0;
    while(cur_str_pos < output_buffer.size()) {
      std::string all_ctr_stat_string(output_buffer.data() + cur_str_pos);
      Log::Warning(all_ctr_stat_string.c_str()); 
      cur_str_pos += max_ctr_values_string_size;
      ++check_num_machines;
      std::stringstream sin(all_ctr_stat_string);
      for(int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
        auto& this_fold_label_sum = fold_label_sum[fold_id];
        auto& this_fold_total_count = fold_total_count[fold_id];
        char ending_char = ' ';
        while(ending_char != '@') {
          CHECK(ending_char == ' ');
          sin >> feature_value;
          CHECK(sin.get() == ':');
          sin >> label_sum;
          if(this_fold_label_sum.count(feature_value) > 0) {
            this_fold_label_sum[feature_value] += label_sum;
          }
          else {
            this_fold_label_sum[feature_value] = label_sum;
          }
          ending_char = sin.get();
        }
        ending_char = ' ';
        while(ending_char != '@') {
          CHECK(ending_char == ' ');
          sin >> feature_value;
          CHECK(sin.get() == ':');
          sin >> count_value;
          if(this_fold_total_count.count(feature_value) > 0) {
            this_fold_total_count[feature_value] += count_value;
          }
          else {
            this_fold_total_count[feature_value] = count_value;
          }
          ending_char = sin.get();
        }
      }
    }
    CHECK(check_num_machines == num_machines);
    CHECK(cur_str_pos == output_buffer.size()); 
  }
}

void CTRProvider::CreatePushDataFunction(const std::vector<int>& used_feature_idx, 
    const std::vector<int>& feature_to_group,
    const std::vector<int>& feature_to_sub_feature,
    const std::function<void(int tid, data_size_t row_idx, 
      int group, int sub_feature, double value)>& feature_group_push_data_func) {
    std::unordered_map<int, std::unordered_map<int, int>> group_subfeature_to_outer_feature;
    for(int feature_index = 0; feature_index < num_original_features_; ++feature_index) {
      const int inner_feature_index = used_feature_idx[feature_index];
      if(inner_feature_index >= 0) {
        const int group = feature_to_group[inner_feature_index];
        const int sub_feature = feature_to_sub_feature[inner_feature_index];
        group_subfeature_to_outer_feature[group][sub_feature] = feature_index;
      }
    }

    push_training_data_func_ = [this, feature_group_push_data_func, feature_to_group, feature_to_sub_feature, used_feature_idx,
      group_subfeature_to_outer_feature]
      (int tid, data_size_t row_idx, int group, int sub_feature, double value) {
      const int outer_feature_index = group_subfeature_to_outer_feature.at(group).at(sub_feature);
      if(is_categorical_feature_[outer_feature_index] && cat_converters_.size() > 0) {
        const int fold_id = training_data_fold_id_[row_idx];
        const int cat_feature_value = static_cast<int>(value);
        const double label_sum = label_info_[outer_feature_index][fold_id][cat_feature_value];
        const double total_count = count_info_[outer_feature_index][fold_id][cat_feature_value];
        const double all_count = count_info_[outer_feature_index][config_.num_ctr_folds][cat_feature_value];
        for (const auto& cat_converter : cat_converters_) {
          const double convert_value = TrimConvertValue(cat_converter->CalcValue(label_sum, total_count, all_count));
          const int convert_fid = cat_converter->GetConvertFid(outer_feature_index);
          const int inner_convert_fid = used_feature_idx[convert_fid];
          if (inner_convert_fid >= 0) {
            const int convert_group = feature_to_group[inner_convert_fid];
            const int convert_sub_feature = feature_to_sub_feature[inner_convert_fid];
            feature_group_push_data_func(tid, row_idx, convert_group, convert_sub_feature, convert_value);
          }
        }
      }
      else {
        feature_group_push_data_func(tid, row_idx, group, sub_feature, value);
      }
    };

    push_valid_data_func_ = [this, feature_group_push_data_func, feature_to_group, feature_to_sub_feature, used_feature_idx,
      group_subfeature_to_outer_feature]
      (int tid, data_size_t row_idx, int group, int sub_feature, double value) {
      const int outer_feature_index = group_subfeature_to_outer_feature.at(group).at(sub_feature);
      if(is_categorical_feature_[outer_feature_index] && cat_converters_.size() > 0) {
        const int cat_feature_value = static_cast<int>(value);
        const auto& label_dict = label_info_[outer_feature_index][config_.num_ctr_folds];
        const auto& count_dict = count_info_[outer_feature_index][config_.num_ctr_folds];
        const double label_sum = label_dict.count(cat_feature_value) == 0 ? 0.0 : label_dict.at(cat_feature_value);
        const double total_count = count_dict.count(cat_feature_value) == 0 ? 0.0 : count_dict.at(cat_feature_value);
        for (const auto& cat_converter : cat_converters_) {
          const double convert_value = TrimConvertValue(cat_converter->CalcValue(label_sum, total_count, total_count));
          const int convert_fid = cat_converter->GetConvertFid(outer_feature_index);
          const int inner_convert_fid = used_feature_idx[convert_fid];
          if (inner_convert_fid >= 0) {
            const int convert_group = feature_to_group[inner_convert_fid];
            const int convert_sub_feature = feature_to_sub_feature[inner_convert_fid];
            feature_group_push_data_func(tid, row_idx, convert_group, convert_sub_feature, convert_value);
          }
        }
      }
      else {
        feature_group_push_data_func(tid, row_idx, group, sub_feature, value);
      }
    };
  }

void CTRProvider::SyncCTRPrior(const double label_sum, const int local_num_data, 
    double& all_label_sum, int& all_num_data, int num_machines) const {
    if(num_machines > 1) {
      all_label_sum = Network::GlobalSyncUpBySum(label_sum);
      all_num_data = Network::GlobalSyncUpBySum(local_num_data); 
    }
    else {
      all_label_sum = label_sum;
      all_num_data = local_num_data;
    }
  }

void CTRProvider::ProcessOneLine(const std::vector<double>& one_line, double label, int line_idx, const int thread_id) {
  const int fold_id = training_data_fold_id_[line_idx];
  auto& count_info = thread_count_info_[thread_id];
  auto& label_info = thread_label_info_[thread_id];
  for (int fid = 0; fid < num_original_features_; ++fid) {
    if (is_categorical_feature_[fid]) {
      const int value = static_cast<int>(one_line[fid]);
      if(count_info[fid][fold_id].count(value) == 0) {
        count_info[fid][fold_id][value] = 1;
        label_info[fid][fold_id][value] = static_cast<label_t>(label);
      } else {
        ++count_info[fid][fold_id][value];
        label_info[fid][fold_id][value] += label;
      }
    }
  }
  thread_label_sum_[thread_id] += label;
}

void CTRProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx, 
  std::vector<bool>& is_feature_processed, const int thread_id) {
  ProcessOneLineInner(one_line, label, line_idx, is_feature_processed, thread_count_info_[thread_id], thread_label_info_[thread_id]);
  thread_label_sum_[thread_id] += label;
}

void CTRProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx, std::vector<bool>& is_feature_processed) {
  ProcessOneLineInner(one_line, label, line_idx, is_feature_processed, count_info_, label_info_);
  prior_ += label;
}

void CTRProvider::ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx, 
  std::vector<bool>& is_feature_processed,
  std::unordered_map<int, std::vector<std::unordered_map<int, int>>>& count_info,
  std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>>& label_info) {
  const int fold_id = training_data_fold_id_[line_idx];
  for (size_t i = 0; i < is_feature_processed.size(); ++i) {
    is_feature_processed[i] = false;
  }
  for (const auto& pair : one_line) {
    const int fid = pair.first;
    if (is_categorical_feature_[fid]) {
      is_feature_processed[fid] = true;
      const int value = static_cast<int>(pair.second);
      if (count_info[fid][fold_id].count(value) == 0) {
        count_info[fid][fold_id][value] = 1;
        label_info[fid][fold_id][value] = static_cast<label_t>(label);
      } else {
        ++count_info[fid][fold_id][value];
        label_info[fid][fold_id][value] += static_cast<label_t>(label);
      }
    }
  }
  // pad the missing values with zeros
  for (const int fid : categorical_features_) {
    if (!is_feature_processed[fid]) {
      if (count_info[fid][fold_id].count(0) == 0) {
        count_info[fid][fold_id][0] = 1;
        label_info[fid][fold_id][0] = static_cast<label_t>(label);
      } else {
        ++count_info[fid][fold_id][0];
        label_info[fid][fold_id][0] += static_cast<label_t>(label);
      }
    }
  }
}

void CTRProvider::FinishProcess(const int num_machines) {
  // gather from threads
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
    const int fid = categorical_features_[i];
    for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
      auto& feature_fold_count_info = count_info_.at(fid)[fold_id];
      auto& feature_fold_label_info = label_info_.at(fid)[fold_id];
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        const auto& thread_feature_fold_count_info = thread_count_info_[thread_id].at(fid)[fold_id];
        const auto& thread_feature_fold_label_info = thread_label_info_[thread_id].at(fid)[fold_id];
        for (const auto& pair : thread_feature_fold_count_info) {
          if (feature_fold_count_info.count(pair.first) == 0) {
            feature_fold_count_info[pair.first] = pair.second;
          } else {
            feature_fold_count_info[pair.first] += pair.second;
          }
        }
        for (const auto& pair : thread_feature_fold_label_info) {
          if (feature_fold_label_info.count(pair.first) == 0) {
            feature_fold_label_info[pair.first] = pair.second;
          } else {
            feature_fold_label_info[pair.first] += pair.second; 
          }
        }
      }
    }
  }
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    prior_ += thread_label_sum_[thread_id];
  }
  thread_count_info_.clear();
  thread_label_info_.clear();
  thread_count_info_.shrink_to_fit();
  thread_label_info_.shrink_to_fit();
  thread_label_sum_.clear();
  thread_label_sum_.shrink_to_fit();

  // gather from machines
  if(num_machines > 1) {
    for(size_t i = 0; i < categorical_features_.size(); ++i) {
      SyncCTRStat(label_info_.at(categorical_features_[i]), count_info_.at(categorical_features_[i]), num_machines);
    }
    const double local_label_sum = prior_;
    const int local_num_data = static_cast<int>(num_data_);
    int global_num_data = 0;
    SyncCTRPrior(local_label_sum, local_num_data, prior_, global_num_data, num_machines);
    prior_ /= global_num_data;
  }
  else {
    prior_ /= num_data_;
  }

  // set prior for label mean ctr converter
  for (size_t i = 0; i < cat_converters_.size(); ++i) {
    cat_converters_[i]->SetPrior(prior_, config_.prior_weight);
  }

  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
    const int fid = categorical_features_[i];
    auto& total_count_info = count_info_.at(fid).at(config_.num_ctr_folds);
    auto& label_count_info = label_info_.at(fid).at(config_.num_ctr_folds);
    // gather from folds
    for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
      const auto& fold_count_info = count_info_.at(fid).at(fold_id);
      const auto& fold_label_info = label_info_.at(fid).at(fold_id);
      for (const auto& pair : fold_count_info) {
        if (total_count_info.count(pair.first) == 0) {
          total_count_info[pair.first] = pair.second;
          label_count_info[pair.first] = fold_label_info.at(pair.first);
        } else {
          total_count_info[pair.first] += pair.second;
          label_count_info[pair.first] += fold_label_info.at(pair.first);
        }
      }
    }
    // replace "fold sum" with "total sum - fold sum", for the convenience of value calculation
    for (const auto& pair : total_count_info) {
      for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
        if (count_info_.at(fid).at(fold_id).count(pair.first) == 0) {
          count_info_[fid][fold_id][pair.first] = total_count_info.at(pair.first);
          label_info_[fid][fold_id][pair.first] = label_count_info.at(pair.first);
        } else {
          count_info_[fid][fold_id][pair.first] = total_count_info.at(pair.first) - count_info_[fid][fold_id][pair.first];
          label_info_[fid][fold_id][pair.first] = label_count_info.at(pair.first) - label_info_[fid][fold_id][pair.first];
        }
      } 
    }
  }
}

void CTRProvider::ReplaceCategoricalValues(const std::vector<data_size_t>& sampled_data_indices,
    std::vector<std::vector<int>>& sampled_non_missing_data_indices,
    std::vector<std::vector<double>>& sampled_non_missing_feature_values,
    std::unordered_set<int>& ignored_features) {
  if (cat_converters_.size() == 0) { return; }
  ExpandCountEncodings(sampled_non_missing_data_indices, sampled_non_missing_feature_values, ignored_features);
  // parallelize by features
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < static_cast<int>(categorical_features_.size()); ++i) {
    const int cat_fid = categorical_features_[i];
    for (size_t j = 0; j < sampled_non_missing_feature_values[cat_fid].size(); ++j) {
      const int feature_value = static_cast<int>(sampled_non_missing_feature_values[cat_fid][j]);
      const data_size_t data_index = sampled_data_indices[sampled_non_missing_data_indices[cat_fid][j]];
      const int fold_id = training_data_fold_id_[data_index];
      const double label_sum = label_info_.at(cat_fid).at(fold_id).at(feature_value);
      const double total_count = count_info_.at(cat_fid).at(fold_id).at(feature_value);
      const double all_total_count = count_info_.at(cat_fid).at(config_.num_ctr_folds).at(feature_value);
      for (const auto& cat_converter : cat_converters_) {
        const double convert_value = TrimConvertValue(cat_converter->CalcValue(label_sum, total_count, all_total_count));
        const int convert_fid = cat_converter->GetConvertFid(cat_fid);
        sampled_non_missing_feature_values[convert_fid][j] = convert_value;
        sampled_non_missing_data_indices[convert_fid][j] = sampled_non_missing_data_indices[cat_fid][j];
      }
    }
  }
}

void CTRProvider::ConvertCatToCTR(double* features) const {
  if (cat_converters_.size() == 0) { return; }
  for (const auto& pair : label_info_) {
    const int cat_value = static_cast<int>(features[pair.first]);
    double label_sum = 0.0f, total_count = 0.0f;
    if (pair.second.back().count(cat_value) > 0) {
      label_sum = pair.second.back().at(cat_value);
      total_count = count_info_.at(pair.first).back().at(cat_value);
    }
    for (const auto& cat_converter : cat_converters_) {
      const double convert_value = cat_converter->CalcValue(label_sum, total_count, total_count);
      const int convert_fid = cat_converter->GetConvertFid(pair.first);
      features[convert_fid] = convert_value;
    }
  }
}

void CTRProvider::ConvertCatToCTR(std::unordered_map<int, double>& features) const {
  if (cat_converters_.size() == 0) { return; }
  for (const auto& pair : label_info_) {
    if (features.count(pair.first) > 0) {
      const int cat_value = static_cast<int>(features[pair.first]);
      double label_sum = 0.0f, total_count = 0.0f;
      if (pair.second.back().count(cat_value)) {
        label_sum = pair.second.back().at(cat_value);
        total_count = count_info_.at(pair.first).back().at(cat_value);
      }
      for (const auto& cat_converter : cat_converters_) {
        const double convert_value = cat_converter->CalcValue(label_sum, total_count, total_count);
        const int convert_fid = cat_converter->GetConvertFid(pair.first);
        features[convert_fid] = convert_value;
      }
    }
  }
}

} // namespace LightGBM
