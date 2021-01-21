/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include <LightGBM/ctr_provider.hpp>

#include <set>

namespace LightGBM {

std::string CTRProvider::DumpModelInfo() const {
  std::stringstream str_buf;
  if (cat_converters_.size() > 0) {
    str_buf << static_cast<int>(keep_raw_cat_method_) << " ";
    str_buf << num_original_features_ << " ";
    str_buf << num_total_features_ << " ";
    for (const int cat_fid : categorical_features_) {
      str_buf << cat_fid;
      // only the information of the full training dataset is kept
      // information per fold is discarded
      for (const auto& pair : label_info_.at(cat_fid).back()) {
        str_buf << " " << pair.first << ":" << pair.second << ":" << count_info_.at(cat_fid).back().at(pair.first);
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
  if (!model_string.empty()) {
    std::unique_ptr<CTRProvider> ret(new CTRProvider(model_string));
    if (ret->cat_converters_.size() > 0) {
      return ret.release();
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

std::unordered_map<int, std::unordered_map<int, double>> CTRProvider::RecoverCTRValues(const std::string str) {
  std::stringstream sin(str);
  std::unordered_map<int, std::unordered_map<int, double>> ctr_values;
  int fid = 0;
  int cat_feature_value = 0;
  double ctr_value = 0.0;
  while (sin >> fid) {
    while (sin.get() != ' ') {
      sin >> cat_feature_value;
      CHECK_EQ(sin.get(), ':');
      sin >> ctr_value;
      ctr_values[fid][cat_feature_value] = ctr_value;
    }
  }
  return ctr_values;
}

void CTRProvider::SyncCTRStat(std::vector<std::unordered_map<int, label_t>>* fold_label_sum_ptr,
    std::vector<std::unordered_map<int, int>>* fold_total_count_ptr, const int num_machines) const {
  auto& fold_label_sum = *fold_label_sum_ptr;
  auto& fold_total_count = *fold_total_count_ptr;
  if (num_machines > 1) {
    std::string ctr_stat_string;
    for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
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

    for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
      fold_label_sum[fold_id].clear();
      fold_total_count[fold_id].clear();
    }

    size_t cur_str_pos = 0;
    int check_num_machines = 0;
    while (cur_str_pos < output_buffer.size()) {
      std::string all_ctr_stat_string(output_buffer.data() + cur_str_pos);
      cur_str_pos += max_ctr_values_string_size;
      ++check_num_machines;
      std::stringstream sin(all_ctr_stat_string);
      for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
        auto& this_fold_label_sum = fold_label_sum[fold_id];
        auto& this_fold_total_count = fold_total_count[fold_id];
        char ending_char = ' ';
        while (ending_char != '@') {
          CHECK_EQ(ending_char, ' ');
          sin >> feature_value;
          CHECK_EQ(sin.get(), ':');
          sin >> label_sum;
          if (this_fold_label_sum.count(feature_value) > 0) {
            this_fold_label_sum[feature_value] += label_sum;
          } else {
            this_fold_label_sum[feature_value] = label_sum;
          }
          ending_char = sin.get();
        }
        ending_char = ' ';
        while (ending_char != '@') {
          CHECK_EQ(ending_char, ' ');
          sin >> feature_value;
          CHECK_EQ(sin.get(), ':');
          sin >> count_value;
          if (this_fold_total_count.count(feature_value) > 0) {
            this_fold_total_count[feature_value] += count_value;
          } else {
            this_fold_total_count[feature_value] = count_value;
          }
          ending_char = sin.get();
        }
      }
    }
    CHECK_EQ(check_num_machines, num_machines);
    CHECK_EQ(cur_str_pos, output_buffer.size());
  }
}

void CTRProvider::SyncCTRPrior(const double label_sum, const int local_num_data,
  double* all_label_sum_ptr, int* all_num_data_ptr, int num_machines) const {
  if (num_machines > 1) {
    *all_label_sum_ptr = Network::GlobalSyncUpBySum(label_sum);
    *all_num_data_ptr = Network::GlobalSyncUpBySum(local_num_data);
  } else {
    *all_label_sum_ptr = label_sum;
    *all_num_data_ptr = local_num_data;
  }
}

void CTRProvider::ProcessOneLine(const std::vector<double>& one_line, double label,
  int /*line_idx*/, const int thread_id, const int fold_id) {
  auto& count_info = thread_count_info_[thread_id];
  auto& label_info = thread_label_info_[thread_id];
  for (int fid = 0; fid < num_original_features_; ++fid) {
    if (is_categorical_feature_[fid]) {
      const int value = static_cast<int>(one_line[fid]);
      if (count_info[fid][fold_id].count(value) == 0) {
        count_info[fid][fold_id][value] = 1;
        label_info[fid][fold_id][value] = static_cast<label_t>(label);
      } else {
        ++count_info[fid][fold_id][value];
        label_info[fid][fold_id][value] += label;
      }
    }
  }
  thread_fold_label_sum_[thread_id][fold_id] += label;
  ++thread_fold_num_data_[thread_id][fold_id];
}

void CTRProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx,
  std::vector<bool>* is_feature_processed_ptr, const int thread_id, const int fold_id) {
  ProcessOneLineInner(one_line, label, line_idx, is_feature_processed_ptr, &thread_count_info_[thread_id],
    &thread_label_info_[thread_id], &thread_fold_label_sum_[thread_id], &thread_fold_num_data_[thread_id], fold_id);
}

void CTRProvider::ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
  int line_idx, std::vector<bool>* is_feature_processed_ptr, const int fold_id) {
  ProcessOneLineInner(one_line, label, line_idx, is_feature_processed_ptr,
    &count_info_, &label_info_, &fold_label_sum_, &fold_num_data_, fold_id);
}

void CTRProvider::ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line,
  double label, int /*line_idx*/,
  std::vector<bool>* is_feature_processed_ptr,
  std::unordered_map<int, std::vector<std::unordered_map<int, int>>>* count_info_ptr,
  std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>>* label_info_ptr,
  std::vector<label_t>* label_sum_ptr,
  std::vector<int>* num_data_ptr,
  const int fold_id) {
  auto& is_feature_processed = *is_feature_processed_ptr;
  auto& count_info = *count_info_ptr;
  auto& label_info = *label_info_ptr;
  auto& label_sum = *label_sum_ptr;
  auto& num_data = *num_data_ptr;
  for (size_t i = 0; i < is_feature_processed.size(); ++i) {
    is_feature_processed[i] = false;
  }
  ++num_data[fold_id];
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
  label_sum[fold_id] += label;
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
  for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      fold_num_data_[fold_id] += thread_fold_num_data_[thread_id][fold_id];
    }
    fold_num_data_.back() += fold_num_data_[fold_id];
  }
  for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      fold_label_sum_[fold_id] += thread_fold_label_sum_[thread_id][fold_id];
    }
    fold_label_sum_.back() += fold_label_sum_[fold_id];
  }
  thread_count_info_.clear();
  thread_label_info_.clear();
  thread_count_info_.shrink_to_fit();
  thread_label_info_.shrink_to_fit();
  thread_fold_label_sum_.clear();
  thread_fold_label_sum_.shrink_to_fit();

  // gather from machines
  if (num_machines > 1) {
    for (size_t i = 0; i < categorical_features_.size(); ++i) {
      SyncCTRStat(&label_info_.at(categorical_features_[i]),
        &count_info_.at(categorical_features_[i]), num_machines);
    }
    for (int fold_id = 0; fold_id < config_.num_ctr_folds + 1; ++fold_id) {
      const double local_label_sum = fold_label_sum_[fold_id];
      const int local_num_data = static_cast<int>(fold_num_data_[fold_id]);
      int global_num_data = 0;
      double global_label_sum = 0.0f;
      SyncCTRPrior(local_label_sum, local_num_data,
        &global_label_sum, &global_num_data, num_machines);
    }
  }
  for (int fold_id = 0; fold_id < config_.num_ctr_folds; ++fold_id) {
    fold_label_sum_[fold_id] = fold_label_sum_.back() - fold_label_sum_[fold_id];
    fold_num_data_[fold_id] = fold_num_data_.back() - fold_num_data_[fold_id];
  }
  for (int fold_id = 0; fold_id < config_.num_ctr_folds + 1; ++fold_id) {
    fold_prior_[fold_id] = fold_label_sum_[fold_id] * 1.0f / fold_num_data_[fold_id];
  }
  prior_ = fold_prior_.back();
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
          count_info_[fid][fold_id][pair.first] =
            total_count_info.at(pair.first) - count_info_[fid][fold_id][pair.first];
          label_info_[fid][fold_id][pair.first] =
            label_count_info.at(pair.first) - label_info_[fid][fold_id][pair.first];
        }
      }
    }
  }
}

void CTRProvider::IterateOverCatConverters(int fid, double fval, int line_idx,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process) const {
  const int fold_id = training_data_fold_id_[line_idx];
  IterateOverCatConvertersInner<true>(fid, fval, fold_id, write_func, post_process);
}

void CTRProvider::IterateOverCatConverters(int fid, double fval,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process) const {
  IterateOverCatConvertersInner<false>(fid, fval, -1, write_func, post_process);
}

void CTRProvider::ConvertCatToCTR(std::vector<double>* features, int line_idx) const {
  if (cat_converters_.size() == 0) { return; }
  auto& features_ref = *features;
  features_ref.resize(num_total_features_);
  for (const auto& pair : label_info_) {
    IterateOverCatConverters(pair.first, features_ref[pair.first], line_idx,

      [&features_ref] (int convert_fid, int, double convert_value) {
        features_ref[convert_fid] = convert_value;
      },

      [] (int) {}
    );
  }
}

void CTRProvider::ConvertCatToCTR(std::vector<double>* features) const {
  if (cat_converters_.size() == 0) { return; }
  auto& features_ref = *features;
  features_ref.resize(num_total_features_);
  features->resize(num_total_features_);
  for (const auto& pair : label_info_) {
    IterateOverCatConverters(pair.first, features_ref[pair.first],

      [&features_ref] (int convert_fid, int, double convert_value) {
        features_ref[convert_fid] = convert_value;
      },

      [] (int) {}
    );
  }
}

void CTRProvider::ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr,
  const int line_idx) const {
  auto& features_ref = *features_ptr;
  std::vector<bool> feature_processed(num_original_features_, false);
  for (const int fid : categorical_features_) {
    feature_processed[fid] = false;
  }
  const size_t n_pairs = features_ref.size();
  for (size_t i = 0; i < n_pairs; ++i) {
    auto& pair = features_ref[i];
    const int fid = pair.first;
    if (is_categorical_feature_[fid]) {
      IterateOverCatConverters(fid, pair.second, line_idx,

        [&features_ref, &pair] (int convert_fid, int fid, double convert_value) {
          if (convert_fid == fid) {
            pair.second = convert_value;
          } else {
            // assert that convert_fid in this case is larger than all the original feature indices
            features_ref.emplace_back(convert_fid, convert_value);
          }
        },

        [&feature_processed] (int fid) { feature_processed[fid] =  true; }
      );
    }
  }
  for (const int fid : categorical_features_) {
    if (!feature_processed[fid]) {
      IterateOverCatConverters(fid, 0.0f, line_idx,

        [&features_ref] (int convert_fid, int, double convert_value) {
          // assert that convert_fid in this case is larger than all the original feature indices
          features_ref.emplace_back(convert_fid, convert_value);
        },

        [] (int) {}
      );
    }
  }
}

void CTRProvider::ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr) const {
  auto& features_ref = *features_ptr;
  std::vector<bool> feature_processed(num_original_features_, false);
  for (const int fid : categorical_features_) {
    feature_processed[fid] = false;
  }
  const size_t n_pairs = features_ref.size();
  for (size_t i = 0; i < n_pairs; ++i) {
    auto& pair = features_ref[i];
    const int fid = pair.first;
    if (is_categorical_feature_[fid]) {
      IterateOverCatConverters(fid, pair.second,

        [&features_ref, &pair] (int convert_fid, int fid, double convert_value) {
          if (convert_fid == fid) {
            pair.second = convert_value;
          } else {
            // assert that convert_fid in this case is larger than all the original feature indices
            features_ref.emplace_back(convert_fid, convert_value);
          }
        },

        [&feature_processed] (int fid) { feature_processed[fid] = true; }
      );
    }
  }
  for (const int fid : categorical_features_) {
    if (!feature_processed[fid]) {
      IterateOverCatConverters(fid, 0.0f,

        [&features_ref] (int convert_fid, int, double convert_value) {
          // assert that convert_fid in this case is larger than all the original feature indices
          features_ref.emplace_back(convert_fid, convert_value);
        },

        [] (int) {}
      );
    }
  }
}

double CTRProvider::ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
  int col_idx, int line_idx) const {
  const int fold_id = training_data_fold_id_[line_idx];
  return HandleOneCatConverter<true>(col_idx, fval, fold_id, cat_converter);
}

double CTRProvider::ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
  int col_idx) const {
  return HandleOneCatConverter<false>(col_idx, fval, -1, cat_converter);
}

void CTRProvider::WrapRowFunctions(
  std::vector<std::function<std::vector<double>(int row_idx)>>* get_row_fun,
  int32_t* ncol, bool is_valid) const {
  const std::vector<std::function<std::vector<double>(int row_idx)>> old_get_row_fun = *get_row_fun;
  get_row_fun->clear();
  for (size_t i = 0; i < old_get_row_fun.size(); ++i) {
    get_row_fun->push_back(WrapRowFunctionInner<double>(&old_get_row_fun[i], is_valid));
  }
  *ncol = static_cast<int32_t>(num_total_features_);
}

void CTRProvider::WrapRowFunction(
  std::function<std::vector<std::pair<int, double>>(int row_idx)>* get_row_fun,
  int64_t* ncol, bool is_valid) const {
  *get_row_fun = WrapRowFunctionInner<std::pair<int, double>>(get_row_fun, is_valid);
  *ncol = static_cast<int64_t>(num_total_features_);
}

void CTRProvider::WrapColIters(
  std::vector<std::unique_ptr<CSC_RowIterator>>* col_iters,
  int64_t* ncol_ptr, bool is_valid, int64_t num_row) const {
  int old_num_col = static_cast<int>(col_iters->size());
  std::vector<std::unique_ptr<CSC_RowIterator>> old_col_iters(col_iters->size());
  for (int i = 0; i < old_num_col; ++i) {
    old_col_iters[i].reset(col_iters->operator[](i).release());
  }
  col_iters->resize(num_total_features_);
  CHECK((*ncol_ptr) - 1 == old_num_col);
  for (int i = 0; i < (*ncol_ptr) - 1; ++i) {
    if (is_categorical_feature_[i]) {
      for (const auto& cat_converter : cat_converters_) {
        const int convert_fid = cat_converter->GetConvertFid(i);
        col_iters->operator[](convert_fid).reset(new CTR_CSC_RowIterator(
          old_col_iters[i].get(), i, cat_converter.get(), this, is_valid, num_row));
      }
      if (keep_raw_cat_method_) {
        col_iters->operator[](i).reset(old_col_iters[i].release());
      }
    } else {
      col_iters->operator[](i).reset(old_col_iters[i].release());
    }
  }
  *ncol_ptr = static_cast<int64_t>(col_iters->size()) + 1;
}

}  // namespace LightGBM
