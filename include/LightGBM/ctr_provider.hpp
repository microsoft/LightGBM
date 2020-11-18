/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_CTR_PROVIDER_H_
#define LIGHTGBM_CTR_PROVIDER_H_

#include <LightGBM/config.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/parser.h>

#include <vector>
#include <string>
#include <random>
#include <unordered_map>

namespace LightGBM {

//transform categorical features to ctr values before the bin construction process
class CTRProvider {
public:
  class CatConverter {
    protected:
      std::unordered_map<int, int> cat_fid_to_convert_fid_;
    public:
      virtual double CalcValue(const double sum_label, const double sum_count, const double all_fold_sum_count, const double prior) = 0;

      virtual double CalcValue(const double sum_label, const double sum_count, const double all_fold_sum_count) = 0;

      virtual std::string DumpToString() const = 0;

      virtual std::string Name() const = 0;

      virtual CatConverter* Copy() const = 0;

      virtual void SetPrior(const double /*prior*/, const double /*prior_weight*/) {}

      void SetCatFidToConvertFid(const std::unordered_map<int, int>& cat_fid_to_convert_fid) {
        cat_fid_to_convert_fid_ = cat_fid_to_convert_fid;
      }

      void RegisterConvertFid(const int cat_fid, const int convert_fid) {
        cat_fid_to_convert_fid_[cat_fid] = convert_fid;
      }

      int GetConvertFid(const int cat_fid) const {
        return cat_fid_to_convert_fid_.at(cat_fid);
      }

      int CalcNumExtraFeatures() const {
        int num_extra_features = 0;
        for (const auto& pair : cat_fid_to_convert_fid_) {
          if (pair.first != pair.second) {
            ++num_extra_features;
          }
        }
        return num_extra_features;
      }

      static CatConverter* CreateFromString(const std::string& model_string, const double prior_weight) {
        std::vector<std::string> split_model_string = Common::Split(model_string.c_str(), ",");
        if (split_model_string.size() > 2 || split_model_string.size() == 0) {
          Log::Fatal("Invalid CatConverter model string %s", model_string.c_str());
        }
        const std::string& cat_converter_name = split_model_string[0];
        CatConverter* cat_converter = nullptr;
        if (Common::StartsWith(cat_converter_name, std::string("label_mean_ctr"))) {
          double prior = 0.0f;
          Common::Atof(Common::Split(cat_converter_name.c_str(), ':')[1].c_str(), &prior);
          cat_converter = new CTRConverterLabelMean();
          cat_converter->SetPrior(prior, prior_weight);
        } else if (Common::StartsWith(cat_converter_name, std::string("ctr"))) {
          double prior = 0.0f;
          Common::Atof(Common::Split(cat_converter_name.c_str(), ':')[1].c_str(), &prior);
          cat_converter = new CTRConverter(prior);
        } else if (cat_converter_name == std::string("count")) {
          cat_converter = new CountConverter();
        } else {
          Log::Fatal("Invalid CatConverter model string %s", model_string.c_str());
        }
        cat_converter->cat_fid_to_convert_fid_.clear();
        if (split_model_string.size() == 2) {
          const std::string& feature_map = split_model_string[1];
          std::stringstream feature_map_stream(feature_map);
          int key = 0, val = 0;
          while (feature_map_stream >> key) {
            CHECK(feature_map_stream.get() == ':');
            feature_map_stream >> val;
            cat_converter->cat_fid_to_convert_fid_[key] = val;
            feature_map_stream.get();
          }
        }
        return cat_converter;
      }
  };

  class CTRConverter: public CatConverter {
    public:
      CTRConverter(const double prior): prior_(prior) {}
      inline virtual double CalcValue(const double sum_label, const double sum_count, const double /*all_fold_sum_count*/) override {
        return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
      }

      inline virtual double CalcValue(const double sum_label, const double sum_count, const double /*all_fold_sum_count*/, const double /*prior*/) override {
        return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
      }

      virtual void SetPrior(const double /*prior*/, const double prior_weight) {
        prior_weight_ = prior_weight;
      }

      std::string Name() const override {
        std::stringstream str_stream;
        str_stream << "ctr:" << prior_;
        return str_stream.str();
      }

      std::string DumpToString() const override {
        std::stringstream str_stream;
        str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
        return str_stream.str();
      }

      CatConverter* Copy() const override {
        CatConverter* ret = new CTRConverter(prior_);
        ret->SetCatFidToConvertFid(cat_fid_to_convert_fid_);
        return ret;
      }

    private:
      const double prior_;
      double prior_weight_;
  };

  class CountConverter: public CatConverter {
    public:
      CountConverter() {}

      CatConverter* Copy() const override {
        CatConverter* ret = new CountConverter();
        ret->SetCatFidToConvertFid(cat_fid_to_convert_fid_);
        return ret;
      }
    private:
      inline virtual double CalcValue(const double /*sum_label*/, const double /*sum_count*/, const double all_fold_sum_count) override {
        return all_fold_sum_count;
      }

      inline virtual double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
        const double all_fold_sum_count, const double /*prior*/) override {
        return all_fold_sum_count;
      }
      
      std::string Name() const override {
        return std::string("count");
      }

      std::string DumpToString() const override {
        std::stringstream str_stream;
        str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
        return str_stream.str();
      }
  };

  class CTRConverterLabelMean: public CatConverter {
    public:
      CTRConverterLabelMean() { prior_set_ = false; }
      virtual void SetPrior(const double prior, const double prior_weight) {
        prior_ = prior;
        prior_weight_ = prior_weight;
        prior_set_ = true;
      }

      inline virtual double CalcValue(const double sum_label, const double sum_count, const double /*all_fold_sum_count*/) override {
        if(!prior_set_) {
          Log::Fatal("CTRConverterLabelMean is not ready since the prior value is not set.");
        }
        return (sum_label + prior_weight_ * prior_) / (sum_count + prior_weight_);
      }

      inline virtual double CalcValue(const double sum_label, const double sum_count, const double /*all_fold_sum_count*/, const double prior) override {
        if(!prior_set_) {
          Log::Fatal("CTRConverterLabelMean is not ready since the prior value is not set.");
        }
        return (sum_label + prior * prior_weight_) / (sum_count + prior_weight_);
      }

      std::string Name() const override {
        std::stringstream str_stream;
        str_stream << "label_mean_ctr:" << prior_;
        return str_stream.str();
      }

      std::string DumpToString() const override {
        std::stringstream str_stream;
        str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
        return str_stream.str();
      }

      CatConverter* Copy() const override {
        CatConverter* ret = new CTRConverterLabelMean();
        ret->SetPrior(prior_, prior_weight_);
        ret->SetCatFidToConvertFid(cat_fid_to_convert_fid_);
        return ret;
      }

    private:
      double prior_;
      double prior_weight_;
      bool prior_set_;
  };

  CTRProvider(const Config& config,
    const int /*num_machines*/, const std::vector<std::string>& /*text_data*/,
    const std::function<void(const char* line, std::vector<std::pair<int, double>>* oneline_features, double* label)> /*parser_func*/):
    CTRProvider(config) {
    /*convert_calc_func_ = [this, &text_data, num_machines, parser_func](const data_size_t num_data, const int num_total_features) {
      if(cat_converters_.size() == 0) { return; }
      CHECK(num_data_ == static_cast<data_size_t>(text_data.size()));
      const data_size_t block_size = (num_data + num_threads_ - 1) / num_threads_;
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for(int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        std::vector<std::pair<int, double>> oneline_features;
        double label;
        std::vector<bool> is_feature_processed(num_total_features, false);
        const data_size_t thread_start = block_size * thread_id;
        const data_size_t thread_end = std::min(thread_start + block_size, num_data);
        for(data_size_t row_idx = thread_start; row_idx < thread_end; ++row_idx) {
          oneline_features.clear();
          parser_func(text_data[row_idx].c_str(), &oneline_features, &label);
          ProcessOneLine(oneline_features, label, row_idx, is_feature_processed, thread_id);
        }
      }
      FinishProcess(num_machines);
    };*/
  }

  CTRProvider(const Config& config, 
    const int /*num_machines*/, const char* /*filename*/,
    const std::function<void(const char* line, std::vector<std::pair<int, double>>* oneline_features, double* label)> /*parser_func*/): 
    CTRProvider(config) {
    //convert_calc_func_ = [this, filename, num_machines, parser_func](const data_size_t /*num_data*/, const int num_total_features) {
    /*  if(cat_converters_.size() == 0) { return; }
      std::vector<std::pair<int, double>> oneline_features;
      std::vector<bool> is_feature_processed(num_total_features, false);
      double label;
      TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
      text_reader.ReadAllAndProcess([parser_func, this, &oneline_features, &label, &is_feature_processed] (data_size_t row_idx, const char* buffer, size_t size) {
        std::string line(buffer, size);
        oneline_features.clear();
        parser_func(line.c_str(), &oneline_features, &label);
        ProcessOneLine(oneline_features, label, row_idx, is_feature_processed);
      });
      FinishProcess(num_machines);
    };*/
  }

  CTRProvider(const Config& config,
    const int /*num_machines*/, const int32_t /*nmat*/, const std::vector<std::function<std::vector<double>(int row_idx)>>& /*get_row_fun*/,
    const std::function<double(int row_idx)>& /*get_label_fun*/, const int32_t* /*nrow*/):
    CTRProvider(config) {
    // we assume that, when constructing datasets from mats under distributed settings, local feature number is the same as global feature number
    // so the size of oneline_features should equal to num_original_features_ when Init of CTRProvider is called
    //convert_calc_func_ = [this, &get_label_fun, &get_row_fun, nmat, nrow, num_machines] (const data_size_t /*num_data*/, const int /*num_total_features*/) {
    /*  if(cat_converters_.size() == 0) { return; }
      int32_t mat_offset = 0;
      for(int32_t i_mat = 0; i_mat < nmat; ++i_mat) {
        const int32_t mat_nrow = nrow[i_mat];
        const auto& mat_get_row_fun = get_row_fun[i_mat];
        Threading::For<int32_t>(0, mat_nrow, 1024, [this, &mat_get_row_fun, &get_label_fun, &mat_offset](int thread_id, int32_t start, int32_t end) {
          for(int32_t j = start; j < end; ++j) {
            const std::vector<double>& oneline_features = mat_get_row_fun(j);
            const int32_t row_idx = j + mat_offset;
            const double label = get_label_fun(row_idx);
            ProcessOneLine(oneline_features, label, row_idx, thread_id);
          }
        });
        mat_offset += mat_nrow;
      }
      FinishProcess(num_machines);
    };*/
  }

  CTRProvider(const Config& config,
    const int /*num_machines*/, const std::function<std::vector<std::pair<int, double>>(int row_idx)>& /*get_row_fun*/,
    const std::function<double(int row_idx)>& /*get_label_fun*/, const int32_t /*nrow*/, const int32_t /*ncol*/):
    CTRProvider(config) {
    //convert_calc_func_ = [this, &get_label_fun, &get_row_fun, nrow, num_machines, ncol] (const data_size_t /*num_data*/, const int /*num_total_features*/) {
    /*  if(cat_converters_.size() == 0) { return; }
      Threading::For<int32_t>(0, nrow, 1024, [this, &get_row_fun, &get_label_fun, ncol](int thread_id, int32_t start, int32_t end) {
        std::vector<bool> is_feature_processed(ncol, false);
        for(int32_t row_idx = start; row_idx < end; ++row_idx) {
          const std::vector<std::pair<int, double>>& oneline_features = get_row_fun(row_idx);
          const double label = get_label_fun(row_idx);
          ProcessOneLine(oneline_features, label, row_idx, is_feature_processed, thread_id);
        }
      });
      FinishProcess(num_machines);
    };*/
  }

  CTRProvider(const Config& config,
    const int /*num_machines*/, const std::vector<std::vector<std::function<double(int row_idx)>>>& /*col_iter_funcs*/,
    const std::function<double(int row_idx)>& /*get_label_fun*/,
    const int32_t /*nrow*/, const int32_t /*ncol*/):
    CTRProvider(config) {
    // we assume that, when constructing datasets from CSC under distributed settings, local feature number is the same as global feature number
    // so the size of oneline_features should equal to num_original_features_ when Init of CTRProvider is called
    //convert_calc_func_ = [this, &col_iter_funcs, &get_label_fun, ncol, nrow, num_machines] (const data_size_t /*num_data*/, const int /*num_total_features*/) {
    /*  if(cat_converters_.size() == 0) { return; }
      int32_t mat_offset = 0;
      Threading::For<int32_t>(0, nrow, 1024, [this, &col_iter_funcs, &get_label_fun, &mat_offset, ncol](int thread_id, int32_t start, int32_t end) {
        std::vector<double> oneline_features(ncol, 0.0f);
        for(int32_t row_idx = start; row_idx < end; ++row_idx) {
          for(int32_t col_idx = 0; col_idx < ncol; ++col_idx) {
            oneline_features[col_idx] = col_iter_funcs[thread_id][col_idx](row_idx);
          }
          const double label = get_label_fun(row_idx);
          ProcessOneLine(oneline_features, label, row_idx, thread_id);
        }
      });
      FinishProcess(num_machines);
    };*/
  }

  CTRProvider(const CTRProvider& other):
    config_(other.config_) {
    num_data_ = other.num_data_;
    categorical_features_ = other.categorical_features_;
    training_data_fold_id_.resize(other.training_data_fold_id_.size());
    num_threads_ = other.num_threads_;
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < static_cast<int>(training_data_fold_id_.size()); ++i) {
      training_data_fold_id_[i] = other.training_data_fold_id_[i];
    }
    convert_fid_to_cat_fid_ = other.convert_fid_to_cat_fid_;
    prior_ = other.prior_;
    fold_prior_ = other.fold_prior_;
    is_categorical_feature_ = other.is_categorical_feature_;
    push_training_data_func_ = other.push_training_data_func_;
    push_valid_data_func_ = other.push_valid_data_func_;
    num_original_features_ = other.num_original_features_;
    num_total_features_ = other.num_total_features_;
    for (const auto& pair : other.count_info_) {
      count_info_[pair.first] = pair.second;
    }
    for (const auto& pair : other.label_info_) {
      label_info_[pair.first] = pair.second;
    }
    thread_count_info_.clear();
    thread_label_info_.clear();
    thread_fold_label_sum_.clear();
    thread_fold_num_data_.clear();
    thread_count_info_.shrink_to_fit();
    thread_label_info_.shrink_to_fit();
    thread_fold_label_sum_.shrink_to_fit();
    thread_fold_num_data_.shrink_to_fit();
    fold_label_sum_ = other.fold_label_sum_;
    fold_num_data_ = other.fold_num_data_;
    max_bin_by_feature_ = other.max_bin_by_feature_;
    cat_converters_.clear();
    for (const std::unique_ptr<CatConverter>& cat_converter: other.cat_converters_) {
      cat_converters_.emplace_back(cat_converter->Copy());
    }
  }

  ~CTRProvider() {
    training_data_fold_id_.clear();
    training_data_fold_id_.shrink_to_fit();
    convert_fid_to_cat_fid_.clear();
    fold_prior_.clear();
    fold_prior_.shrink_to_fit();
    is_categorical_feature_.clear();
    is_categorical_feature_.shrink_to_fit();
    count_info_.clear();
    label_info_.clear();
    max_bin_by_feature_.clear();
    max_bin_by_feature_.shrink_to_fit();
    cat_converters_.clear();
    cat_converters_.shrink_to_fit();
  }

  static CTRProvider* CreateCTRProvider(Config& config,
    const int num_machines, const char* filename) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, num_machines, filename));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  void Init() {
    num_total_features_ = num_original_features_;
    CHECK(max_bin_by_feature_.empty() || static_cast<int>(max_bin_by_feature_.size()) == num_original_features_);
    fold_prior_.resize(config_.num_ctr_folds + 1, 0.0f);
    if (cat_converters_.size() > 0) {
      // prepare to accumulate ctr statistics
      fold_label_sum_.resize(config_.num_ctr_folds + 1, 0.0f);
      fold_num_data_.resize(config_.num_ctr_folds + 1, 0);
      thread_fold_label_sum_.resize(num_threads_);
      thread_fold_num_data_.resize(num_threads_);
      thread_count_info_.resize(num_threads_);
      thread_label_info_.resize(num_threads_);
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        thread_fold_label_sum_[thread_id].resize(config_.num_ctr_folds + 1, 0.0f);
        thread_fold_num_data_[thread_id].resize(config_.num_ctr_folds + 1, 0.0f);
      }
      for (const int fid : categorical_features_) {
        count_info_[fid].resize(config_.num_ctr_folds + 1);
        label_info_[fid].resize(config_.num_ctr_folds + 1);
        for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
          thread_count_info_[thread_id][fid].resize(config_.num_ctr_folds + 1);
          thread_label_info_[thread_id][fid].resize(config_.num_ctr_folds + 1);
        }
      }

      size_t append_from = 0;
      if (!config_.keep_old_cat_method) {
        auto& cat_converter = cat_converters_[0];
        for (int fid : categorical_features_) {
          cat_converter->RegisterConvertFid(fid, fid);
          convert_fid_to_cat_fid_[fid] = fid;
        }
        append_from = 1;
      }
      for (size_t i = append_from; i < cat_converters_.size(); ++i) {
        auto& cat_converter = cat_converters_[i];
        for (const int& fid : categorical_features_) {
          cat_converter->RegisterConvertFid(fid, num_total_features_);
          convert_fid_to_cat_fid_[num_total_features_] = fid;
          if (!max_bin_by_feature_.empty()) {
            max_bin_by_feature_.push_back(max_bin_by_feature_[fid]);
          }
          ++num_total_features_;
        }
      }
    }
  }

  void ProcessOneLine(const std::vector<double>& one_line, double label, int line_idx, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>& is_feature_processed, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>& is_feature_processed, const int fold_id);

  std::string DumpModelInfo() const;

  static CTRProvider* RecoverFromModelString(const std::string model_string);

  // recover ctr values from string
  static std::unordered_map<int, std::unordered_map<int, double>> RecoverCTRValues(const std::string str);

  // create ctr convert function
  void CreatePushDataFunction(const std::vector<int>& used_feature_idx, 
    const std::vector<int>& feature_to_group,
    const std::vector<int>& feature_to_sub_feature,
    const std::function<void(int tid, data_size_t row_idx, 
      int group, int sub_feature, double value)>& feature_group_push_data_func); 

  void PushTrainingOneData(int tid, data_size_t row_idx, int group, int sub_feature, double value) const {
    push_training_data_func_(tid, row_idx, group, sub_feature, value);
  }

  void PushValidOneData(int tid, data_size_t row_idx, int group, int sub_feature, double value) const {
    push_valid_data_func_(tid, row_idx, group, sub_feature, value);
  }

  bool IsCategorical(const int fid) const { 
    if(fid < num_original_features_) {
      return is_categorical_feature_[fid]; 
    }
    else {
      return false;
    }
  }

  int ConvertFidToCatFid(int convert_fid) const { return convert_fid_to_cat_fid_.at(convert_fid); }

  inline int GetNumOriginalFeatures() const { return num_original_features_; }

  inline int GetNumTotalFeatures() const { 
    return num_total_features_;
  }

  inline int GetNumCatConverters() const {
    return static_cast<int>(cat_converters_.size());
  }

  inline int GetMaxBinForFeature(const int fid) const {
    max_bin_by_feature_[fid];
  }

  // replace categorical feature values of sampled data with ctr values and count values
  void ReplaceCategoricalValues(const std::vector<data_size_t>& sampled_data_indices, 
    std::vector<std::vector<int>>& sampled_non_missing_data_indices,
    std::vector<std::vector<double>>& sampled_non_missing_feature_values,
    std::unordered_set<int>& ignored_features);

  void ConvertCatToCTR(double* features) const;

  void ConvertCatToCTR(std::unordered_map<int, double>& features) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>& features, const int fold_id) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>& features) const;

  int CalcNumExtraFeatures() const {
    int num_extra_features = 0;
    for (const auto& cat_converter : cat_converters_) {
      num_extra_features += cat_converter->CalcNumExtraFeatures();
    }
    return num_extra_features;
  }

  void ExtendFeatureNames(std::vector<std::string>& feature_names) const {
    CHECK(static_cast<int>(feature_names.size()) == num_original_features_);
    const std::vector<std::string> old_feature_names = feature_names;
    feature_names.resize(num_total_features_);
    for (int fid = 0; fid < num_original_features_; ++fid) {
      if (is_categorical_feature_[fid]) {
        for (const auto& cat_converter : cat_converters_) {
          const int convert_fid = cat_converter->GetConvertFid(fid);
          feature_names[convert_fid] = old_feature_names[fid] + std::string("[") + cat_converter->Name() + std::string("]");
        }
      }
    }
  }

private:
  CTRProvider(const Config& config): config_(config) {
    num_threads_ = config_.num_threads > 0 ? config_.num_threads : OMP_NUM_THREADS();
    const std::string ctr_string = std::string("ctr");
    if(config_.cat_converters.size() > 0) {
      for (auto token : Common::Split(config_.cat_converters.c_str(), ',')) {
        if (Common::StartsWith(token, "ctr")) {
          if (token.size() == ctr_string.size()) {
            cat_converters_.emplace_back(new CTRProvider::CTRConverterLabelMean());
          } else {
            double prior = 0.0f;
            if (!Common::AtofAndCheck(token.c_str() + ctr_string.size() + 1, &prior)) {
              Log::Fatal("CTR prior of cat_converter specification %s is not a valid float value.", token.c_str());
            }
            cat_converters_.emplace_back(new CTRProvider::CTRConverter(prior));
          }
        } else if(token == std::string("count")) {
          cat_converters_.emplace_back(new CTRProvider::CountConverter());
        }
        else {
          Log::Fatal("Unknown cat_converters specification %s.", token.c_str());
        }
      }
    }

    max_bin_by_feature_ = config_.max_bin_by_feature;
    prior_ = 0.0f;
  }

  CTRProvider(const std::string model_string) {
    std::stringstream str_stream(model_string);
    int cat_fid = 0, cat_value = 0;
    double label_sum = 0.0f, total_count = 0.0f;
    while (str_stream >> cat_fid) {
      CHECK(str_stream.get() == ' ');
      label_info_[cat_fid].clear();
      count_info_[cat_fid].clear();
      label_info_[cat_fid].resize(1);
      count_info_[cat_fid].resize(1);
      while (str_stream >> cat_value) {
        CHECK(str_stream.get() == ':');
        str_stream >> label_sum;
        CHECK(str_stream.get() == ':');
        str_stream >> total_count;
        label_info_[cat_fid][0][cat_value] = label_sum;
        count_info_[cat_fid][0][cat_value] = total_count;
      }
      str_stream.clear();
      CHECK(str_stream.get() == '@');
    }
    str_stream.clear();
    cat_converters_.clear();
    std::string cat_converter_string;
    while (str_stream >> cat_converter_string) {
      cat_converters_.emplace_back(CatConverter::CreateFromString(cat_converter_string, config_.prior_weight));
    }
  }

  CTRProvider(Config& config, const int num_machines, const char* filename): 
    CTRProvider(config) {
    const int label_idx = ParseMetaInfo(filename, config);
    auto parser = std::unique_ptr<Parser>(Parser::CreateParser(filename, config_.header, 0, label_idx));
    auto categorical_features = categorical_features_;
    categorical_features_.clear();
    categorical_features_.shrink_to_fit();
    num_original_features_ = parser->NumFeatures();
    if (num_machines > 1) {
      num_original_features_ = Network::GlobalSyncUpByMax(num_original_features_);
    }
    for (const int fid : categorical_features) {
      if (fid < num_original_features_) {
        categorical_features_.push_back(fid);
      }
    }
    categorical_features.clear();
    categorical_features.shrink_to_fit();
    is_categorical_feature_.clear();
    is_categorical_feature_.resize(num_original_features_, false);
    for (const int fid : categorical_features_) {
      is_categorical_feature_[fid] = true;
    }
    Init();
    if(cat_converters_.size() == 0) { return; }
    std::vector<std::pair<int, double>> oneline_features;
    std::vector<bool> is_feature_processed(num_total_features_, false);
    double label;
    TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
    num_data_ = 0;
    std::mt19937 mt_generator;
    const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    training_data_fold_id_.clear();
    text_reader.ReadAllAndProcess([&parser, this, &oneline_features, &label, &is_feature_processed,
      &fold_distribution, &mt_generator] 
      (data_size_t row_idx, const char* buffer, size_t size) {
      std::string line(buffer, size);
      oneline_features.clear();
      parser->ParseOneLine(line.c_str(), &oneline_features, &label);
      const int fold_id = fold_distribution(mt_generator);
      training_data_fold_id_.push_back(fold_id);
      ProcessOneLine(oneline_features, label, row_idx, is_feature_processed, fold_id);
      ++num_data_;
    });
    FinishProcess(num_machines);
  }

  void ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line, double label, int line_idx, std::vector<bool>& is_feature_processed,
    std::unordered_map<int, std::vector<std::unordered_map<int, int>>>& count_info,
    std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>>& label_info,
    std::vector<label_t>& label_sum, std::vector<int>& num_data, const int fold_id);

  // generate fold ids for training data, each data point will be randomly allocated into one fold
  void GenTrainingDataFoldID();

  // expand to use count encodings, adding new feature columns to the sampled feature values
  void ExpandCountEncodings(std::vector<std::vector<int>>& sampled_non_missing_data_indices,
    std::vector<std::vector<double>>& sampled_non_missing_feature_values,
    std::unordered_set<int>& ignored_features); 

  // sync up ctr values by gathering statistics from all machines in distributed scenario
  void SyncCTRStat(std::vector<std::unordered_map<int, label_t>>& fold_label_sum,
    std::vector<std::unordered_map<int, int>>& fold_total_count, const int num_machines) const; 
  
  // sync up statistics to calculate the ctr prior by gathering statistics from all machines in distributed scenario
  void SyncCTRPrior(const double label_sum, const int local_num_data, double& all_label_sum, int& all_num_data, int num_machines) const;

  // dump a dictionary to string
  template <typename T>
  static std::string DumpDictToString(const std::unordered_map<int, T>& dict, const char delimiter) {
    std::stringstream str_buf;
    if(dict.empty()) {
      return str_buf.str();
    }
    auto iter = dict.begin();
    str_buf << iter->first << ":" << iter->second;
    ++iter;
    for(; iter != dict.end(); ++iter) {
      str_buf << delimiter << iter->first << ":" << iter->second;
    }
    return str_buf.str();
  }

  void FinishProcess(const int num_machines);

  inline double TrimConvertValue(const double convert_value) {
    if (std::fabs(convert_value) > kZeroThreshold || std::isnan(convert_value)) {
      return convert_value;
    }
    if (convert_value < 0.0f) {
      return -2 * kZeroThreshold;
    }
    else {
      return 2 * kZeroThreshold;
    }
  }

  int ParseMetaInfo(const char* filename, Config& config) {
    std::unordered_map<std::string, int> name2idx;
    std::string name_prefix("name:");
    TextReader<data_size_t> text_reader(filename, config.header);
    // get column names
    std::vector<std::string> feature_names;
    if (config.header) {
      std::string first_line = text_reader.first_line();
      feature_names = Common::Split(first_line.c_str(), "\t,");
    }
    int label_idx = 0;
    // load label idx first
    if (config.label_column.size() > 0) {
      if (Common::StartsWith(config.label_column, name_prefix)) {
        std::string name = config.label_column.substr(name_prefix.size());
        label_idx = -1;
        for (int i = 0; i < static_cast<int>(feature_names.size()); ++i) {
          if (name == feature_names[i]) {
            label_idx = i;
            break;
          }
        }
        if (label_idx >= 0) {
          Log::Info("Using column %s as label", name.c_str());
        } else {
          Log::Fatal("Could not find label column %s in data file \n"
                     "or data file doesn't contain header", name.c_str());
        }
      } else {
        if (!Common::AtoiAndCheck(config.label_column.c_str(), &label_idx)) {
          Log::Fatal("label_column is not a number,\n"
                     "if you want to use a column name,\n"
                     "please add the prefix \"name:\" to the column name");
        }
        Log::Info("Using column number %d as label", label_idx);
      }
    }

    if (!feature_names.empty()) {
      // erase label column name
      feature_names.erase(feature_names.begin() + label_idx);
      for (size_t i = 0; i < feature_names.size(); ++i) {
        name2idx[feature_names[i]] = static_cast<int>(i);
      }
    }

    // load ignore columns
    std::unordered_set<int> ignore_features;
    if (config.ignore_column.size() > 0) {
      if (Common::StartsWith(config.ignore_column, name_prefix)) {
        std::string names = config.ignore_column.substr(name_prefix.size());
        for (auto name : Common::Split(names.c_str(), ',')) {
          if (name2idx.count(name) > 0) {
            int tmp = name2idx[name];
            ignore_features.emplace(tmp);
          } else {
            Log::Fatal("Could not find ignore column %s in data file", name.c_str());
          }
        }
      } else {
        for (auto token : Common::Split(config.ignore_column.c_str(), ',')) {
          int tmp = 0;
          if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
            Log::Fatal("ignore_column is not a number,\n"
                       "if you want to use a column name,\n"
                       "please add the prefix \"name:\" to the column name");
          }
          ignore_features.emplace(tmp);
        }
      }
    }
    std::unordered_set<int> categorical_features;
    if (config.categorical_feature.size() > 0) {
      if (Common::StartsWith(config.categorical_feature, name_prefix)) {
        std::string names = config.categorical_feature.substr(name_prefix.size());
        for (auto name : Common::Split(names.c_str(), ',')) {
          if (name2idx.count(name) > 0) {
            int tmp = name2idx[name];
            categorical_features.emplace(tmp);
          } else {
            Log::Fatal("Could not find categorical_feature %s in data file", name.c_str());
          }
        }
      } else {
        for (auto token : Common::Split(config.categorical_feature.c_str(), ',')) {
          int tmp = 0;
          if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
            Log::Fatal("categorical_feature is not a number,\n"
                      "if you want to use a column name,\n"
                      "please add the prefix \"name:\" to the column name");
          }
          categorical_features.emplace(tmp);
        }
      }
    }

    for (const int fid : ignore_features) {
      if (categorical_features.count(fid)) {
        categorical_features.erase(fid);
      }
    }

    // reset categorical features for Config
    if (config.categorical_feature.size() > 0) {
      if (!config_.keep_old_cat_method) {
        config.categorical_feature.clear();
        config.categorical_feature.shrink_to_fit();
      }
    }

    categorical_features_.clear();
    for (const int fid : categorical_features) {
      categorical_features_.push_back(fid);
    }
    std::sort(categorical_features_.begin(), categorical_features_.end());

    return label_idx;
  }

  // parameter configuration
  const Config config_;
  
  // size of training data
  data_size_t num_data_;
  // list of categorical feature indices (real index, not inner index of Dataset)
  std::vector<int> categorical_features_;

  // maps training data index to fold index
  std::vector<int> training_data_fold_id_;
  // maps converted feature index to the feature index of original categorical feature
  std::unordered_map<int, int> convert_fid_to_cat_fid_;
  // prior used per fold
  std::vector<double> fold_prior_;
  // mean of labels of sampled data
  double prior_;
  // record whether a feature is categorical in the original data
  std::vector<bool> is_categorical_feature_; 
  
  // push one feature value of one data into the dataset
  std::function<void(int tid, data_size_t row_idx,
      int group, int sub_feature, double value)> push_training_data_func_, push_valid_data_func_;

  // number of features in the original dataset, without adding count features
  int num_original_features_;
  // number of features after converting categorical features
  int num_total_features_;

  // number of threads used for ctr encoding
  int num_threads_;

  // the accumulated count information for ctr
  std::unordered_map<int, std::vector<std::unordered_map<int, int>>> count_info_;
  // the accumulated label sum information for ctr
  std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>> label_info_;
  // the accumulated count information for ctr per thread
  std::vector<std::unordered_map<int, std::vector<std::unordered_map<int, int>>>> thread_count_info_;
  // the accumulated label sum information for ctr per thread
  std::vector<std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>>> thread_label_info_;
  // the accumulated label sum per fold
  std::vector<label_t> fold_label_sum_;
  // the accumulated label sum per thread per fold
  std::vector<std::vector<label_t>> thread_fold_label_sum_;
  // the accumulated number of data per fold per thread
  std::vector<std::vector<int>> thread_fold_num_data_;
  // number of data per fold
  std::vector<data_size_t> fold_num_data_;
  // categorical value converters
  std::vector<std::unique_ptr<CatConverter>> cat_converters_;
  // max bin by feature
  std::vector<int> max_bin_by_feature_;
};

class CTRParser : public Parser {
  public:
    explicit CTRParser(const Parser* inner_parser,
    const std::unique_ptr<CTRProvider>& ctr_provider, const bool is_valid):
    inner_parser_(inner_parser), ctr_provider_(ctr_provider), is_valid_(is_valid) {

    }

    inline void ParseOneLine(const char* str,
      std::vector<std::pair<int, double>>* out_features,
      double* out_label, const int line_idx = -1) const override {
      inner_parser_->ParseOneLine(str, out_features, out_label);
      if (is_valid_) {
        ctr_provider_->ConvertCatToCTR(*out_features);
      } else {
        ctr_provider_->ConvertCatToCTR(*out_features, line_idx);
      }
    }

    inline int NumFeatures() const override {
      return ctr_provider_->GetNumTotalFeatures();
    }

  private:
    std::unique_ptr<const Parser> inner_parser_;
    const std::unique_ptr<CTRProvider>& ctr_provider_;
    const bool is_valid_;
};

} //namespace LightGBM
#endif //LightGBM_CTR_PROVIDER_H_
