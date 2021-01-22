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
#include <LightGBM/parser_base.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace LightGBM {

// transform categorical features to ctr values before the bin construction process
class CTRProvider {
 public:
  class CatConverter {
   protected:
      std::unordered_map<int, int> cat_fid_to_convert_fid_;

   public:
      virtual ~CatConverter() {}

      virtual double CalcValue(const double sum_label, const double sum_count,
        const double all_fold_sum_count, const double prior) const = 0;

      virtual double CalcValue(const double sum_label, const double sum_count,
        const double all_fold_sum_count) const = 0;

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

      static CatConverter* CreateFromString(const std::string& model_string, const double prior_weight) {
        std::vector<std::string> split_model_string = Common::Split(model_string.c_str(), ",");
        if (split_model_string.size() != 2) {
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
          cat_converter->SetPrior(prior, prior_weight);
        } else if (cat_converter_name == std::string("count")) {
          cat_converter = new CountConverter();
        } else {
          Log::Fatal("Invalid CatConverter model string %s", model_string.c_str());
        }
        cat_converter->cat_fid_to_convert_fid_.clear();

        const std::string& feature_map = split_model_string[1];
        std::stringstream feature_map_stream(feature_map);
        int key = 0, val = 0;
        while (feature_map_stream >> key) {
          CHECK_EQ(feature_map_stream.get(), ':');
          feature_map_stream >> val;
          cat_converter->cat_fid_to_convert_fid_[key] = val;
          feature_map_stream.get();
        }

        return cat_converter;
      }
  };

  class CTRConverter: public CatConverter {
   public:
      explicit CTRConverter(const double prior): prior_(prior) {}
      inline double CalcValue(const double sum_label, const double sum_count,
        const double /*all_fold_sum_count*/) const override {
        return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
      }

      inline double CalcValue(const double sum_label, const double sum_count,
        const double /*all_fold_sum_count*/, const double /*prior*/) const override {
        return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
      }

      void SetPrior(const double /*prior*/, const double prior_weight) override {
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
        ret->SetPrior(prior_, prior_weight_);
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
      inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
        const double all_fold_sum_count) const override {
        return all_fold_sum_count;
      }

      inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
        const double all_fold_sum_count, const double /*prior*/) const override {
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

      void SetPrior(const double prior, const double prior_weight) override {
        prior_ = prior;
        prior_weight_ = prior_weight;
        prior_set_ = true;
      }

      inline double CalcValue(const double sum_label, const double sum_count,
        const double /*all_fold_sum_count*/) const override {
        if (!prior_set_) {
          Log::Fatal("CTRConverterLabelMean is not ready since the prior value is not set.");
        }
        return (sum_label + prior_weight_ * prior_) / (sum_count + prior_weight_);
      }

      inline double CalcValue(const double sum_label, const double sum_count,
        const double /*all_fold_sum_count*/, const double prior) const override {
        if (!prior_set_) {
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

  // for file data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const int num_machines, const char* filename) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, num_machines, filename));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for pandas/numpy array data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    int32_t nmat, int32_t* nrow, int32_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, get_row_fun, get_label_fun, nmat, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for csr sparse matrix data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, get_row_fun, get_label_fun, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for csc sparse matrix data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_func,
    const std::function<double(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, csc_func, get_label_fun, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  void Init(Config* config) {
    num_total_features_ = num_original_features_;
    CHECK(max_bin_by_feature_.empty() || static_cast<int>(max_bin_by_feature_.size()) == num_original_features_);
    CHECK(config->max_bin_by_feature.empty() || static_cast<int>(config->max_bin_by_feature.size()) == num_original_features_);
    is_categorical_feature_.clear();
    is_categorical_feature_.resize(num_original_features_, false);
    for (const int fid : categorical_features_) {
      is_categorical_feature_[fid] = true;
    }
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
      if (!keep_raw_cat_method_) {
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
            config->max_bin_by_feature.push_back(max_bin_by_feature_[fid]);
          }
          ++num_total_features_;
        }
      }
    }
  }

  void ProcessOneLine(const std::vector<double>& one_line, double label,
    int line_idx, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, int thread_id, const int fold_id);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, const int fold_id);

  std::string DumpModelInfo() const;

  static CTRProvider* RecoverFromModelString(const std::string model_string);

  // recover ctr values from string
  static std::unordered_map<int, std::unordered_map<int, double>> RecoverCTRValues(const std::string str);

  bool IsCategorical(const int fid) const {
    if (fid < num_original_features_) {
      return is_categorical_feature_[fid];
    } else {
      return false;
    }
  }

  int ConvertFidToCatFid(int convert_fid) const { return convert_fid_to_cat_fid_.at(convert_fid); }

  inline int GetNumOriginalFeatures() const {
    return num_original_features_;
  }

  inline int GetNumTotalFeatures() const {
    return num_total_features_;
  }

  inline int GetNumCatConverters() const {
    return static_cast<int>(cat_converters_.size());
  }

  void IterateOverCatConverters(int fid, double fval, int line_idx,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  void IterateOverCatConverters(int fid, double fval,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  template <bool IS_TRAIN>
  void IterateOverCatConvertersInner(int fid, double fval, int fold_id,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const {
    const int int_fval = static_cast<int>(fval);
    double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
    const auto& fold_label_info = IS_TRAIN ?
      label_info_.at(fid).at(fold_id) : label_info_.at(fid).back();
    const auto& fold_count_info = IS_TRAIN ?
      count_info_.at(fid).at(fold_id) : count_info_.at(fid).back();
    if (fold_count_info.count(int_fval) > 0) {
      label_sum = fold_label_info.at(int_fval);
      total_count = fold_count_info.at(int_fval);
    }
    if (IS_TRAIN) {
      const auto& all_fold_count_info = count_info_.at(fid).back();
      if (all_fold_count_info.count(int_fval) > 0) {
        all_fold_total_count = all_fold_count_info.at(int_fval);
      }
    } else {
      all_fold_total_count = total_count;
    }
    for (const auto& cat_converter : cat_converters_) {
      const double convert_value =
        cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
      const int convert_fid = cat_converter->GetConvertFid(fid);
      write_func(convert_fid, fid, convert_value);
    }
    post_process_func(fid);
  }

  template <bool IS_TRAIN>
  double HandleOneCatConverter(int fid, double fval, int fold_id,
    const CTRProvider::CatConverter* cat_converter) const {
    const int int_fval = static_cast<int>(fval);
    double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
    const auto& fold_label_info = IS_TRAIN ?
      label_info_.at(fid).at(fold_id) : label_info_.at(fid).back();
    const auto& fold_count_info = IS_TRAIN ?
      count_info_.at(fid).at(fold_id) : count_info_.at(fid).back();
    if (fold_count_info.count(int_fval) > 0) {
      label_sum = fold_label_info.at(int_fval);
      total_count = fold_count_info.at(int_fval);
    }
    if (IS_TRAIN) {
      const auto& all_fold_count_info = count_info_.at(fid).back();
      if (all_fold_count_info.count(int_fval) > 0) {
        all_fold_total_count = all_fold_count_info.at(int_fval);
      }
    } else {
      all_fold_total_count = total_count;
    }
    return cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
  }

  void ConvertCatToCTR(std::vector<double>* features, int line_idx) const;

  void ConvertCatToCTR(std::vector<double>* features) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr, const int fold_id) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr) const;

  double ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
    int col_idx, int line_idx) const;

  double ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
    int col_idx) const;

  void ExtendFeatureNames(std::vector<std::string>* feature_names_ptr) const {
    auto& feature_names = *feature_names_ptr;
    if (feature_names.empty()) {
      for (int i = 0; i < num_original_features_; ++i) {
        std::stringstream str_buf;
        str_buf << "Column_" << i;
        feature_names.push_back(str_buf.str());
      }
    }
    int feature_names_size = static_cast<int>(feature_names.size());
    const std::vector<std::string> old_feature_names = feature_names;
    std::vector<std::string> new_feature_names(num_total_features_);
    for (int fid = 0; fid < num_original_features_; ++fid) {
      new_feature_names[fid] = old_feature_names[fid];
      if (is_categorical_feature_[fid]) {
        for (const auto& cat_converter : cat_converters_) {
          const int convert_fid = cat_converter->GetConvertFid(fid);
          std::string cat_converter_name = cat_converter->Name();
          std::replace(cat_converter_name.begin(), cat_converter_name.end(), ':', '_');
          new_feature_names[convert_fid] = old_feature_names[fid] + std::string("_") + cat_converter_name;
        }
      }
    }
    if (feature_names_size == num_original_features_) {
      feature_names = new_feature_names;
    } else if (feature_names_size == num_original_features_ +
      static_cast<int>(cat_converters_.size()) * static_cast<int>(categorical_features_.size())) {
      for (size_t i = 0; i < new_feature_names.size(); ++i) {
        CHECK_EQ(new_feature_names[i], feature_names[i]);
      }
    } else {
      Log::Fatal("wrong length of feature_names");
    }
  }

  void WrapRowFunctions(
    std::vector<std::function<std::vector<double>(int row_idx)>>* get_row_fun,
    int32_t* ncol, bool is_valid) const;

  void WrapRowFunction(
    std::function<std::vector<std::pair<int, double>>(int row_idx)>* get_row_fun,
    int64_t* ncol, bool is_valid) const;

  template <typename T>
  std::function<std::vector<T>(int row_idx)> WrapRowFunctionInner(
    const std::function<std::vector<T>(int row_idx)>* get_row_fun, bool is_valid) const {
  std::function<std::vector<T>(int row_idx)> old_get_row_fun = *get_row_fun;
    if (is_valid) {
      return [old_get_row_fun, this] (int row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToCTR(&row);
        return row;
      };
    } else {
      return [old_get_row_fun, this] (int row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToCTR(&row, row_idx);
        return row;
      };
    }
  }

  void WrapColIters(
    std::vector<std::unique_ptr<CSC_RowIterator>>* col_iters,
    int64_t* ncol_ptr, bool is_valid, int64_t num_row) const;

 private:
  void SetConfig(const Config* config) {
    config_ = *config;
    num_threads_ = config_.num_threads > 0 ? config_.num_threads : OMP_NUM_THREADS();
    keep_raw_cat_method_ = false;
    const std::string ctr_string = std::string("ctr");
    if (config_.cat_converters.size() > 0) {
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
        } else if (token == std::string("count")) {
          cat_converters_.emplace_back(new CTRProvider::CountConverter());
        } else if (token == std::string("raw")) {
          keep_raw_cat_method_ = true;
        } else {
          Log::Fatal("Unknown cat_converters specification %s.", token.c_str());
        }
      }
    }

    max_bin_by_feature_ = config_.max_bin_by_feature;
    prior_ = 0.0f;
    prior_weight_ = config_.prior_weight;
  }

  explicit CTRProvider(const std::string model_string) {
    std::stringstream str_stream(model_string);
    int cat_fid = 0, cat_value = 0;
    double label_sum = 0.0f, total_count = 0.0f;
    int keep_raw_cat_method;
    str_stream >> keep_raw_cat_method;
    keep_raw_cat_method_ = static_cast<bool>(keep_raw_cat_method);
    str_stream >> num_original_features_;
    str_stream >> num_total_features_;
    str_stream >> prior_weight_;
    is_categorical_feature_.clear();
    is_categorical_feature_.resize(num_original_features_, false);
    categorical_features_.clear();
    while (str_stream >> cat_fid) {
      CHECK_EQ(str_stream.get(), ' ');
      is_categorical_feature_[cat_fid] = true;
      categorical_features_.push_back(cat_fid);
      label_info_[cat_fid].clear();
      count_info_[cat_fid].clear();
      label_info_[cat_fid].resize(1);
      count_info_[cat_fid].resize(1);
      while (str_stream >> cat_value) {
        CHECK_EQ(str_stream.get(), ':');
        str_stream >> label_sum;
        CHECK_EQ(str_stream.get(), ':');
        str_stream >> total_count;
        label_info_[cat_fid][0][cat_value] = label_sum;
        count_info_[cat_fid][0][cat_value] = total_count;
      }
      str_stream.clear();
      CHECK_EQ(str_stream.get(), '@');
    }
    std::sort(categorical_features_.begin(), categorical_features_.end());
    str_stream.clear();
    cat_converters_.clear();
    std::string cat_converter_string;
    while (str_stream >> cat_converter_string) {
      cat_converters_.emplace_back(CatConverter::CreateFromString(cat_converter_string, config_.prior_weight));
    }
  }

  CTRProvider(Config* config, const int num_machines, const char* filename) {
    SetConfig(config);
    const auto bin_filename = Parser::CheckCanLoadFromBin(filename);
    if (bin_filename.size() > 0) {
      cat_converters_.clear();
      return;
    }
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
    Init(config);
    if (cat_converters_.size() == 0) { return; }
    std::vector<std::pair<int, double>> oneline_features;
    std::vector<bool> is_feature_processed(num_total_features_, false);
    double label;
    TextReader<data_size_t> text_reader(filename, config_.header, config_.file_load_progress_interval_bytes);
    num_data_ = text_reader.CountLine();
    std::mt19937 mt_generator(config_.seed);
    const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    training_data_fold_id_.resize(num_data_, 0);
    text_reader.ReadAllAndProcess([&parser, this, &oneline_features, &label, &is_feature_processed,
      &fold_distribution, &mt_generator]
      (data_size_t row_idx, const char* buffer, size_t size) {
      std::string line(buffer, size);
      oneline_features.clear();
      parser->ParseOneLine(line.c_str(), &oneline_features, &label);
      const int fold_id = fold_distribution(mt_generator);
      training_data_fold_id_[row_idx] = fold_id;
      ProcessOneLine(oneline_features, label, row_idx, &is_feature_processed, fold_id);
    });
    FinishProcess(num_machines);
  }

  CTRProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun, const int32_t nmat,
    const int32_t* nrow, const int32_t ncol) {
    SetConfig(config);
    ParseMetaInfo(nullptr, config);
    num_original_features_ = ncol;
    Init(config);
    if (cat_converters_.size() == 0) { return; }
    if (get_label_fun == nullptr) {
      Log::Fatal("Please specify the label before the dataset is constructed to use CTR");
    }
    int32_t mat_offset = 0;
    num_data_ = 0;
    for (int32_t i = 0; i < nmat; ++i) {
      num_data_ += nrow[i];
    }
    std::vector<std::mt19937> mt_generators;
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      mt_generators.emplace_back(config_.seed + thread_id);
    }
    const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    training_data_fold_id_.resize(num_data_, 0);
    for (int32_t i_mat = 0; i_mat < nmat; ++i_mat) {
      const int32_t mat_nrow = nrow[i_mat];
      const auto& mat_get_row_fun = get_row_fun[i_mat];
      Threading::For<int32_t>(0, mat_nrow, 1024,
        [this, &mat_get_row_fun, &get_label_fun, &mat_offset, &fold_distribution, &mt_generators]
        (int thread_id, int32_t start, int32_t end) {
        for (int32_t j = start; j < end; ++j) {
          const std::vector<double>& oneline_features = mat_get_row_fun(j);
          const int32_t row_idx = j + mat_offset;
          const double label = get_label_fun(row_idx);
          const int fold_id = fold_distribution(mt_generators[thread_id]);
          training_data_fold_id_[row_idx] = fold_id;
          ProcessOneLine(oneline_features, label, row_idx, thread_id, fold_id);
        }
      });
      mat_offset += mat_nrow;
    }
    FinishProcess(1);
  }

  CTRProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol) {
    SetConfig(config);
    ParseMetaInfo(nullptr, config);
    num_original_features_ = ncol;
    num_data_ = nrow;
    training_data_fold_id_.resize(num_data_);
    Init(config);
    if (cat_converters_.size() == 0) { return; }
    if (get_label_fun == nullptr) {
      Log::Fatal("Please specify the label before the dataset is constructed to use CTR");
    }
    std::vector<std::mt19937> mt_generators;
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      mt_generators.emplace_back(config_.seed + thread_id);
    }
    const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    std::vector<std::vector<bool>> is_feature_processed(num_threads_);
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      is_feature_processed[thread_id].resize(num_total_features_, false);
    }
    Threading::For<int64_t>(0, nrow, 1024,
    [this, &get_row_fun, &get_label_fun, &fold_distribution, &mt_generators, &is_feature_processed]
    (int thread_id, int64_t start, int64_t end) {
      for (int64_t j = start; j < end; ++j) {
        const std::vector<std::pair<int, double>>& oneline_features = get_row_fun(j);
        const int32_t row_idx = j;
        const double label = get_label_fun(row_idx);
        const int fold_id = fold_distribution(mt_generators[thread_id]);
        training_data_fold_id_[row_idx] = fold_id;
        ProcessOneLine(oneline_features, label, row_idx, &is_feature_processed[thread_id], thread_id, fold_id);
      }
    });
    FinishProcess(1);
  }

  CTRProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_iters,
    const std::function<double(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol) {
    SetConfig(config);
    ParseMetaInfo(nullptr, config);
    num_original_features_ = ncol;
    num_data_ = nrow;
    training_data_fold_id_.resize(num_data_);
    Init(config);
    if (cat_converters_.size() == 0) { return; }
    if (get_label_fun == nullptr) {
      Log::Fatal("Please specify the label before the dataset is constructed to use CTR");
    }
    std::vector<std::mt19937> mt_generators;
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      mt_generators.emplace_back(config_.seed + thread_id);
    }
    const std::vector<double> fold_probs(config_.num_ctr_folds, 1.0 / config_.num_ctr_folds);
    std::discrete_distribution<int> fold_distribution(fold_probs.begin(), fold_probs.end());
    std::vector<std::vector<std::unique_ptr<CSC_RowIterator>>> thread_csc_iters(num_threads_);
    // copy csc row iterators for each thread
    #pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
      for (size_t i = 0; i < csc_iters.size(); ++i) {
        thread_csc_iters[thread_id].emplace_back(new CSC_RowIterator(*csc_iters[i].get()));
      }
    }
    Threading::For<int32_t>(0, nrow, 1024,
      [this, &thread_csc_iters, &get_label_fun, ncol, &fold_distribution, &mt_generators]
      (int thread_id, int32_t start, int32_t end) {
      std::vector<double> oneline_features(ncol, 0.0f);
      for (int32_t row_idx = start; row_idx < end; ++row_idx) {
        for (int32_t col_idx = 0; col_idx < ncol; ++col_idx) {
          oneline_features[col_idx] = thread_csc_iters[thread_id][col_idx]->Get(row_idx);
        }
        const int fold_id = fold_distribution(mt_generators[thread_id]);
        training_data_fold_id_[row_idx] = fold_id;
        const double label = get_label_fun(row_idx);
        ProcessOneLine(oneline_features, label, row_idx, thread_id, fold_id);
      }
    });
    FinishProcess(1);
  }

  void ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line,
    double label, int line_idx, std::vector<bool>* is_feature_processed_ptr,
    std::unordered_map<int, std::vector<std::unordered_map<int, int>>>* count_info_ptr,
    std::unordered_map<int, std::vector<std::unordered_map<int, label_t>>>* label_info_ptr,
    std::vector<label_t>* label_sum_ptr, std::vector<int>* num_data_ptr, const int fold_id);

  // sync up ctr values by gathering statistics from all machines in distributed scenario
  void SyncCTRStat(std::vector<std::unordered_map<int, label_t>>* fold_label_sum_ptr,
    std::vector<std::unordered_map<int, int>>* fold_total_count_ptr, const int num_machines) const;

  // sync up statistics to calculate the ctr prior by gathering statistics from all machines in distributed scenario
  void SyncCTRPrior(const double label_sum, const int local_num_data, double* all_label_sum_ptr,
    int* all_num_data_ptr, int num_machines) const;

  // dump a dictionary to string
  template <typename T>
  static std::string DumpDictToString(const std::unordered_map<int, T>& dict, const char delimiter) {
    std::stringstream str_buf;
    if (dict.empty()) {
      return str_buf.str();
    }
    auto iter = dict.begin();
    str_buf << iter->first << ":" << iter->second;
    ++iter;
    for (; iter != dict.end(); ++iter) {
      str_buf << delimiter << iter->first << ":" << iter->second;
    }
    return str_buf.str();
  }

  void FinishProcess(const int num_machines);

  int ParseMetaInfo(const char* filename, Config* config) {
    std::unordered_set<int> ignore_features;
    std::unordered_map<std::string, int> name2idx;
    std::string name_prefix("name:");
    int label_idx = 0;
    if (filename != nullptr) {
      TextReader<data_size_t> text_reader(filename, config->header);
      // get column names
      std::vector<std::string> feature_names;
      if (config->header) {
        std::string first_line = text_reader.first_line();
        feature_names = Common::Split(first_line.c_str(), "\t,");
      }
      // load label idx first
      if (config->label_column.size() > 0) {
        if (Common::StartsWith(config->label_column, name_prefix)) {
          std::string name = config->label_column.substr(name_prefix.size());
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
          if (!Common::AtoiAndCheck(config->label_column.c_str(), &label_idx)) {
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
      if (config->ignore_column.size() > 0) {
        if (Common::StartsWith(config->ignore_column, name_prefix)) {
          std::string names = config->ignore_column.substr(name_prefix.size());
          for (auto name : Common::Split(names.c_str(), ',')) {
            if (name2idx.count(name) > 0) {
              int tmp = name2idx[name];
              ignore_features.emplace(tmp);
            } else {
              Log::Fatal("Could not find ignore column %s in data file", name.c_str());
            }
          }
        } else {
          for (auto token : Common::Split(config->ignore_column.c_str(), ',')) {
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
    }
    std::unordered_set<int> categorical_features;
    if (config->categorical_feature.size() > 0) {
      if (Common::StartsWith(config->categorical_feature, name_prefix)) {
        std::string names = config->categorical_feature.substr(name_prefix.size());
        for (auto name : Common::Split(names.c_str(), ',')) {
          if (name2idx.count(name) > 0) {
            int tmp = name2idx[name];
            categorical_features.emplace(tmp);
          } else {
            Log::Fatal("Could not find categorical_feature %s in data file", name.c_str());
          }
        }
      } else {
        for (auto token : Common::Split(config->categorical_feature.c_str(), ',')) {
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
    if (config->categorical_feature.size() > 0) {
      if (!keep_raw_cat_method_) {
        config->categorical_feature.clear();
        config->categorical_feature.shrink_to_fit();
      }
    }

    categorical_features_.clear();
    for (const int fid : categorical_features) {
      categorical_features_.push_back(fid);
    }
    std::sort(categorical_features_.begin(), categorical_features_.end());
    is_categorical_feature_.clear();

    return label_idx;
  }

  // parameter configuration
  Config config_;

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
  // weight of the prior in ctr calculation
  double prior_weight_;
  // record whether a feature is categorical in the original data
  std::vector<bool> is_categorical_feature_;

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
  std::vector<std::vector<data_size_t>> thread_fold_num_data_;
  // number of data per fold
  std::vector<data_size_t> fold_num_data_;
  // categorical value converters
  std::vector<std::unique_ptr<CatConverter>> cat_converters_;
  // max bin by feature
  std::vector<int> max_bin_by_feature_;
  // whether the old categorical handling method is used
  bool keep_raw_cat_method_;
};

class CTRParser : public Parser {
 public:
    explicit CTRParser(const Parser* inner_parser,
      const CTRProvider* ctr_provider, const bool is_valid):
      inner_parser_(inner_parser), ctr_provider_(ctr_provider), is_valid_(is_valid) {}

    inline void ParseOneLine(const char* str,
      std::vector<std::pair<int, double>>* out_features,
      double* out_label, const int line_idx = -1) const override {
      inner_parser_->ParseOneLine(str, out_features, out_label);
      if (is_valid_) {
        ctr_provider_->ConvertCatToCTR(out_features);
      } else {
        ctr_provider_->ConvertCatToCTR(out_features, line_idx);
      }
    }

    inline int NumFeatures() const override {
      return ctr_provider_->GetNumTotalFeatures();
    }

 private:
    std::unique_ptr<const Parser> inner_parser_;
    const CTRProvider* ctr_provider_;
    const bool is_valid_;
};

class CTR_CSC_RowIterator: public CSC_RowIterator {
 public:
  CTR_CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx,
                  const CTRProvider::CatConverter* cat_converter,
                  const CTRProvider* ctr_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, col_idx),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    ctr_provider_ = ctr_provider;
  }

  CTR_CSC_RowIterator(CSC_RowIterator* csc_iter, const int col_idx,
                  const CTRProvider::CatConverter* cat_converter,
                  const CTRProvider* ctr_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(*csc_iter),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    ctr_provider_ = ctr_provider;
  }

  double Get(int row_idx) override {
    const double value = CSC_RowIterator::Get(row_idx);
    if (is_valid_) {
      return ctr_provider_->ConvertCatToCTR(value, cat_converter_, col_idx_);
    } else {
      return ctr_provider_->ConvertCatToCTR(value, cat_converter_, col_idx_, row_idx);
    }
  }

  std::pair<int, double> NextNonZero() override {
    if (cur_row_idx_ + 1 < static_cast<int>(num_row_)) {
      auto pair = cached_pair_;
      if (cur_row_idx_ == cached_pair_.first) {
        cached_pair_ = CSC_RowIterator::NextNonZero();
      }
      if ((cur_row_idx_ + 1 < cached_pair_.first) || is_end_) {
        pair = std::make_pair(cur_row_idx_ + 1, 0.0f);
      } else {
        pair = cached_pair_;
      }
      ++cur_row_idx_;
      double value = 0.0f;
      if (is_valid_) {
        value = ctr_provider_->ConvertCatToCTR(pair.second, cat_converter_, col_idx_);
      } else {
        value = ctr_provider_->ConvertCatToCTR(pair.second, cat_converter_, col_idx_, pair.first);
      }
      pair.second = value;
      return pair;
    } else {
      return std::make_pair(-1, 0.0f);
    }
  }

  void Reset() override {
    CSC_RowIterator::Reset();
    cur_row_idx_ = -1;
    cached_pair_ = std::make_pair(-1, 0.0f);
  }

 private:
  const CTRProvider::CatConverter* cat_converter_;
  const CTRProvider* ctr_provider_;
  const int col_idx_;
  const bool is_valid_;
  const int64_t num_row_;

  int cur_row_idx_ = -1;
  std::pair<int, double> cached_pair_ = std::make_pair(-1, 0.0f);
};

}  // namespace LightGBM
#endif  // LightGBM_CTR_PROVIDER_H_
