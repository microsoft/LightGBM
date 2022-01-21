/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_ENCODER_HPP_
#define LIGHTGBM_ENCODER_HPP_

#include <LightGBM/utils/json11.h>
#include <LightGBM/meta.h>

#include <string>
#include <unordered_map>

namespace LightGBM {

using json11::Json;
  
  class CategoryFeatureEncoder {
  public:
    CategoryFeatureEncoder(int type) : type_(type){}

    CategoryFeatureEncoder(const std::string feature_name, int type) : feature_name_(feature_name), type_(type) {}

    inline std::string GetFeatureName() {
      return feature_name_;
    }

    inline int GetTypeId() {
      return type_;
    }

    virtual double Encode(double feature_value) = 0;

    virtual json11::Json::object DumpToJsonObject() = 0;

    static std::unique_ptr<CategoryFeatureEncoder> RecoverFromModelStringInJsonFormat(json11::Json input);

  protected:
    std::string feature_name_;
    int type_;
  };

  class CategoryFeatureCountEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureCountEncoder(std::unordered_map<int, int> count_information) : CategoryFeatureEncoder(count_encoder_type), count_information_(count_information) {}

    CategoryFeatureCountEncoder(std::string feature_name, std::unordered_map<int, int> count_information) : CategoryFeatureEncoder(feature_name, count_encoder_type), count_information_(count_information){}

    double Encode(double feature_value) override;

    json11::Json::object DumpToJsonObject() override;

    static std::unique_ptr<CategoryFeatureEncoder> RecoverFromModelStringInJsonFormat(json11::Json input);

    // public constant value
    static const int count_encoder_type = 1;

  private:
    std::unordered_map<int, int> count_information_;

    // constant value
    const double default_value = 0.0;
  };

  class CategoryFeatureTargetEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureTargetEncoder(double prior, double prior_weight, std::unordered_map<int, int> count_information, std::unordered_map<int, double> label_information)
    : CategoryFeatureEncoder(target_encoder_type), prior_(prior), prior_weight_(prior_weight), count_information_(count_information), label_information_(label_information){}

    CategoryFeatureTargetEncoder(std::string feature_name, double prior, double prior_weight, std::unordered_map<int, int> count_information, std::unordered_map<int, double> label_information)
      : CategoryFeatureEncoder(feature_name, target_encoder_type), prior_(prior), prior_weight_(prior_weight), count_information_(count_information), label_information_(label_information) {}

    double Encode(double feature_value) override;

    json11::Json::object DumpToJsonObject() override;

    static std::unique_ptr<CategoryFeatureEncoder> RecoverFromModelStringInJsonFormat(json11::Json input);

    // public constant value
    static const int target_encoder_type = 2;

  private:
    std::unordered_map<int, int> count_information_;
    std::unordered_map<int, double> label_information_;
    double prior_;
    double prior_weight_;

    // constant value
    const double default_value = 0.0;
  };

  struct CategoryFeatureTargetInformation {
    // <category_id, category_total_count>
    std::unordered_map<int, int> category_count; 

    // <category_id, label_sum>
    std::unordered_map<int, double> category_label_sum; 

    int total_count;

    double label_sum;
  };

  class CategoryFeatureTargetInformationCollector {
  public:
    CategoryFeatureTargetInformationCollector(std::vector<int> categorical_features, int fold_count) : count_(fold_count), label_sum_(fold_count), category_target_information_(fold_count){
      categorical_features_ = categorical_features;
    }

    void HandleRecord(int fold_id, const std::vector<double>& record, double label);

    void AppendFrom(CategoryFeatureTargetInformationCollector& collector);

    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>> GetCategoryTargetInformation() {
      return category_target_information_;
    }

  std::vector<int> GetCategoricalFeatures() {
    return categorical_features_;
  }

  std::vector<data_size_t> GetCounts() {
    return count_;
  }

  std::vector<double> GetLabelSum() {
    return label_sum_;
  }

  std::unordered_map<int, CategoryFeatureTargetInformation> GetGlobalCategoryTargetInformation() {
    return global_category_target_information_;
  }

  private:
    std::vector<int> categorical_features_;

    // <fold_id, row_count>
    std::vector<data_size_t> count_;

    // <fold_id, label_sum>
    std::vector<double> label_sum_;

    // <fold_id, <feature_id, CategoryFeatureTargetInformation>>
    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>> category_target_information_;

    // <feature_id, CategoryFeatureTargetInformation>
    std::unordered_map<int, CategoryFeatureTargetInformation> global_category_target_information_;
  };

  struct EncodeResult {
    double value;
    std::string feature_name;
  };

  class CategoryFeatureEncoderManager {
  public:
    CategoryFeatureEncoderManager(std::vector<std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>>& train_category_feature_encoders, std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>& category_feature_encoders)
    : train_category_feature_encoders_(std::move(train_category_feature_encoders)), category_feature_encoders_(std::move(category_feature_encoders)) {
    };

    std::vector<EncodeResult> Encode(int fold_id, int feature_id, double feature_value);

    std::vector<EncodeResult> Encode(int feature_id, double feature_value);

    std::string DumpToModelStringInJsonFormat();

    static std::unique_ptr<CategoryFeatureEncoderManager> RecoverFromModelStringInJsonFormat(std::string input);

    static std::unique_ptr<CategoryFeatureEncoderManager> Create(json11::Json settings, CategoryFeatureTargetInformationCollector& informationCollector);

  private:
    // <fold_id, <feature_id, Encoders>>
    std::vector<std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>> train_category_feature_encoders_;

    // <feature_id, Encoders>
    std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> category_feature_encoders_;
  };
}

#endif
