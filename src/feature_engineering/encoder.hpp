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
    CategoryFeatureEncoder(const std::string feature_name) : feature_name_(feature_name){}

    std::string GetFeatureName() {
      return feature_name_;
    }

    virtual double Encode(double feature_value) const = 0;

    virtual json11::Json::object DumpToJsonObject() {
      json11::Json::object result {
        {encoder_type_key, json11::Json(default_encoder_type)},
        {feature_name_key, json11::Json(feature_name_)},
      };

      return result;
    }

  protected:
    std::string feature_name_;

    // property name keys
    const std::string feature_name_key = "feature_name";
    const std::string encoder_type_key = "encoder_type";

    // constant value
    const int default_encoder_type = 0;
  };

  class CategoryFeatureCountEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureCountEncoder(std::string feature_name, std::unordered_map<int, double> count_information) : CategoryFeatureEncoder(feature_name), count_information_(count_information){}

    double Encode(double feature_value);

    json11::Json::object DumpToJsonObject();

    // public constant value
    const int count_encoder_type = 1;

  private:
    std::unordered_map<int, double> count_information_;

    // property name keys
    const std::string count_information_key = "count_information";
    const std::string count_information_category_key = "cat";
    const std::string count_information_value_key = "value";

    // constant value
    const double default_value = 0.0;
  };

  class CategoryFeatureTargetEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureTargetEncoder(std::string feature_name, double prior, double prior_weight, double total_count, std::unordered_map<int, double> count_information)
      : CategoryFeatureEncoder(feature_name), prior_(prior), prior_weight_(prior_weight), total_count_(total_count), count_information_(count_information) {}

    double Encode(double feature_value);

    json11::Json::object DumpToJsonObject();

    // public constant value
    const int target_encoder_type = 1;

  private:
    std::unordered_map<int, double> count_information_;
    double prior_;
    double prior_weight_;
    double total_count_;

    // property name keys
    const std::string count_information_key = "count_information";
    const std::string count_information_category_key = "cat";
    const std::string count_information_value_key = "value";
    const std::string count_prior_key = "prior";
    const std::string count_prior_weight_key = "prior_weight";
    const std::string count_total_count_key = "total_count";

    // constant value
    const double default_value = 0.0;
  };

  struct CategoryFeatureTargetInformation {
    // <category_id, category_total_count>
    std::unordered_map<int, int> category_count; 

    // <category_id, label_sum>
    std::unordered_map<int, double> category_label_sum; 
  };

  class CategoryFeatureTargetInformationCollector {
  public:
    CategoryFeatureTargetInformationCollector(std::vector<int> categorical_features, int fold_count) {
      categorical_features_ = categorical_features;

      count_.resize(fold_count);
      label_sum_.resize(fold_count);
      category_target_information_.resize(fold_count);
    }

    void HandleRecord(int fold_id, const std::vector<double>& record, double label);

    void AppendFrom(CategoryFeatureTargetInformationCollector collector);

    std::vector<data_size_t> GetCounts() {
      return count_;
    }

    std::vector<double> GetLabelSum() {
      return label_sum_;
    }

    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>> GetCategoryTargetInformation() {
      return category_target_information_;
    }

  private:
    std::vector<int> categorical_features_;

    // <fold_id, row_count>
    std::vector<data_size_t> count_; 

    // <fold_id, label_sum>
    std::vector<double> label_sum_; 

    // <fold_id, <feature_id, CategoryFeatureTargetInformation>>
    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>> category_target_information_;
  };

  class CategoryFeatureEncoderDeserializer {
  public:
    static CategoryFeatureEncoder* ParseFromJsonString(std::string content) {
      std::string error_message;
      json11::Json inputJson = json11::Json::parse(content, &error_message);

      return nullptr;
    }
  };
}

#endif
