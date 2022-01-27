/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include "category_feature_encoder.hpp"
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/log.h>

#include <unordered_map>
#include <string>
#include <algorithm>

// property name keys
static const char train_category_feature_encoders_key[] = "train_category_feature_encoders";
static const char category_feature_encoders_key[] = "category_feature_encoders";
static const char encorders_key[] = "encoders";
static const char feature_id_key[] = "fid";
static const char fold_id_key[] = "fold_id";
static const char encoder_type_key[] = "encoder_type";
static const char prior_key[] = "prior";
static const char prior_weight_key[] = "prior_weight";
static const char count_encoder_type[] = "count";
static const char taregt_label_encoder_type[] = "taregt_label";

namespace LightGBM {
  std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int fold_id, int feature_id, double feature_value) {
    std::vector<std::unique_ptr<CategoryFeatureEncoder>>& encoders = train_category_feature_encoders_[fold_id][feature_id];
    std::vector<EncodeResult> result(encoders.size());

    for (int i = 0; i < encoders.size(); ++i) {
      result[i].value = encoders[i]->Encode(feature_value);
      result[i].feature_name = encoders[i]->GetFeatureName();
    }

    return result;
  }

  std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int feature_id, double feature_value) {
    std::vector<std::unique_ptr<CategoryFeatureEncoder>>& encoders = category_feature_encoders_[feature_id];
    std::vector<EncodeResult> result(encoders.size());

    for (int i = 0; i < encoders.size(); ++i) {
      result[i].value = encoders[i]->Encode(feature_value);
      result[i].feature_name = encoders[i]->GetFeatureName();
    }

    return result;
  }

  std::string CategoryFeatureEncoderManager::DumpToModelStringInJsonFormat() {
    json11::Json::array category_feature_encoders_json;

    for (auto it = category_feature_encoders_.begin(); it != category_feature_encoders_.end(); ++it) {
      json11::Json::array category_feature_encoders_per_feature;

      for (size_t i = 0; i < it->second.size(); i++) {
        category_feature_encoders_per_feature.emplace_back(it->second[i]->DumpToJsonObject());
      }

      category_feature_encoders_json.emplace_back(
        json11::Json::object {
          { feature_id_key, json11::Json(it->first) },
          { encorders_key, json11::Json(category_feature_encoders_per_feature) },
        });
    }

    json11::Json::array train_category_feature_encoders_json;

    for (int fold_id = 0; fold_id < train_category_feature_encoders_.size(); ++fold_id) {
      json11::Json::array category_feature_encoders_per_fold;

      for (auto it = train_category_feature_encoders_[fold_id].begin(); it != train_category_feature_encoders_[fold_id].end(); ++it) {
        json11::Json::array category_feature_encoders_per_feature;

        for (size_t i = 0; i < it->second.size(); i++) {
          category_feature_encoders_per_feature.emplace_back(it->second[i]->DumpToJsonObject());
        }

        category_feature_encoders_per_fold.emplace_back(
          json11::Json::object {
            { feature_id_key, json11::Json(it->first) },
            { encorders_key, json11::Json(category_feature_encoders_per_feature) },
        });
      }

      train_category_feature_encoders_json.emplace_back(json11::Json(category_feature_encoders_per_fold));
    }

    json11::Json::object result {
      { category_feature_encoders_key, json11::Json(category_feature_encoders_json) },
      { train_category_feature_encoders_key, json11::Json(train_category_feature_encoders_json) },
    };

    return json11::Json(result).dump();
  }

  std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> ParseCategoryFeatureEncoders(Json input_json) {
    std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> result;

    for each (Json encoders_per_feature_json in input_json.array_items()) {
      int featureId = encoders_per_feature_json[feature_id_key].int_value();

      for each (Json encoder_json in encoders_per_feature_json[encorders_key].array_items()) {
        result[featureId].push_back(std::move(CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(encoder_json)));
      }
    }

    return result;
  }

  std::unique_ptr<CategoryFeatureEncoderManager> CategoryFeatureEncoderManager::RecoverFromModelStringInJsonFormat(std::string input) {
    std::string err;
    Json input_json = json11::Json::parse(input, &err);

    if (!err.empty()) {
      Log::Fatal("Invalid CategoryFeatureEncoderManager model: %s. Please check if follow json format.", err.c_str());
    }

    std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>&& category_feature_encoders_tmp = ParseCategoryFeatureEncoders(input_json[category_feature_encoders_key]);
    std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> category_feature_encoders(std::move(category_feature_encoders_tmp));

    std::vector<Json> train_category_feature_encoders_json = input_json[train_category_feature_encoders_key].array_items();
    std::vector<std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>> train_category_feature_encoders(train_category_feature_encoders_json.size());
    for (int fold_id = 0; fold_id < train_category_feature_encoders_json.size(); fold_id++) {
      Json entry = train_category_feature_encoders_json[fold_id];
      std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>&& train_category_feature_encoders_tmp = ParseCategoryFeatureEncoders(entry);
      std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> category_feature_encoders_fold(std::move(train_category_feature_encoders_tmp));

      for (auto it = category_feature_encoders_fold.begin(); it != category_feature_encoders_fold.end(); ++it) {
        train_category_feature_encoders[fold_id][it->first] = std::move(it->second);
      }
    }

    return std::unique_ptr<CategoryFeatureEncoderManager>(new CategoryFeatureEncoderManager(train_category_feature_encoders, category_feature_encoders));
  }

  std::unique_ptr<CategoryFeatureEncoder> CreateEncoder(std::string featureName, json11::Json encoder_setting, CategoryFeatureTargetInformation targetInformation) {
    if (std::strcmp(count_encoder_type, encoder_setting[encoder_type_key].string_value().c_str()) == 0) {
      return std::unique_ptr<CategoryFeatureEncoder>(new CategoryFeatureCountEncoder(featureName, targetInformation.category_count));
    }

    if (std::strcmp(taregt_label_encoder_type, encoder_setting[encoder_type_key].string_value().c_str()) == 0) {
      double prior = encoder_setting[prior_key].is_null() ? (targetInformation.label_sum / targetInformation.total_count) : encoder_setting[prior_key].number_value();
      double prior_weight = encoder_setting[prior_weight_key].is_null() ? 0 : encoder_setting[prior_weight_key].number_value();

      return std::unique_ptr<CategoryFeatureEncoder>(new CategoryFeatureTargetEncoder(featureName, prior, prior_weight, targetInformation.category_count, targetInformation.category_label_sum));
    }

    return std::unique_ptr<CategoryFeatureEncoder>();
  }

  std::unique_ptr<CategoryFeatureEncoderManager> CategoryFeatureEncoderManager::Create(json11::Json settings, const CategoryFeatureTargetInformationCollector& informationCollector) {
    std::vector<int>&& categorical_features = informationCollector.GetCategoricalFeatures();
    std::vector<std::unordered_map<int, CategoryFeatureTargetInformation>>&& category_target_information = informationCollector.GetCategoryTargetInformation();
    std::unordered_map<int, CategoryFeatureTargetInformation>&& global_category_target_information = informationCollector.GetGlobalCategoryTargetInformation();
    int fold_count = category_target_information.size();
    std::vector<std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>> train_category_feature_encoders(fold_count);
    std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> category_feature_encoders;

    for (int e_index = 0; e_index < settings.array_items().size(); ++e_index) {
      Json encoder_setting = settings.array_items()[e_index];

      for each (int f_id in categorical_features) {
        const std::string feature_name = std::to_string(f_id) + "_" + std::to_string(e_index);

        for (int fold_id = 0; fold_id < fold_count; ++fold_id) {
          CategoryFeatureTargetInformation targetInformation = category_target_information[fold_id][f_id];

          train_category_feature_encoders[fold_id][f_id].push_back(std::move(CreateEncoder(feature_name, encoder_setting, targetInformation)));
        }

        category_feature_encoders[f_id].push_back(std::move(CreateEncoder(feature_name, encoder_setting, global_category_target_information[f_id])));
      }
    }

    return std::unique_ptr<CategoryFeatureEncoderManager>(new CategoryFeatureEncoderManager(train_category_feature_encoders, category_feature_encoders));
  }
}  // namespace LightGBM
