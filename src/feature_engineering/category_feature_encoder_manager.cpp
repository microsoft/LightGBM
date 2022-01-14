/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include <unordered_map>
#include <string>
#include <algorithm>

#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/log.h>
#include "category_feature_encoder.hpp"

// property name keys
static const std::string train_category_feature_encoders_key = "train_category_feature_encoders";
static const std::string category_feature_encoders_key = "category_feature_encoders";
static const std::string encorders_key = "encoders";
static const std::string feature_id_key = "fid";
static const std::string fold_id_key = "fold_id";

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
		
        for (auto it = category_feature_encoders_.begin(); it != category_feature_encoders_.end(); ++it)
		{
			json11::Json::array category_feature_encoders_per_feature;

			for (size_t i = 0; i < it->second.size(); i++)
			{
				category_feature_encoders_per_feature.emplace_back(it->second[i]->DumpToJsonObject());
			}

			category_feature_encoders_json.emplace_back(
				json11::Json::object{
					{ feature_id_key, json11::Json(it->first) },
					{ encorders_key, json11::Json(category_feature_encoders_per_feature) },
			});
		};

		json11::Json::array train_category_feature_encoders_json;

		for (int fold_id = 0; fold_id < train_category_feature_encoders_.size(); ++fold_id) {
			std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>& category_feature_encoders = train_category_feature_encoders_[fold_id];
			json11::Json::array category_feature_encoders_per_fold;

			for (auto it = category_feature_encoders_.begin(); it != category_feature_encoders_.end(); ++it)
			{
				json11::Json::array category_feature_encoders_per_feature;

				for (size_t i = 0; i < it->second.size(); i++)
				{
					category_feature_encoders_per_feature.emplace_back(it->second[i]->DumpToJsonObject());
				}

				category_feature_encoders_per_fold.emplace_back(
					json11::Json::object{
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

		for each (Json encoders_per_feature_json in input_json.array_items())
		{
			int featureId = encoders_per_feature_json[feature_id_key].int_value();

			for each (Json encoder_json in encoders_per_feature_json[feature_id_key].array_items())
			{
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

		std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>> category_feature_encoders; // = ParseCategoryFeatureEncoders(input_json[category_feature_encoders_key]);
		std::vector<std::unordered_map<int, std::vector<std::unique_ptr<CategoryFeatureEncoder>>>> train_category_feature_encoders;

		CategoryFeatureEncoderManager result(train_category_feature_encoders, category_feature_encoders);
		Json train_category_feature_encoders_json = input_json[train_category_feature_encoders_key];
		for each (Json entry in train_category_feature_encoders_json.array_items())
		{
			train_category_feature_encoders.push_back(ParseCategoryFeatureEncoders(entry));
		}

		return std::unique_ptr<CategoryFeatureEncoderManager>(&result);
	}
}