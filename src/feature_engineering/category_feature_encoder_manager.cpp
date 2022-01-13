/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include <unordered_map>
#include <string>
#include <algorithm>

#include "LightGBM/utils/json11.h"
#include "encoder.hpp"

namespace LightGBM {
	std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int fold_id, int feature_id, double feature_value) {
		std::vector<CategoryFeatureEncoder> encoders = train_category_feature_encoders_[fold_id][feature_id];
		std::vector<EncodeResult> result(encoders.size());

		for (int i = 0; i < encoders.size(); ++i) {
			result[i].value = encoders[i].Encode(feature_value);
			result[i].feature_name = encoders[i].GetFeatureName();
		}

		return result;
	}

	std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int feature_id, double feature_value) {
		std::vector<CategoryFeatureEncoder> encoders = category_feature_encoders_[feature_id];
		std::vector<EncodeResult> result(encoders.size());

		for (int i = 0; i < encoders.size(); ++i) {
			result[i].value = encoders[i].Encode(feature_value);
			result[i].feature_name = encoders[i].GetFeatureName();
		}

		return result;
	}

	std::string CategoryFeatureEncoderManager::DumpToModelStringInJsonFormat() {
		json11::Json::array category_feature_encoders_json;
		
		for each (std::pair<int, std::vector<CategoryFeatureEncoder>> record in category_feature_encoders_)
		{
			json11::Json::array category_feature_encoders_per_feature;

			for each (CategoryFeatureEncoder encoder in record.second)
			{
				category_feature_encoders_per_feature.emplace_back(encoder.DumpToJsonObject());
			}

			category_feature_encoders_json.emplace_back(
				json11::Json::object{
					{ feature_id_key, json11::Json(record.first) },
					{ encorders_key, json11::Json(category_feature_encoders_per_feature) },
			});
		};

		json11::Json::array train_category_feature_encoders_json;
		for each (std::unordered_map<int, std::vector<CategoryFeatureEncoder>> category_feature_encoders in train_category_feature_encoders_)
		{
			json11::Json::array category_feature_encoders_per_fold;

			for each (std::pair<int, std::vector<CategoryFeatureEncoder>> record in category_feature_encoders)
			{
				json11::Json::array category_feature_encoders_per_feature;

				for each (CategoryFeatureEncoder encoder in record.second)
				{
					category_feature_encoders_per_feature.emplace_back(encoder.DumpToJsonObject());
				}

				category_feature_encoders_per_fold.emplace_back(
					json11::Json::object{
						{ feature_id_key, json11::Json(record.first) },
						{ encorders_key, json11::Json(category_feature_encoders_per_feature) },
				});
			}

			train_category_feature_encoders_json.emplace_back(json11::Json(category_feature_encoders_per_fold));
		};

		json11::Json::object result{
			{ category_feature_encoders_key, json11::Json(category_feature_encoders_json) },
			{ train_category_feature_encoders_key, json11::Json(train_category_feature_encoders_json) },
		};

		return json11::Json(result).dump();
	}
}