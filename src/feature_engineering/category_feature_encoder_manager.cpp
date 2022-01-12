/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include "encoder.hpp"

namespace LightGBM {
	std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int fold_id, int feature_id, double feature_value) {
		std::vector<CategoryFeatureEncoder> encoders = train_categoryFeatureEncoders_[fold_id][feature_id];
		std::vector<EncodeResult> result(encoders.size());

		for (int i = 0; i < encoders.size(); ++i) {
			result[i].value = encoders[i].Encode(feature_value);
			result[i].feature_name = encoders[i].GetFeatureName();
		}

		return result;
	}

	std::vector<EncodeResult> CategoryFeatureEncoderManager::Encode(int feature_id, double feature_value) {
		std::vector<CategoryFeatureEncoder> encoders = categoryFeatureEncoders_[feature_id];
		std::vector<EncodeResult> result(encoders.size());

		for (int i = 0; i < encoders.size(); ++i) {
			result[i].value = encoders[i].Encode(feature_value);
			result[i].feature_name = encoders[i].GetFeatureName();
		}

		return result;
	}

	CategoryFeatureEncoderManager CategoryFeatureEncoderManager::Create(CategoryFeatureTargetInformationCollector informationCollector) {

	}
}