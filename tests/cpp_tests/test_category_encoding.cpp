/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <string>
#include <unordered_map>
#include <gtest/gtest.h>
#include "../src/feature_engineering/category_feature_encoder.hpp"
#include "../include/LightGBM/utils/json11.h"

 // property name keys
const std::string feature_name_key = "feature_name";
const std::string encoder_type_key = "encoder_type";

class CategoryFeatureCountEncoderTests : public testing::Test { };

TEST_F(CategoryFeatureCountEncoderTests, GivenCategoryValue_WhenEncoding_ThenEncodedValueShouldBeReturned) {
  const std::string feature_name = "TestFeature";
  std::unordered_map<int, int> count_information;
  count_information[1] = 2;
  count_information[2] = 3;

  LightGBM::CategoryFeatureCountEncoder encoder(feature_name, count_information);

  EXPECT_EQ(encoder.Encode(1), 2);
  EXPECT_EQ(encoder.Encode(2), 3);
  EXPECT_EQ(encoder.Encode(-1), 0);
  EXPECT_EQ(encoder.GetFeatureName(), feature_name);
}

TEST_F(CategoryFeatureCountEncoderTests, GivenCategoryFeatureCountEncoder_WhenRecoverFromDumpedJson_ThenEncoderWithAllInformationShouldBeReturned) {
	const std::string feature_name = "TestFeature";
	
	std::unordered_map<int, int> count_information;
	count_information[1] = 2;

	LightGBM::CategoryFeatureCountEncoder encoder1(feature_name, count_information);
	json11::Json::object encoder1_in_Json = encoder1.DumpToJsonObject();
	std::unique_ptr<LightGBM::CategoryFeatureEncoder> encoder1_from_json = LightGBM::CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(json11::Json::Json(encoder1_in_Json));

	EXPECT_EQ(encoder1_from_json->Encode(1), 2);
	EXPECT_EQ(encoder1_from_json->GetTypeId(), LightGBM::CategoryFeatureCountEncoder::count_encoder_type);
	EXPECT_EQ(encoder1_from_json->GetFeatureName(), feature_name);

	// Empty encoder
	LightGBM::CategoryFeatureCountEncoder encoder2(feature_name, std::unordered_map<int, int>());
	json11::Json::object encoder2_in_Json = encoder2.DumpToJsonObject();
	EXPECT_EQ(encoder2_in_Json[feature_name_key].string_value(), feature_name);

	std::unique_ptr<LightGBM::CategoryFeatureEncoder> encoder2_from_json = LightGBM::CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(json11::Json::Json(encoder2_in_Json));
	EXPECT_EQ(encoder2_from_json->Encode(1), 0);
}
