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
const std::string feature_name = "TestFeature";
const std::string prior_key = "prior";
const std::string prior_weight_key = "prior_weight";
const std::string count_encoder_type = "count";
const std::string taregt_label_encoder_type = "taregt_label";

// test records
std::vector<int> categorical_features {0, 2, 4};
const int fold_count = 2;
std::vector<double> fold0_record0 = {1, 0.2, 2, 0.3, 1};
double fold0_record0_label = 0.12;
std::vector<double> fold0_record1 = {0, 0.3, 1, 0.4, 1};
double fold0_record1_label = 0.32;
std::vector<double> fold1_record0 = {1, -1.2, 0, 1.4, 1};
double fold1_record0_label = -0.8;
std::vector<double> fold1_record1 = {3, 0.7, 9, 0.14, 1};
double fold1_record1_label = 1.2;
std::vector<double> fold1_record2 = {10, 0.32, 0, 0.43, 1};
double fold1_record2_label = -3.1;

class CategoryFeatureCountEncoderTests : public testing::Test { };

TEST_F(CategoryFeatureCountEncoderTests, GivenCategoryValue_WhenEncoding_ThenEncodedValueShouldBeReturned) {
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

class CategoryFeatureTargetEncoderTests : public testing::Test { };

TEST_F(CategoryFeatureTargetEncoderTests, GivenCategoryValue_WhenEncoding_ThenEncodedValueShouldBeReturned) {
  std::unordered_map<int, int> count_information;
  int count1 = 2;
  int count2 = 3;
  count_information[1] = 2;
  count_information[2] = 3;

  std::unordered_map<int, double> label_information;
  double label1 = 3.0;
  double label2 = 4.0;
  label_information[1] = 3.0;
  label_information[2] = 4.0;

  double prior = 2.0;
  double prior_weight = 0.3;

  LightGBM::CategoryFeatureTargetEncoder encoder(feature_name, prior, prior_weight, count_information, label_information);

  EXPECT_EQ(encoder.Encode(1), (label1 + prior * prior_weight) / (count1 + prior_weight));
  EXPECT_EQ(encoder.Encode(2), (label2 + prior * prior_weight) / (count2 + prior_weight));
  EXPECT_EQ(encoder.Encode(-1), 0);
  EXPECT_EQ(encoder.GetFeatureName(), feature_name);
}

TEST_F(CategoryFeatureTargetEncoderTests, GivenCategoryFeatureTargetEncoder_WhenRecoverFromDumpedJson_ThenEncoderWithAllInformationShouldBeReturned) {
  std::unordered_map<int, int> count_information;
  int count1 = 2;
  count_information[1] = 2;

  std::unordered_map<int, double> label_information;
  double label1 = 3.0;
  label_information[1] = 3.0;

  double prior = 2.0;
  double prior_weight = 0.3;

  LightGBM::CategoryFeatureTargetEncoder encoder1(feature_name, prior, prior_weight, count_information, label_information);
  json11::Json::object encoder1_in_Json = encoder1.DumpToJsonObject();
  std::unique_ptr<LightGBM::CategoryFeatureEncoder> encoder1_from_json = LightGBM::CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(json11::Json::Json(encoder1_in_Json));

  EXPECT_EQ(encoder1_from_json->Encode(1), (label1 + prior * prior_weight) / (count1 + prior_weight));
  EXPECT_EQ(encoder1_from_json->GetTypeId(), LightGBM::CategoryFeatureTargetEncoder::target_encoder_type);
  EXPECT_EQ(encoder1_from_json->GetFeatureName(), feature_name);

  // Empty encoder
  LightGBM::CategoryFeatureTargetEncoder encoder2(feature_name, prior, prior_weight, std::unordered_map<int, int>(), std::unordered_map<int, double>());
  json11::Json::object encoder2_in_Json = encoder2.DumpToJsonObject();
  std::unique_ptr<LightGBM::CategoryFeatureEncoder> encoder2_from_json = LightGBM::CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(json11::Json::Json(encoder2_in_Json));
  EXPECT_EQ(encoder2_from_json->Encode(1), 0);
}

class CategoryFeatureTargetInformationCollectorTests : public testing::Test { };

TEST_F(CategoryFeatureTargetInformationCollectorTests, GivenCollector_WhenHandleRecords_ThenRecordsInformationShouldBeCollected) {
  LightGBM::CategoryFeatureTargetInformationCollector collector(categorical_features, fold_count);
  EXPECT_EQ(collector.GetCategoryTargetInformation().size(), 2);
  collector.HandleRecord(0, fold0_record0, fold0_record0_label);
  collector.HandleRecord(0, fold0_record1, fold0_record1_label);
  collector.HandleRecord(1, fold1_record0, fold1_record0_label);
  collector.HandleRecord(1, fold1_record1, fold1_record1_label);
  collector.HandleRecord(1, fold1_record2, fold1_record2_label);

  std::vector<std::unordered_map<int, LightGBM::CategoryFeatureTargetInformation>> categoty_information = collector.GetCategoryTargetInformation();
  EXPECT_EQ(categoty_information[0][0].total_count, 2);
  EXPECT_EQ(categoty_information[0][0].label_sum, fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(categoty_information[0][0].category_count[1], 1);
  EXPECT_EQ(categoty_information[0][0].category_count[0], 1);
  EXPECT_EQ(categoty_information[0][0].category_label_sum[1], fold0_record0_label);
  EXPECT_EQ(categoty_information[0][0].category_label_sum[0], fold0_record1_label);
  EXPECT_EQ(categoty_information[0][0].category_label_sum[-1], 0);
  EXPECT_EQ(categoty_information[0][4].total_count, 2);
  EXPECT_EQ(categoty_information[0][4].label_sum, fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(categoty_information[0][4].category_count[1], 2);
  EXPECT_EQ(categoty_information[0][4].category_label_sum[1], fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(categoty_information[1][2].total_count, 3);
  EXPECT_EQ(categoty_information[1][2].label_sum, fold1_record0_label + fold1_record1_label + fold1_record2_label);
  EXPECT_EQ(categoty_information[1][2].category_count[0], 2);

  std::unordered_map<int, LightGBM::CategoryFeatureTargetInformation> global_categoty_information = collector.GetGlobalCategoryTargetInformation();
  EXPECT_EQ(global_categoty_information[0].total_count, 5);
  EXPECT_EQ(global_categoty_information[4].total_count, 5);
  EXPECT_EQ(global_categoty_information[2].label_sum, fold0_record0_label + fold0_record1_label + fold1_record0_label + fold1_record1_label + fold1_record2_label);
  EXPECT_EQ(global_categoty_information.size(), 3);
  EXPECT_EQ(global_categoty_information[0].category_count[1], 2);
  EXPECT_EQ(global_categoty_information[0].category_label_sum[1], fold0_record0_label + fold1_record0_label);

  std::vector<LightGBM::data_size_t> count = collector.GetCounts();
  EXPECT_EQ(count[0], 2);
  EXPECT_EQ(count[1], 3);

  std::vector<double> label_sum = collector.GetLabelSum();
  EXPECT_EQ(label_sum[0], fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(label_sum[1], fold1_record0_label + fold1_record1_label + fold1_record2_label);
}

TEST_F(CategoryFeatureTargetInformationCollectorTests, GivenCollector_WhenAppendNewCollector_ThenRecordsInformationShouldBeMerged) {
  LightGBM::CategoryFeatureTargetInformationCollector collector(categorical_features, 1);
  EXPECT_EQ(collector.GetCategoryTargetInformation().size(), 1);
  LightGBM::CategoryFeatureTargetInformationCollector fold1_collector(categorical_features, 1);
  collector.HandleRecord(0, fold0_record0, fold0_record0_label);
  collector.HandleRecord(0, fold0_record1, fold0_record1_label);
  fold1_collector.HandleRecord(0, fold1_record0, fold1_record0_label);
  fold1_collector.HandleRecord(0, fold1_record1, fold1_record1_label);
  fold1_collector.HandleRecord(0, fold1_record2, fold1_record2_label);
  collector.AppendFrom(fold1_collector);

  std::vector<std::unordered_map<int, LightGBM::CategoryFeatureTargetInformation>> categoty_information = collector.GetCategoryTargetInformation();
  EXPECT_EQ(categoty_information[0][4].category_label_sum[1], fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(categoty_information[1][2].total_count, 3);
  EXPECT_EQ(categoty_information[1][2].label_sum, fold1_record0_label + fold1_record1_label + fold1_record2_label);
  EXPECT_EQ(categoty_information[1][2].category_count[0], 2);
  EXPECT_EQ(categoty_information.size(), 2);

  std::unordered_map<int, LightGBM::CategoryFeatureTargetInformation> global_categoty_information = collector.GetGlobalCategoryTargetInformation();
  EXPECT_EQ(global_categoty_information[0].total_count, 5);
  EXPECT_EQ(global_categoty_information[2].label_sum, fold0_record0_label + fold0_record1_label + fold1_record0_label + fold1_record1_label + fold1_record2_label);
  EXPECT_EQ(global_categoty_information.size(), 3);

  std::vector<LightGBM::data_size_t> count = collector.GetCounts();
  EXPECT_EQ(count[0], 2);
  EXPECT_EQ(count[1], 3);

  std::vector<double> label_sum = collector.GetLabelSum();
  EXPECT_EQ(label_sum[0], fold0_record0_label + fold0_record1_label);
  EXPECT_EQ(label_sum[1], fold1_record0_label + fold1_record1_label + fold1_record2_label);
}

class CategoryFeatureEncoderManagerTests : public testing::Test { };

TEST_F(CategoryFeatureEncoderManagerTests, GivenCollectorAndSettings_WhenCreateManager_ThenManagerShouldBeCreatedWithEncoders) {
  LightGBM::CategoryFeatureTargetInformationCollector collector(categorical_features, fold_count);
  EXPECT_EQ(collector.GetCategoryTargetInformation().size(), fold_count);
  collector.HandleRecord(0, fold0_record0, fold0_record0_label);
  collector.HandleRecord(0, fold0_record1, fold0_record1_label);
  collector.HandleRecord(1, fold1_record0, fold1_record0_label);
  collector.HandleRecord(1, fold1_record1, fold1_record1_label);
  collector.HandleRecord(1, fold1_record2, fold1_record2_label);

  json11::Json::array encoder_settings;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(count_encoder_type) },
    });

  double prior_weight = 0.3; 
  double prior = 0.5;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(taregt_label_encoder_type) },
      { prior_weight_key, json11::Json(prior_weight) },
      { prior_key, json11::Json(prior) },
    });
  std::unique_ptr<LightGBM::CategoryFeatureEncoderManager> manager = LightGBM::CategoryFeatureEncoderManager::Create(json11::Json(encoder_settings), collector);
  std::string manager_string_in_json_string = manager->DumpToModelStringInJsonFormat();

  EXPECT_EQ(manager->Encode(0, 0)[0].value, 1);
  EXPECT_EQ(manager->Encode(0, 0)[1].value, (fold0_record1_label + prior * prior_weight) / (1 + prior_weight));
  EXPECT_EQ(manager->Encode(0, 1)[0].value, 2);
  EXPECT_EQ(manager->Encode(0, 1)[1].value, (fold0_record0_label + fold1_record0_label + prior * prior_weight) / (2 + prior_weight));
  EXPECT_EQ(manager->Encode(2, 1)[0].value, 1);
  EXPECT_EQ(manager->Encode(2, 1)[1].value, (fold0_record1_label + prior * prior_weight) / (1 + prior_weight));

  EXPECT_EQ(manager->Encode(0, 4, 1)[0].value, 2);
  EXPECT_EQ(manager->Encode(0, 4, 1)[1].value, (fold0_record1_label + fold0_record0_label + prior * prior_weight) / (2 + prior_weight));
  EXPECT_EQ(manager->Encode(1, 2, 0)[0].value, 2);
  EXPECT_EQ(manager->Encode(1, 2, 9)[1].value, (fold1_record1_label + prior * prior_weight) / (1 + prior_weight));
}

TEST_F(CategoryFeatureEncoderManagerTests, GivenCollectorAndSettings_WhenRecoverFromModelInJson_ThenManagerShouldBeRecovered) {
  LightGBM::CategoryFeatureTargetInformationCollector collector(categorical_features, fold_count);
  EXPECT_EQ(collector.GetCategoryTargetInformation().size(), fold_count);
  collector.HandleRecord(0, fold0_record0, fold0_record0_label);
  collector.HandleRecord(0, fold0_record1, fold0_record1_label);
  collector.HandleRecord(1, fold1_record0, fold1_record0_label);
  collector.HandleRecord(1, fold1_record1, fold1_record1_label);
  collector.HandleRecord(1, fold1_record2, fold1_record2_label);

  json11::Json::array encoder_settings;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(count_encoder_type) },
    });

  double prior_weight = 0.3; 
  double prior = 0.5;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(taregt_label_encoder_type) },
      { prior_weight_key, json11::Json(prior_weight) },
      { prior_key, json11::Json(prior) },
    });
  std::unique_ptr<LightGBM::CategoryFeatureEncoderManager> manager = LightGBM::CategoryFeatureEncoderManager::Create(json11::Json(encoder_settings), collector);
  std::string manager_string_in_json_string = manager->DumpToModelStringInJsonFormat();
  std::unique_ptr<LightGBM::CategoryFeatureEncoderManager> manager_recovered = LightGBM::CategoryFeatureEncoderManager::RecoverFromModelStringInJsonFormat(manager_string_in_json_string);

  EXPECT_EQ(manager_recovered->Encode(0, 0)[0].value, 1);
  EXPECT_EQ(manager_recovered->Encode(2, 1)[1].value, (fold0_record1_label + prior * prior_weight) / (1 + prior_weight));

  EXPECT_EQ(manager_recovered->Encode(0, 4, 1)[0].value, 2);
  EXPECT_EQ(manager_recovered->Encode(1, 2, 9)[1].value, (fold1_record1_label + prior * prior_weight) / (1 + prior_weight));
}

TEST_F(CategoryFeatureEncoderManagerTests, GivenCategoryFeatureEncoderManager_WhenCreateWithPrior_ThenPriorShouldBeSameAsTargetLabelMean) {
  LightGBM::CategoryFeatureTargetInformationCollector collector(categorical_features, fold_count);
  EXPECT_EQ(collector.GetCategoryTargetInformation().size(), fold_count);
  collector.HandleRecord(0, fold0_record0, fold0_record0_label);
  collector.HandleRecord(0, fold0_record1, fold0_record1_label);
  collector.HandleRecord(1, fold1_record0, fold1_record0_label);
  collector.HandleRecord(1, fold1_record1, fold1_record1_label);
  collector.HandleRecord(1, fold1_record2, fold1_record2_label);

  json11::Json::array encoder_settings;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(count_encoder_type) },
    });

  double prior_weight = 0.3; 
  double fold0_expect_prior = (fold0_record0_label + fold0_record1_label)/ 2.0;
  double fold1_expect_prior = (fold1_record0_label + fold1_record1_label + fold1_record2_label)/ 3.0;
  double global_expect_prior = (fold0_record0_label + fold0_record1_label + fold1_record0_label + fold1_record1_label + fold1_record2_label)/ 5.0;
  encoder_settings.emplace_back(json11::Json::object {
      { encoder_type_key, json11::Json(taregt_label_encoder_type) },
      { prior_weight_key, json11::Json(prior_weight) },
    });
  std::unique_ptr<LightGBM::CategoryFeatureEncoderManager> manager = LightGBM::CategoryFeatureEncoderManager::Create(json11::Json(encoder_settings), collector);
  std::string manager_string_in_json_string = manager->DumpToModelStringInJsonFormat();
  std::unique_ptr<LightGBM::CategoryFeatureEncoderManager> manager_recovered = LightGBM::CategoryFeatureEncoderManager::RecoverFromModelStringInJsonFormat(manager_string_in_json_string);

  EXPECT_EQ(manager_recovered->Encode(0, 0)[0].value, 1);
  EXPECT_EQ(manager_recovered->Encode(2, 1)[1].value, (fold0_record1_label + global_expect_prior * prior_weight) / (1 + prior_weight));

  EXPECT_EQ(manager_recovered->Encode(0, 4, 1)[0].value, 2);
  EXPECT_EQ(manager_recovered->Encode(0, 4, 1)[1].value, (fold0_record1_label + fold0_record0_label + fold0_expect_prior * prior_weight) / (2 + prior_weight));
  EXPECT_EQ(manager_recovered->Encode(1, 2, 9)[1].value, (fold1_record1_label + fold1_expect_prior * prior_weight) / (1 + prior_weight));
}

