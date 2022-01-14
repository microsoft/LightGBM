/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include "category_feature_encoder.hpp"

// property name keys
const std::string count_information_key = "count_information";
const std::string count_information_category_key = "cat";
const std::string count_information_value_key = "value";

namespace LightGBM {
  double CategoryFeatureCountEncoder::Encode(double feature_value) {
    int category = static_cast<int>(feature_value);

    if (count_information_.find(category) != count_information_.end()) {
      return count_information_[category];
    }

    return default_value;
  }

  json11::Json::object CategoryFeatureCountEncoder::DumpToJsonObject()
  {
    json11::Json::object result = CategoryFeatureEncoder::DumpToJsonObject();

    json11::Json::array count_information_json;
    for (const auto& count_pair : count_information_) {
      count_information_json.emplace_back(
        json11::Json::object{
          {count_information_category_key, json11::Json(count_pair.first)},
          {count_information_value_key, json11::Json(count_pair.second)},
        });
    }
    result[count_information_key] = json11::Json(count_information_json);

    return result;
  }

  std::unique_ptr<CategoryFeatureEncoder> CategoryFeatureCountEncoder::RecoverFromModelStringInJsonFormat(json11::Json input)
  {
	  std::unordered_map<int, double> count_information;

	  std::vector<Json> count_information_json = input[count_information_key].array_items();
	  for (Json entry : count_information_json) {
		  int count_information_category = entry[count_information_category_key].int_value();
		  double count_information_value = entry[count_information_value_key].number_value();

		  count_information[count_information_category] = count_information_value;
	  }

	  return std::make_unique<CategoryFeatureCountEncoder>(CategoryFeatureCountEncoder(count_information));
  }
}
