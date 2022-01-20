/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include "category_feature_encoder.hpp"

// property name keys
const std::string count_information_key = "count_information";
const std::string label_information_key = "label_information";
const std::string category_key = "cat";
const std::string value_key = "value";
const std::string count_prior_key = "prior";
const std::string count_prior_weight_key = "prior_weight";

namespace LightGBM {
  double CategoryFeatureTargetEncoder::Encode(double feature_value) {
    int category = static_cast<int>(feature_value);

    if (count_information_.find(category) != count_information_.end()) {
      return (label_information_[category] + prior_ * prior_weight_) / (count_information_[category] + prior_weight_);
    }

    return default_value;
  }

  json11::Json::object CategoryFeatureTargetEncoder::DumpToJsonObject()
  {
    json11::Json::object result = CategoryFeatureEncoder::DumpToJsonObject();
    result[count_prior_key] = json11::Json(prior_);
    result[count_prior_weight_key] = json11::Json(prior_weight_);

    json11::Json::array count_information_json;
    for (const auto& count_pair : count_information_) {
      count_information_json.emplace_back(
        json11::Json::object{
          {category_key, json11::Json(count_pair.first)},
          {value_key, json11::Json(count_pair.second)}
        });
    }
    result[count_information_key] = json11::Json(count_information_json);

	json11::Json::array label_information_json;
	for (const auto& count_pair : label_information_) {
		label_information_json.emplace_back(
			json11::Json::object{
				{ category_key, json11::Json(count_pair.first) },
				{ value_key, json11::Json(count_pair.second) }
		});
	}
	result[label_information_key] = json11::Json(label_information_json);

    return result;
  }

  std::unique_ptr<CategoryFeatureEncoder> CategoryFeatureTargetEncoder::RecoverFromModelStringInJsonFormat(json11::Json input)
  {
	  double prior = input[count_prior_key].number_value();
	  double prior_weight = input[count_prior_weight_key].number_value();
	  
	  std::unordered_map<int, int> count_information;
	  std::vector<Json> count_information_json = input[count_information_key].array_items();
	  for (Json entry : count_information_json) {
		  int category = entry[category_key].int_value();
		  int category_value = entry[value_key].int_value();

		  count_information[category] = category_value;
	  }

	  std::unordered_map<int, double> label_information;
	  std::vector<Json> label_information_json = input[label_information_key].array_items();
	  for (Json entry : label_information_json) {
		  int category = entry[category_key].int_value();
		  double category_value = entry[value_key].number_value();

		  label_information[category] = category_value;
	  }

	  return std::unique_ptr<CategoryFeatureEncoder>(new CategoryFeatureTargetEncoder(prior, prior_weight, count_information, label_information));
  }
}
