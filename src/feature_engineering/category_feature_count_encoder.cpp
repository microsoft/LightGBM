/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include "encoder.hpp"

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
    result[encoder_type_key] = json11::Json(count_encoder_type);

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
}
