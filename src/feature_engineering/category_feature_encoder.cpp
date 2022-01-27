/*!
* Copyright (c) 2016 Microsoft Corporation. All rights reserved.
* Licensed under the MIT License. See LICENSE file in the project root for license information.
*/

#include "category_feature_encoder.hpp"
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/json11.h>

#include <memory>
#include <utility>

// property name keys
const char feature_name_key[] = "feature_name";
const char encoder_type_key[] = "encoder_type";

namespace LightGBM {
  json11::Json::object CategoryFeatureEncoder::DumpToJsonObject() {
    json11::Json::object result {
      { encoder_type_key, json11::Json(type_) },
      { feature_name_key, json11::Json(feature_name_) },
    };

    return result;
  }

  std::unique_ptr<CategoryFeatureEncoder> CategoryFeatureEncoder::RecoverFromModelStringInJsonFormat(json11::Json input) {
    int type = input[encoder_type_key].int_value();
    std::string feature_name = input[feature_name_key].string_value();
    std::unique_ptr<CategoryFeatureEncoder> result;

    if (type == CategoryFeatureCountEncoder::count_encoder_type) {
      result = std::move(CategoryFeatureCountEncoder::RecoverFromModelStringInJsonFormat(input));
    } else if (type == CategoryFeatureTargetEncoder::target_encoder_type) {
      result = std::move(CategoryFeatureTargetEncoder::RecoverFromModelStringInJsonFormat(input));
    } else {
      Log::Fatal("Unknown encoder type %d", type);
    }

    result->feature_name_ = feature_name;
    return result;
  }
}  // namespace LightGBM
