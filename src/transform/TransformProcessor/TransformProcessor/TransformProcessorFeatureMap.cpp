/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "TransformProcessorFeatureMap.h"

FeatureMap::FeatureMap(Parser *parser_interface) {
  parser_interface_ = parser_interface;
}

string FeatureMap::GetRawFeatureName(int index) {
  return this->parser_interface_->GetFeatureName(index);
}