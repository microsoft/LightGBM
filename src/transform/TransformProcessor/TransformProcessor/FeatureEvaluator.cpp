/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FeatureEvaluator.h"

#include <algorithm>

FeatureEvaluator::FeatureEvaluator(Parser *parser_interface, uint32_t id) {
  parser_interface_ = parser_interface;
  input_id_ = id;
  inputptr_ = parser_interface_->GetInput((uint32_t)this->input_id_);
}

vector<uint32_t> FeatureEvaluator::GetRequiredRawFeatureIndices() {
  vector<uint32_t> features(parser_interface_->GetFeatureCount());
  uint32_t feature_count;
  this->parser_interface_->GetInputFeatures(
      this->inputptr_, reinterpret_cast<uint32_t *>(features.data()),
      (uint32_t)features.size(), &feature_count);
  std::vector<uint32_t> subvector(features.begin(),
                                  features.begin() + feature_count);
  sort(subvector.begin(), subvector.end());
  subvector.erase(unique(subvector.begin(), subvector.end()), subvector.end());
  return subvector;
}

bool FeatureEvaluator::IsRawFeatureEvaluator() {
  return this->parser_interface_->IsCopyInput(this->inputptr_);
}

double FeatureEvaluator::Evaluate(uint32_t *input) {
  uint32_t *input1 = input;
  return this->parser_interface_->EvaluateInput(this->inputptr_, input1);
}