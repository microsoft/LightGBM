/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once
#include <vector>

#include "Parser.h"

using namespace std;

class FeatureEvaluator {
 public:
  FeatureEvaluator(Parser *parser_interface, uint32_t input_id);
  vector<uint32_t> GetRequiredRawFeatureIndices();
  double Evaluate(uint32_t *input);
  bool IsRawFeatureEvaluator();

 private:
  uint32_t input_id_;
  Parser *parser_interface_;
  void *inputptr_;
};