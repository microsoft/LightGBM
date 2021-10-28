/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once
#include <string>
#include <vector>

#include "Parser.h"

using namespace std;

class FeatureMap {
 public:
  FeatureMap(Parser *parser_interface);
  string GetRawFeatureName(int index);
  ~FeatureMap(void);

 private:
  Parser *parser_interface_;
};