/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#pragma once
#include <vector>

#include "FeatureEvaluator.h"
#include "IniFileParserInterface.h"
#include "TransformProcessorFeatureMap.h"

using namespace std;

class FeatureEvaluatorExtendedInfo {
 public:
  FeatureEvaluator *feature_evaluator_;
  FeatureMap *featuremap_;
  vector<int> required_feature_mapped_indexes_;
  int index_ = -1;
  bool israw_ = false;
};
