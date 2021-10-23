#pragma once
#include <vector>
#include "IniFileParserInterface.h"
#include "FeatureEvaluator.h"
#include "TransformProcessorFeatureMap.h"

using namespace std;


class FeatureEvaluatorExtendedInfo {
 public:
  FeatureEvaluator* feature_evaluator_;
  FeatureMap* featuremap_;
  vector<int> required_feature_mapped_indexes_;
  int index_ = -1;
  bool israw_ = false;
};
