#include "TransformProcessorFeatureMap.h"


FeatureMap::FeatureMap(Parser* parser_interface) {
  parser_interface_ = parser_interface;
}

string FeatureMap::GetRawFeatureName(int index) {
  return this->parser_interface_->GetFeatureName(index);
}