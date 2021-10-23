#pragma once
#include <vector>
#include <string>
#include "Parser.h"

using namespace std;


class FeatureMap {
 public:
  FeatureMap(Parser* parser_interface);
  string GetRawFeatureName(int index);
  ~FeatureMap(void);

 private:
   Parser* parser_interface_;
};